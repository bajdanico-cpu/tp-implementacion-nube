"""Identidad canónica de los equipos — la pieza que sostiene todos los joins.

El problema
-----------
Las tres fuentes nombran e identifican a los equipos distinto, y ninguna de las
formas obvias sirve como clave:

- El `id` de FPL **se reasigna cada temporada** (verificado: cambia en 18 de 27
  equipos). Usarlo para cruzar temporadas mezcla equipos entre sí.
- El nombre tampoco: football-data dice `Man United` / `Tottenham` /
  `Sheffield United` donde FPL dice `Man Utd` / `Spurs` / `Sheffield Utd`.
- Ni siquiera el nombre de FPL es estable consigo mismo: el código 40 aparece como
  `Ipswich` en 2024-25 y como `Ipswich Town` en 2026-27.

La solución
-----------
`short_name` de FPL (ARS, MUN, TOT…) resultó **100% estable** entre temporadas para
los 27 equipos observados, y el `code` numérico lo respalda. Ésa es la clave
canónica del proyecto.

Cualquier nombre que no resuelva hace fallar el build. Es a propósito: un equipo
sin mapear que pasa silenciosamente se convierte en filas perdidas en el join, y
eso aparece recién al final, disfrazado de "faltan datos".
"""

from __future__ import annotations

import io
import re

import pandas as pd

from common.config import CFG
from common.logging_setup import get_logger
from common.storage import read_raw

log = get_logger(__name__)


class UnmappedTeam(ValueError):
    """Un nombre de equipo que no se pudo resolver a la clave canónica."""


# Alias que ninguna normalización razonable resuelve: son nombres genuinamente
# distintos para el mismo club. Mapean nombre de football-data -> short_name FPL.
FD_ALIASES: dict[str, str] = {
    "Tottenham": "TOT",  # FPL lo llama "Spurs"
}

# Transfermarkt usa el nombre OFICIAL completo, que casi nunca coincide con el de FPL.
# Once de los veintisiete equipos de la ventana no resuelven sin esto, y la normalización no
# puede arreglarlo: `Manchester` vs `Man`, `Hotspur`, `& Hove Albion` y `Wanderers` no son
# sufijos genéricos, son parte del nombre.
#
# ⚠️ El cruce con esta fuente va por nombre EXACTO. En `player_valuations` conviven
# `Manchester City` con `Manchester City U21`, `... Reserves` y `... U18`, y —peor—
# `Newcastle United Jets`, que es un club **australiano**. Un `contains` mete filiales y
# homónimos en el plantel.
TM_ALIASES: dict[str, str] = {
    "AFC Bournemouth": "BOU",
    "Brighton & Hove Albion": "BHA",
    "Leeds United": "LEE",
    "Manchester City": "MCI",
    "Manchester United": "MUN",
    "Newcastle United": "NEW",
    "Nottingham Forest": "NFO",
    "Tottenham Hotspur": "TOT",
    "West Ham United": "WHU",
    "Wolverhampton Wanderers": "WOL",
}

# Sufijos genéricos que una fuente pone y la otra no ("Ipswich" vs "Ipswich Town").
# No se quitan a lo bruto: sólo se usan para un segundo intento de match, y si el
# resultado es ambiguo se rechaza.
_GENERIC_SUFFIXES = ("city", "town", "fc", "afc")


def normalize(name: str) -> str:
    """Forma comparable de un nombre de equipo.

    `Man Utd` y `Man United` colapsan al mismo string; `Man City` no colisiona con
    ninguno de los dos porque los sufijos genéricos se tratan aparte.
    """
    s = str(name).lower().strip()
    s = s.replace("'", "").replace(".", "").replace("-", " ")
    s = re.sub(r"\butd\b", "united", s)
    return re.sub(r"\s+", " ", s).strip()


def strip_generic(name: str) -> str:
    """Quita un sufijo genérico final. `ipswich town` -> `ipswich`."""
    parts = normalize(name).split()
    while len(parts) > 1 and parts[-1] in _GENERIC_SUFFIXES:
        parts.pop()
    return " ".join(parts)


# --------------------------------------------------------------------------- #
#  Registro canónico
# --------------------------------------------------------------------------- #


def build_registry() -> pd.DataFrame:
    """Registro canónico de equipos, a partir de los `teams.csv` de todas las temporadas.

    Devuelve: team_code | short_name | team_name | seasons | n_seasons
    """
    frames = []
    for season in CFG.seasons_to_ingest():
        raw = read_raw("vaastav", season, "raw", "teams.csv")
        if raw is None:
            log.warning("[%s] sin teams.csv en Bronze — se omite del registro", season)
            continue
        df = pd.read_csv(io.BytesIO(raw))
        df["season"] = season
        frames.append(df[["season", "code", "id", "name", "short_name"]])

    if not frames:
        raise FileNotFoundError(
            "No hay ningún teams.csv en Bronze. Corré `python -m ingestion.run` primero."
        )

    allteams = pd.concat(frames, ignore_index=True)

    # `short_name` tiene que ser único por `code`; si no, la clave canónica no sirve.
    inconsistent = allteams.groupby("code")["short_name"].nunique()
    bad = inconsistent[inconsistent > 1]
    if not bad.empty:
        raise UnmappedTeam(
            f"short_name dejó de ser estable para los codes {list(bad.index)}. "
            f"Hay que revisar la clave canónica antes de seguir."
        )

    registry = (
        allteams.groupby(["code", "short_name"], as_index=False)
        .agg(
            # El nombre más reciente gana: `Ipswich Town` en vez de `Ipswich`.
            team_name=("name", "last"),
            seasons=("season", lambda x: ",".join(sorted(set(x)))),
            n_seasons=("season", "nunique"),
        )
        .rename(columns={"code": "team_code"})
        .sort_values("short_name")
        .reset_index(drop=True)
    )
    return registry


def _lookup_tables(registry: pd.DataFrame) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Índices de búsqueda: normalizado -> short_name, y sin-sufijo -> [short_names]."""
    exact: dict[str, str] = {}
    stripped: dict[str, list[str]] = {}

    for _, row in registry.iterrows():
        short = row["short_name"]
        exact[normalize(row["team_name"])] = short
        # El propio short_name también es un alias válido ("TOT").
        exact[normalize(short)] = short
        stripped.setdefault(strip_generic(row["team_name"]), []).append(short)

    return exact, stripped


def resolve(name: str, registry: pd.DataFrame, aliases: dict[str, str] | None = None) -> str:
    """Resuelve un nombre de equipo (de cualquier fuente) a su `short_name` canónico.

    Orden: alias explícito -> match normalizado -> match sin sufijo genérico.
    Levanta `UnmappedTeam` si no resuelve o si el match es ambiguo.
    """
    aliases = aliases or FD_ALIASES
    if name in aliases:
        return aliases[name]

    exact, stripped = _lookup_tables(registry)

    key = normalize(name)
    if key in exact:
        return exact[key]

    candidates = stripped.get(strip_generic(name), [])
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise UnmappedTeam(
            f"'{name}' es ambiguo: podría ser {candidates}. Agregalo a FD_ALIASES."
        )

    raise UnmappedTeam(
        f"'{name}' no se pudo mapear a ningún equipo conocido. "
        f"Agregá una entrada a FD_ALIASES en transform/team_mapping.py."
    )


def resolve_many(
    names: list[str], registry: pd.DataFrame, aliases: dict[str, str] | None = None
) -> dict[str, str]:
    """Resuelve una lista de nombres. Falla informando TODOS los que no resolvieron,
    no sólo el primero — así se arregla el mapeo de una sola pasada."""
    resolved: dict[str, str] = {}
    failures: list[str] = []

    for name in sorted(set(names)):
        try:
            resolved[name] = resolve(name, registry, aliases)
        except UnmappedTeam as exc:
            failures.append(str(exc))

    if failures:
        raise UnmappedTeam(
            "No se pudieron mapear %d nombres de equipo:\n  - %s"
            % (len(failures), "\n  - ".join(failures))
        )
    return resolved


# --------------------------------------------------------------------------- #
#  dim_team
# --------------------------------------------------------------------------- #


def build_dim_team() -> pd.DataFrame:
    """Dimensión de equipos con grano `(season, team_code)`.

    Es una dimensión de cambio lento: el `fpl_team_id` y las `strength_*` cambian
    todas las temporadas, pero `team_code` / `short_name` son el hilo que las une.

    Columnas: season | team_code | short_name | team_name | fpl_team_id |
              fd_name | strength* | promoted
    """
    registry = build_registry()
    rows = []

    for season in CFG.seasons_to_ingest():
        raw = read_raw("vaastav", season, "raw", "teams.csv")
        if raw is None:
            continue
        teams = pd.read_csv(io.BytesIO(raw))

        # Nombres tal como los escribe football-data en esa temporada.
        fd_map: dict[str, str] = {}
        fd_raw = read_raw("football_data", season, "raw", "E0.csv")
        if fd_raw is not None:
            fd = pd.read_csv(io.BytesIO(fd_raw), encoding="utf-8-sig", encoding_errors="ignore")
            fd_names = sorted(set(fd["HomeTeam"].dropna()) | set(fd["AwayTeam"].dropna()))
            # short_name -> nombre football-data
            fd_map = {v: k for k, v in resolve_many(fd_names, registry).items()}

        for _, t in teams.iterrows():
            short = t["short_name"]
            rows.append({
                "season": season,
                "team_code": int(t["code"]),
                "short_name": short,
                "team_name": t["name"],
                "fpl_team_id": int(t["id"]),
                "fd_name": fd_map.get(short),
                "strength": t.get("strength"),
                "strength_overall_home": t.get("strength_overall_home"),
                "strength_overall_away": t.get("strength_overall_away"),
                "strength_attack_home": t.get("strength_attack_home"),
                "strength_attack_away": t.get("strength_attack_away"),
                "strength_defence_home": t.get("strength_defence_home"),
                "strength_defence_away": t.get("strength_defence_away"),
            })

    dim = pd.DataFrame(rows)

    # `promoted`: primera temporada del equipo dentro de la ventana ingestada.
    # Es un proxy imperfecto (un equipo puede haber estado en Premier antes de la
    # ventana) pero marca justamente los casos de cold-start que importan.
    first_season = dim.groupby("team_code")["season"].min()
    dim["promoted"] = dim.apply(
        lambda r: r["season"] == first_season[r["team_code"]] and r["season"] != CFG.ingest_from,
        axis=1,
    )

    return dim.sort_values(["season", "short_name"]).reset_index(drop=True)


def report() -> None:
    """Imprime el estado del mapeo. Sirve como chequeo manual rápido."""
    registry = build_registry()
    print(f"Registro canónico: {len(registry)} equipos\n")
    print(registry.to_string(index=False))

    dim = build_dim_team()
    print(f"\n\ndim_team: {len(dim)} filas ({dim['season'].nunique()} temporadas)")

    sin_fd = dim[dim["fd_name"].isna()]
    if not sin_fd.empty:
        print(f"\nEquipos sin nombre de football-data ({len(sin_fd)}):")
        print(sin_fd[["season", "short_name", "team_name"]].to_string(index=False))
        print("  (esperado si esa temporada todavía no publicó su E0.csv)")

    distinto = dim[dim["fd_name"].notna() & (dim["fd_name"] != dim["team_name"])]
    if not distinto.empty:
        print("\nNombres que difieren entre fuentes:")
        print(distinto[["season", "short_name", "team_name", "fd_name"]]
              .drop_duplicates(["short_name", "team_name", "fd_name"])
              .to_string(index=False))

    promo = dim[dim["promoted"]]
    if not promo.empty:
        print("\nEquipos que aparecen por primera vez (cold-start):")
        print(promo[["season", "short_name", "team_name"]].to_string(index=False))


if __name__ == "__main__":
    report()
