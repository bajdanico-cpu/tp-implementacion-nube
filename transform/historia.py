"""Silver — `fact_match_historico`: veinte años de resultados crudos, sólo para el rating.

    python -m transform.historia

Una fila por partido de E0/E1/E2 desde `historia.desde`. Sin cuotas, sin estadísticas, sin
xG: **fecha, dos equipos y un marcador**, que es todo lo que un Elo consume.

## La identidad de los equipos acá es otra

El resto del pipeline usa `short_name` de FPL (ARS, MUN, TOT) como clave canónica, y el
registro que la resuelve se construye con los `teams.csv` de las temporadas ingestadas: sabe
de 27 equipos. Esta tabla tiene **cientos** — todo el que pasó por las tres divisiones desde
2000, incluidos clubes que nunca estuvieron en la Premier.

Pedirles `short_name` a todos sería absurdo y haría fallar el build. Así que acá la clave es
`team_key`: el nombre de football-data **normalizado** con `team_mapping.normalize`, que ya
resuelve `Man Utd` ↔ `Man United`. Es una clave interna del rating, no canónica.

El puente entre los dos mundos es `short_name`, que se completa **sólo** donde el registro
lo resuelve. Así el Elo puede calcularse sobre las tres divisiones y después entregarle a
Gold el rating de los 27 equipos que le importan.

## Y por qué se guarda la división

Porque la separación entre divisiones no se calibra a mano: emerge de los ascensos y
descensos. El rating necesita saber en qué división juega cada equipo cada año para regresar
a la media **de su división** entre temporadas, que es lo que evita que la regresión aplaste
la diferencia que tanto costó estimar.
"""

from __future__ import annotations

import io

import pandas as pd

from common.config import CFG
from common.logging_setup import get_logger, setup
from common.storage import read_raw, read_table, write_table
from transform import team_mapping

log = get_logger(__name__)

TABLA = "fact_match_historico"
DATASET = "historia"

RENOMBRE = {
    "Div": "division", "Date": "match_date",
    "HomeTeam": "fd_home", "AwayTeam": "fd_away",
    "FTHG": "home_goals", "FTAG": "away_goals", "FTR": "result",
}


def _leer(season: str, division: str) -> pd.DataFrame | None:
    raw = read_raw("football_data", season, DATASET, f"{division}.csv")
    if raw is None:
        return None
    df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig",
                     encoding_errors="ignore", on_bad_lines="skip")
    faltan = [c for c in RENOMBRE if c not in df.columns]
    if faltan:
        log.warning("[%s %s] faltan columnas %s — se omite", season, division, faltan)
        return None
    d = df[list(RENOMBRE)].rename(columns=RENOMBRE)
    d["season"] = season
    return d


def construir() -> pd.DataFrame:
    frames = []
    for season in CFG.seasons_historia():
        for division in CFG.divisiones_historia:
            d = _leer(season, division)
            if d is not None:
                frames.append(d)

    if not frames:
        raise FileNotFoundError(
            "No hay nada en Bronze historia. Corre `python -m ingestion.bronze_fd_historia`.")

    d = pd.concat(frames, ignore_index=True)

    # football-data usa dd/mm/yyyy, con año de 2 y de 4 digitos segun la temporada.
    # `dayfirst` cubre las dos sin tener que ramificar por año.
    d["match_date"] = pd.to_datetime(d["match_date"], dayfirst=True, errors="coerce")

    antes = len(d)
    d = d.dropna(subset=["match_date", "fd_home", "fd_away", "home_goals", "away_goals"])
    if len(d) < antes:
        log.info("Descartadas %d filas sin fecha o sin marcador", antes - len(d))

    d["home_goals"] = d["home_goals"].astype(int)
    d["away_goals"] = d["away_goals"].astype(int)
    d["target_1x2"] = d["result"].map({"H": "home", "D": "draw", "A": "away"})

    # La clave interna del rating: el nombre normalizado. No es canonica y no pretende
    # serlo -- hay cientos de clubes aca y el registro conoce 27.
    d["home_key"] = d["fd_home"].map(team_mapping.normalize)
    d["away_key"] = d["fd_away"].map(team_mapping.normalize)

    # El puente al mundo canonico, solo donde se puede.
    puente = _puente(sorted(set(d["fd_home"]) | set(d["fd_away"])))
    d["home_short"] = d["fd_home"].map(puente)
    d["away_short"] = d["fd_away"].map(puente)

    d = d.sort_values(["match_date", "division", "fd_home"]).reset_index(drop=True)
    return d[["season", "division", "match_date", "fd_home", "fd_away",
              "home_key", "away_key", "home_short", "away_short",
              "home_goals", "away_goals", "result", "target_1x2"]]


def _puente(nombres: list[str]) -> dict[str, str]:
    """Nombre de football-data -> `short_name` canonico, donde el registro lo resuelva.

    Usa `team_mapping.resolve`, que es el resolvedor del proyecto, **no** un match exacto
    contra el nombre del registro. La diferencia no es cosmetica y costo un bug: la Premier
    de 2026-27 tiene a Coventry, Hull e Ipswich, football-data los llama `Coventry`, `Hull`
    e `Ipswich` y el registro `Coventry City`, `Hull City` e `Ipswich Town`. Con match
    exacto los tres quedaban **sin una sola fila de historia** -- justo los tres ascendidos,
    que son la razon de ser de toda esta fase.

    Se resuelve nombre por nombre y **sin levantar**: que un club del League One no tenga
    `short_name` es lo normal, no un error. El build sigue fallando ruidosamente donde
    corresponde, en `fact_match`, que es la tabla canonica.
    """
    registry = team_mapping.build_registry()
    puente: dict[str, str] = {}
    for nombre in sorted(set(nombres)):
        try:
            puente[nombre] = team_mapping.resolve(nombre, registry)
        except team_mapping.UnmappedTeam:
            continue          # club de ascenso que nunca paso por la ventana: esperado
    return puente


def equipos_sin_historia(d: pd.DataFrame, esperados: list[str]) -> list[str]:
    """Equipos canonicos que no aparecen ni una vez en la historia.

    Es el control que faltaba: un equipo sin historia arranca el Elo en 1500 y anula, en
    silencio, exactamente el problema que la Fase 1 vino a resolver.
    """
    presentes = set(d["home_short"].dropna()) | set(d["away_short"].dropna())
    return sorted(set(esperados) - presentes)


def run(escribir: bool = True) -> pd.DataFrame:
    d = construir()

    # Control: todo equipo de la ventana tiene que tener historia. Si alguno no la tiene,
    # su Elo arranca en 1500 y la fase no sirvio para el, en silencio.
    try:
        m = read_table("fact_match")
        faltan = equipos_sin_historia(d, sorted(set(m["home_short"]) | set(m["away_short"])))
        if faltan:
            log.warning("SIN historia y son de la ventana: %s. "
                        "Su Elo va a arrancar en 1500 -- revisar el puente de nombres.",
                        faltan)
        else:
            log.info("Los %d equipos de la ventana tienen historia.",
                     len(set(m["home_short"]) | set(m["away_short"])))
    except FileNotFoundError:
        pass                      # sin Silver todavia: el control no aplica

    if escribir:
        write_table(d, TABLA, layer="silver")
    return d


def main() -> None:
    setup(CFG.log_level, CFG.log_format)
    d = run()
    print(f"\n{TABLA}: {len(d):,} partidos, "
          f"{d['match_date'].min().date()} a {d['match_date'].max().date()}")
    print(f"\nPor division:\n{d['division'].value_counts().sort_index().to_string()}")
    print(f"\nEquipos distintos: {len(set(d['home_key']) | set(d['away_key']))}")
    con_short = d["home_short"].notna().sum() + d["away_short"].notna().sum()
    print(f"Filas-equipo con short_name canonico: {con_short:,} de {2 * len(d):,} "
          f"({con_short / (2 * len(d)):.1%})")


if __name__ == "__main__":
    main()
