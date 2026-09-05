"""Bronze — historia profunda de football-data: resultados crudos de E0, E1 y E2.

    python -m ingestion.bronze_fd_historia
    python -m ingestion.bronze_fd_historia --desde 2010-11 --force

## Qué es y qué NO es

**No entra al entrenamiento.** Un partido de 2005 no tiene xG, ni datos de FPL, ni nada de
lo que el modelo usa; y el fútbol de entonces no predice el de hoy. Usar estas filas como
objetivo de entrenamiento sería exactamente el error que `docs/PLAN-MEJORAS.md` descarta.

Lo que sí aporta es lo único que un **rating** necesita y hoy no tiene: partidos anteriores.

1. **Burn-in.** El Elo arranca a todos en 1500 en 2022-23 y tarda media temporada en
   converger. O sea que la primera temporada del train se entrena con una feature que
   todavía no significa nada, y `dif_elo` es la feature más importante del modelo.
2. **Los ascendidos.** Con E1 y E2, un equipo que sube entra con el rating que se ganó en
   el Championship en lugar de 1500. Es el fallo medido en la GW1 de 2026-27: los dos
   errores más caros fueron ascendidos ganando de local (HUL-MUN, IPS-SUN).

## Por qué tres divisiones y por qué tan atrás

El Elo es **uno solo y global**: no hay offset de división puesto a mano. La separación
entre divisiones **emerge** de los ascensos y descensos, que son los únicos eventos que
conectan los grupos — dentro de una división el Elo es de suma cero y nunca sabría que la
Premier es mejor que el Championship.

Se mueven 3+3 equipos por año, así que estimar bien esa separación necesita muchas
temporadas. De ahí `desde: "2000-01"`: son ~80 archivos de 150 KB, se bajan una vez y no
cambian nunca más.

## Bronze aparte

Va a `football_data/<season>/historia/`, no a `.../raw/`. El dataset `raw` es el E0 de la
ventana ingestada, con cuotas, que alimenta `fact_match` y el baseline del mercado; tocarlo
para meter 20 años de partidos sin cuotas rompería ese contrato. Son dos cosas distintas
que comparten fuente.

Las temporadas de la ventana normal quedan duplicadas entre `raw` y `historia` a propósito:
`historia` se lee entera y sola, sin tener que saber dónde termina un dataset y empieza el
otro.
"""

from __future__ import annotations

import argparse
import io
import time

import pandas as pd

from common.config import CFG, season_to_football_data, utc_stamp
from common.logging_setup import get_logger, setup
from common.storage import latest_snapshot, sha256, write_manifest, write_raw
from ingestion.bronze_footballdata import DivisionMismatch, validate_division
from ingestion.http_utils import fetch

log = get_logger(__name__)

SOURCE = "football_data"
DATASET = "historia"
PAUSA_S = 0.4

# Lo único que se le pide a estas filas. Las cuotas y las estadísticas de tiros existen
# sólo en las temporadas recientes, y el rating no las usa: pedirlas obligaría a tratar
# como error una ausencia que es normal.
COLUMNAS = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]


def _read_csv(content: bytes) -> pd.DataFrame:
    """`utf-8-sig` porque los archivos traen BOM, y `on_bad_lines='skip'` porque algunas
    temporadas viejas tienen filas de basura al final (separadores sueltos)."""
    return pd.read_csv(io.BytesIO(content), encoding="utf-8-sig",
                       encoding_errors="ignore", on_bad_lines="skip")


def _utilizable(df: pd.DataFrame, season: str, division: str) -> pd.DataFrame:
    """Filas con lo mínimo para un rating: dos equipos y un marcador."""
    faltan = [c for c in COLUMNAS if c not in df.columns]
    if faltan:
        raise DivisionMismatch(
            f"[{season} {division}] al CSV le faltan columnas basicas: {faltan}")
    d = df[COLUMNAS].dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    return d


def ingest(season: str, division: str, stamp: str, force: bool = False) -> dict:
    cfg = CFG.football_data
    url = f"{cfg['base_url'].rstrip('/')}/{season_to_football_data(season)}/{division}.csv"
    filename = f"{division}.csv"

    # Las temporadas cerradas no cambian nunca: una vez bajadas, no se vuelven a pedir.
    # La actual si, porque se completa fecha a fecha.
    es_actual = season == CFG.current_season
    snap = latest_snapshot(SOURCE, season, DATASET)
    if not force and not es_actual and snap and (snap / filename).exists():
        return {"season": season, "division": division, "skipped": True}

    res = fetch(url, timeout=cfg["timeout_s"])
    if not res.ok or not res.content:
        nivel = log.info if es_actual else log.warning
        nivel("[%s %s] no disponible (HTTP %s)", season, division, res.status)
        return {"season": season, "division": division, "rejected": True,
                "status": res.status}

    df = _read_csv(res.content)
    if df.empty:
        log.info("[%s %s] archivo vacio — la temporada no arranco", season, division)
        return {"season": season, "division": division, "rejected": True, "rows": 0}

    try:
        # El mismo guard que el ingestor principal: un URL inexistente sirve OTRA
        # division con status 200, y sin esto entran datos ajenos en silencio.
        validate_division(df, season, expected=division)
        d = _utilizable(df, season, division)
    except DivisionMismatch as exc:
        log.warning("[%s %s] RECHAZADO — %s", season, division, exc)
        return {"season": season, "division": division, "rejected": True,
                "reason": str(exc)}

    write_raw(SOURCE, season, DATASET, filename, res.content, stamp=stamp)
    log.info("[%s %s] %d partidos utilizables de %d filas (%.0f KB)",
             season, division, len(d), len(df), len(res.content) / 1024)
    return {"season": season, "division": division, "rejected": False,
            "rows": len(d), "bytes": len(res.content), "sha256": sha256(res.content)}


def run(desde: str | None = None, divisiones: list[str] | None = None,
        force: bool = False) -> pd.DataFrame:
    seasons = CFG.seasons_historia()
    if desde:
        seasons = [s for s in seasons if s >= desde]
    divisiones = divisiones or CFG.divisiones_historia
    if not seasons:
        log.warning("No hay temporadas de historia configuradas "
                    "(sources.football_data.historia.desde)")
        return pd.DataFrame()

    stamp = utc_stamp()
    log.info("=== Bronze football-data HISTORIA — %d temporadas x %d divisiones ===",
             len(seasons), len(divisiones))

    filas = []
    for season in seasons:
        entradas = []
        for division in divisiones:
            r = ingest(season, division, stamp, force=force)
            filas.append(r)
            if not r.get("skipped"):
                entradas.append(r)
                time.sleep(PAUSA_S)
        if entradas:
            write_manifest(SOURCE, season, DATASET, entradas, stamp)

    df = pd.DataFrame(filas)
    # `df.get(col, False)` devuelve un **bool** cuando la columna no existe, no una Serie.
    # En la primera corrida no hay ningun `skipped`, asi que la columna falta y encadenar
    # `.fillna()` revienta. Se normaliza a Serie antes de contar.
    def _flag(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        return df[col].fillna(False).astype(bool)

    saltados, rechazados = _flag("skipped"), _flag("rejected")
    log.info("Historia: %d archivos nuevos, %d ya estaban, %d rechazados",
             int((~saltados & ~rechazados).sum()), int(saltados.sum()),
             int(rechazados.sum()))
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Historia profunda de football-data.")
    ap.add_argument("--desde", default=None, help='p.ej. "2010-11"')
    ap.add_argument("--divisiones", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    setup(CFG.log_level, CFG.log_format)
    df = run(desde=args.desde, divisiones=args.divisiones, force=args.force)
    if not df.empty and "rows" in df:
        total = int(df["rows"].fillna(0).sum())
        print(f"\n{total:,} partidos utilizables en Bronze historia.")


if __name__ == "__main__":
    main()
