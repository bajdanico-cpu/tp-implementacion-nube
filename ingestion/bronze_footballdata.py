"""Bronze — resultados y cuotas de football-data.co.uk.

Es el spine a nivel partido y, sobre todo, **la única fuente de las cuotas de
cierre**, que son el baseline duro contra el que se mide el modelo (~53-55%).

Trampa verificada el 17/08/2026
-------------------------------
El URL de una temporada que todavía no arrancó **no devuelve 404**: hace un 301 a
otra división. `mmz4281/2627/E0.csv` servía en ese momento partidos de la National
League (Altrincham, Southend, Boreham Wood…) con status 200.

Sin validar el contenido, esas filas entran a Bronze en silencio y aparecen mucho
más tarde como equipos imposibles de mapear. Por eso el guard de `Div` no es
opcional: es la diferencia entre fallar ruidosamente ahora y depurar datos raros
tres capas más adelante.
"""

from __future__ import annotations

import io

import pandas as pd

from common.config import CFG, season_to_football_data, utc_stamp
from common.logging_setup import get_logger
from common.storage import latest_snapshot, sha256, write_manifest, write_raw
from ingestion.http_utils import fetch

log = get_logger(__name__)

SOURCE = "football_data"
DATASET = "raw"
FILENAME = "E0.csv"


class DivisionMismatch(ValueError):
    """El archivo descargado no es de la división esperada."""


def _read_csv(content: bytes) -> pd.DataFrame:
    """Parsea el CSV. `utf-8-sig` es necesario: los archivos traen BOM y sin esto
    la primera columna queda llamada '﻿Div' y ninguna validación la encuentra."""
    return pd.read_csv(io.BytesIO(content), encoding="utf-8-sig", encoding_errors="ignore")


def validate_division(df: pd.DataFrame, season: str, expected: str = "E0") -> None:
    """Verifica que el archivo sea realmente de la división pedida.

    Levanta `DivisionMismatch` si no lo es. Es el guard descrito arriba.
    """
    if "Div" not in df.columns:
        raise DivisionMismatch(
            f"[{season}] el CSV no tiene columna 'Div'. Columnas: {list(df.columns)[:8]}"
        )

    found = set(df["Div"].dropna().unique())
    if found != {expected}:
        raise DivisionMismatch(
            f"[{season}] se esperaba Div='{expected}' y se encontró {sorted(found)}. "
            f"Casi seguro el URL redirigió a otra división porque la temporada "
            f"todavía no arrancó."
        )


def ingest_season(season: str, stamp: str, force: bool = False) -> dict:
    cfg = CFG.football_data
    fd_season = season_to_football_data(season)
    division = cfg["division"]
    url = f"{cfg['base_url'].rstrip('/')}/{fd_season}/{division}.csv"

    is_current = season == CFG.current_season
    if not force and not is_current and latest_snapshot(SOURCE, season, DATASET):
        log.info("[%s] ya está en Bronze y es temporada cerrada — se omite (usar --force)", season)
        return {"season": season, "skipped": True}

    res = fetch(url, timeout=cfg["timeout_s"])

    if not res.ok or not res.content:
        # Que la temporada en curso todavía no tenga archivo es lo normal hasta que
        # se juega la primera fecha: no es un fallo del pipeline.
        not_published_yet = is_current and (res.status == 404 or 300 <= res.status < 400)
        if not_published_yet:
            log.info("[%s] todavía no publicado (HTTP %s) — la temporada no arrancó",
                     season, res.status)
        else:
            log.warning("[%s] %s -> %s (%s)", season, url, res.status, res.error)

        entry = {"file": FILENAME, "url": url, "status": res.status,
                 "error": res.error, "rejected": True, "expected": not_published_yet}
        write_manifest(SOURCE, season, DATASET, [entry], stamp)
        return {"season": season, "skipped": False, "rejected": True}

    if res.redirected:
        log.warning("[%s] el request redirigió: %s -> %s", season, url, res.final_url)

    df = _read_csv(res.content)

    if df.empty:
        log.info("[%s] archivo vacío — la temporada todavía no tiene partidos jugados", season)
        entry = {"file": FILENAME, "url": url, "status": 200, "rows": 0,
                 "rejected": True, "reason": "sin partidos jugados"}
        write_manifest(SOURCE, season, DATASET, [entry], stamp)
        return {"season": season, "skipped": False, "rejected": True}

    if cfg.get("validate_division", True):
        try:
            validate_division(df, season, expected=division)
        except DivisionMismatch as exc:
            # No es un error del pipeline: es la fuente sirviendo otra cosa.
            # Se rechaza, se deja constancia y se sigue con las demás temporadas.
            log.warning("[%s] RECHAZADO — %s", season, exc)
            entry = {"file": FILENAME, "url": url, "status": 200, "rows": len(df),
                     "rejected": True, "reason": str(exc), "final_url": res.final_url}
            write_manifest(SOURCE, season, DATASET, [entry], stamp)
            return {"season": season, "skipped": False, "rejected": True}

    write_raw(SOURCE, season, DATASET, FILENAME, res.content, stamp=stamp)
    entry = {
        "file": FILENAME, "url": url, "status": 200,
        "bytes": len(res.content), "sha256": sha256(res.content),
        "rows": len(df), "cols": len(df.columns), "rejected": False,
    }
    write_manifest(SOURCE, season, DATASET, [entry], stamp)
    log.info("[%s] %s -> %d partidos, %d columnas (%.1f KB)",
             season, division, len(df), len(df.columns), len(res.content) / 1024)
    return {"season": season, "skipped": False, "rejected": False, "rows": len(df)}


def run(seasons: list[str] | None = None, force: bool = False) -> None:
    seasons = seasons or CFG.seasons_to_ingest()
    stamp = utc_stamp()
    log.info("=== Bronze football-data — temporadas %s ===", seasons)
    for season in seasons:
        ingest_season(season, stamp, force=force)


if __name__ == "__main__":
    run()
