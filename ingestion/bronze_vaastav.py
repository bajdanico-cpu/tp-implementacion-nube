"""Bronze — histórico de FPL desde el repo de vaastav.

Es la ÚNICA fuente del detalle jugador-por-fecha del pasado. La API oficial no lo
sirve: `/element-summary/{id}/` devuelve `history` (sólo temporada actual) y
`history_past` (una fila por temporada, totales). vaastav archiva semanalmente el
`history` de cada jugador, y por eso el pasado sólo existe acá.

Nota sobre `2026-27`: la carpeta existe con `teams.csv`, `players_raw.csv` y
`fixtures.csv` (snapshot de pretemporada) pero **sin `gws/`**, porque todavía no se
jugó ninguna fecha. Ese 404 es estado normal y no debe romper la corrida.
"""

from __future__ import annotations

from common.config import CFG, utc_stamp
from common.logging_setup import get_logger
from common.storage import latest_snapshot, sha256, write_manifest, write_raw
from ingestion.http_utils import fetch

log = get_logger(__name__)

SOURCE = "vaastav"
DATASET = "raw"


def _filename_for(remote_path: str) -> str:
    """`gws/merged_gw.csv` -> `merged_gw.csv` (Bronze guarda plano, sin subcarpetas)."""
    return remote_path.split("/")[-1]


def ingest_season(season: str, stamp: str, force: bool = False) -> dict:
    """Baja los archivos de una temporada. Devuelve el resumen para el manifest."""
    cfg = CFG.vaastav
    base = cfg["base_url"].rstrip("/")
    timeout = cfg["timeout_s"]

    # Las temporadas cerradas no cambian: si ya están, no se vuelven a bajar.
    # La actual sí, porque se actualiza cada fecha.
    is_current = season == CFG.current_season
    if not force and not is_current and latest_snapshot(SOURCE, season, DATASET):
        log.info("[%s] ya está en Bronze y es temporada cerrada — se omite (usar --force)", season)
        return {"season": season, "skipped": True, "entries": []}

    entries: list[dict] = []
    for remote in cfg["season_files"]:
        url = f"{base}/{season}/{remote}"
        res = fetch(url, timeout=timeout)

        if res.ok and res.content:
            fname = _filename_for(remote)
            write_raw(SOURCE, season, DATASET, fname, res.content, stamp=stamp)
            entries.append({
                "file": remote,
                "url": url,
                "status": 200,
                "bytes": len(res.content),
                "sha256": sha256(res.content),
            })
            log.info("[%s] %-22s %7.1f KB", season, remote, len(res.content) / 1024)

        elif res.status == 404:
            # Esperado para `gws/merged_gw.csv` de una temporada que no arrancó.
            expected = remote.startswith("gws/") and is_current
            level = log.info if expected else log.warning
            level(
                "[%s] %-22s 404 %s",
                season, remote,
                "(esperado: la temporada todavía no arrancó)" if expected else "NO ENCONTRADO",
            )
            entries.append({
                "file": remote, "url": url, "status": 404,
                "expected": expected, "error": "not found",
            })

        else:
            log.error("[%s] %-22s FALLÓ: %s", season, remote, res.error)
            entries.append({
                "file": remote, "url": url, "status": res.status, "error": res.error,
            })

    write_manifest(SOURCE, season, DATASET, entries, stamp)
    return {"season": season, "skipped": False, "entries": entries}


def ingest_global(stamp: str, force: bool = False) -> dict:
    """Archivos que no dependen de temporada.

    `master_team_list.csv` mapea id de equipo FPL -> nombre, pero **sólo llega hasta
    2023-24**. De 2024-25 en adelante el mapeo sale del `teams.csv` de cada temporada.
    """
    cfg = CFG.vaastav
    base = cfg["base_url"].rstrip("/")
    season = "_global"

    if not force and latest_snapshot(SOURCE, season, DATASET):
        log.info("[_global] ya está en Bronze — se omite (usar --force)")
        return {"season": season, "skipped": True, "entries": []}

    entries: list[dict] = []
    for remote in cfg["global_files"]:
        url = f"{base}/{remote}"
        res = fetch(url, timeout=cfg["timeout_s"])
        if res.ok and res.content:
            write_raw(SOURCE, season, DATASET, _filename_for(remote), res.content, stamp=stamp)
            entries.append({
                "file": remote, "url": url, "status": 200,
                "bytes": len(res.content), "sha256": sha256(res.content),
            })
            log.info("[_global] %-22s %7.1f KB", remote, len(res.content) / 1024)
        else:
            log.error("[_global] %-22s FALLÓ: %s", remote, res.error)
            entries.append({"file": remote, "url": url, "status": res.status, "error": res.error})

    write_manifest(SOURCE, season, DATASET, entries, stamp)
    return {"season": season, "skipped": False, "entries": entries}


def run(seasons: list[str] | None = None, force: bool = False) -> None:
    seasons = seasons or CFG.seasons_to_ingest()
    stamp = utc_stamp()
    log.info("=== Bronze vaastav — temporadas %s ===", seasons)

    ingest_global(stamp, force=force)
    for season in seasons:
        ingest_season(season, stamp, force=force)


if __name__ == "__main__":
    run()
