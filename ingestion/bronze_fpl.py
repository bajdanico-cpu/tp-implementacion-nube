"""Bronze — snapshots crudos de la API oficial de FPL.

Esta fuente no sirve para entrenar (el histórico está en vaastav): sirve para
**operar**. Es la que da los fixtures de la fecha que viene, los deadlines y el
resultado apenas termina el partido.

Por qué cada corrida escribe un snapshot nuevo
----------------------------------------------
El contenido de `/bootstrap-static/` cambia a lo largo de la semana: el `status` de
un jugador puede pasar de "duda" a "lesionado" el viernes a la tarde. El snapshot
tomado ANTES del deadline es el único que refleja lo que se sabía al momento de
predecir; el tomado después ya está contaminado.

Guardar los dos, particionados por `ingested_at`, es lo que permite demostrar
después que la predicción no usó información del futuro. Por eso Bronze es
append-only y esta fuente nunca se cachea.
"""

from __future__ import annotations

import json

from common.config import CFG, utc_stamp
from common.logging_setup import get_logger
from common.storage import sha256, write_manifest, write_raw
from ingestion.fpl_client import FPLClient, FPLClientError

log = get_logger(__name__)

SOURCE = "fpl"


def _snapshot(season: str, dataset: str, filename: str, content: bytes, stamp: str) -> dict:
    write_raw(SOURCE, season, dataset, filename, content, stamp=stamp)
    entry = {"file": filename, "bytes": len(content), "sha256": sha256(content)}
    write_manifest(SOURCE, season, dataset, [entry], stamp)
    return entry


def ingest_bootstrap(client: FPLClient, season: str, stamp: str) -> dict:
    """`/bootstrap-static/` — equipos, jugadores, disponibilidad y deadlines."""
    content, payload = client.bootstrap_static()
    _snapshot(season, "bootstrap", "bootstrap_static.json", content, stamp)

    current, nxt = client.current_and_next_gw(payload)
    n_teams = len(payload.get("teams", []))
    n_players = len(payload.get("elements", []))
    deadlines = client.deadlines(payload)

    log.info("bootstrap: %d equipos, %d jugadores, %d gameweeks (%.1f KB)",
             n_teams, n_players, len(deadlines), len(content) / 1024)
    log.info("gameweek en curso: %s | próxima: %s", current or "ninguna (pretemporada)", nxt)
    if nxt and nxt in deadlines:
        log.info("deadline de la GW%s: %s", nxt, deadlines[nxt])

    return {"current_gw": current, "next_gw": nxt, "n_teams": n_teams,
            "n_players": n_players, "deadlines": deadlines}


def ingest_fixtures(client: FPLClient, season: str, stamp: str) -> dict:
    """`/fixtures/` — los 380 partidos, con FDR y (una vez jugados) el marcador."""
    content, payload = client.fixtures()
    _snapshot(season, "fixtures", "fixtures.json", content, stamp)

    played = sum(1 for f in payload if f.get("finished"))
    log.info("fixtures: %d totales, %d jugados (%.1f KB)",
             len(payload), played, len(content) / 1024)
    return {"n_fixtures": len(payload), "n_finished": played}


def ingest_event_live(client: FPLClient, season: str, gameweek: int, stamp: str) -> dict:
    """`/event/{GW}/live/` — el ground truth automático de una fecha.

    Antes de que se juegue devuelve los elementos en cero. No es error: se guarda
    igual y queda registrado en el manifest cuántos jugadores tenían minutos.
    """
    try:
        content, payload = client.event_live(gameweek)
    except FPLClientError as exc:
        log.warning("event/%d/live no disponible: %s", gameweek, exc)
        return {"gameweek": gameweek, "available": False, "error": str(exc)}

    _snapshot(season, f"event_live_gw{gameweek}", f"live_gw{gameweek}.json", content, stamp)

    elements = payload.get("elements", [])
    with_minutes = sum(1 for e in elements if e.get("stats", {}).get("minutes", 0) > 0)
    if with_minutes == 0:
        log.info("event/%d/live: %d jugadores, ninguno con minutos "
                 "(la fecha todavía no se jugó)", gameweek, len(elements))
    else:
        log.info("event/%d/live: %d jugadores, %d con minutos",
                 gameweek, len(elements), with_minutes)

    return {"gameweek": gameweek, "available": True,
            "n_elements": len(elements), "n_with_minutes": with_minutes}


def run(gameweek: int | None = None, season: str | None = None, force: bool = False) -> dict:
    """Snapshot completo de la API. `force` no aplica: esta fuente nunca se cachea."""
    season = season or CFG.current_season
    stamp = utc_stamp()
    client = FPLClient()

    log.info("=== Bronze FPL API — temporada %s, snapshot %s ===", season, stamp)

    boot = ingest_bootstrap(client, season, stamp)
    fixtures = ingest_fixtures(client, season, stamp)

    # Si no se pide una GW puntual, se intenta la que esté en curso.
    target_gw = gameweek if gameweek is not None else boot.get("current_gw")
    live = None
    if target_gw is not None:
        live = ingest_event_live(client, season, target_gw, stamp)
    else:
        log.info("no hay gameweek en curso — se omite event/live "
                 "(usar --gw N para forzar una fecha puntual)")

    return {"stamp": stamp, "bootstrap": boot, "fixtures": fixtures, "live": live}


if __name__ == "__main__":
    result = run()
    print(json.dumps({k: v for k, v in result.items() if k != "bootstrap"}, indent=2))
