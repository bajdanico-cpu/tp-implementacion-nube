"""Bronze de la API oficial de la Premier: fixtures y estadísticas de TODAS las competencias.

`footballapi.pulselive.com` es la API pública que alimenta premierleague.com. Es gratuita,
no pide clave y sólo exige el header `Origin`. Resuelve de una vez dos pendientes que
ninguna otra fuente cubría:

**1 · Los partidos de copa y de Europa.** El pipeline sólo veía la Premier, así que un
equipo que sigue en Champions y en la Copa de la Liga aparecía con la misma carga que uno
que sólo juega liga. Medido sobre 2025-26, los equipos de Premier jugaron **36 partidos de
EFL Cup, 43 de FA Cup y 69 de Champions**: 148 partidos por temporada que el modelo no veía.

**2 · Estadísticas de Opta por partido.** `/stats/match/{id}` devuelve entre 120 y 187
estadísticas por equipo, verificadas en las cuatro temporadas de entrenamiento y también en
copas. Trae tres grupos que no teníamos: ubicación del tiro (`attempts_ibox` / `_obox`),
defensivas reales (`total_tackle`, `interception`, `total_clearance`) y dominio territorial
(`possession_percentage`, `touches_in_opp_box`). No trae xG: eso Opta lo licencia aparte.

**Frescura, medida el 25/08/2026.** Los diez partidos de la fecha 1 tenían estadísticas
completas pocas horas después del último — antes incluso de que FPL marcara la fecha como
`finished`. Es la fuente más fresca del proyecto.

⚠️ **Lo que esta fuente NO permite.** El calendario de copa sólo llega hasta la ronda en
curso: las rondas se sortean al terminar la anterior y están espaciadas 20-67 días. Al
25/08/2026 la EFL Cup tenía 60 fixtures publicados, todos de primera ronda, con **dos días**
de anticipación — contra los 278 días de la Premier. Por eso las features de congestión se
construyen **sólo con partidos ya jugados**: una feature del tipo "juega copa la semana que
viene" estaría siempre completa en entrenamiento y faltaría en producción durante los días
entre el fin de una ronda y el sorteo de la siguiente, y no hay forma de reconstruir
retrospectivamente qué estaba publicado en cada momento.
"""

from __future__ import annotations

import json
import time

from common.config import CFG, utc_stamp
from common.logging_setup import get_logger, setup
from common.storage import latest_snapshot, write_manifest, write_raw
from ingestion.http_utils import fetch

log = get_logger(__name__)

SOURCE = "pulselive"
BASE = "https://footballapi.pulselive.com/football"

# La API rechaza el pedido sin este header: valida el origen del navegador.
HEADERS = {
    "Origin": "https://www.premierleague.com",
    "Referer": "https://www.premierleague.com/",
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
}

# id -> nombre corto de la competencia. La Premier va incluida a propósito: sirve para
# cruzar y validar contra lo que ya tenemos de football-data.
COMPETENCIAS = {1: "premier", 2: "champions", 3: "europa", 4: "facup", 5: "eflcup"}

# Entre llamadas. La API no publica rate limit; se es conservador igual que con FPL.
PAUSA_S = 0.25


def _get(path: str, params: dict | None = None) -> dict | None:
    url = f"{BASE}/{path}"
    if params:
        url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
    r = fetch(url, timeout=30, max_retries=3)
    if not r.ok or not r.content:
        return None
    try:
        return json.loads(r.content)
    except json.JSONDecodeError:
        return None


def _get_headers(path: str, params: dict | None = None) -> tuple[bytes, dict] | None:
    """Igual que `_get` pero devuelve también el crudo, para guardarlo en Bronze."""
    import requests

    url = f"{BASE}/{path}"
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=30)
    except requests.RequestException as exc:
        log.warning("fallo %s: %s", path, exc)
        return None
    if r.status_code != 200:
        return None
    try:
        return r.content, r.json()
    except json.JSONDecodeError:
        return None


def temporadas(comp_id: int) -> list[dict]:
    """Las temporadas disponibles de una competencia, de la más nueva a la más vieja."""
    r = _get_headers(f"competitions/{comp_id}/compseasons", {"pageSize": 100})
    return r[1].get("content", []) if r else []


def _season_label_a_nuestro(label: str) -> str | None:
    """'English Premier League Season 2025/2026' o '2025/26' -> '2025-26'.

    Las etiquetas no son consistentes entre competencias ni entre temporadas de la misma
    competencia, así que se extrae el año de arranque y se reconstruye nuestro formato.
    """
    import re

    m = re.search(r"(\d{4})/(\d{2,4})", label)
    if not m:
        return None
    ini = int(m.group(1))
    return f"{ini}-{str(ini + 1)[-2:]}"


def ingest_fixtures(comp_id: int, nombre: str, stamp: str,
                    seasons: list[str] | None = None) -> dict:
    """Fixtures de una competencia, una temporada por archivo."""
    seasons = seasons or CFG.seasons_to_ingest()
    out = {}
    for s in temporadas(comp_id):
        etiqueta = _season_label_a_nuestro(str(s.get("label", "")))
        if etiqueta not in seasons:
            continue
        sid = int(s["id"])
        r = _get_headers("fixtures", {"comps": comp_id, "compSeasons": sid,
                                      "pageSize": 500, "sort": "asc"})
        if not r:
            log.warning("[%s] %s: sin respuesta", etiqueta, nombre)
            continue
        crudo, data = r
        n = len(data.get("content", []))
        write_raw(SOURCE, etiqueta, f"fixtures_{nombre}", f"{nombre}.json", crudo, stamp)
        out[etiqueta] = n
        log.info("[%s] %-10s %3d fixtures (%.1f KB)", etiqueta, nombre, n, len(crudo) / 1024)
        time.sleep(PAUSA_S)
    return out


def ids_de_partidos(season: str, nombre: str) -> list[int]:
    """Los `id` de los fixtures ya ingestados de una competencia-temporada."""
    from common.storage import read_raw

    crudo = read_raw(SOURCE, season, f"fixtures_{nombre}", f"{nombre}.json")
    if not crudo:
        return []
    data = json.loads(crudo)
    return [int(x["id"]) for x in data.get("content", [])
            if str(x.get("status", "")).upper() == "C"]


def ingest_stats(season: str, nombre: str, stamp: str, force: bool = False) -> int:
    """Las estadísticas de Opta de cada partido terminado, en un solo archivo por competencia.

    Es una llamada por partido, así que se cachea agresivamente: una temporada cerrada no
    cambia. Sólo se re-baja con `force` o si aparecen partidos nuevos.
    """
    from common.storage import read_raw

    ids = ids_de_partidos(season, nombre)
    if not ids:
        return 0

    previos = {}
    if not force:
        crudo = read_raw(SOURCE, season, f"stats_{nombre}", f"{nombre}.json")
        if crudo:
            previos = {int(k): v for k, v in json.loads(crudo).items()}

    faltan = [i for i in ids if i not in previos]
    if not faltan:
        log.info("[%s] %-10s stats: %d partidos ya en cache", season, nombre, len(previos))
        return len(previos)

    log.info("[%s] %-10s stats: %d en cache, bajando %d...",
             season, nombre, len(previos), len(faltan))
    for j, fid in enumerate(faltan, 1):
        r = _get_headers(f"stats/match/{fid}")
        if r:
            previos[fid] = r[1]
        time.sleep(PAUSA_S)
        if j % 100 == 0:
            log.info("   %d/%d", j, len(faltan))

    payload = json.dumps({str(k): v for k, v in previos.items()}).encode()
    write_raw(SOURCE, season, f"stats_{nombre}", f"{nombre}.json", payload, stamp)
    log.info("[%s] %-10s stats: %d partidos (%.1f MB)",
             season, nombre, len(previos), len(payload) / 1024 ** 2)
    return len(previos)


def run(seasons: list[str] | None = None, comps: list[str] | None = None,
        con_stats: bool = True, force: bool = False) -> None:
    stamp = utc_stamp()
    seasons = seasons or CFG.seasons_to_ingest()
    elegidas = {k: v for k, v in COMPETENCIAS.items()
                if comps is None or v in comps}

    log.info("=== Bronze pulselive — %s ===", ", ".join(elegidas.values()))
    for cid, nombre in elegidas.items():
        ingest_fixtures(cid, nombre, stamp, seasons)

    if not con_stats:
        return
    for cid, nombre in elegidas.items():
        for s in seasons:
            if latest_snapshot(SOURCE, s, f"fixtures_{nombre}") is None:
                continue
            ingest_stats(s, nombre, stamp, force)


if __name__ == "__main__":
    setup(CFG.log_level, CFG.log_format)
    run()
