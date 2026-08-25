"""Los partidos de TODAS las competencias, en un grano equipo x partido.

Produce `silver.fact_match_comp`: una fila por equipo y por partido, cubriendo Premier,
Champions, Europa League, FA Cup y EFL Cup. Es la tabla que le permite al pipeline ver la
carga real de un equipo — hasta ahora sólo contaba partidos de liga, así que un equipo en
semifinales de dos copas figuraba igual de descansado que uno que sólo juega el fin de
semana.

Sólo se conservan las filas de equipos de Premier: los rivales de otras divisiones
(76 de los 83 clubes de la EFL Cup) entran como `rival_short = None`. No se los mapea
porque no tienen historia en nuestro sistema y forzarlos a un `short_name` inventado sería
peor que dejarlos anónimos: lo que importa de un partido de copa es **que se jugó**, no
contra quién.

⚠️ Los nombres de esta API son los oficiales completos (`Manchester City`,
`Tottenham Hotspur`) mientras que nuestro registro usa los cortos de FPL (`Man City`,
`Spurs`). Siete de veinte no cruzaban. Peor: `resolve` quita sufijos genéricos, así que
`Manchester City` colapsaba a `manchester` y colisionaba con `Manchester United`. De ahí el
mapa explícito de abajo, que es la misma solución que ya existe para football-data.
"""

from __future__ import annotations

import json

import pandas as pd

from common.config import CFG
from common.logging_setup import get_logger, setup
from common.storage import read_raw, write_table
from ingestion.bronze_pulselive import COMPETENCIAS, SOURCE
from transform import team_mapping

log = get_logger(__name__)

TABLA = "fact_match_comp"

# Nombre oficial de pulselive -> short_name canónico. Sólo los que no resuelven solos.
ALIAS = {
    "Brighton & Hove Albion": "BHA",
    "Leeds United": "LEE",
    "Manchester City": "MCI",
    "Manchester United": "MUN",
    "Newcastle United": "NEW",
    "Nottingham Forest": "NFO",
    "Tottenham Hotspur": "TOT",
    "Wolverhampton Wanderers": "WOL",
    "West Ham United": "WHU",
    "Sheffield United": "SHU",
    "Luton Town": "LUT",
    "Ipswich Town": "IPS",
    "Coventry City": "COV",
    "Hull City": "HUL",
    "Leicester City": "LEI",
    "Norwich City": "NOR",
    "Cardiff City": "CAR",
    "Stoke City": "STK",
    "Swansea City": "SWA",
}

# Peso de la instancia: más adelante en el torneo, más carga y más importancia. Se usa
# como feature ordinal, no como etiqueta categórica.
IMPORTANCIA_RONDA = {
    "1st Round": 1, "2nd Round": 2, "3rd Round": 3, "4th Round": 4, "5th Round": 5,
    "Round of 16": 6, "Quarter-Finals": 7, "Semi-Finals": 8, "Final": 9,
    "League Phase": 3, "Group Stage": 3, "Play-Offs": 5,
}


def _resolver(nombre: str, registry) -> str | None:
    """short_name canónico, o None si el equipo no es de Premier en la ventana."""
    if nombre in ALIAS:
        return ALIAS[nombre]
    try:
        return team_mapping.resolve(nombre, registry)
    except Exception:  # noqa: BLE001 — un equipo de otra división no es un error
        return None


def _leer(season: str, nombre: str) -> list[dict]:
    crudo = read_raw(SOURCE, season, f"fixtures_{nombre}", f"{nombre}.json")
    return json.loads(crudo).get("content", []) if crudo else []


def build() -> pd.DataFrame:
    registry = team_mapping.build_registry()
    filas = []

    for season in CFG.seasons_to_ingest():
        for comp in COMPETENCIAS.values():
            for x in _leer(season, comp):
                ms = (x.get("kickoff") or {}).get("millis")
                if not ms:
                    continue
                equipos = x.get("teams", [])
                if len(equipos) != 2:
                    continue

                gw = x.get("gameweek") or {}
                fase = (gw.get("competitionPhase") or {}).get("label")
                terminado = str(x.get("status", "")).upper() == "C"
                goles = {str(g.get("teamId")): g.get("score") for g in (x.get("goals") or [])}

                for i, e in enumerate(equipos):
                    nom = e["team"]["name"]
                    corto = _resolver(nom, registry)
                    if corto is None:
                        continue  # rival de otra división: no nos interesa su fila
                    rival = equipos[1 - i]["team"]["name"]
                    filas.append({
                        "season": season,
                        "competencia": comp,
                        "es_premier": comp == "premier",
                        "fixture_pl_id": int(x["id"]),
                        "kickoff_time": pd.to_datetime(ms, unit="ms", utc=True),
                        "team_short": corto,
                        "rival_short": _resolver(rival, registry),
                        "rival_nombre": rival,
                        "es_local": i == 0,
                        "ronda": fase,
                        "importancia_ronda": IMPORTANCIA_RONDA.get(fase),
                        "terminado": terminado,
                        "gf_comp": e.get("score"),
                    })

    d = pd.DataFrame(filas)
    if d.empty:
        raise RuntimeError("No hay fixtures de pulselive en Bronze. "
                           "Corré: python -m ingestion.bronze_pulselive")
    d = d.sort_values(["team_short", "kickoff_time"]).reset_index(drop=True)

    log.info("fact_match_comp: %d filas equipo-partido", len(d))
    resumen = (d[d["terminado"]].groupby(["season", "competencia"]).size()
                .unstack(fill_value=0))
    log.info("partidos TERMINADOS de equipos de Premier, por competencia:\n%s",
             resumen.to_string())
    return d


def run() -> pd.DataFrame:
    d = build()
    write_table(d, TABLA)
    return d


if __name__ == "__main__":
    setup(CFG.log_level, CFG.log_format)
    run()
