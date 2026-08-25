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

**La clave de equipo sale de `club.abbr`, no del nombre.** Los nombres de esta API son los
oficiales completos (`Manchester City`, `Tottenham Hotspur`) mientras que nuestro registro
usa los cortos de FPL (`Man City`, `Spurs`): siete de veinte no cruzaban, y `Manchester
City` colapsaba a `manchester` colisionando con `Manchester United` porque `resolve` quita
sufijos genéricos.

Pero la API trae `club.abbr` y **coincide exactamente con nuestro `short_name`**: verificado
sobre las cinco temporadas, 27 abreviaturas contra 27 `short_name`, cero discrepancias.
Usarla evita una tabla de alias mantenida a mano, que es justo el tipo de cosa que se
desactualiza en silencio cuando asciende un equipo nuevo. El emparejamiento por nombre queda
sólo como respaldo.
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

# Peso de la instancia: más adelante en el torneo, más carga y más importancia. Se usa
# como feature ordinal, no como etiqueta categórica.
IMPORTANCIA_RONDA = {
    "1st Round": 1, "2nd Round": 2, "3rd Round": 3, "4th Round": 4, "5th Round": 5,
    "Round of 16": 6, "Quarter-Finals": 7, "Semi-Finals": 8, "Final": 9,
    "League Phase": 3, "Group Stage": 3, "Play-Offs": 5,
}


def _resolver(equipo: dict, registry, conocidos: set[str]) -> str | None:
    """short_name canónico, o None si el equipo no es de Premier en la ventana.

    Primero `club.abbr`, que la API ya entrega en nuestro formato. El emparejamiento por
    nombre queda de respaldo para el caso improbable de que falte la abreviatura.
    """
    abbr = ((equipo.get("club") or {}).get("abbr") or "").strip().upper()
    if abbr and abbr in conocidos:
        return abbr
    try:
        return team_mapping.resolve(equipo.get("name", ""), registry)
    except Exception:  # noqa: BLE001 — un equipo de otra división no es un error
        return None


def _leer(season: str, nombre: str) -> list[dict]:
    crudo = read_raw(SOURCE, season, f"fixtures_{nombre}", f"{nombre}.json")
    return json.loads(crudo).get("content", []) if crudo else []


def build() -> pd.DataFrame:
    from common.storage import read_table

    registry = team_mapping.build_registry()
    conocidos = set(read_table("dim_team")["short_name"])
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

                for i, e in enumerate(equipos):
                    corto = _resolver(e["team"], registry, conocidos)
                    if corto is None:
                        continue  # rival de otra división: no nos interesa su fila
                    otro = equipos[1 - i]
                    rival = otro["team"]["name"]
                    # Los goles del rival se toman ACA, del otro lado del fixture. Hacerlo
                    # despues con un self-join perdia todos los partidos contra equipos
                    # que no son de Premier -- o sea TODA Europa, donde ningun rival esta
                    # en nuestro registro, y la mayor parte de las copas nacionales.
                    filas.append({
                        "season": season,
                        "competencia": comp,
                        "es_premier": comp == "premier",
                        "fixture_pl_id": int(x["id"]),
                        "kickoff_time": pd.to_datetime(ms, unit="ms", utc=True),
                        "team_short": corto,
                        "rival_short": _resolver(otro["team"], registry, conocidos),
                        "team_id_pl": int(float(e["team"]["id"])),
                        "rival_nombre": rival,
                        "es_local": i == 0,
                        "ronda": fase,
                        "importancia_ronda": IMPORTANCIA_RONDA.get(fase),
                        "terminado": terminado,
                        "gf_comp": e.get("score"),
                        "gc_comp": otro.get("score"),
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
