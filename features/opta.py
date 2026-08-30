"""Ventanas móviles sobre las estadísticas de Opta.

Cubren tres huecos que el feature set no tenía, y que ninguna otra fuente del proyecto
podía llenar:

**Ubicación del tiro.** `attempts_ibox` / `attempts_obox` separan el remate desde dentro
del área del de afuera. Es el proxy de calidad del xG que se había dado por inalcanzable
sin Understat: un remate dentro del área vale unas cuatro veces uno de afuera, y el xG
agregado de FPL no distingue "2,0 en tres ocasiones claras" de "2,0 en veinte remates
lejanos".

**Defensa como acción, no como consecuencia.** Hasta ahora la defensa se medía por lo que
el rival lograba —tiros concedidos, goles, tasa de atajadas—. `total_tackle`,
`interception`, `total_clearance` y `outfielder_block` son lo que el equipo *hace*.

**Dominio territorial.** `possession_percentage` y `touches_in_opp_box` no tenían
equivalente.

Las ventanas se calculan **sólo sobre partidos de Premier**, igual que el resto de las
features de forma. Mezclar competencias distorsiona: 25 tiros contra un equipo de cuarta
división en la FA Cup no dicen lo mismo que 25 contra el City. La carga de las otras
competencias ya entra por `features/competencias.py`, que es donde corresponde.
"""

from __future__ import annotations

import pandas as pd

from common.config import CFG
from common.logging_setup import get_logger
from transform.opta_stats import DERIVADAS, STATS

log = get_logger(__name__)

# De las ~180 que devuelve la API se rueda un subconjunto. Muchas son derivadas triviales
# de otras o tan específicas que con 1.004 filas sólo aportan ruido.
# Cobertura verificada por temporada antes de elegirlas: todas estan por encima del 94 %
# en las cinco, y esa falta es uniforme -- no un artefacto de temporada.
#
# Quedaron AFUERA tres que Opta agrego recien en los ultimos anios y que en el historico
# de entrenamiento no existen:
#
#   conducciones_prog     0 % en 2022-24, 41 % en 2025-26, 100 % en 2026-27
#   recuperaciones        0 % en todas menos la actual
#   atajadas_clarisimas   6 % global
#
# Es la misma trampa que el xG hardcodeado en cero de 2022-23: una feature que solo existe
# en las temporadas recientes le ensenia al modelo a reconocer la temporada, no el futbol.
A_RODAR = [
    "tiros_area", "tiros_fuera", "tiros_area_conc", "tiros_fuera_conc",
    "quites", "intercepciones", "rechazos", "bloqueos",
    "posesion", "toques_area_rival",
    "prop_tiros_area", "prop_tiros_area_conc", "precision_pases", "prop_aereos_ganados",
]

VENTANAS = (3, 5)
COLUMNAS = [f"{c}_u{n}" for n in VENTANAS for c in A_RODAR]


def historia(stats: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    """Estadísticas por equipo-partido de Premier, con su kickoff, listas para el asof.

    El cruce va por `(season, fixture_pl_id, team_id_pl)`: las estadísticas vienen
    indexadas por el id de equipo de la API, no por nombre.
    """
    clave = ["season", "fixture_pl_id", "team_id_pl"]
    base = comp.loc[comp["es_premier"], clave + ["team_short", "kickoff_time", "terminado"]]
    d = base.merge(stats, on=clave, how="left")

    # La cobertura se mide SOLO sobre partidos ya jugados: un fixture futuro de la
    # temporada en curso no tiene estadisticas porque todavia no se jugo, no porque
    # falte ingesta. Sin este filtro el aviso salta al 20 % en agosto -- 740 fixtures
    # de 2026-27 que aun no ocurrieron -- y se vuelve ruido que se aprende a ignorar.
    jugados = d[d["terminado"].fillna(False)] if "terminado" in d else d
    faltan = jugados[A_RODAR[0]].isna().mean() if len(jugados) else 0.0
    if faltan > 0.05:
        log.warning("El %.0f%% de los partidos de Premier YA JUGADOS no tiene "
                    "estadisticas de Opta. Puede faltar ingesta: "
                    "python -m ingestion.bronze_pulselive", faltan * 100)

    d = d.sort_values(["team_short", "kickoff_time"]).reset_index(drop=True)
    g = d.groupby("team_short", sort=False)
    for n in VENTANAS:
        for c in A_RODAR:
            # Inclusiva, igual que el resto: el "correrse" lo hace el merge_asof del corte.
            d[f"{c}_u{n}"] = g[c].transform(lambda s: s.rolling(n, min_periods=1).mean())

    return d[["season", "team_short", "kickoff_time"] + COLUMNAS].rename(
        columns={"kickoff_time": "hist_kickoff"})


def disponible() -> bool:
    """Si hay estadísticas ingestadas. Permite que Gold funcione sin ellas."""
    from common.storage import table_exists

    return table_exists("fact_opta_stats")


def construir() -> pd.DataFrame | None:
    from common.storage import read_table

    if not disponible():
        log.info("Sin fact_opta_stats: las features de Opta se omiten. "
                 "Para tenerlas: python -m ingestion.bronze_pulselive && "
                 "python -m transform.opta_stats")
        return None
    return historia(read_table("fact_opta_stats"), read_table("fact_match_comp"))
