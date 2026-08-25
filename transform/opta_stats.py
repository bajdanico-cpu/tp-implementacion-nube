"""Estadísticas de Opta por equipo y partido, desde la API oficial de la Premier.

`/stats/match/{id}` devuelve entre 120 y 187 estadísticas por equipo. Están verificadas en
las cuatro temporadas de entrenamiento y también en copas y Champions, y se publican **pocas
horas después del partido** — antes incluso de que FPL marque la fecha como `finished`.

De las ~180 disponibles se conserva un subconjunto elegido, no todas. La razón: muchas son
derivadas triviales de otras (`att_lf_total` + `att_rf_total` + `att_hd_total` ≈
`total_scoring_att`) o tan específicas que con 1.004 filas de entrenamiento sólo aportan
ruido. Las que quedan cubren tres huecos concretos del feature set:

**Ubicación del tiro.** `attempts_ibox` / `attempts_obox` separan el remate desde dentro
del área del de afuera. Es el proxy de calidad del xG que se había dado por inalcanzable sin
Understat: un remate dentro del área vale unas cuatro veces uno de afuera, y el xG agregado
no lo distingue.

**Defensa real.** `total_tackle`, `interception`, `total_clearance`, `outfielder_block`.
Hasta ahora la defensa se aproximaba con tiros concedidos y tasa de atajadas, que son
consecuencias; esto son acciones.

**Dominio territorial.** `possession_percentage`, `touches_in_opp_box`, `total_pass`,
`accurate_pass`. No teníamos nada de esto.

⚠️ **No hay xG.** Opta lo licencia aparte y la API pública no lo expone. Tampoco
`big_chance_created/missed/scored` — sólo `big_chance_saves`. El xG sigue viniendo de FPL.
"""

from __future__ import annotations

import json

import pandas as pd

from common.config import CFG
from common.logging_setup import get_logger, setup
from common.storage import read_raw, write_table
from ingestion.bronze_pulselive import COMPETENCIAS, SOURCE

log = get_logger(__name__)

TABLA = "fact_opta_stats"

# Las que se conservan, agrupadas por lo que aportan. El nombre corto es el que llega a las
# features; el largo es como lo llama Opta.
STATS = {
    # --- ubicación del tiro: la calidad de la situación, no la cantidad ---
    "tiros_area": "attempts_ibox",
    "tiros_fuera": "attempts_obox",
    "tiros_area_arco": "att_ibox_target",
    "tiros_area_conc": "attempts_conceded_ibox",
    "tiros_fuera_conc": "attempts_conceded_obox",
    "goles_conc_area": "goals_conceded_ibox",
    # --- defensa: acciones, no consecuencias ---
    "quites": "total_tackle",
    "quites_ganados": "won_tackle",
    "intercepciones": "interception",
    "rechazos": "total_clearance",
    "bloqueos": "outfielder_block",
    "bloqueos_remate": "blocked_scoring_att",
    "recuperaciones": "recoveries",
    # --- dominio territorial ---
    "posesion": "possession_percentage",
    "pases": "total_pass",
    "pases_ok": "accurate_pass",
    "toques_area_rival": "touches_in_opp_box",
    "conducciones_prog": "progressive_carries",
    # --- arquero ---
    "atajadas_opta": "saves",
    "atajadas_clarisimas": "big_chance_saves",
    # --- disciplina y juego aéreo ---
    "duelos_aereos_ganados": "aerial_won",
    "duelos_aereos_perdidos": "aerial_lost",
    "offsides": "total_offside",
}

DERIVADAS = {
    # Proporciones, que son más comparables entre partidos que los conteos crudos.
    "prop_tiros_area": ("tiros_area", "tiros_totales_opta"),
    "prop_tiros_area_conc": ("tiros_area_conc", "tiros_conc_opta"),
    "precision_pases": ("pases_ok", "pases"),
    "prop_quites_ganados": ("quites_ganados", "quites"),
    "prop_aereos_ganados": ("duelos_aereos_ganados", "duelos_aereos_totales"),
}


def _leer(season: str, comp: str) -> dict:
    crudo = read_raw(SOURCE, season, f"stats_{comp}", f"{comp}.json")
    return json.loads(crudo) if crudo else {}


def build() -> pd.DataFrame:
    filas = []
    for season in CFG.seasons_to_ingest():
        for comp in COMPETENCIAS.values():
            datos = _leer(season, comp)
            for fid, payload in datos.items():
                data = (payload or {}).get("data") or {}
                for team_id, blk in data.items():
                    stats = {s["name"]: s["value"] for s in (blk.get("M") or [])}
                    if not stats:
                        continue
                    fila = {"season": season, "competencia": comp,
                            "fixture_pl_id": int(fid), "team_id_pl": int(float(team_id))}
                    for corto, largo in STATS.items():
                        fila[corto] = stats.get(largo)
                    filas.append(fila)

    d = pd.DataFrame(filas)
    if d.empty:
        raise RuntimeError("No hay estadísticas de pulselive en Bronze. "
                           "Corré: python -m ingestion.bronze_pulselive")

    d["tiros_totales_opta"] = d["tiros_area"].fillna(0) + d["tiros_fuera"].fillna(0)
    d["tiros_conc_opta"] = d["tiros_area_conc"].fillna(0) + d["tiros_fuera_conc"].fillna(0)
    d["duelos_aereos_totales"] = (d["duelos_aereos_ganados"].fillna(0)
                                  + d["duelos_aereos_perdidos"].fillna(0))

    for nombre, (num, den) in DERIVADAS.items():
        d[nombre] = d[num] / d[den].replace(0, pd.NA)

    log.info("fact_opta_stats: %d filas equipo-partido, %d columnas", len(d), d.shape[1])
    cobertura = d.groupby("season").size()
    log.info("por temporada:\n%s", cobertura.to_string())
    faltantes = d[list(STATS)].isna().mean().sort_values(ascending=False)
    log.info("las 5 stats con más faltantes:\n%s", faltantes.head(5).round(3).to_string())
    return d


def run() -> pd.DataFrame:
    d = build()
    write_table(d, TABLA)
    return d


if __name__ == "__main__":
    setup(CFG.log_level, CFG.log_format)
    run()
