"""Tests de las interacciones de matchup (Fase 4).

La prueba de fuego contesta la única pregunta que justifica el bloque: **¿estas features
dicen algo que `dif_X` no puede decir?** Si un matchup fuera reproducible como resta de la
misma columna, sería una columna redundante más — que es exactamente lo que la Fase 5 midió
y por lo que se rechazó.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features import estilos as est


def _gold(filas: list[dict]) -> pd.DataFrame:
    """Un Gold mínimo con las columnas de Opta que los matchups consumen."""
    base = {
        "local_prop_tiros_area_u5": 0.5, "visita_prop_tiros_area_u5": 0.5,
        "local_prop_tiros_area_conc_u5": 0.5, "visita_prop_tiros_area_conc_u5": 0.5,
        "local_posesion_u5": 50.0, "visita_posesion_u5": 50.0,
        "local_quites_u5": 20.0, "visita_quites_u5": 20.0,
        "local_precision_pases_u5": 80.0, "visita_precision_pases_u5": 80.0,
        "local_intercepciones_u5": 10.0, "visita_intercepciones_u5": 10.0,
        "local_tiros_fuera_u5": 6.0, "visita_tiros_fuera_u5": 6.0,
        "local_bloqueos_u5": 4.0, "visita_bloqueos_u5": 4.0,
        "local_toques_area_rival_u5": 25.0, "visita_toques_area_rival_u5": 25.0,
        "local_prop_aereos_ganados_u5": 0.5, "visita_prop_aereos_ganados_u5": 0.5,
    }
    return pd.DataFrame([{**base, **f} for f in filas])


# ---------------------------------------------------------------------------
# PRUEBA DE FUEGO: un matchup no es un `dif_`
# ---------------------------------------------------------------------------

def test_prueba_de_fuego_dos_partidos_con_el_mismo_dif_tienen_matchups_distintos():
    """La razón de ser del bloque, en un caso construido.

    Los dos partidos tienen **exactamente la misma diferencia de posesión** (+20), así que
    `dif_posesion_u5` no los distingue. Pero en uno el visitante roba mucho y en el otro
    casi nada: el choque de estilos es otro. Si el matchup no los separara, no estaría
    aportando nada sobre las columnas que ya existen.
    """
    g = _gold([
        {"local_posesion_u5": 60.0, "visita_posesion_u5": 40.0, "visita_quites_u5": 30.0},
        {"local_posesion_u5": 60.0, "visita_posesion_u5": 40.0, "visita_quites_u5": 10.0},
    ])
    dif = g["local_posesion_u5"] - g["visita_posesion_u5"]
    assert dif.iloc[0] == dif.iloc[1], "el diferencial NO los distingue: ese es el punto"

    out = est.construir(g)
    assert out["mu_posesion_quites_l"].iloc[0] != out["mu_posesion_quites_l"].iloc[1]
    assert out["mu_posesion_quites_l"].iloc[0] == pytest.approx(60.0 * 30.0)


def test_el_matchup_cruza_estadisticas_DISTINTAS_de_los_dos_lados():
    """Un matchup nunca puede ser `X_local * X_visita` de la misma estadística: eso sería
    otra forma de comparar lo mismo. Cada término cruza dos columnas de nombre distinto."""
    for nombre, (a, b, _) in est.MATCHUPS.items():
        base_a = a.replace("local_", "").replace("visita_", "")
        base_b = b.replace("local_", "").replace("visita_", "")
        assert base_a != base_b, f"{nombre} cruza la misma estadistica consigo misma"
        assert a.split("_")[0] != b.split("_")[0], f"{nombre} usa dos columnas del mismo lado"


# ---------------------------------------------------------------------------
# La asimetría es una magnitud, no una resta
# ---------------------------------------------------------------------------

def test_la_asimetria_es_valor_absoluto_y_no_distingue_el_signo():
    """Dos partidos espejo tienen la misma asimetría de estilo. Si fuera con signo sería
    `dif_posesion` otra vez, que ya existe."""
    g = _gold([{"local_posesion_u5": 65.0, "visita_posesion_u5": 35.0},
               {"local_posesion_u5": 35.0, "visita_posesion_u5": 65.0}])
    out = est.construir(g)
    assert out[est.ASIMETRIA].iloc[0] == out[est.ASIMETRIA].iloc[1] == pytest.approx(30.0)


def test_dos_equipos_que_quieren_la_pelota_dan_asimetria_baja():
    g = _gold([{"local_posesion_u5": 55.0, "visita_posesion_u5": 53.0}])
    assert est.construir(g)[est.ASIMETRIA].iloc[0] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Los cruces, uno por uno
# ---------------------------------------------------------------------------

def test_el_cruce_de_area_va_en_las_dos_direcciones_y_no_son_lo_mismo():
    """El partido no es simétrico: que el local entre al área contra un visitante que lo
    permite es un hecho distinto del recíproco."""
    g = _gold([{"local_prop_tiros_area_u5": 0.7, "visita_prop_tiros_area_conc_u5": 0.6,
                "visita_prop_tiros_area_u5": 0.3, "local_prop_tiros_area_conc_u5": 0.2}])
    out = est.construir(g)
    assert out["mu_area_l"].iloc[0] == pytest.approx(0.42)
    assert out["mu_area_v"].iloc[0] == pytest.approx(0.06)


def test_todos_los_matchups_se_calculan():
    out = est.construir(_gold([{}]))
    for c in est.COLUMNAS:
        assert c in out.columns
        assert out[c].notna().all(), f"{c} quedo en NaN con todas las columnas presentes"


def test_construir_no_toca_las_columnas_que_ya_estaban():
    g = _gold([{"local_posesion_u5": 60.0}])
    out = est.construir(g)
    for c in g.columns:
        pd.testing.assert_series_equal(out[c], g[c])


# ---------------------------------------------------------------------------
# Degradación honesta
# ---------------------------------------------------------------------------

def test_si_falta_una_columna_base_el_matchup_queda_en_NaN_y_no_rompe():
    """Las de Opta existen sólo desde que la API oficial las publica. Un NaN es información
    honesta —"no sabemos"— y los árboles aprenden una dirección para el faltante; romper el
    build dejaría al pipeline sin correr por una fuente que puede faltar."""
    g = _gold([{}]).drop(columns=["visita_quites_u5"])
    out = est.construir(g)
    assert out["mu_posesion_quites_l"].isna().all()
    assert out["mu_area_l"].notna().all(), "los demas se calculan igual"


def test_las_columnas_declaradas_son_las_que_se_producen():
    """El spec y el módulo tienen que decir lo mismo, o Gold queda con columnas que el
    contrato no conoce."""
    out = est.construir(_gold([{}]))
    producidas = [c for c in out.columns if c.startswith("mu_")]
    assert sorted(producidas) == sorted(est.COLUMNAS)
