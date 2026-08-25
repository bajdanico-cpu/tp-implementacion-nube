"""Contrato de la tabla Gold-TP.

Verifica sobre los datos reales lo que `features/gold_tp.py` promete. Varios de estos
tests existen para que una decisión no se revierta sin que alguien se entere: por qué las
cuotas no son features, por qué el xG de 2022-23 tiene que estar en NaN, por qué
`strength_*` quedó afuera.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from common.config import CFG
from features import spec
from transform import leakage

TEMPORADAS_CERRADAS = ("2022-23", "2023-24", "2024-25", "2025-26")


# ---------------------------------------------------------------------------
# Forma y tamaño
# ---------------------------------------------------------------------------

def test_gold_tiene_una_fila_por_partido(gold_tp):
    """Una fila por partido jugado, sin duplicados.

    No se fija un total: Gold CRECE cada fecha, porque los partidos de la temporada en
    curso entran para servir de historia. Fijar 1.520 funcionaba mientras el dataset
    estaba congelado y se rompio el dia que arranco 2026-27 -- que es exactamente el tipo
    de supuesto que un test tiene que atrapar.
    """
    cerradas = len(TEMPORADAS_CERRADAS) * 380
    assert len(gold_tp) >= cerradas
    assert not gold_tp.duplicated(spec.CLAVE_PARTIDO).any()


def test_la_temporada_en_curso_no_entra_al_entrenamiento(gold_tp):
    """Sus partidos estan en Gold como historia, pero no como objetivo de entrenamiento."""
    actual = gold_tp[gold_tp["split"] == "actual"]
    if actual.empty:
        pytest.skip("todavia no arranco la temporada en curso")
    assert not set(actual["season"]) & set(CFG.seasons_for_training())
    assert CFG.holdout_season not in set(actual["season"])


@pytest.mark.parametrize("season", TEMPORADAS_CERRADAS)
def test_cada_temporada_cerrada_tiene_380_partidos(gold_tp, season):
    assert (gold_tp["season"] == season).sum() == 380


@pytest.mark.parametrize("columna", spec.GOLD_COLUMNS)
def test_la_columna_del_contrato_existe(gold_tp, columna):
    """Parametrizado sobre el spec, igual que `test_schemas.py` con Silver."""
    assert columna in gold_tp.columns


def test_no_hay_columnas_fuera_del_contrato(gold_tp):
    sobran = set(gold_tp.columns) - set(spec.GOLD_COLUMNS)
    assert not sobran, f"columnas no declaradas en features/spec.py: {sorted(sobran)}"


# ---------------------------------------------------------------------------
# Anti-leakage
# ---------------------------------------------------------------------------

def test_gold_no_tiene_columnas_prohibidas(gold_tp):
    leakage.assert_no_banned_columns(gold_tp, context="gold_tp_match")


@pytest.mark.parametrize("lado", ["local", "visita"])
def test_la_historia_usada_es_anterior_al_corte(gold_tp, lado):
    """`hist_kickoff_*` es la prueba auditable de que no se miró el futuro."""
    hk = gold_tp[f"hist_kickoff_{lado}"]
    con_dato = hk.notna()
    assert (hk[con_dato] < gold_tp.loc[con_dato, "corte"]).all()


def test_el_corte_es_el_inicio_de_la_fecha(gold_tp):
    """Todos los partidos de una gameweek comparten corte, incluidas las dobles fechas."""
    por_fecha = gold_tp.groupby(["season", "gameweek"])["corte"].nunique()
    assert (por_fecha == 1).all()


def test_el_corte_nunca_es_posterior_al_partido(gold_tp):
    assert (gold_tp["corte"] <= gold_tp["kickoff_time"]).all()


def test_ninguna_feature_es_una_cuota(gold_tp):
    """Las cuotas están en Gold para la simulación de ROI, pero NO son features.

    Si el modelo las usa aprende a copiar al mercado: el valor esperado da ~0 por
    construcción y el sistema nunca encontraría una apuesta con valor. Detectar una
    discrepancia exige que las dos estimaciones sean independientes.
    """
    sospechosas = [f for f in spec.FEATURES
                   if "odds" in f or f.startswith("p_mercado")]
    assert not sospechosas, f"features que son cuotas: {sospechosas}"
    for c in spec.MERCADO:
        assert c in gold_tp.columns and c not in spec.FEATURES


def test_ninguna_feature_es_el_resultado(gold_tp):
    prohibidas = {"target_1x2", "home_goals", "away_goals", "goal_diff"}
    assert not prohibidas & set(spec.FEATURES)


def test_ninguna_feature_es_strength():
    """`strength_*` quedó afuera por skew de escala entre histórico y API en vivo.

    Medido en `dim_team`: `strength_overall_home` promedia ~1.130 en 2022-26 y **2,85** en
    2026-27, con `strength_attack_*` en cero para los 20 equipos. Entrenar con valores de
    cuatro cifras y servir con valores de un dígito sería train/serve skew silencioso.
    Este test impide que alguien lo re-agregue sin volver a discutirlo.
    """
    assert not [f for f in spec.FEATURES if "strength" in f]


# ---------------------------------------------------------------------------
# Los hallazgos, fijados
# ---------------------------------------------------------------------------

def test_el_xg_de_2022_23_antes_de_la_gw16_es_nulo_y_no_cero(gold_tp):
    """Viene hardcodeado en 0,0 para los 20 equipos: es dato ausente, no xG bajo."""
    temprano = gold_tp[(gold_tp["season"] == "2022-23") & (gold_tp["gameweek"] < 16)]
    assert len(temprano) > 0
    assert (~temprano["xg_available"]).all()

    tarde = gold_tp[(gold_tp["season"] == "2022-23") & (gold_tp["gameweek"] >= 16)]
    assert tarde["xg_available"].all()

    # Y las ventanas del arranque de la temporada siguiente sí tienen xG.
    otra = gold_tp[gold_tp["season"] == "2023-24"]
    assert otra["xg_available"].all()


def test_la_posicion_en_la_tabla_va_de_1_a_20(gold_tp):
    for lado in ("local", "visita"):
        pos = gold_tp[f"{lado}_pos_tabla_camp"].dropna()
        assert pos.min() >= 1 and pos.max() <= 20


def test_la_posicion_no_se_repite_dentro_de_una_fecha(gold_tp):
    """Los 20 equipos de una fecha tienen 20 posiciones distintas."""
    d = pd.concat([
        gold_tp[["season", "gameweek", "home_short", "local_pos_tabla_camp"]]
        .rename(columns={"home_short": "eq", "local_pos_tabla_camp": "pos"}),
        gold_tp[["season", "gameweek", "away_short", "visita_pos_tabla_camp"]]
        .rename(columns={"away_short": "eq", "visita_pos_tabla_camp": "pos"}),
    ]).dropna(subset=["pos"]).drop_duplicates(["season", "gameweek", "eq"])

    # Se mira una fecha bien entrada la temporada, donde ya no hay empate a 0 puntos.
    muestra = d[(d["season"] == "2024-25") & (d["gameweek"] == 30)]
    assert muestra["pos"].nunique() == len(muestra)


def test_la_ventana_intra_temporada_arranca_vacia_para_todos(gold_tp):
    """En la fecha 1 nadie jugó todavía esta temporada: ni los ascendidos.

    El prior de ascendidos rellena las ventanas que CRUZAN temporadas —las que a un recién
    llegado le quedarían vacías para siempre— pero no las intra-temporada, que están
    vacías para los veinte equipos por igual.
    """
    primera = gold_tp[gold_tp["gameweek"] == 1]
    assert len(primera) > 0
    assert primera["local_pts_u5_temp"].isna().all()
    assert primera["visita_pts_u5_temp"].isna().all()


def test_los_ascendidos_sin_historia_reciben_el_prior(gold_tp):
    """Coventry y Hull no tienen NINGÚN partido en la ventana: sin prior serían todo NaN."""
    sin_historia = gold_tp[(gold_tp["local_es_ascendido"] == True)  # noqa: E712
                           & (gold_tp["local_n_hist"] == 0)]
    if sin_historia.empty:
        pytest.skip("no hay ascendidos sin historia en la ventana ingestada")
    assert sin_historia["local_pts_u5"].notna().all(), "el prior tiene que haber rellenado"
    # y el modelo tiene con qué descontar esa señal
    assert (sin_historia["local_n_hist"] == 0).all()


def test_el_head_to_head_no_supera_lo_posible(gold_tp):
    """Con 4 temporadas: máximo 7 enfrentamientos, y sólo 3 con la condición fija."""
    assert gold_tp["h2h_n"].max() <= 7
    assert gold_tp["h2h_cond_n"].max() <= 3
    assert (gold_tp["h2h_cond_n"] <= gold_tp["h2h_n"]).all()


# ---------------------------------------------------------------------------
# Target y mercado
# ---------------------------------------------------------------------------

def test_el_target_es_coherente_con_los_goles(gold_tp):
    esperado = np.where(gold_tp["home_goals"] > gold_tp["away_goals"], "home",
                        np.where(gold_tp["home_goals"] == gold_tp["away_goals"],
                                 "draw", "away"))
    assert (gold_tp["target_1x2"] == esperado).all()


def test_las_probabilidades_del_mercado_suman_uno(gold_tp):
    s = gold_tp[["p_mercado_home", "p_mercado_draw", "p_mercado_away"]].sum(axis=1)
    assert np.allclose(s.dropna(), 1.0, atol=1e-9)


def test_el_split_es_temporal_y_no_se_solapa(gold_tp):
    train = gold_tp[gold_tp["split"] == "train"]
    hold = gold_tp[gold_tp["split"] == "holdout"]
    assert set(train["season"]) & set(hold["season"]) == set()
    assert set(hold["season"]) == {CFG.holdout_season}
    assert train["kickoff_time"].max() < hold["kickoff_time"].max()


# ---------------------------------------------------------------------------
# La documentación no puede mentir
# ---------------------------------------------------------------------------

def test_docs_features_esta_sincronizado_con_spec():
    """`docs/FEATURES.md` se genera desde el spec; si quedó viejo, este test lo dice."""
    if not spec.DOCS_PATH.exists():
        pytest.skip("Falta docs/FEATURES.md. Corré `python -m features.spec --docs`.")
    actual = spec.DOCS_PATH.read_text(encoding="utf-8")
    assert actual == spec.render_docs(), (
        "docs/FEATURES.md quedó desfasado del spec. "
        "Regeneralo con `python -m features.spec --docs`.")


def test_la_version_del_feature_set_se_deriva_del_contenido():
    """Mantenerla a mano fallo: quedo pegada en "v2" mientras el set pasaba por 159, 164,
    171, 175, 184 y 192 columnas. Seis modelos guardados con la misma etiqueta.

    Derivandola de un hash de la lista, cambiar una sola feature cambia la version.
    """
    actual = spec.FEATURE_SET_VERSION
    assert actual == spec._version_features(spec.FEATURES)
    assert actual.endswith(str(len(spec.FEATURES)))
    # y una lista distinta produce una version distinta
    assert spec._version_features(spec.FEATURES[:-1]) != actual


def test_gold_registra_la_version_con_la_que_se_construyo(gold_tp):
    assert (gold_tp["feature_set_version"] == spec.FEATURE_SET_VERSION).all()
