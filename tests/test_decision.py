"""Tests de la regla de decisión: el paso de las tres probabilidades a UNA clase.

Varios son **pruebas de fuego**. Las dos que más importan:

- **El umbral no puede implementarse enmascarando la columna del empate.** Cuando `p_draw`
  queda debajo del umbral pero sigue siendo el máximo de las tres, un argmax crudo devuelve
  `draw` igual y el umbral no hace nada. Se construye ese caso a mano.
- **El backfill no puede tocar el registro.** Etiquetar hacia atrás es legítimo porque la
  regla es una función pura de las probabilidades guardadas; deja de serlo en el segundo en
  que se toca `predicted_at`, `p_*` o `model_version`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eda.baselines import CLASES_ORD
from serving import decision


def _P(filas: list[dict]) -> np.ndarray:
    """Matriz de probabilidades en el orden de CLASES_ORD a partir de dicts legibles."""
    return np.array([[f[c] for c in CLASES_ORD] for f in filas], dtype=float)


def _df(filas: list[dict]) -> pd.DataFrame:
    d = pd.DataFrame(filas)
    return d.rename(columns={c: f"p_{c}" for c in CLASES_ORD})


# ---------------------------------------------------------------------------
# Las reglas
# ---------------------------------------------------------------------------

def test_argmax_devuelve_la_clase_mas_probable():
    P = _P([{"home": 0.6, "draw": 0.3, "away": 0.1},
            {"home": 0.2, "draw": 0.3, "away": 0.5}])
    r = decision.Regla("argmax", "argmax")
    assert list(r.aplicar(P)) == ["home", "away"]


def test_el_umbral_convierte_en_empate_solo_al_que_lo_supera():
    P = _P([{"home": 0.50, "draw": 0.31, "away": 0.19},    # 0,31 >= 0,30 -> draw
            {"home": 0.52, "draw": 0.29, "away": 0.19}])   # 0,29 <  0,30 -> home
    r = decision.Regla("u030", "umbral_empate", {"umbral": 0.30})
    assert list(r.aplicar(P)) == ["draw", "home"]


def test_el_umbral_es_mayor_o_igual_no_mayor_estricto():
    """Justo en el umbral cuenta como empate. Es arbitrario, pero tiene que estar fijado:
    si no, el mismo partido cambia de predicción según el redondeo de la impresión."""
    P = _P([{"home": 0.40, "draw": 0.30, "away": 0.30}])
    assert decision.Regla("u", "umbral_empate", {"umbral": 0.30}).aplicar(P)[0] == "draw"


def test_prueba_de_fuego_el_umbral_no_es_enmascarar_la_columna_del_empate():
    """El caso que rompe la implementación ingenua.

    `p_draw = 0,29` está debajo del umbral 0,30 **y sin embargo es el máximo de las tres**.
    Un `argmax` sobre la matriz con la columna del empate puesta a cero funcionaría, pero
    la versión que sólo compara contra el umbral y después hace argmax de las tres devuelve
    `draw`: el umbral no habría hecho nada. Hay que elegir entre las OTRAS dos clases.
    """
    P = _P([{"home": 0.29, "draw": 0.42, "away": 0.29}])
    assert P[0].argmax() == list(CLASES_ORD).index("draw")     # el empate ES el maximo
    fuera = decision.Regla("u", "umbral_empate", {"umbral": 0.50}).aplicar(P)
    assert fuera[0] != "draw", "con umbral 0,50 y p_draw 0,42 no puede decir empate"
    assert fuera[0] in ("home", "away")


def test_un_umbral_alto_no_predice_ningun_empate():
    P = _P([{"home": 0.4, "draw": 0.35, "away": 0.25},
            {"home": 0.2, "draw": 0.33, "away": 0.47}])
    etq = decision.Regla("u", "umbral_empate", {"umbral": 0.99}).aplicar(P)
    assert "draw" not in set(etq)


def test_bajar_el_umbral_nunca_reduce_los_empates_predichos():
    """Monotonía: es la propiedad que hace del umbral un dial y no una perilla caprichosa."""
    rng = np.random.default_rng(0)
    P = rng.dirichlet(np.ones(3), size=300)
    previos = -1
    for u in (0.50, 0.40, 0.35, 0.30, 0.25, 0.20):
        n = int((decision.Regla("u", "umbral_empate", {"umbral": u}).aplicar(P) == "draw").sum())
        assert n >= previos
        previos = n


def test_una_regla_desconocida_falla_al_construirse():
    """Un typo en config.yaml tiene que romper al arrancar, no predecir cualquier cosa."""
    with pytest.raises(ValueError, match="desconocida"):
        decision.Regla("x", "umbral_de_la_suerte", {"umbral": 0.3})


# ---------------------------------------------------------------------------
# El orden de las columnas — la misma familia de error que el del log-loss
# ---------------------------------------------------------------------------

def test_la_matriz_respeta_el_orden_de_clases_ord():
    d = _df([{"home": 0.6, "draw": 0.3, "away": 0.1}])
    P = decision.matriz(d[["p_home", "p_draw", "p_away"]])   # columnas en OTRO orden
    assert list(P[0]) == [0.1, 0.3, 0.6]                     # away, draw, home
    assert list(decision._argmax(P)) == ["home"]


def test_faltar_una_probabilidad_falla_en_vez_de_adivinar():
    with pytest.raises(ValueError, match="probabilidad"):
        decision.matriz(pd.DataFrame({"p_home": [0.5], "p_draw": [0.3]}))


# ---------------------------------------------------------------------------
# Etiquetado
# ---------------------------------------------------------------------------

def test_etiquetar_deja_una_columna_por_regla_y_no_toca_las_probabilidades():
    d = _df([{"home": 0.37, "draw": 0.32, "away": 0.31}])
    antes = d.copy()
    out = decision.etiquetar(d)
    for c in ("p_home", "p_draw", "p_away"):
        pd.testing.assert_series_equal(out[c], antes[c])
    assert out[decision.COL_PRODUCCION].iloc[0] == "home"
    for r in decision.candidatos():
        assert r.columna in out.columns


def test_etiquetar_es_idempotente():
    d = _df([{"home": 0.37, "draw": 0.32, "away": 0.31},
             {"home": 0.20, "draw": 0.25, "away": 0.55}])
    una = decision.etiquetar(d)
    dos = decision.etiquetar(una)
    pd.testing.assert_frame_equal(una, dos)


def test_la_produccion_por_defecto_es_argmax():
    """Si esto falla, cambió lo que el sistema ANUNCIA — y eso no es un cambio menor."""
    assert decision.produccion().tipo == "argmax"


# ---------------------------------------------------------------------------
# `desde`: qué fechas cuentan como evidencia
# ---------------------------------------------------------------------------

def test_una_regla_no_cuenta_para_las_fechas_anteriores_a_su_desde():
    r = decision.Regla("u", "umbral_empate", {"umbral": 0.3}, desde="2026-27 GW3")
    assert not r.cuenta_para("2026-27", 2)
    assert r.cuenta_para("2026-27", 3)
    assert r.cuenta_para("2026-27", 10)
    assert r.cuenta_para("2027-28", 1)
    assert not r.cuenta_para("2025-26", 38)


def test_una_regla_sin_desde_cuenta_siempre():
    assert decision.Regla("u", "argmax").cuenta_para("2022-23", 1)


# ---------------------------------------------------------------------------
# Backfill — PRUEBA DE FUEGO: reproducir no es reescribir
# ---------------------------------------------------------------------------

def _registro() -> pd.DataFrame:
    return pd.DataFrame({
        "season": ["2026-27"] * 2, "gameweek": [3, 3], "fixture_id": [1, 2],
        "home_short": ["NFO", "ARS"], "away_short": ["TOT", "CHE"],
        "p_away": [0.313, 0.151], "p_draw": [0.320, 0.163], "p_home": [0.367, 0.685],
        "prediccion": ["home", "home"], "confianza": [0.367, 0.685],
        "predicted_at": ["2026-09-01T23:32:03+00:00"] * 2,
        "model_name": ["xgb_gbt"] * 2, "model_version": ["20260825T024144Z"] * 2,
    })


def test_el_backfill_no_toca_probabilidades_ni_trazabilidad(tmp_path):
    antes = _registro()
    antes.to_parquet(tmp_path / "2026-27_GW03_20260901T233203Z.parquet", index=False)

    decision.backfill(tmp_path)
    despues = pd.read_parquet(tmp_path / "2026-27_GW03_20260901T233203Z.parquet")

    intocables = ["p_away", "p_draw", "p_home", "predicted_at",
                  "model_name", "model_version", "prediccion"]
    pd.testing.assert_frame_equal(despues[intocables], antes[intocables])


def test_el_backfill_agrega_la_columna_del_candidato(tmp_path):
    _registro().to_parquet(tmp_path / "2026-27_GW03_20260901T233203Z.parquet", index=False)
    decision.backfill(tmp_path)
    d = pd.read_parquet(tmp_path / "2026-27_GW03_20260901T233203Z.parquet")
    for r in decision.candidatos():
        assert r.columna in d.columns


def test_el_backfill_es_idempotente(tmp_path):
    p = tmp_path / "2026-27_GW03_20260901T233203Z.parquet"
    _registro().to_parquet(p, index=False)
    decision.backfill(tmp_path)
    una = pd.read_parquet(p)
    res = decision.backfill(tmp_path)
    pd.testing.assert_frame_equal(pd.read_parquet(p), una)
    assert (res["columnas_agregadas"] == "-").all(), "la segunda pasada no agrega nada"


def test_el_backfill_en_dry_run_no_escribe(tmp_path):
    p = tmp_path / "2026-27_GW03_20260901T233203Z.parquet"
    _registro().to_parquet(p, index=False)
    antes = p.read_bytes()
    decision.backfill(tmp_path, dry_run=True)
    assert p.read_bytes() == antes


def test_ultimas_por_fecha_se_queda_con_el_stamp_mas_nuevo(tmp_path):
    for stamp in ("20260824T234847Z", "20260830T150911Z"):
        _registro().to_parquet(tmp_path / f"2026-27_GW01_{stamp}.parquet", index=False)
    ultimas = decision.ultimas_por_fecha(tmp_path)
    assert ultimas[("2026-27", 1)].name.endswith("20260830T150911Z.parquet")


# ---------------------------------------------------------------------------
# La comparación en paralelo — que sea PAREADA es lo que la hace valer
# ---------------------------------------------------------------------------

def test_el_log_loss_no_puede_cambiar_entre_reglas():
    """La propiedad que ordena cómo se lee toda la evaluación.

    La regla no toca las probabilidades, así que **ninguna métrica basada en probabilidad
    puede moverse**. Si alguna vez esto falla, es que una regla empezó a reescribir `P` —
    y entonces dejó de ser una regla de decisión y pasó a ser otro modelo.
    """
    from training import decision_eval

    rng = np.random.default_rng(3)
    P = rng.dirichlet(np.ones(3), size=400)
    y = np.asarray(CLASES_ORD)[rng.integers(0, 3, 400)]
    d = decision_eval.evaluar_reglas(P, y)
    assert d["log_loss"].nunique() == 1


def test_la_comparacion_contra_produccion_es_pareada():
    """`n` tiene que ser el total de filas y los discordantes un subconjunto: si la
    comparación dejara de ser pareada, McNemar no seria el test correcto."""
    from training import decision_eval

    rng = np.random.default_rng(5)
    P = rng.dirichlet(np.ones(3), size=200)
    y = np.asarray(CLASES_ORD)[rng.integers(0, 3, 200)]
    c = decision_eval.comparar_contra_produccion(P, y)
    for fila in c.itertuples():
        assert fila.n == 200
        assert fila.discordantes <= 200
        assert fila.gana_candidato + fila.gana_produccion == fila.discordantes


def test_el_candidato_solo_puede_discrepar_donde_movio_el_empate():
    """Si dos reglas discrepan en un partido donde ninguna dice `draw`, hay un bug: la
    única diferencia entre argmax y umbral_empate es a quién llama empate."""
    rng = np.random.default_rng(7)
    P = rng.dirichlet(np.ones(3), size=500)
    a = decision.Regla("a", "argmax").aplicar(P)
    b = decision.Regla("b", "umbral_empate", {"umbral": 0.30}).aplicar(P)
    distintos = a != b
    assert ((a[distintos] == "draw") | (b[distintos] == "draw")).all()


def test_con_umbral_050_el_candidato_no_puede_ganarle_al_argmax_por_empates():
    """Con umbral 0,50 la regla sólo anuncia empate cuando el empate ya era mayoría
    absoluta, así que coincide con el argmax salvo en casos degenerados."""
    rng = np.random.default_rng(11)
    P = rng.dirichlet(np.ones(3), size=500)
    a = decision.Regla("a", "argmax").aplicar(P)
    b = decision.Regla("b", "umbral_empate", {"umbral": 0.50}).aplicar(P)
    # Donde el empate supera 0,5 las dos dicen draw; donde no, b nunca dice draw.
    assert (b[P[:, decision.I_DRAW] >= 0.5] == "draw").all()
    assert (b[P[:, decision.I_DRAW] < 0.5] != "draw").all()
    assert (a[P[:, decision.I_DRAW] >= 0.5] == "draw").all()
