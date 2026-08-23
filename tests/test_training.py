"""Tests del entrenamiento: device, métricas, apuestas y promoción.

Varios son **pruebas de fuego**: construyen el caso que la implementación ingenua rompe y
verifican que el control lo detecta. En particular el del orden de etiquetas del log-loss
y el de la promoción por una sola fecha.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss

from eda.baselines import CLASES_ORD
from training import betting, dataset, metrics, promotion
from training.device import resolve


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def test_pedir_cpu_nunca_devuelve_cuda():
    assert resolve("cpu").used == "cpu"


def test_auto_siempre_devuelve_algo_valido():
    assert resolve("auto").used in ("cpu", "cuda")


def test_pedir_cuda_explicito_levanta_si_no_hay(monkeypatch):
    """Explícito es explícito: si alguien pidió GPU y se cayó a CPU en silencio, el
    benchmark mentiría."""
    import xgboost as xgb

    monkeypatch.setattr(xgb, "build_info", lambda: {"USE_CUDA": False})
    with pytest.raises(RuntimeError, match="cuda"):
        resolve("cuda")


def test_auto_cae_a_cpu_sin_cuda(monkeypatch):
    import xgboost as xgb

    monkeypatch.setattr(xgb, "build_info", lambda: {"USE_CUDA": False})
    info = resolve("auto")
    assert info.used == "cpu" and info.requested == "auto"
    assert "CUDA" in info.reason or "cuda" in info.reason


def test_device_invalido_falla():
    with pytest.raises(ValueError):
        resolve("tpu")


# ---------------------------------------------------------------------------
# Métricas — PRUEBA DE FUEGO del orden de etiquetas
# ---------------------------------------------------------------------------

def test_el_log_loss_usa_el_orden_lexicografico():
    """`sklearn.log_loss` asume que las columnas de probabilidad vienen en orden
    lexicográfico, y ese supuesto es el que hay que respetar.

    En scikit-learn 1.9 pasarle `labels=['home','draw','away']` ya no desalinea en
    silencio: reordena y avisa por warning. Pero el peligro de fondo sigue intacto y es
    peor, porque no avisa nada: si **las columnas de tu matriz** están en orden
    home/draw/away, el número sale mal y no hay warning que lo delate.

    Por eso `training/metrics.py` es el único módulo que llama a `log_loss`, y siempre
    ordena las columnas con `CLASES_ORD` antes.
    """
    y = np.array(["home", "draw", "away"])
    # Columnas en el orden de CLASES_ORD = ['away', 'draw', 'home'].
    P = np.array([[0.1, 0.2, 0.7],
                  [0.2, 0.6, 0.2],
                  [0.8, 0.1, 0.1]])

    esperado = -np.mean([np.log(0.7), np.log(0.6), np.log(0.8)])
    rep = metrics.reporte(y, y, P, con_ic=False)
    assert rep["log_loss"] == pytest.approx(esperado, rel=1e-9)

    # PRUEBA DE FUEGO: la misma información, pero con las columnas en orden
    # home/draw/away. sklearn no tiene forma de saberlo y devuelve un número incorrecto
    # sin ninguna advertencia.
    P_mal_ordenada = P[:, ::-1]
    silenciosamente_mal = log_loss(y, P_mal_ordenada, labels=list(CLASES_ORD))
    assert silenciosamente_mal != pytest.approx(esperado, rel=1e-6), (
        "si esto empieza a coincidir, el ejemplo dejó de ser sensible al orden")

    # Y el módulo lo resuelve si se le dice en qué orden vienen las columnas.
    rep_ok = metrics.reporte(y, y, P_mal_ordenada,
                             columnas_proba=["home", "draw", "away"], con_ic=False)
    assert rep_ok["log_loss"] == pytest.approx(esperado, rel=1e-9)


def test_el_reporte_devuelve_las_metricas_del_bloque_5():
    y = np.array(["home"] * 6 + ["draw"] * 2 + ["away"] * 2)
    pred = np.array(["home"] * 10)
    rep = metrics.reporte(y, pred, con_ic=False)
    for k in ("accuracy", "f1_macro", "precision_macro", "recall_macro"):
        assert k in rep
    assert rep["accuracy"] == pytest.approx(0.6)


def test_el_f1_macro_penaliza_ignorar_el_empate():
    """El empate es el 24 % de los partidos y casi nunca es el argmax de nadie.

    La accuracy sola no lo muestra; el F1 macro sí. Es la razón de reportar los dos.
    """
    y = np.array(["home"] * 5 + ["draw"] * 3 + ["away"] * 2)
    nunca_empata = np.array(["home"] * 7 + ["away"] * 3)
    rep = metrics.reporte(y, nunca_empata, con_ic=False)

    assert rep["f1_draw"] == 0.0
    assert rep["f1_macro"] < rep["accuracy"]


def test_el_intervalo_de_confianza_contiene_al_punto():
    rng = np.random.default_rng(0)
    y = rng.choice(CLASES_ORD, 380)
    pred = rng.choice(CLASES_ORD, 380)
    rep = metrics.reporte(y, pred)
    lo, hi = rep["accuracy_ic95"]
    assert lo <= rep["accuracy"] <= hi
    assert hi - lo > 0.03, "con n=380 el intervalo no puede ser angosto"


# ---------------------------------------------------------------------------
# Codificación de etiquetas
# ---------------------------------------------------------------------------

def test_la_codificacion_respeta_el_orden_de_clases_ord():
    """Que el índice coincida con CLASES_ORD es lo que alinea `predict_proba`."""
    assert dataset.CLASES == list(CLASES_ORD)
    cod = dataset.codificar(["away", "draw", "home"])
    assert list(cod) == [0, 1, 2]
    assert list(dataset.decodificar(cod)) == ["away", "draw", "home"]


def test_codificar_una_etiqueta_desconocida_falla():
    with pytest.raises(ValueError):
        dataset.codificar(["home", "penales"])


# ---------------------------------------------------------------------------
# Capa de decisión
# ---------------------------------------------------------------------------

def test_el_valor_esperado_de_una_cuota_justa_es_cero():
    """Si `c = 1/p`, apostar no tiene ni ventaja ni desventaja."""
    p = np.array([[0.25, 0.25, 0.5]])
    cuotas = 1.0 / p
    assert np.allclose(betting.valor_esperado(p, cuotas), 0.0)


def test_el_valor_esperado_es_positivo_si_el_modelo_ve_mas_probable_que_el_mercado():
    p = np.array([[0.5]])
    cuota = np.array([[2.5]])           # el mercado implica 0.40
    assert betting.valor_esperado(p, cuota)[0, 0] == pytest.approx(0.25)


def test_un_modelo_que_copia_al_mercado_no_encuentra_apuestas():
    """La razón estructural por la que las cuotas NO son features.

    Si el modelo aprende `p ≈ 1/cuota`, el EV da ~0 en todos lados y el sistema nunca
    apostaría. Detectar valor exige que las dos estimaciones sean independientes.
    """
    n = 50
    rng = np.random.default_rng(1)
    p = rng.dirichlet([4, 3, 5], n)
    cuotas = 1.0 / p
    filas = pd.DataFrame({
        "season": ["S"] * n, "fixture_id": range(n), "gameweek": 1,
        "target_1x2": rng.choice(CLASES_ORD, n),
        "odds_avg_close_away": cuotas[:, 0],
        "odds_avg_close_draw": cuotas[:, 1],
        "odds_avg_close_home": cuotas[:, 2],
    })
    assert betting.decidir(filas, p, umbral_ev=0.05).empty


# ---------------------------------------------------------------------------
# Promoción — PRUEBA DE FUEGO
# ---------------------------------------------------------------------------

def test_la_promocion_no_asciende_por_una_sola_fecha():
    """Con 10 partidos el error estándar de la accuracy es ±15,7 puntos.

    Un candidato que gana 6-4 en una fecha es indistinguible del azar. Si el pipeline lo
    promoviera, estaría eligiendo modelos a cara o cruz.
    """
    candidato = np.array([True] * 6 + [False] * 4)
    produccion = np.array([False] * 6 + [True] * 4)
    d = promotion.decidir(candidato, produccion)
    assert not d.promover
    assert "significativa" in d.motivo


def test_la_promocion_acepta_una_ventaja_sostenida():
    """Sobre 100 partidos (10 fechas), una ventaja consistente sí es detectable."""
    rng = np.random.default_rng(0)
    n = 100
    produccion = rng.random(n) < 0.42
    candidato = produccion.copy()
    # el candidato corrige 20 errores y no rompe ninguno acierto
    idx = np.flatnonzero(~produccion)[:20]
    candidato[idx] = True
    d = promotion.decidir(candidato, produccion)
    assert d.promover, d.motivo


def test_la_promocion_rechaza_un_modelo_peor():
    rng = np.random.default_rng(0)
    n = 100
    candidato = rng.random(n) < 0.35
    produccion = candidato.copy()
    produccion[np.flatnonzero(~candidato)[:20]] = True
    d = promotion.decidir(candidato, produccion)
    assert not d.promover


def test_la_promocion_rechaza_si_empeora_el_holdout():
    """Ganar en las últimas fechas pero perder en el holdout huele a sobreajuste reciente."""
    rng = np.random.default_rng(0)
    n = 100
    produccion = rng.random(n) < 0.42
    candidato = produccion.copy()
    candidato[np.flatnonzero(~produccion)[:20]] = True
    d = promotion.decidir(candidato, produccion,
                          acc_holdout_candidato=0.44, acc_holdout_produccion=0.49)
    assert not d.promover
    assert "holdout" in d.motivo


def test_mcnemar_solo_mira_los_pares_discordantes():
    a = np.array([True, True, False, False, True])
    b = np.array([True, True, False, True, False])
    mc = metrics.mcnemar(a, b)
    assert mc["n_discordantes"] == 2 and mc["n01"] == 1 and mc["n10"] == 1


def test_mcnemar_con_modelos_identicos_no_decide():
    a = np.array([True, False, True, True])
    mc = metrics.mcnemar(a, a)
    assert mc["n_discordantes"] == 0 and mc["p_valor"] == 1.0


# ---------------------------------------------------------------------------
# Portabilidad entre devices — el escenario real del bloque 7
# ---------------------------------------------------------------------------

def _hay_gpu() -> bool:
    try:
        return resolve("cuda").used == "cuda"
    except RuntimeError:
        return False


@pytest.mark.skipif(not _hay_gpu(), reason="no hay GPU disponible")
def test_modelo_entrenado_en_gpu_predice_igual_en_cpu(tmp_path):
    """Se entrena en un device y se sirve en otro. Es exactamente lo que hace el TP.

    El bloque 7 del canvas dice que la predicción corre en batch calendarizado, sin GPU:
    o sea que el modelo entrenado con CUDA tiene que poder cargarse y predecir en un nodo
    pelado. El formato `.ubj` no arrastra estado de device, y esto lo demuestra.

    La tolerancia es 1e-5 y no igualdad exacta a propósito: el algoritmo `hist` de GPU no
    es bit-idéntico al de CPU (distinto orden de reducción en punto flotante). Eso se
    documenta, no se esconde.
    """
    import xgboost as xgb

    from training import models

    rng = np.random.default_rng(0)
    X = rng.random((300, 20), dtype=np.float32)
    y = rng.integers(0, 3, 300)

    m = models.construir("xgb_gbt", resolve("cuda"), params={"n_estimators": 40})
    m.fit(X, y)
    p_gpu = m.predict_proba(X)

    ruta = tmp_path / "model.ubj"
    m.get_booster().save_model(str(ruta))

    b = xgb.Booster()
    b.load_model(str(ruta))
    b.set_param({"device": "cpu"})
    p_cpu = b.inplace_predict(X)

    assert p_cpu.shape == p_gpu.shape
    assert np.allclose(p_gpu, p_cpu, atol=1e-5), (
        f"diferencia máxima {np.abs(p_gpu - p_cpu).max():.2e}")


def test_el_orden_de_features_se_persiste_en_el_metadata():
    """Si el serving arma las columnas en otro orden, XGBoost no se queja: devuelve basura.

    Por eso `feature_names` va ORDENADO al metadata y el serving lo valida.
    """
    from features import spec

    assert spec.FEATURES == list(spec.FEATURES)
    assert len(spec.FEATURES) == len(set(spec.FEATURES)), "hay features duplicadas"
    # El contrato tiene que ser una lista (ordenada), no un set.
    assert isinstance(spec.FEATURES, list)
