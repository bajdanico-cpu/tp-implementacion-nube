"""El modelo de goles y la corrección de Dixon-Coles.

`PoissonBivariado` no predice la clase: predice **cuántos goles hace cada equipo** y deriva
el 1X2 de la distribución conjunta. Es la lógica más delicada del repo después del control
anti-leakage, porque un error acá no rompe nada — devuelve probabilidades que suman 1 y
parecen razonables.

Los tests van sobre `lam` y `mu` sintéticos, sin tocar Gold: corren en milisegundos y no
dependen de que el pipeline esté construido.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import poisson

from eda.baselines import CLASES_ORD
from training import models_alt as ma


class _RegresorFijo:
    """Devuelve siempre la misma tasa. Reemplaza a XGBoost para aislar la matemática."""

    def __init__(self, tasa: float):
        self.tasa = tasa

    def predict(self, X):
        return np.full(len(X), self.tasa, dtype=float)


def _modelo(lam: float, mu: float, dixon_coles: bool = False, rho: float = 0.0):
    m = ma.PoissonBivariado.__new__(ma.PoissonBivariado)
    m.m_local = _RegresorFijo(lam)
    m.m_visita = _RegresorFijo(mu)
    m.dixon_coles = dixon_coles
    m.rho = rho
    return m


X = np.zeros((3, 1))          # el contenido no importa: los regresores son fijos


# --------------------------------------------------------------------------- #
#  El contrato de salida
# --------------------------------------------------------------------------- #

def test_las_tres_probabilidades_suman_uno():
    P = _modelo(1.6, 1.2).predict_proba(X)
    assert P.shape == (3, 3)
    assert np.allclose(P.sum(axis=1), 1.0)


def test_las_columnas_salen_en_el_orden_de_clases_ord():
    """`CLASES_ORD` es (away, draw, home): el orden lexicográfico que espera `log_loss`.

    Es la trampa que el repo ya documenta para `sklearn.metrics.log_loss`. Con un local
    mucho más fuerte, la columna de `home` tiene que ser la más alta.
    """
    assert list(CLASES_ORD) == ["away", "draw", "home"]

    P = _modelo(3.0, 0.4).predict_proba(X)[0]
    assert P[2] > P[1] and P[2] > P[0], "gana el local: p_home tiene que ser la mayor"

    P = _modelo(0.4, 3.0).predict_proba(X)[0]
    assert P[0] > P[1] and P[0] > P[2], "gana el visitante: p_away tiene que ser la mayor"


def test_sin_correccion_el_empate_es_exactamente_la_diagonal():
    """Fija la independencia como comportamiento del modelo base.

    Sin corrección, P(empate) tiene que ser exactamente `sum_k P(local=k)·P(visita=k)`.
    Si alguien toca la construcción de la conjunta, esto lo detecta.
    """
    lam, mu = 1.6, 1.2
    P = _modelo(lam, mu).predict_proba(X)
    k = np.arange(ma.MAX_GOLES + 1)
    esperado = float((poisson.pmf(k, lam) * poisson.pmf(k, mu)).sum())

    # La conjunta se normaliza, así que se compara contra el mismo denominador.
    total = float((poisson.pmf(k, lam).sum()) * (poisson.pmf(k, mu).sum()))
    assert P[0, 1] == pytest.approx(esperado / total, rel=1e-9)


# --------------------------------------------------------------------------- #
#  La tau de Dixon-Coles
# --------------------------------------------------------------------------- #

def test_tau_vale_uno_fuera_de_las_cuatro_celdas():
    for x, y in ((2, 3), (0, 2), (2, 0), (5, 5)):
        assert ma.tau_dixon_coles(x, y, 1.5, 1.2, -0.13) == pytest.approx(1.0)


def test_con_rho_negativo_tau_sube_los_empates_y_baja_el_1_0():
    """Es la razón de ser de la corrección, escrita como test."""
    lam, mu, rho = 1.5, 1.2, -0.13
    assert ma.tau_dixon_coles(0, 0, lam, mu, rho) > 1.0
    assert ma.tau_dixon_coles(1, 1, lam, mu, rho) > 1.0
    assert ma.tau_dixon_coles(1, 0, lam, mu, rho) < 1.0
    assert ma.tau_dixon_coles(0, 1, lam, mu, rho) < 1.0


def test_con_rho_cero_la_correccion_es_un_no_op():
    """Garantiza que activar Dixon-Coles no cambie nada por sí solo."""
    P_sin = _modelo(1.6, 1.2, dixon_coles=False).predict_proba(X)
    P_con = _modelo(1.6, 1.2, dixon_coles=True, rho=0.0).predict_proba(X)
    assert np.allclose(P_sin, P_con)


def test_con_rho_negativo_sube_la_probabilidad_de_empate():
    """El efecto que justifica todo el cambio."""
    P_sin = _modelo(1.6, 1.2, dixon_coles=True, rho=0.0).predict_proba(X)
    P_con = _modelo(1.6, 1.2, dixon_coles=True, rho=-0.13).predict_proba(X)
    assert P_con[0, 1] > P_sin[0, 1]


def test_la_conjunta_corregida_no_tiene_probabilidades_negativas():
    """Un rho fuera de rango se clipea en vez de colarse como probabilidad negativa."""
    P = _modelo(3.0, 3.0, dixon_coles=True, rho=0.9).predict_proba(X)
    assert (P >= 0).all()
    assert np.allclose(P.sum(axis=1), 1.0)


# --------------------------------------------------------------------------- #
#  El ajuste de rho
# --------------------------------------------------------------------------- #

def test_los_limites_de_rho_garantizan_tau_no_negativa():
    lam = np.full(50, 2.0)
    mu = np.full(50, 1.5)
    lo, hi = ma._limites_rho(lam, mu)
    assert lo < hi
    for rho in (lo, hi, 0.0, (lo + hi) / 2):
        for x, y in ((0, 0), (0, 1), (1, 0), (1, 1)):
            assert ma.tau_dixon_coles(x, y, 2.0, 1.5, rho) >= -1e-12


def test_ajustar_rho_recupera_el_signo_cuando_los_empates_sobran():
    """Con exceso de 0-0 y de 1-1 respecto de la independencia, rho tiene que dar < 0."""
    rng = np.random.default_rng(0)
    n = 4000
    lam = np.full(n, 1.4)
    mu = np.full(n, 1.2)
    x = rng.poisson(1.4, n)
    y = rng.poisson(1.2, n)

    # Se fuerza el exceso: una porción de los partidos pasa a 0-0 y a 1-1.
    idx = rng.choice(n, size=500, replace=False)
    x[idx[:250]], y[idx[:250]] = 0, 0
    x[idx[250:]], y[idx[250:]] = 1, 1

    assert ma.ajustar_rho(x, y, lam, mu) < 0


def test_ajustar_rho_queda_dentro_del_intervalo_acotado():
    rng = np.random.default_rng(1)
    n = 2000
    lam, mu = np.full(n, 1.5), np.full(n, 1.3)
    rho = ma.ajustar_rho(rng.poisson(1.5, n), rng.poisson(1.3, n), lam, mu)
    lo, hi = ma._limites_rho(lam, mu)
    assert lo <= rho <= hi
    assert ma.RHO_MIN <= rho <= ma.RHO_MAX


# --------------------------------------------------------------------------- #
#  El ensamble
# --------------------------------------------------------------------------- #

def test_el_ensamble_promedia_y_renormaliza():
    A = np.array([[0.2, 0.3, 0.5]])
    B = np.array([[0.6, 0.2, 0.2]])
    out = ma.ensamble([A, B])
    assert np.allclose(out, [[0.4, 0.25, 0.35]])
    assert np.allclose(out.sum(axis=1), 1.0)


def test_el_ensamble_respeta_los_pesos():
    A = np.array([[0.0, 0.0, 1.0]])
    B = np.array([[1.0, 0.0, 0.0]])
    assert np.allclose(ma.ensamble([A, B], [1.0, 0.0]), A)
    assert np.allclose(ma.ensamble([A, B], [0.0, 1.0]), B)


# --------------------------------------------------------------------------- #
#  El clasificador de marcadores
# --------------------------------------------------------------------------- #

def _datos_marcador(n: int = 400):
    """Marcadores sinteticos: unos pocos frecuentes y una cola larga de raros."""
    rng = np.random.default_rng(7)
    frec = [(1, 1), (1, 0), (2, 1), (0, 1), (2, 0)]
    gl, gv = [], []
    for i in range(n):
        if i % 5 < 4:                       # 80 % de los partidos, marcadores comunes
            a, b = frec[i % len(frec)]
        else:                               # 20 %, cola larga
            a, b = int(rng.integers(3, 7)), int(rng.integers(0, 7))
        gl.append(a)
        gv.append(b)
    return np.array(gl), np.array(gv), rng.normal(size=(n, 4))


def test_el_mapa_a_1x2_es_deterministico_y_cubre_todas_las_clases():
    """Cada etiqueta tiene un unico resultado posible, bolsas de 'otro' incluidas.

    Es la razon por la que las raras van a TRES bolsas y no a una: una sola mezclaria
    4-2 con 2-4 y con 3-3, y no habria forma de mapearla.
    """
    gl, gv, X = _datos_marcador()
    m = ma.ClasificadorMarcador(min_frecuencia=20).fit(X, gl, gv)

    assert m.mapa_ is not None
    assert m.mapa_.shape == (len(m.clases_), 3)
    # Exactamente un 1 por fila: el mapeo no es ambiguo para ninguna clase.
    assert np.all(m.mapa_.sum(axis=1) == 1.0)


def test_las_bolsas_de_otro_mapean_al_resultado_que_dice_su_nombre():
    gl, gv, X = _datos_marcador()
    m = ma.ClasificadorMarcador(min_frecuencia=20).fit(X, gl, gv)

    for i, c in enumerate(m.clases_):
        esperado = (c.split("_")[1] if c.startswith("otro_")
                    else ma.ClasificadorMarcador._resultado(*(int(v) for v in c.split("-"))))
        assert m.mapa_[i, list(CLASES_ORD).index(esperado)] == 1.0, c


def test_el_clasificador_de_marcadores_devuelve_probabilidades_validas():
    gl, gv, X = _datos_marcador()
    m = ma.ClasificadorMarcador(min_frecuencia=20).fit(X, gl, gv)
    P = m.predict_proba(X[:10])
    assert P.shape == (10, 3)
    assert np.allclose(P.sum(axis=1), 1.0)
    assert (P >= 0).all()


def test_un_marcador_raro_cae_en_la_bolsa_de_su_resultado():
    gl, gv, X = _datos_marcador()
    m = ma.ClasificadorMarcador(min_frecuencia=20).fit(X, gl, gv)
    # 6-0 es raro por construccion: no puede ser una clase propia, y su bolsa es la de home.
    assert "6-0" not in m.clases_
    assert "otro_home" in m.clases_
