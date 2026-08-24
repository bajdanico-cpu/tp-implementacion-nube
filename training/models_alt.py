"""Modelos alternativos, elegidos para atacar el problema del empate.

El análisis de `training/analysis.py` mostró dos cosas:

1. **El empate es la única clase de apuesta rentable** (ROI +0,092 contra −0,106 del
   visitante y −0,038 del local), porque el mercado lo subestima sistemáticamente.
2. **Es también donde el modelo pierde más log-loss** (1,472 contra 1,388 del mercado).

O sea que el empate es a la vez la debilidad del modelo y la oportunidad del negocio. Los
modelos de acá lo atacan de tres formas distintas:

**Poisson bivariado.** No predice la clase: predice **cuántos goles hace cada equipo**, y
de ahí deriva las tres probabilidades. El empate deja de ser una etiqueta arbitraria y
pasa a ser lo que realmente es: `P(empate) = suma sobre k de P(local=k)·P(visita=k)`. Es
la familia Dixon-Coles, el modelo clásico del fútbol, y es la respuesta estructuralmente
correcta a "¿hace falta predecir el empate?".

**Logit ordinal.** Aprovecha que las tres clases tienen un **orden natural**: derrota <
empate < victoria, sobre un eje latente de superioridad. Un multiclase común trata a las
tres como categorías sin relación; el ordinal sabe que el empate está *en el medio*, que es
exactamente por qué es difícil de predecir como argmax.

**Red neuronal (MLP).** La pregunta obligada. Con 1.140 filas y 159 features hay razones
para esperar que sobreajuste, pero se mide en vez de suponerlo.

Se agrega además `hgb` (HistGradientBoosting de scikit-learn) como contraste de
implementación contra XGBoost, y un **ensamble** que promedia probabilidades.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from common.config import CFG
from eda.baselines import CLASES_ORD

MAX_GOLES = 10


class PoissonBivariado:
    """Modela los goles de cada equipo y deriva el 1X2. El empate sale solo.

    Dos regresores de Poisson —uno para los goles del local, otro para los del visitante—
    y después la convolución de las dos distribuciones. `P(empate)` es la diagonal.

    La virtud para este proyecto: el empate no se aprende como clase, se **deduce**. Un
    clasificador tiene que descubrir que "empate" significa "los dos marcan lo mismo"; acá
    eso está impuesto por la estructura del modelo.
    """

    def __init__(self, device: str = "cpu", seed: int | None = None, **params):
        import xgboost as xgb

        base = dict(objective="count:poisson", tree_method="hist", device=device,
                    max_depth=3, min_child_weight=10, learning_rate=0.03,
                    n_estimators=400, subsample=0.8, colsample_bytree=0.5,
                    reg_lambda=5.0, max_bin=64,
                    random_state=CFG.seed if seed is None else seed, verbosity=0)
        base.update(params)
        self.m_local = xgb.XGBRegressor(**base)
        self.m_visita = xgb.XGBRegressor(**base)

    def fit(self, X, goles_local, goles_visita, **kw):
        self.m_local.fit(X, goles_local)
        self.m_visita.fit(X, goles_visita)
        return self

    def predict_proba(self, X) -> np.ndarray:
        lam_l = np.clip(self.m_local.predict(X), 1e-6, None)
        lam_v = np.clip(self.m_visita.predict(X), 1e-6, None)

        k = np.arange(MAX_GOLES + 1)
        p_l = poisson.pmf(k[None, :], lam_l[:, None])      # (n, k)
        p_v = poisson.pmf(k[None, :], lam_v[:, None])

        # Matriz conjunta bajo independencia: filas = goles del local, cols = del visitante.
        conj = p_l[:, :, None] * p_v[:, None, :]
        idx = np.arange(MAX_GOLES + 1)
        gana_local = conj[:, idx[:, None] > idx[None, :]].sum(axis=1)
        empate = conj[:, idx[:, None] == idx[None, :]].sum(axis=1)
        gana_visita = conj[:, idx[:, None] < idx[None, :]].sum(axis=1)

        P = np.stack([gana_visita, empate, gana_local], axis=1)  # orden CLASES_ORD
        return P / P.sum(axis=1, keepdims=True)


class LogitOrdinal:
    """Clasificación ordinal por el método de umbrales acumulados.

    Se entrenan dos clasificadores binarios sobre el orden away < draw < home:

        A: P(resultado > away)      = P(draw) + P(home)
        B: P(resultado > draw)      = P(home)

    y de ahí P(away) = 1 - A, P(draw) = A - B, P(home) = B. El empate queda definido como
    **la franja entre dos umbrales**, que es la forma correcta de tratar una clase que vive
    en el medio de un continuo latente.
    """

    def __init__(self, seed: int | None = None, C: float = 0.1):
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        def _crear():
            return Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
                ("lr", LogisticRegression(C=C, max_iter=2000,
                                          random_state=CFG.seed if seed is None else seed)),
            ])

        self.a, self.b = _crear(), _crear()

    def fit(self, X, y, **kw):
        y = np.asarray(y)
        self.a.fit(X, (y > 0).astype(int))   # > away
        self.b.fit(X, (y > 1).astype(int))   # > draw
        return self

    def predict_proba(self, X) -> np.ndarray:
        pa = self.a.predict_proba(X)[:, 1]
        pb = self.b.predict_proba(X)[:, 1]
        # Los umbrales pueden cruzarse; se fuerza la monotonía antes de restar.
        pb = np.minimum(pb, pa)
        P = np.stack([1 - pa, pa - pb, pb], axis=1)
        P = np.clip(P, 1e-9, 1)
        return P / P.sum(axis=1, keepdims=True)


def mlp(seed: int | None = None):
    """Red neuronal chica. Con 1.140 filas es la que más riesgo de sobreajuste tiene.

    Dos capas ocultas de 64 y 32, `alpha` alto (regularización L2 fuerte) y early stopping
    interno. Escalar es obligatorio: sin eso una red no converge con features de escalas
    tan distintas (Elo ~1.500 contra probabilidades ~0,3).
    """
    from sklearn.impute import SimpleImputer
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("nn", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1.0,
                             learning_rate_init=1e-3, max_iter=1500,
                             early_stopping=True, n_iter_no_change=30,
                             validation_fraction=0.15,
                             random_state=CFG.seed if seed is None else seed)),
    ])


def hgb(seed: int | None = None):
    """HistGradientBoosting de scikit-learn: contraste de implementación contra XGBoost."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_depth=3, min_samples_leaf=20, learning_rate=0.03, max_iter=400,
        l2_regularization=5.0, max_bins=64, early_stopping=True,
        validation_fraction=0.15, random_state=CFG.seed if seed is None else seed)


def ensamble(probas: list[np.ndarray], pesos: list[float] | None = None) -> np.ndarray:
    """Promedio ponderado de probabilidades.

    Se promedian probabilidades y no votos: preserva la confianza, que es lo que después
    usa la capa de decisión para el valor esperado.
    """
    P = np.stack(probas, axis=0)
    if pesos is None:
        out = P.mean(axis=0)
    else:
        w = np.asarray(pesos, dtype=float)
        w /= w.sum()
        out = np.tensordot(w, P, axes=(0, 0))
    return out / out.sum(axis=1, keepdims=True)


ORDEN_CLASES = list(CLASES_ORD)
