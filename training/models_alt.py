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

**Clasificador de marcadores.** No supone **nada** sobre la distribucion de goles: ni
forma ni independencia. Clasifica el marcador exacto —que se concentra muchisimo: 40
distintos en 1.004 partidos— y lo mapea a 1X2. Es el unico de la familia de goles que
**puede** poner un empate como resultado mas probable, porque 1-1 es el marcador mas
frecuente del dataset.

**Red neuronal (MLP).** La pregunta obligada. Con 1.140 filas y 159 features hay razones
para esperar que sobreajuste, pero se mide en vez de suponerlo.

Se agrega además `hgb` (HistGradientBoosting de scikit-learn) como contraste de
implementación contra XGBoost, y un **ensamble** que promedia probabilidades.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson

from common.config import CFG
from eda.baselines import CLASES_ORD

MAX_GOLES = 10

# Intervalo donde se busca `rho`. Los limites duros salen de exigir que las cuatro celdas
# corregidas queden >= 0 y se calculan con los datos (`_limites_rho`); esto es un cerco
# adicional, porque un rho grande es sintoma de sobreajuste a cuatro celdas y no de una
# correlacion real. Dixon-Coles reportan alrededor de -0,13 para el futbol ingles.
RHO_MIN, RHO_MAX = -0.4, 0.4


def tau_dixon_coles(x, y, lam, mu, rho: float):
    """El factor de correccion de Dixon-Coles para las cuatro celdas de marcador bajo.

        tau(0,0) = 1 - lam*mu*rho        tau(0,1) = 1 + lam*rho
        tau(1,0) = 1 + mu*rho            tau(1,1) = 1 - rho
        tau(x,y) = 1                     en el resto

    `lam` es la tasa de goles del local y `mu` la del visitante.

    **Para que sirve.** El modelo base multiplica las dos distribuciones, o sea asume que
    los goles de un equipo no dicen nada de los del otro. Eso es falso justo en los
    marcadores bajos: un 0-0 y un 1-1 ocurren mas seguido de lo que predice la
    independencia, y un 1-0 menos. Y esas son precisamente las celdas donde vive el empate.

    Con **rho negativo** —el signo que da en el futbol ingles— sube 0-0 y 1-1 y baja 1-0 y
    0-1: **sube P(empate)**, que es el efecto buscado.
    """
    x, y = np.asarray(x), np.asarray(y)
    lam, mu = np.asarray(lam, dtype=float), np.asarray(mu, dtype=float)
    t = np.ones(np.broadcast(x, y, lam, mu).shape, dtype=float)
    t = np.where((x == 0) & (y == 0), 1.0 - lam * mu * rho, t)
    t = np.where((x == 0) & (y == 1), 1.0 + lam * rho, t)
    t = np.where((x == 1) & (y == 0), 1.0 + mu * rho, t)
    t = np.where((x == 1) & (y == 1), 1.0 - rho, t)
    return t


def _limites_rho(lam, mu) -> tuple[float, float]:
    """Los rho que mantienen las cuatro celdas corregidas >= 0, dados lam y mu.

        1 - lam*mu*rho >= 0   ->   rho <= 1/max(lam*mu)
        1 + lam*rho    >= 0   ->   rho >= -1/max(lam)
        1 + mu*rho     >= 0   ->   rho >= -1/max(mu)
        1 - rho        >= 0   ->   rho <= 1
    """
    lam, mu = np.asarray(lam, dtype=float), np.asarray(mu, dtype=float)
    lo = max(-1.0 / max(lam.max(), 1e-9), -1.0 / max(mu.max(), 1e-9))
    hi = min(1.0, 1.0 / max((lam * mu).max(), 1e-9))
    return max(lo, RHO_MIN), min(hi, RHO_MAX)


def ajustar_rho(x, y, lam, mu) -> float:
    """`rho` por maxima verosimilitud sobre el train.

    El termino de Poisson **no depende de rho**, asi que maximizar la verosimilitud
    completa se reduce a maximizar `sum(log tau)`. Y como tau vale 1 fuera de las cuatro
    celdas, sólo aportan los partidos que terminaron 0-0, 1-0, 0-1 o 1-1: es una
    optimizacion de UNA sola variable sobre un puñado de filas.
    """
    from scipy.optimize import minimize_scalar

    lo, hi = _limites_rho(lam, mu)
    if not np.isfinite([lo, hi]).all() or lo >= hi:
        return 0.0

    def neg_ll(rho: float) -> float:
        t = tau_dixon_coles(x, y, lam, mu, float(rho))
        return -np.log(np.clip(t, 1e-12, None)).sum()

    # `bounded` no necesita derivada y no se escapa del intervalo: con una sola variable
    # y un intervalo chico es de sobra, y no puede devolver un rho que rompa la positividad.
    r = minimize_scalar(neg_ll, bounds=(lo, hi), method="bounded")
    return float(r.x) if r.success else 0.0


class PoissonBivariado:
    """Modela los goles de cada equipo y deriva el 1X2. El empate sale solo.

    Dos regresores de Poisson —uno para los goles del local, otro para los del visitante—
    y después la convolución de las dos distribuciones. `P(empate)` es la diagonal.

    La virtud para este proyecto: el empate no se aprende como clase, se **deduce**. Un
    clasificador tiene que descubrir que "empate" significa "los dos marcan lo mismo"; acá
    eso está impuesto por la estructura del modelo.

    Con `dixon_coles=True` se aplica además la corrección de `tau_dixon_coles`, que levanta
    la suposición de independencia en los marcadores bajos. Va apagada por defecto para que
    el modelo sin corregir siga existiendo y la comparación sea contra sí mismo.
    """

    def __init__(self, device: str = "cpu", seed: int | None = None,
                 dixon_coles: bool = False, **params):
        import xgboost as xgb

        base = dict(objective="count:poisson", tree_method="hist", device=device,
                    max_depth=3, min_child_weight=10, learning_rate=0.03,
                    n_estimators=400, subsample=0.8, colsample_bytree=0.5,
                    reg_lambda=5.0, max_bin=64,
                    random_state=CFG.seed if seed is None else seed, verbosity=0)
        base.update(params)
        self.m_local = xgb.XGBRegressor(**base)
        self.m_visita = xgb.XGBRegressor(**base)
        self.dixon_coles = dixon_coles
        self.rho = 0.0

    def fit(self, X, goles_local, goles_visita, **kw):
        self.m_local.fit(X, goles_local)
        self.m_visita.fit(X, goles_visita)

        if self.dixon_coles:
            # El rho se ajusta con las tasas PREDICHAS sobre el propio train, que es lo
            # que despues va a ver `predict_proba`. Con las tasas reales el rho saldria
            # optimista: estaria corrigiendo un modelo que no es el que se usa.
            lam = np.clip(self.m_local.predict(X), 1e-6, None)
            mu = np.clip(self.m_visita.predict(X), 1e-6, None)
            self.rho = ajustar_rho(np.asarray(goles_local), np.asarray(goles_visita),
                                   lam, mu)
        return self

    def predict_proba(self, X) -> np.ndarray:
        lam_l = np.clip(self.m_local.predict(X), 1e-6, None)
        lam_v = np.clip(self.m_visita.predict(X), 1e-6, None)

        k = np.arange(MAX_GOLES + 1)
        p_l = poisson.pmf(k[None, :], lam_l[:, None])      # (n, k)
        p_v = poisson.pmf(k[None, :], lam_v[:, None])

        # Matriz conjunta bajo independencia: filas = goles del local, cols = del visitante.
        conj = p_l[:, :, None] * p_v[:, None, :]

        if self.dixon_coles and self.rho != 0.0:
            # Sólo las cuatro celdas de marcador bajo; el resto queda intacto.
            conj[:, 0, 0] *= 1.0 - lam_l * lam_v * self.rho
            conj[:, 0, 1] *= 1.0 + lam_l * self.rho
            conj[:, 1, 0] *= 1.0 + lam_v * self.rho
            conj[:, 1, 1] *= 1.0 - self.rho
            # `ajustar_rho` ya acota rho para que esto no pase, pero un rho pasado a mano
            # sí podría: una probabilidad negativa se cortaría en silencio al normalizar.
            conj = np.clip(conj, 0.0, None)

        idx = np.arange(MAX_GOLES + 1)
        gana_local = conj[:, idx[:, None] > idx[None, :]].sum(axis=1)
        empate = conj[:, idx[:, None] == idx[None, :]].sum(axis=1)
        gana_visita = conj[:, idx[:, None] < idx[None, :]].sum(axis=1)

        P = np.stack([gana_visita, empate, gana_local], axis=1)  # orden CLASES_ORD
        return P / P.sum(axis=1, keepdims=True)


class ClasificadorMarcador:
    """Clasifica el MARCADOR EXACTO y despues lo mapea a 1X2.

    Es la tercera forma de atacar el empate, y la unica que no supone **nada** sobre la
    distribucion de goles: ni forma (Poisson) ni independencia entre los dos equipos. El
    modelo aprende `P(marcador)` directamente de los datos.

    **Hay que enumerar las clases, no alcanza con "son enteros".** Un clasificador necesita
    un conjunto finito de etiquetas; no puede emitir un par de numeros. Y eso es viable
    porque los marcadores se concentran muchisimo: en las 1.004 filas de entrenamiento hay
    **40 marcadores distintos**, los 20 mas frecuentes cubren el 91,5 % y solo 4 aparecen
    una vez.

    Los marcadores por debajo de `min_frecuencia` van a tres bolsas —`otro_home`,
    `otro_draw`, `otro_away`— y no a una sola. **La diferencia importa**: una bolsa unica
    mezclaria 4-2 con 2-4 y con 3-3, y no habria forma de mapearla a un resultado. Con tres
    bolsas el mapeo a 1X2 sigue siendo deterministico y cubre el 100 % de los partidos.

    La virtud sobre el Poisson: como 1-1 es el marcador mas frecuente del dataset (107 de
    1.004), este modelo **puede** poner un empate como resultado mas probable, cosa que el
    Poisson no hace nunca. La medicion de `training/empate.py` dice si eso sirve de algo.
    """

    def __init__(self, device: str = "cpu", seed: int | None = None,
                 min_frecuencia: int = 20, **params):
        self.device = device
        self.seed = CFG.seed if seed is None else seed
        self.min_frecuencia = min_frecuencia
        self.params = params
        self.clases_: list[str] = []
        self.mapa_: np.ndarray | None = None

    @staticmethod
    def _resultado(gl: int, gv: int) -> str:
        return "home" if gl > gv else ("away" if gl < gv else "draw")

    def _etiquetar(self, gl, gv) -> np.ndarray:
        marc = np.array([f"{a}-{b}" for a, b in zip(gl, gv)])
        return np.where(np.isin(marc, list(self.frecuentes_)), marc,
                        np.array([f"otro_{self._resultado(a, b)}"
                                  for a, b in zip(gl, gv)]))

    def fit(self, X, goles_local, goles_visita, **kw):
        import xgboost as xgb

        gl = np.asarray(goles_local, dtype=int)
        gv = np.asarray(goles_visita, dtype=int)

        marc = pd.Series([f"{a}-{b}" for a, b in zip(gl, gv)])
        frec = marc.value_counts()
        self.frecuentes_ = set(frec[frec >= self.min_frecuencia].index)

        etiquetas = self._etiquetar(gl, gv)
        self.clases_ = sorted(str(e) for e in set(etiquetas))
        indice = {c: i for i, c in enumerate(self.clases_)}

        # Mapa clase -> 1X2, en el orden de CLASES_ORD. Es deterministico: cada etiqueta
        # tiene un unico resultado posible, incluidas las tres bolsas de "otro".
        self.mapa_ = np.zeros((len(self.clases_), 3))
        for c, i in indice.items():
            r = c.split("_")[1] if c.startswith("otro_") else self._resultado(
                *(int(v) for v in c.split("-")))
            self.mapa_[i, list(CLASES_ORD).index(r)] = 1.0

        base = dict(objective="multi:softprob", num_class=len(self.clases_),
                    tree_method="hist", device=self.device, max_depth=3,
                    min_child_weight=10, learning_rate=0.03, n_estimators=400,
                    subsample=0.8, colsample_bytree=0.5, reg_lambda=5.0, max_bin=64,
                    random_state=self.seed, verbosity=0)
        base.update(self.params)
        self.modelo = xgb.XGBClassifier(**base)
        self.modelo.fit(X, np.array([indice[e] for e in etiquetas]))
        return self

    def predict_proba(self, X) -> np.ndarray:
        P = self.modelo.predict_proba(X) @ self.mapa_
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
