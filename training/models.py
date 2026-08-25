"""Los modelos del canvas: XGBoost y Random Forest, más una logística de referencia.

El bloque 3 del canvas nombra **"XGBoost, RF"**. Los dos se implementan sobre el mismo
motor: XGBoost hace Random Forest con `num_parallel_tree` y `n_estimators=1`, así que
corren en GPU con el mismo código y sin agregar dependencias. (cuML de RAPIDS, que es el
Random Forest en GPU "de verdad", es **sólo Linux**: no es opción en Windows.)

La logística no está en el canvas, pero es el piso contra el cual se justifica la
complejidad de los árboles: si un modelo lineal de una línea empata con el boosting, el
boosting no está aportando.

## El problema de este dataset, y las cinco palancas

1.140 filas de entrenamiento y 143 features son ~8 observaciones por feature. El riesgo no
es falta de capacidad, es memorizar. Contra eso:

1. **Árboles chatos** (`max_depth=3`) y hojas gordas (`min_child_weight=10`, ~1 % de las
   filas).
2. **`colsample_bytree=0.5`** — cada árbol ve la mitad de las features. Con 143 columnas
   correlacionadas entre sí, es la palanca que más decorrelaciona.
3. **`max_bin=64`.** Merece explicación: con el default de 256 y 1.140 filas, cada bin del
   histograma tiene **4 observaciones** y los cortes son ruido. A 64 son ~18 obs/bin. Es
   regularización estructural, no una optimización de velocidad.
4. **Early stopping temporal** contra 2024-25, nunca contra el holdout.
5. **Promediado de semillas**: cada fit tarda menos de un segundo y con ±5 puntos de error
   estándar en el holdout, reducir varianza sale gratis.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from common.config import CFG
from training.device import DeviceInfo

MODELOS = ("xgb_gbt", "xgb_rf", "rf_sklearn", "hgb", "logreg")

# Alternativos, en training/models_alt.py: Poisson bivariado (deriva el 1X2 de las
# distribuciones de goles), logit ordinal (trata el empate como la franja entre dos
# umbrales) y red neuronal. Se comparan con `python -m training.compare_models`.


def hiperparametros(nombre: str, info: DeviceInfo, seed: int | None = None) -> dict[str, Any]:
    """Los hiperparámetros de partida, ya resueltos para el device elegido."""
    seed = CFG.seed if seed is None else seed

    if nombre == "xgb_gbt":
        p = dict(
            objective="multi:softprob", num_class=3, eval_metric="mlogloss",
            tree_method="hist", device=info.used,
            max_depth=3, min_child_weight=10,
            learning_rate=0.03, n_estimators=2000,
            subsample=0.8, colsample_bytree=0.5, colsample_bylevel=0.8,
            reg_lambda=5.0, reg_alpha=0.5, gamma=0.5,
            max_bin=64, random_state=seed, verbosity=0,
        )
    elif nombre == "xgb_rf":
        # Random Forest sobre el motor de XGBoost: un solo "round" de 300 árboles en
        # paralelo, sin learning rate. subsample=0.632 es la fracción esperada de una
        # muestra bootstrap, que es lo que hace un RF clásico.
        p = dict(
            objective="multi:softprob", num_class=3, eval_metric="mlogloss",
            tree_method="hist", device=info.used,
            num_parallel_tree=300, n_estimators=1, learning_rate=1.0,
            subsample=0.632, colsample_bynode=0.5,
            max_depth=6, min_child_weight=5, reg_lambda=1.0,
            max_bin=64, random_state=seed, verbosity=0,
        )
    else:
        raise ValueError(f"{nombre} no tiene hiperparámetros de XGBoost.")

    if info.used == "cpu" and info.n_jobs:
        p["n_jobs"] = info.n_jobs
    return p


def construir(nombre: str, info: DeviceInfo, seed: int | None = None,
              params: dict[str, Any] | None = None):
    """Fábrica de modelos."""
    if nombre not in MODELOS:
        raise ValueError(f"Modelo desconocido: {nombre!r}. Válidos: {MODELOS}")

    if nombre == "rf_sklearn":
        # El Random Forest "de verdad", para contrastar contra `xgb_rf`, que es XGBoost
        # en modo bosque. Diferencias que importan:
        #   - sklearn NO maneja NaN: hay que imputar, y eso borra la senial de "este
        #     equipo no tiene historia", que en este dataset es informacion real.
        #   - submuestrea features por nodo igual que xgb_rf, pero usa criterio de Gini
        #     en vez de la funcion de perdida con regularizacion.
        #   - se serializa con joblib (fragil entre versiones), no en un formato nativo
        #     portable. Para servir en Cloud Run eso es una desventaja concreta.
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline

        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("rf", RandomForestClassifier(
                n_estimators=400, max_depth=8, min_samples_leaf=10,
                max_features="sqrt", n_jobs=info.n_jobs or -1,
                random_state=CFG.seed if seed is None else seed)),
        ])

    if nombre == "hgb":
        # HistGradientBoosting de scikit-learn. Mismo espíritu que XGBoost pero otra
        # implementación; sobre este dataset dio el mejor resultado medido (accuracy
        # 0,500 y log-loss 1,031). Maneja NaN nativamente, igual que XGBoost.
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_depth=3, min_samples_leaf=20, learning_rate=0.03, max_iter=400,
            l2_regularization=5.0, max_bins=64, early_stopping=True,
            validation_fraction=0.15,
            random_state=CFG.seed if seed is None else seed)

    if nombre == "logreg":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        # ⚠️ En scikit-learn 1.9 el parámetro `multi_class` YA NO EXISTE. Con `lbfgs` y
        # 3 clases el ajuste es multinomial por defecto: pasarlo hace fallar el fit.
        #
        # La imputación va DENTRO del pipeline para que las medianas se ajusten sólo con
        # el train; calcularlas sobre todo el dataset sería leakage. Los árboles no la
        # necesitan: manejan el NaN nativamente, y "este equipo no tiene historia" es
        # información real que imputar borraría.
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            # `penalty="l2"` quedó deprecado en scikit-learn 1.8 (y `multi_class` ya no
            # existe en 1.9). L2 es el default de lbfgs, así que basta con regular C.
            ("lr", LogisticRegression(C=0.1, solver="lbfgs", max_iter=2000,
                                      random_state=CFG.seed if seed is None else seed)),
        ])

    import xgboost as xgb

    p = hiperparametros(nombre, info, seed)
    if params:
        p.update(params)
    return xgb.XGBClassifier(**p)


def promediar_probabilidades(probas: list[np.ndarray]) -> np.ndarray:
    """Media aritmética de las probabilidades de varias semillas.

    Se promedia la probabilidad, no el voto: preserva la información de confianza, que es
    lo que después usa la capa de decisión para calcular el valor esperado de la apuesta.
    """
    return np.mean(np.stack(probas, axis=0), axis=0)
