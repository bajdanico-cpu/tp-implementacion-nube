"""Experimentos de ablación, para decidir con evidencia en vez de con intuición.

Cada variante cambia UNA cosa respecto de la base y se evalúa sobre el mismo holdout
2025-26, con el mismo protocolo (early stopping temporal contra 2024-25, promediado de
semillas). Así la comparación significa algo.

    python -m training.experiments

Las preguntas que responde:

- ¿Entrenar más rondas mejora, o el early stopping ya cortó donde había que cortar?
- ¿Conviene sacar del entrenamiento las fechas de 2022-23 sin xG, **pero conservarlas como
  historia** para las ventanas de los partidos siguientes?
- ¿Pesar la clase empate hace que el modelo la prediga, y a qué costo?
- ¿La calibración por temperatura arregla el sesgo del empate?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common.config import CFG, PROJECT_ROOT
from common.logging_setup import get_logger, setup
from features import spec
from training import dataset, evaluate, metrics, models
from training.device import resolve

log = get_logger(__name__)

SALIDA = PROJECT_ROOT / "training" / "output"


def _entrenar_y_evaluar(gold: pd.DataFrame, info, nombre: str,
                        filtro_train=None, params: dict | None = None,
                        pesos_clase: dict | None = None,
                        temperatura: bool = False,
                        features: list[str] | None = None) -> dict:
    """Un experimento: entrena con lo que se le indique y evalúa sobre el holdout.

    `filtro_train` recorta las filas que se usan **como objetivo de entrenamiento**. NO
    toca las features: éstas ya vienen calculadas en Gold sobre la historia completa, así
    que un partido excluido del train sigue contando como historia para los posteriores.
    Ésa es justamente la separación que se quiere probar.
    """
    features = features or spec.FEATURES
    val_season, test_season = CFG.valid_season, CFG.holdout_season
    train_seasons = [s for s in CFG.seasons_for_training() if s != val_season]

    tr = gold[gold["season"].isin(train_seasons)]
    va = gold[gold["season"] == val_season]
    te = gold[gold["season"] == test_season]
    if filtro_train is not None:
        antes = len(tr)
        tr = tr[filtro_train(tr)]
        log.info("  %s: train %d -> %d filas", nombre, antes, len(tr))

    X_tr, y_tr = dataset.matriz(tr, features), dataset.codificar(tr["target_1x2"])
    X_va, y_va = dataset.matriz(va, features), dataset.codificar(va["target_1x2"])

    w_tr = _pesos(y_tr, pesos_clase)
    m = models.construir(nombre, info, params=params)
    m.set_params(early_stopping_rounds=100)
    m.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_va, y_va)], verbose=False)
    best = int(m.best_iteration)

    # La temperatura se ajusta con ESTE modelo, que NO vio la validación. Hacerlo con el
    # refiteado —que sí la incluye— la ajustaría contra predicciones artificialmente
    # buenas y saldría un T que agudiza en vez de aplanar. Es un leakage sutil que
    # empeoró el log-loss de 1,040 a 1,205 en la primera corrida.
    T = _ajustar_temperatura(m.predict_proba(X_va), y_va) if temperatura else None

    # Refit con train + validación, con el nº de rondas ya fijado.
    full = pd.concat([tr, va])
    X_f, y_f = dataset.matriz(full, features), dataset.codificar(full["target_1x2"])
    w_f = _pesos(y_f, pesos_clase)

    p_full = {**(params or {}), "n_estimators": best + 1}
    modelos = []
    for i in range(CFG.n_seeds):
        mm = models.construir(nombre, info, seed=CFG.seed + i, params=p_full)
        mm.fit(X_f, y_f, sample_weight=w_f)
        modelos.append(mm)

    X_te = dataset.matriz(te, features)
    proba = models.promediar_probabilidades([mm.predict_proba(X_te) for mm in modelos])

    if temperatura:
        log.info("  temperatura ajustada: T=%.3f", T)
        proba = _aplicar_temperatura(proba, T)

    y_te = te["target_1x2"].to_numpy()
    pred = dataset.decodificar(proba.argmax(axis=1))
    rep = metrics.reporte(y_te, pred, proba)
    rep.update(n_train=len(full), best_iteration=best,
               empates_predichos=int((proba.argmax(axis=1) == 1).sum()),
               p_draw_media=float(proba[:, 1].mean()))
    return rep


def _pesos(y: np.ndarray, pesos_clase: dict | None) -> np.ndarray | None:
    if not pesos_clase:
        return None
    return np.array([pesos_clase.get(dataset.CLASES[i], 1.0) for i in y])


def _ajustar_temperatura(proba_val: np.ndarray, y_val: np.ndarray) -> float:
    """Un solo escalar que aplana o agudiza las probabilidades. Minimiza log-loss."""
    from scipy.optimize import minimize_scalar

    eps = 1e-12
    logp = np.log(np.clip(proba_val, eps, 1))

    def perdida(T):
        z = logp / max(T, 1e-3)
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        return -np.mean(np.log(np.clip(p[np.arange(len(y_val)), y_val], eps, 1)))

    return float(minimize_scalar(perdida, bounds=(0.3, 5.0), method="bounded").x)


def _aplicar_temperatura(proba: np.ndarray, T: float) -> np.ndarray:
    z = np.log(np.clip(proba, 1e-12, 1)) / T
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Las variantes
# ---------------------------------------------------------------------------

def variantes() -> dict:
    """Cada entrada cambia UNA cosa respecto de la base."""
    return {
        "base": {},

        # ¿El early stopping cortó donde había que cortar, o falta entrenar?
        "lr_lento_3x_rondas": {"params": {"learning_rate": 0.01, "n_estimators": 6000}},
        "lr_rapido": {"params": {"learning_rate": 0.10}},
        "arboles_profundos": {"params": {"max_depth": 6, "min_child_weight": 3}},

        # ¿Conviene sacar del train las fechas sin xG, conservándolas como historia?
        "sin_gw_sin_xg": {
            "filtro_train": lambda d: d["xg_available"],
        },
        "sin_2022_23_entera": {
            "filtro_train": lambda d: d["season"] != "2022-23",
        },

        # ¿Se puede forzar al modelo a predecir empates, y a qué costo?
        "peso_empate_x1.5": {"pesos_clase": {"draw": 1.5}},
        "peso_empate_x2.5": {"pesos_clase": {"draw": 2.5}},

        # ¿La calibración arregla el sesgo del empate?
        "temperatura": {"temperatura": True},

        # Combinaciones de lo que funcionó por separado.
        "sin_xg_falso + peso_1.5": {
            "filtro_train": lambda d: d["xg_available"],
            "pesos_clase": {"draw": 1.5},
        },
        "sin_2022_23 + peso_1.5": {
            "filtro_train": lambda d: d["season"] != "2022-23",
            "pesos_clase": {"draw": 1.5},
        },
        "sin_xg_falso + peso_1.5 + temp": {
            "filtro_train": lambda d: d["xg_available"],
            "pesos_clase": {"draw": 1.5},
            "temperatura": True,
        },
    }


def correr(nombre_modelo: str = "xgb_gbt") -> pd.DataFrame:
    info = resolve("auto")
    gold = dataset.cargar()

    filas = []
    for etiqueta, kwargs in variantes().items():
        log.info("--- %s ---", etiqueta)
        rep = _entrenar_y_evaluar(gold, info, nombre_modelo, **kwargs)
        filas.append({
            "variante": etiqueta,
            "n_train": rep["n_train"],
            "rondas": rep["best_iteration"],
            "accuracy": rep["accuracy"],
            "ic_bajo": rep["accuracy_ic95"][0],
            "ic_alto": rep["accuracy_ic95"][1],
            "f1_macro": rep["f1_macro"],
            "f1_draw": rep["f1_draw"],
            "log_loss": rep["log_loss"],
            "empates_pred": rep["empates_predichos"],
            "p_draw_media": rep["p_draw_media"],
        })
    return pd.DataFrame(filas)


def main() -> None:
    setup(CFG.log_level, CFG.log_format)
    SALIDA.mkdir(parents=True, exist_ok=True)
    df = correr()
    df.to_csv(SALIDA / "experimentos.csv", index=False)

    base = df[df.variante == "base"].iloc[0]
    df["d_acc"] = (df["accuracy"] - base["accuracy"]).round(4)
    df["d_ll"] = (df["log_loss"] - base["log_loss"]).round(4)

    print("\nHoldout 2025-26 (380 partidos). El empate ocurre en el 27,4 %.")
    print("Referencia: el MERCADO nunca pone el empate como argmax (0 de 380).\n")
    print(df.round(4).to_string(index=False))
    print(f"\nGuardado en {SALIDA / 'experimentos.csv'}")


if __name__ == "__main__":
    main()
