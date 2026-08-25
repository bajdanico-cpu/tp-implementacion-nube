"""Búsqueda de hiperparámetros con validación cruzada TEMPORAL.

## Por qué no se puede usar `KFold` ni `cross_val_score`

La validación cruzada estándar **mezcla el tiempo**: cada fold toma filas al azar, así que
el modelo entrena con partidos de mayo y valida con partidos de agosto del mismo año. Ve el
futuro, la métrica sale optimista y la decisión de hiperparámetros queda tomada sobre un
número que no significa nada.

Peor todavía en este dataset: las features de un partido se calculan con los partidos
anteriores de los mismos equipos. Un fold aleatorio pone en validación un partido cuyas
features ya "vieron" partidos que quedaron en el train. La contaminación no es sutil.

Acá se usa **expanding window**: se ordena por tiempo, se corta en bloques, y cada fold
entrena con todo lo anterior y valida con el bloque siguiente.

```
fold 1   train [====]              valida [==]
fold 2   train [========]          valida [==]
fold 3   train [============]      valida [==]
fold 4   train [================]  valida [==]
```

Es exactamente `sklearn.model_selection.TimeSeriesSplit`, pero cortando por **gameweek**
en lugar de por fila, para que un fold nunca parta una fecha al medio.

**El holdout 2025-26 no se toca.** Toda la búsqueda ocurre dentro de las temporadas de
entrenamiento; si se eligieran hiperparámetros mirando el holdout, dejaría de ser holdout.

    python -m training.tune --model xgb_gbt --n-iter 60
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from common.config import CFG, PROJECT_ROOT
from common.logging_setup import get_logger, setup
from eda.baselines import CLASES_ORD
from features import spec
from training import dataset, metrics, models
from training.device import resolve

log = get_logger(__name__)

SALIDA = PROJECT_ROOT / "training" / "output"
N_FOLDS = 4


# ---------------------------------------------------------------------------
# Los espacios de búsqueda
# ---------------------------------------------------------------------------
# Cada rango está centrado en el valor actual y explora hacia los dos lados. Los valores
# actuales se eligieron razonando sobre el tamaño del dataset (1.140 filas, 159 features),
# no buscando: esto es lo que verifica si ese razonamiento era correcto.

ESPACIOS = {
    "xgb_gbt": {
        # Profundidad del árbol. Actual 3. Con 1.140 filas, 6 memoriza.
        "max_depth": [2, 3, 4, 5],
        # Mínimo de peso por hoja. Actual 10 (~1 % de las filas).
        "min_child_weight": [3, 5, 10, 15, 20],
        # Actual 0,03. Más bajo necesita más rondas; el early stopping se encarga.
        "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.10],
        # Fracción de filas por árbol. Actual 0,8.
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        # Fracción de FEATURES por árbol. Actual 0,5, la palanca que más decorrelaciona
        # cuando hay 159 columnas correlacionadas entre sí.
        "colsample_bytree": [0.3, 0.4, 0.5, 0.7, 0.9],
        "colsample_bylevel": [0.6, 0.8, 1.0],
        # Regularización L2 y L1 sobre los pesos de las hojas. Actuales 5,0 y 0,5.
        "reg_lambda": [1.0, 3.0, 5.0, 10.0, 20.0],
        "reg_alpha": [0.0, 0.5, 1.0, 3.0],
        # Ganancia mínima para abrir un nodo. Actual 0,5.
        "gamma": [0.0, 0.5, 1.0, 2.0],
        # Bins del histograma. Actual 64. Con el default de 256 y 1.140 filas cada bin
        # tiene 4 observaciones y los cortes son ruido.
        "max_bin": [16, 32, 64, 128, 256],
    },
    "xgb_rf": {
        # Cantidad de árboles del bosque. Actual 300.
        "num_parallel_tree": [100, 200, 300, 500],
        "max_depth": [4, 6, 8, 10, 12],
        "min_child_weight": [1, 3, 5, 10],
        # 0,632 es la fracción esperada de una muestra bootstrap: el RF clásico.
        "subsample": [0.5, 0.632, 0.8, 1.0],
        # En un RF la submuestra de features se toma POR NODO, no por árbol.
        "colsample_bynode": [0.2, 0.3, 0.5, 0.7],
        "reg_lambda": [0.0, 1.0, 3.0, 10.0],
        "max_bin": [32, 64, 128, 256],
    },
    "rf_sklearn": {
        "n_estimators": [200, 400, 800],
        "max_depth": [4, 6, 8, 12, None],
        "min_samples_leaf": [1, 5, 10, 20],
        "max_features": ["sqrt", "log2", 0.3, 0.5],
        "class_weight": [None, "balanced"],
    },
}

FIJOS = {"xgb_gbt": {"n_estimators": 400}, "xgb_rf": {}, "rf_sklearn": {}}


@dataclass(frozen=True)
class Fold:
    train_idx: np.ndarray
    valid_idx: np.ndarray
    etiqueta: str


def folds_temporales(df: pd.DataFrame, n_folds: int = N_FOLDS) -> list[Fold]:
    """Expanding window cortando por gameweek, nunca por fila.

    Cortar por fila partiría una fecha al medio: la mitad de los partidos de una gameweek
    en train y la otra mitad en validación. Como todos comparten el mismo corte temporal,
    eso es exactamente el tipo de mezcla que la validación temporal viene a evitar.
    """
    d = df.reset_index(drop=True)
    fechas = (d[["season", "gameweek", "corte"]].drop_duplicates()
               .sort_values("corte").reset_index(drop=True))
    bloques = np.array_split(np.arange(len(fechas)), n_folds + 1)

    out = []
    for i in range(1, len(bloques)):
        fechas_tr = fechas.iloc[np.concatenate(bloques[:i])]
        fechas_va = fechas.iloc[bloques[i]]
        corte_tr = set(map(tuple, fechas_tr[["season", "gameweek"]].values))
        corte_va = set(map(tuple, fechas_va[["season", "gameweek"]].values))

        clave = list(map(tuple, d[["season", "gameweek"]].values))
        tr = np.array([j for j, k in enumerate(clave) if k in corte_tr])
        va = np.array([j for j, k in enumerate(clave) if k in corte_va])
        if len(tr) < 150 or len(va) < 50:
            continue
        out.append(Fold(tr, va,
                        f"train {len(tr):4d} -> valida {len(va):3d} "
                        f"({fechas_va.season.iloc[0]} GW{int(fechas_va.gameweek.iloc[0])}"
                        f"-{int(fechas_va.gameweek.iloc[-1])})"))
    return out


def evaluar_config(nombre: str, params: dict, d: pd.DataFrame, folds: list[Fold],
                   info, features: list[str]) -> dict:
    """Log-loss y accuracy promediados sobre los folds temporales."""
    X = dataset.matriz(d, features)
    y = dataset.codificar(d["target_1x2"])

    lls, accs = [], []
    for f in folds:
        m = models.construir(nombre, info, params={**FIJOS[nombre], **params})
        m.fit(X[f.train_idx], y[f.train_idx])
        P = m.predict_proba(X[f.valid_idx])
        y_txt = np.array(CLASES_ORD)[y[f.valid_idx]]
        pred = np.array(CLASES_ORD)[P.argmax(1)]
        r = metrics.reporte(y_txt, pred, P, con_ic=False)
        lls.append(r["log_loss"])
        accs.append(r["accuracy"])

    return {"log_loss_cv": float(np.mean(lls)), "log_loss_std": float(np.std(lls)),
            "accuracy_cv": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
            **params}


def buscar(nombre: str = "xgb_gbt", n_iter: int = 60, seed: int = 42,
           features: list[str] | None = None) -> pd.DataFrame:
    features = features or spec.FEATURES
    info = resolve("auto")
    gold = dataset.cargar()

    # SOLO las temporadas de entrenamiento. El holdout no se mira ni de reojo.
    d = gold[gold["season"].isin(CFG.seasons_for_training())]
    d = dataset.filtrar_train(d).sort_values("corte").reset_index(drop=True)
    folds = folds_temporales(d)

    log.info("Busqueda sobre %d filas, %d folds temporales:", len(d), len(folds))
    for f in folds:
        log.info("   %s", f.etiqueta)

    rng = np.random.default_rng(seed)
    espacio = ESPACIOS[nombre]

    # La configuración actual entra como primera candidata, para que la comparación sea
    # contra ella y no contra un punto arbitrario.
    actual = _config_actual(nombre, info)
    candidatas = [actual] if actual else []
    vistas = {json.dumps(actual, sort_keys=True, default=str)} if actual else set()
    while len(candidatas) < n_iter:
        c = {k: v[rng.integers(len(v))] for k, v in espacio.items()}
        c = {k: (None if v is None else (v.item() if hasattr(v, "item") else v))
             for k, v in c.items()}
        clave = json.dumps(c, sort_keys=True, default=str)
        if clave not in vistas:
            vistas.add(clave)
            candidatas.append(c)

    filas = []
    for i, c in enumerate(candidatas, 1):
        try:
            filas.append({"i": i, "es_actual": i == 1 and actual is not None,
                          **evaluar_config(nombre, c, d, folds, info, features)})
        except Exception as exc:  # noqa: BLE001
            log.warning("config %d fallo: %s", i, str(exc)[:120])
        if i % 10 == 0:
            log.info("  %d/%d configuraciones", i, len(candidatas))

    return pd.DataFrame(filas).sort_values("log_loss_cv").reset_index(drop=True)


def _config_actual(nombre: str, info) -> dict | None:
    """Los hiperparámetros que están hoy en el código, para tenerlos de referencia."""
    if nombre not in ("xgb_gbt", "xgb_rf"):
        return None
    hp = models.hiperparametros(nombre, info)
    return {k: hp[k] for k in ESPACIOS[nombre] if k in hp}


def main() -> None:
    ap = argparse.ArgumentParser(description="Busqueda de hiperparametros con CV temporal.")
    ap.add_argument("--model", default="xgb_gbt", choices=list(ESPACIOS))
    ap.add_argument("--n-iter", type=int, default=60)
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    args = ap.parse_args()

    setup(CFG.log_level, CFG.log_format)
    SALIDA.mkdir(parents=True, exist_ok=True)

    df = buscar(args.model, args.n_iter)
    df.to_csv(SALIDA / f"tune_{args.model}.csv", index=False)

    cols = ["i", "es_actual", "log_loss_cv", "log_loss_std", "accuracy_cv"]
    hp = [c for c in df.columns if c not in cols and c not in ("accuracy_std",)]

    print(f"\n{'=' * 92}")
    print(f"BUSQUEDA DE HIPERPARAMETROS — {args.model}, {len(df)} configuraciones, "
          f"CV temporal de {args.folds} folds")
    print("El holdout 2025-26 NO se toco: toda la busqueda es dentro del train.")
    print("=" * 92 + "\n")

    print("--- las 8 mejores por log-loss ---")
    print(df.head(8)[cols + hp].round(4).to_string(index=False))

    act = df[df.es_actual]
    if not act.empty:
        pos = int(act.index[0]) + 1
        print(f"\n--- la configuracion ACTUAL quedo {pos}a de {len(df)} ---")
        print(act[cols + hp].round(4).to_string(index=False))
        mejor = df.iloc[0]
        print(f"\n  mejora del mejor sobre el actual: "
              f"log-loss {mejor.log_loss_cv - act.log_loss_cv.iloc[0]:+.4f}   "
              f"accuracy {mejor.accuracy_cv - act.accuracy_cv.iloc[0]:+.4f}")
        print(f"  desvio entre folds del mejor: +-{mejor.log_loss_std:.4f}")
        print("  (si la mejora es menor al desvio entre folds, no es distinguible)")

    print(f"\nCSV en {SALIDA / f'tune_{args.model}.csv'}")


if __name__ == "__main__":
    main()
