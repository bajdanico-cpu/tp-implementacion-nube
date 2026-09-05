"""Holdout, walk-forward semanal y comparación contra los baselines del canvas.

Dos evaluaciones distintas, que responden preguntas distintas:

**Holdout fijo** (train 2022-25 → test 2025-26). Es la foto que se reporta. Con 380
partidos el error estándar de la accuracy ronda ±5 puntos, así que va siempre con su
intervalo de confianza: diferencias chicas entre modelos no son distinguibles y reportar
sólo el punto invita a concluir de más.

**Walk-forward semanal.** Para cada gameweek del holdout se reentrena con todo lo anterior
al corte y se predice esa fecha. No es sólo validación: **es la simulación del ciclo
operativo del bloque 9** — reentrenar al cierre de cada fecha y comparar contra el modelo
en producción. El mismo código sirve para las dos cosas, y de acá sale la serie de
predicciones pareadas que alimenta el McNemar de `training/promotion.py`.

El baseline del canvas es *"el promedio de resultado del dataset"*: el prior de clase, que
sobre el holdout 2025-26 da **42,6 % de accuracy y log-loss 1,085**. Se calcula con
`eda.baselines`, sin reimplementarlo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common.config import CFG
from common.logging_setup import get_logger
from eda.baselines import (CLASES_ORD, baseline_cuotas, baseline_prior_de_clase,
                           baseline_siempre_local)
from features import spec
from training import dataset, metrics, models

log = get_logger(__name__)


def entrenar(nombre: str, X, y, info, X_val=None, y_val=None,
             n_seeds: int | None = None, params: dict | None = None):
    """Entrena promediando semillas. Devuelve (modelos, best_iteration)."""
    n_seeds = CFG.n_seeds if n_seeds is None else n_seeds
    entrenados, best = [], None
    for i in range(n_seeds):
        m = models.construir(nombre, info, seed=CFG.seed + i, params=params)
        if nombre == "xgb_gbt" and X_val is not None:
            m.set_params(early_stopping_rounds=100)
            m.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
            best = int(m.best_iteration) if best is None else best
        else:
            m.fit(X, y)
        entrenados.append(m)
    return entrenados, best


def predecir(entrenados: list, X) -> np.ndarray:
    """Probabilidades promediadas entre semillas, en el orden de CLASES_ORD."""
    return models.promediar_probabilidades([m.predict_proba(X) for m in entrenados])


def evaluar_holdout(nombre: str, info, features: list[str] | None = None,
                    gold: pd.DataFrame | None = None,
                    incluir_holdout: bool = False) -> dict:
    """Entrena y evalua sobre el holdout 2025-26.

    ⚠️ Con `incluir_holdout=True` el modelo entrena CON el holdout, asi que las metricas
    que devuelve son **de entrenamiento, no de generalizacion**. Se usa unicamente para
    producir el artefacto de produccion; el numero que se reporta sale siempre de la
    variante sin holdout.
    """
    gold = dataset.cargar() if gold is None else gold
    features = features or spec.FEATURES
    sp = dataset.preparar(gold, features)

    # 1) Early stopping contra la temporada de validación (nunca contra el holdout).
    _, best = entrenar(nombre, sp.X_train, sp.y_train, info, sp.X_valid, sp.y_valid,
                       n_seeds=1)

    # 2) Refit con TODO el train (incluida la de validación), con el nº de rondas ya fijo.
    params = {"n_estimators": best + 1} if best else None
    X_full, y_full = dataset.train_completo(gold, features,
                                            incluir_holdout=incluir_holdout)
    entrenados, _ = entrenar(nombre, X_full, y_full, info, params=params)

    proba = predecir(entrenados, sp.X_test)
    pred = dataset.decodificar(proba.argmax(axis=1))

    rep = metrics.reporte(sp.y_test_txt, pred, proba)
    rep["best_iteration"] = best
    rep["n_train"] = int(len(y_full))
    rep["baselines"] = _baselines(sp.filas_test)
    rep["calibracion"] = metrics.curva_de_confiabilidad(
        sp.y_test_txt, proba).to_dict("records")
    rep["incluye_holdout"] = bool(incluir_holdout)
    rep["metricas_son_de_generalizacion"] = not incluir_holdout
    rep["criterio_bloque5"] = {
        "accuracy_baseline_prior": rep["baselines"]["prior_de_clase"]["accuracy"],
        "cumple": bool(rep["accuracy"] > rep["baselines"]["prior_de_clase"]["accuracy"]),
    }
    return {"reporte": rep, "modelos": entrenados, "proba": proba,
            "split": sp, "features": features}


def _baselines(filas: pd.DataFrame) -> dict:
    """Las varas del canvas, calculadas sobre las MISMAS filas que se evalúa el modelo."""
    d = filas.copy()
    train = dataset.cargar()
    train = train[train["season"].isin(CFG.seasons_for_training())]
    out = {
        "siempre_local": baseline_siempre_local(d),
        "prior_de_clase": baseline_prior_de_clase(train, d),
    }
    try:
        out["cuotas_de_cierre"] = baseline_cuotas(d)
    except Exception as exc:  # noqa: BLE001
        out["cuotas_de_cierre"] = {"error": str(exc)}
    return out


def walk_forward(nombre: str, info, features: list[str] | None = None,
                 gold: pd.DataFrame | None = None,
                 temporada: str | None = None,
                 n_estimators: int | None = None,
                 guardar_proba: bool = False) -> pd.DataFrame:
    """Reentrena fecha a fecha y predice la siguiente. Simula el ciclo del bloque 9.

    Para cada gameweek se entrena con **todo lo anterior a su corte** y se predice esa
    fecha. El refit completo por fold es caro conceptualmente pero acá tarda segundos.

    ⚠️ **El número de rondas se fija UNA vez, fuera del bucle.** Si cada fold entrenara con
    `n_estimators=2000` sin early stopping, el walk-forward mediría un modelo distinto del
    que se reporta en el holdout —uno mucho más sobreajustado— y la comparación no querría
    decir nada. Fijarlo también es lo que haría un reentrenamiento semanal real: se
    reentrena con los datos nuevos, no se re-tunea de cero cada semana.

    Con `guardar_proba=True` cada fila se lleva además la matriz de probabilidades y las
    etiquetas reales de su fecha. Es lo que permite evaluar una **regla de decisión**
    distinta sobre exactamente la misma simulación, sin volver a entrenar 38 veces
    (`training/decision_eval.py`). Va apagado por defecto porque engorda el CSV.
    """
    gold = dataset.cargar() if gold is None else gold
    features = features or spec.FEATURES
    temporada = temporada or CFG.holdout_season

    if n_estimators is None and nombre == "xgb_gbt":
        sp = dataset.preparar(gold, features)
        _, best = entrenar(nombre, sp.X_train, sp.y_train, info,
                           sp.X_valid, sp.y_valid, n_seeds=1)
        n_estimators = (best + 1) if best else None
        log.info("Walk-forward con n_estimators=%s (early stopping temporal)", n_estimators)

    params = {"n_estimators": n_estimators} if n_estimators else None

    obj = gold[gold["season"] == temporada].sort_values("corte")
    fechas = obj[["gameweek", "corte"]].drop_duplicates().sort_values("corte")

    filas = []
    for _, f in fechas.iterrows():
        prev = gold[gold["corte"] < f["corte"]]
        test = obj[obj["gameweek"] == f["gameweek"]]
        if len(prev) < 200 or test.empty:
            continue

        X_tr = dataset.matriz(prev, features)
        y_tr = dataset.codificar(prev["target_1x2"])
        entrenados, _ = entrenar(nombre, X_tr, y_tr, info, n_seeds=1, params=params)

        proba = predecir(entrenados, dataset.matriz(test, features))
        pred = dataset.decodificar(proba.argmax(axis=1))
        y = test["target_1x2"].to_numpy()

        rep = metrics.reporte(y, pred, proba, con_ic=False)
        base = baseline_siempre_local(test)
        prior = baseline_prior_de_clase(prev, test)
        filas.append({
            "season": temporada, "gameweek": int(f["gameweek"]),
            "n": len(test), "n_train": len(prev),
            "accuracy": rep["accuracy"], "f1_macro": rep["f1_macro"],
            "log_loss": rep["log_loss"],
            "acc_siempre_local": base["accuracy"],
            "acc_prior": prior["accuracy"], "ll_prior": prior.get("log_loss"),
            "gana_a_siempre_local": rep["accuracy"] > base["accuracy"],
            "aciertos": (pred == y).tolist(),
            "fixture_ids": test["fixture_id"].tolist(),
        })
        if guardar_proba:
            filas[-1]["proba"] = proba.tolist()
            filas[-1]["y"] = y.tolist()
        log.info("WF %s GW%02d | n=%d acc=%.3f LL=%.3f (local %.3f)",
                 temporada, f["gameweek"], len(test), rep["accuracy"],
                 rep["log_loss"], base["accuracy"])
    return pd.DataFrame(filas)


def resumen_walk_forward(wf: pd.DataFrame) -> dict:
    """Las métricas de creación de valor del bloque 10."""
    if wf.empty:
        return {}
    return {
        "fechas": int(len(wf)),
        "accuracy_media": float(wf["accuracy"].mean()),
        "log_loss_media": float(wf["log_loss"].mean()),
        "pct_fechas_gana_a_siempre_local": float(wf["gana_a_siempre_local"].mean()),
        "pct_fechas_gana_al_prior": float((wf["log_loss"] < wf["ll_prior"]).mean()),
        "accuracy_media_siempre_local": float(wf["acc_siempre_local"].mean()),
    }
