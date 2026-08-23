"""Las métricas del bloque 5 del canvas, en un solo lugar.

El canvas pide **accuracy, F1, recall y precision**. Se agregan el log-loss (porque la
apuesta se decide con la probabilidad, no con la clase) y la matriz de confusión (porque
es donde se ve qué pasa con el empate, que es el 24,1 % de los partidos y casi nunca es
el argmax de nadie).

⚠️ **Este módulo es el ÚNICO que puede llamar a `log_loss`.** `sklearn.metrics.log_loss`
asume las etiquetas en orden lexicográfico y alinea las columnas de probabilidad con ese
orden: pasarle `['home','draw','away']` devuelve un número silenciosamente incorrecto y
sólo avisa por warning. Concentrar la llamada acá, siempre con `labels=CLASES_ORD`, es la
mitigación estructural. Hay un test que lo verifica inyectando el orden equivocado.

El intervalo de confianza no es decorativo: con 380 partidos de holdout, el error estándar
de la accuracy ronda **±5 puntos**, así que diferencias chicas entre modelos no son
distinguibles y reportar sólo el punto invita a concluir de más.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)

from eda.baselines import CLASES_ORD

N_BOOT = 2000


def _proba_ordenada(proba: np.ndarray | pd.DataFrame,
                    columnas: list[str] | None = None) -> np.ndarray:
    """Devuelve la matriz con las columnas en el orden de CLASES_ORD."""
    if isinstance(proba, pd.DataFrame):
        return proba[CLASES_ORD].to_numpy()
    if columnas is None:
        return np.asarray(proba)
    idx = [list(columnas).index(c) for c in CLASES_ORD]
    return np.asarray(proba)[:, idx]


def reporte(y_true, y_pred, proba=None, columnas_proba: list[str] | None = None,
            con_ic: bool = True) -> dict:
    """Todas las métricas del bloque 5, más el log-loss y la matriz de confusión."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=CLASES_ORD, average="macro",
                                   zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=CLASES_ORD,
                                                 average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=CLASES_ORD,
                                           average="macro", zero_division=0)),
    }
    for clase, f1, pr, rc in zip(
            CLASES_ORD,
            f1_score(y_true, y_pred, labels=CLASES_ORD, average=None, zero_division=0),
            precision_score(y_true, y_pred, labels=CLASES_ORD, average=None, zero_division=0),
            recall_score(y_true, y_pred, labels=CLASES_ORD, average=None, zero_division=0)):
        out[f"f1_{clase}"] = float(f1)
        out[f"precision_{clase}"] = float(pr)
        out[f"recall_{clase}"] = float(rc)

    out["matriz_confusion"] = confusion_matrix(y_true, y_pred, labels=CLASES_ORD).tolist()
    out["clases"] = list(CLASES_ORD)

    if proba is not None:
        P = _proba_ordenada(proba, columnas_proba)
        out["log_loss"] = float(log_loss(y_true, P, labels=CLASES_ORD))

    if con_ic and len(y_true) > 1:
        lo, hi = ic_bootstrap(y_true, y_pred)
        out["accuracy_ic95"] = [lo, hi]
    return out


def ic_bootstrap(y_true, y_pred, n_boot: int = N_BOOT, seed: int = 42) -> tuple[float, float]:
    """IC del 95 % de la accuracy, remuestreando las filas del holdout."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    accs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        accs[i] = np.mean(y_true[idx] == y_pred[idx])
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def curva_de_confiabilidad(y_true, proba, columnas_proba=None,
                           n_bins: int = 10) -> pd.DataFrame:
    """Calibración: de los partidos donde dijimos 60 %, ¿cuántos salieron?"""
    P = _proba_ordenada(proba, columnas_proba)
    y_true = np.asarray(y_true)
    filas = []
    for j, clase in enumerate(CLASES_ORD):
        p = P[:, j]
        real = (y_true == clase).astype(float)
        bins = np.clip(np.digitize(p, np.linspace(0, 1, n_bins + 1)[1:-1]), 0, n_bins - 1)
        for b in range(n_bins):
            m = bins == b
            if not m.any():
                continue
            filas.append({"clase": clase, "bin": b, "n": int(m.sum()),
                          "p_media": float(p[m].mean()),
                          "frecuencia_real": float(real[m].mean())})
    return pd.DataFrame(filas)


def mcnemar(aciertos_a: np.ndarray, aciertos_b: np.ndarray) -> dict:
    """Test de McNemar pareado entre dos modelos sobre los MISMOS partidos.

    Es el test correcto para decidir una promoción. Como ambos modelos predicen los mismos
    partidos, sólo aportan información los casos en que **discrepan**: los pares
    discordantes (A acierta y B no, o al revés). Eso lo hace mucho más potente que comparar
    dos accuracies independientes — que es lo que haría falta si cada modelo hubiera visto
    partidos distintos.

    Con 10 partidos por fecha, el error estándar de una accuracy suelta es ±15,7 puntos:
    comparar así sobre una sola fecha es tirar una moneda. De ahí que la promoción use una
    ventana de varias fechas y este test.

    Devuelve el p-valor exacto (binomial sobre los discordantes) para no depender de la
    aproximación chi-cuadrado, que con pocos casos no vale.
    """
    from scipy.stats import binomtest

    a = np.asarray(aciertos_a, dtype=bool)
    b = np.asarray(aciertos_b, dtype=bool)
    n01 = int(np.sum(a & ~b))   # A acierta, B no
    n10 = int(np.sum(~a & b))   # B acierta, A no
    n_disc = n01 + n10
    if n_disc == 0:
        return {"n01": 0, "n10": 0, "n_discordantes": 0, "p_valor": 1.0,
                "gana": "empate"}
    p = binomtest(n01, n_disc, 0.5).pvalue
    gana = "a" if n01 > n10 else ("b" if n10 > n01 else "empate")
    return {"n01": n01, "n10": n10, "n_discordantes": n_disc,
            "p_valor": float(p), "gana": gana}
