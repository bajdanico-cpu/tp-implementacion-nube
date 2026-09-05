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

# El ORDEN del resultado, que no es lo mismo que el orden de las clases.
#
# `CLASES_ORD` es lexicográfico y existe para alinear las columnas con `sklearn.log_loss`.
# Que coincida con el orden ordinal del fútbol (visita < empate < local) es **casualidad**
# —salen las dos de ordenar alfabéticamente 'away','draw','home'—, y el RPS depende de que
# el orden sea el correcto, no de que sea alfabético. Se declara aparte y hay un test que
# lo fija, para que renombrar una clase no rompa la métrica en silencio.
ORDEN_ORDINAL = ["away", "draw", "home"]


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
        # El RPS va al lado del log-loss, no en su lugar: el log-loss es el que castiga
        # la mala calibracion que le importa a la capa de apuestas, y el RPS el que sabe
        # que el empate esta en el medio. Se reportan los dos.
        out["rps"] = rps(y_true, P)

    if con_ic and len(y_true) > 1:
        lo, hi = ic_bootstrap(y_true, y_pred)
        out["accuracy_ic95"] = [lo, hi]
    return out


def rps(y_true, proba, columnas_proba: list[str] | None = None) -> float:
    """Ranked Probability Score: el log-loss, pero sabiendo que las clases están ORDENADAS.

    Es la métrica estándar de las dos Soccer Prediction Challenges, y por eso permite
    comparar este proyecto contra números publicados sobre 300.000 partidos.

    ## Qué calcula

    El resultado no es una etiqueta suelta: hay un orden natural **visita < empate <
    local**. El log-loss lo ignora — sólo mira la probabilidad que le pusiste a la clase
    correcta, y castiga igual cualquier error. El RPS mira las **acumuladas**:

        RPS = 1/(r-1) * suma_{i=1}^{r-1} ( P(<=i) - O(<=i) )^2

    donde `P(<=i)` es la probabilidad acumulada hasta la categoría `i` y `O(<=i)` la del
    resultado real (0 antes de la clase que salió, 1 desde ahí). Con tres clases son dos
    términos, promediados. Va de 0 (perfecto) a 1 (todo a la clase del extremo opuesto).

    ## Por qué importa acá

    **El log-loss mira UNA sola cosa: la probabilidad que le pusiste a la clase que salió.**
    Cómo repartiste el resto le da exactamente igual. El RPS sí lo mira, y ahí está toda
    la diferencia. Las dos predicciones de abajo salen **local** y le dan la misma
    probabilidad al local, así que el log-loss no puede separarlas:

                                        visita  empate  local | log-loss |    RPS
        error pegado al empate            0,00    0,40   0,60 |   0,5108 | 0,0800
        error tirado a la visita          0,40    0,00   0,60 |   0,5108 | 0,1600

    La segunda es peor y cualquiera lo diría: puso 40 % en el resultado **opuesto** en vez
    de en el de al lado. El RPS la castiga el doble; el log-loss no las distingue.

    Para este proyecto eso importa por una razón concreta: el empate está **en el medio**,
    así que es el destino natural de la masa de probabilidad de un modelo que duda. Bajo
    log-loss, dudar hacia el empate no vale más que dudar hacia el extremo equivocado.
    Bajo RPS, sí. Es la métrica que le da valor a la única cosa que el modelo sí sabe
    hacer con el empate.

    ## Las varas publicadas, en esta misma métrica

        0,2063   consenso de las casas de apuestas (ganó la Challenge 2023)
        0,2085   CatBoost + pi-ratings (mejor ML de esa edición, en validación)
        0,2149   k-NN sobre Berrar ratings (ganador de la Challenge 2017)
        0,2303   baseline: el prior de clase

    Ojo con compararse de más: esos números son sobre decenas de ligas mezcladas, y una
    liga más predecible que la Premier baja el RPS de todos. Sirven como escala de
    magnitud —el rango entero entre no saber nada y lo mejor del mundo son 0,024— no como
    tabla de posiciones.
    """
    P = _proba_ordenada(proba, columnas_proba)
    # De CLASES_ORD al orden del RESULTADO. Hoy coinciden; el codigo no lo asume.
    P = P[:, [list(CLASES_ORD).index(c) for c in ORDEN_ORDINAL]]

    y_true = np.asarray(y_true)
    O = np.zeros_like(P)
    for j, c in enumerate(ORDEN_ORDINAL):
        O[y_true == c, j] = 1.0
    if not np.isclose(O.sum(), len(y_true)):
        desconocidas = set(np.unique(y_true)) - set(ORDEN_ORDINAL)
        raise ValueError(f"Etiquetas fuera de {ORDEN_ORDINAL}: {sorted(desconocidas)}")

    # Las acumuladas. El ultimo termino siempre vale (1 - 1)^2 = 0, asi que se descarta:
    # de ahi el r-1 del denominador.
    dif = np.cumsum(P, axis=1)[:, :-1] - np.cumsum(O, axis=1)[:, :-1]
    return float(np.mean(np.sum(dif ** 2, axis=1) / (len(ORDEN_ORDINAL) - 1)))


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
