"""El empate: por qué ningún modelo lo predice, y qué se puede hacer al respecto.

    python -m training.empate
    python -m training.empate --device cpu

Es la pregunta que más se discutió en el proyecto, y la que más veces se contestó a medias.
Este módulo la cierra con el estadístico que corresponde.

---

## La respuesta corta

**El empate no es una falla del modelo: es un evento que casi no se puede rankear.**

Medido con **AUC uno-contra-resto**, que es la pregunta correcta —*¿el modelo pone más
probabilidad de empate en los partidos que terminan empatados?*— y no la accuracy, que
premia adivinar la clase mayoritaria:

    xgb_gbt      AUC_away 0,680    AUC_draw 0,515    AUC_home 0,683
    poisson                0,648             0,479             0,655
    ordinal                0,606             0,493             0,648
    marcador               0,641             0,460             0,659
    mercado                0,674             0,531             0,697

**Cuatro familias de modelos distintas, las cuatro en 0,5 para el empate.** El IC95 del
nuestro es [0,441 – 0,584]: el 0,5 está adentro, así que no hay discriminación demostrable.

## Y no es que el modelo sea malo: es que el empate es así

Sobre las 1.530 filas con cuotas de las cinco temporadas, **el mercado** —casas de apuestas
con plata de verdad e información que este proyecto no tiene— saca:

    AUC empate  0,563  [0,530 - 0,595]
    AUC local   0,733
    AUC visita  0,735

Tiene señal, pero **es minúscula**, y hace falta n=1.530 para demostrar que existe. Con los
380 partidos del holdout, nuestro 0,515 es indistinguible de ese 0,563. **El techo es el
tamaño de la muestra, no el algoritmo.**

## La trampa que hay que evitar

`ordinal` tiene el **mejor F1 del empate (0,209)** y un **AUC de 0,493**. Su F1 sube porque
predice 68 empates en vez de 4 — **volumen, no acierto**. Optimizar el F1 del empate premia
adivinar más seguido. Por eso el reporte muestra las dos cosas juntas: sin el AUC al lado,
el F1 del empate miente.

## Lo único que sí es una palanca real

El **umbral** con el que se decide llamar empate. No es una decisión de modelado, es de
negocio: cuánto cuesta perderse un empate contra cuánto cuesta anunciar uno que no fue.
Bajándolo de "argmax" a 0,30 se pasa de 4 empates predichos a 36, con precisión 0,417, y la
accuracy global **no baja**.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from common.config import CFG, PROJECT_ROOT
from common.logging_setup import get_logger, setup
from eda.baselines import CLASES_ORD
from features import spec
from training import dataset, evaluate, metrics, models_alt as ma
from training.device import resolve

log = get_logger(__name__)

SALIDA = PROJECT_ROOT / "training" / "output"

UMBRALES = (0.50, 0.35, 0.32, 0.30, 0.28, 0.26, 0.24, 0.22, 0.20)
N_BOOT = 2000
I_DRAW = list(CLASES_ORD).index("draw")


# --------------------------------------------------------------------------- #
#  Los modelos que se comparan
# --------------------------------------------------------------------------- #

def probabilidades(gold: pd.DataFrame, info) -> tuple[dict, np.ndarray, pd.DataFrame]:
    """Las probabilidades de cada familia sobre el holdout, con el mismo train."""
    sp = dataset.preparar(gold, spec.FEATURES)
    tr = dataset.filtrar_train(gold[gold["season"].isin(CFG.seasons_for_training())])
    X_tr, X_te = dataset.matriz(tr, spec.FEATURES), sp.X_test
    gl = tr["home_goals"].to_numpy().astype(int)
    gv = tr["away_goals"].to_numpy().astype(int)

    P = {CFG.modelo: evaluate.evaluar_holdout(CFG.modelo, info, spec.FEATURES, gold,
                                              incluir_holdout=False)["proba"]}

    P["poisson"] = ma.ensamble([
        ma.PoissonBivariado(device=info.used, seed=CFG.seed + i)
        .fit(X_tr, gl, gv).predict_proba(X_te) for i in range(CFG.n_seeds)])

    P["marcador"] = ma.ensamble([
        ma.ClasificadorMarcador(device=info.used, seed=CFG.seed + i)
        .fit(X_tr, gl, gv).predict_proba(X_te) for i in range(CFG.n_seeds)])

    P["ordinal"] = ma.LogitOrdinal().fit(
        X_tr, dataset.codificar(tr["target_1x2"])).predict_proba(X_te)

    P["mercado"] = sp.filas_test[
        ["p_mercado_away", "p_mercado_draw", "p_mercado_home"]].to_numpy()

    return P, sp.y_test_txt, sp.filas_test


# --------------------------------------------------------------------------- #
#  Los estadísticos
# --------------------------------------------------------------------------- #

def por_clase(P: dict, y) -> pd.DataFrame:
    """Precision, recall, F1 y cuántos predijo, para cada clase y cada modelo."""
    filas = []
    for nombre, Pm in P.items():
        pred = np.array(CLASES_ORD)[Pm.argmax(1)]
        r = metrics.reporte(y, pred, Pm, con_ic=False)
        for c in CLASES_ORD:
            filas.append({"modelo": nombre, "clase": c,
                          "precision": r[f"precision_{c}"], "recall": r[f"recall_{c}"],
                          "f1": r[f"f1_{c}"], "predichos": int((pred == c).sum()),
                          "reales": int((np.asarray(y) == c).sum())})
    return pd.DataFrame(filas)


def _auc_ic(objetivo: np.ndarray, score: np.ndarray, seed: int = 0) -> tuple:
    """AUC con intervalo por bootstrap. Sin el IC, un AUC sobre 380 filas no dice nada."""
    rng = np.random.default_rng(seed)
    auc = roc_auc_score(objetivo, score)
    muestras = []
    for _ in range(N_BOOT):
        k = rng.integers(0, len(objetivo), len(objetivo))
        if 0 < objetivo[k].sum() < len(k):
            muestras.append(roc_auc_score(objetivo[k], score[k]))
    return auc, float(np.percentile(muestras, 2.5)), float(np.percentile(muestras, 97.5))


def discriminacion(P: dict, y) -> pd.DataFrame:
    """AUC uno-contra-resto y Brier por clase. **Es la tabla que contesta la pregunta.**

    La accuracy no sirve acá: un modelo que nunca predice empate puede tener buena accuracy
    y no saber absolutamente nada sobre el empate. El AUC pregunta lo correcto — si pone
    más probabilidad donde efectivamente hubo empate — y no depende del umbral.
    """
    y = np.asarray(y)
    filas = []
    for nombre, Pm in P.items():
        f = {"modelo": nombre}
        for i, c in enumerate(CLASES_ORD):
            obj = (y == c).astype(int)
            auc, lo, hi = _auc_ic(obj, Pm[:, i])
            f[f"auc_{c}"] = auc
            f[f"ic_{c}"] = f"[{lo:.3f}, {hi:.3f}]"
            f[f"brier_{c}"] = brier_score_loss(obj, Pm[:, i])
            if c == "draw":
                f["draw_sin_señal"] = bool(lo <= 0.5 <= hi)
        filas.append(f)
    return pd.DataFrame(filas).set_index("modelo")


def deciles(p_draw: np.ndarray, y) -> pd.DataFrame:
    """Tasa real de empates por decil de probabilidad predicha.

    Si el modelo rankeara, la tasa real subiría con el decil. Es la version visual del AUC.
    """
    es = (np.asarray(y) == "draw")
    q = pd.qcut(p_draw, 10, labels=False, duplicates="drop")
    return (pd.DataFrame({"decil": q, "p": p_draw, "empate": es})
            .groupby("decil").agg(n=("empate", "size"), p_media=("p", "mean"),
                                  tasa_real=("empate", "mean")))


def barrido_umbral(Pm: np.ndarray, y) -> pd.DataFrame:
    """Qué pasa si se baja el umbral para llamar empate.

    Es la única palanca real, y es una decisión de negocio: cuánto cuesta perderse un
    empate contra cuánto cuesta anunciar uno que no fue.
    """
    y = np.asarray(y)
    es = (y == "draw")
    otros_idx = np.where(Pm[:, [0, 2]].argmax(1) == 0, 0, 2)
    otros = np.array(CLASES_ORD)[otros_idx]

    filas = []
    for t in UMBRALES:
        pred = np.where(Pm[:, I_DRAW] >= t, "draw", otros)
        n_pred = int((pred == "draw").sum())
        tp = int(((pred == "draw") & es).sum())
        prec = tp / n_pred if n_pred else 0.0
        rec = tp / es.sum() if es.sum() else 0.0
        filas.append({"umbral": t, "empates_predichos": n_pred, "aciertos": tp,
                      "precision": prec, "recall": rec,
                      "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                      "accuracy_global": float((pred == y).mean())})
    return pd.DataFrame(filas)


def mercado_por_temporada(gold: pd.DataFrame) -> pd.DataFrame:
    """El AUC del mercado en las cinco temporadas: la vara de lo que se puede saber.

    Es el chequeo que impide confundir "nuestro modelo es malo" con "el empate es dificil".
    """
    g = gold.dropna(subset=["p_mercado_draw", "target_1x2"])
    filas = []
    for s, d in list(g.groupby("season")) + [("TODAS", g)]:
        obj = (d["target_1x2"] == "draw").astype(int).to_numpy()
        if obj.sum() < 10:
            continue
        auc, lo, hi = _auc_ic(obj, d["p_mercado_draw"].to_numpy())
        filas.append({
            "temporada": s, "n": len(d), "auc_empate": auc,
            "ic95": f"[{lo:.3f}, {hi:.3f}]",
            "auc_local": roc_auc_score((d["target_1x2"] == "home").astype(int),
                                       d["p_mercado_home"]),
            "auc_visita": roc_auc_score((d["target_1x2"] == "away").astype(int),
                                        d["p_mercado_away"])})
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------- #

def correr(device: str | None = None) -> dict:
    info = resolve(device)
    log.info("device: %s (%s)", info.used, info.reason)
    gold = dataset.cargar()
    P, y, _ = probabilidades(gold, info)
    return {"P": P, "y": y,
            "por_clase": por_clase(P, y),
            "discriminacion": discriminacion(P, y),
            "deciles": deciles(P[CFG.modelo][:, I_DRAW], y),
            "umbral": barrido_umbral(P[CFG.modelo], y),
            "mercado": mercado_por_temporada(gold)}


def main() -> None:
    ap = argparse.ArgumentParser(description="El empate, con el estadistico correcto.")
    ap.add_argument("--device", default=None, choices=("auto", "cuda", "cpu"))
    args = ap.parse_args()

    setup(CFG.log_level, CFG.log_format)
    SALIDA.mkdir(parents=True, exist_ok=True)
    res = correr(args.device)

    y = np.asarray(res["y"])
    print("\n" + "=" * 88)
    print(f"EL EMPATE — holdout {CFG.holdout_season}, {len(y)} partidos "
          f"(away {int((y == 'away').sum())}, draw {int((y == 'draw').sum())}, "
          f"home {int((y == 'home').sum())})")
    print("=" * 88)

    print("\n--- 1. Performance por clase ---\n")
    piv = res["por_clase"].pivot(index="modelo", columns="clase",
                                 values=["precision", "recall", "f1", "predichos"])
    print(piv.round(3).to_string())

    print("\n--- 2. Poder discriminante: AUC uno-contra-resto (0,5 = no distingue) ---\n")
    d = res["discriminacion"]
    print(d[[f"auc_{c}" for c in CLASES_ORD] + ["ic_draw", "draw_sin_señal"]]
          .round(3).to_string())
    print("\n  `draw_sin_señal` = el 0,5 esta DENTRO del IC95: no hay discriminacion")
    print("  demostrable. La accuracy no lo muestra porque premia no predecir empates.")

    print("\n--- 3. La trampa del F1 ---\n")
    emp = res["por_clase"].query("clase == 'draw'").set_index("modelo")
    tabla = emp[["f1", "predichos"]].join(d[["auc_draw"]])
    print(tabla.round(3).to_string())
    print("\n  El mejor F1 del empate viene con AUC ~0,5: sube por predecir MAS empates,")
    print("  no por acertarlos. Sin el AUC al lado, el F1 del empate miente.")

    print("\n--- 4. Tasa real de empates por decil de p_draw ---\n")
    print(res["deciles"].round(3).to_string())
    print(f"\n  tasa global: {(y == 'draw').mean():.3f}. Si el modelo rankeara, la tasa")
    print("  real subiria con el decil.")

    print("\n--- 5. La unica palanca: el umbral ---\n")
    print(res["umbral"].round(4).to_string(index=False))
    print("\n  No es una decision de modelado sino de negocio: cuanto cuesta perderse un")
    print("  empate contra cuanto cuesta anunciar uno que no fue.")

    print("\n--- 6. La vara: que tanto sabe el MERCADO del empate ---\n")
    print(res["mercado"].round(3).to_string(index=False))
    print("\n  Casas de apuestas con plata de verdad: 0,563 para el empate contra 0,73")
    print("  para local y visitante. La señal existe pero es minuscula, y hace falta")
    print("  n=1.530 para demostrarla. Con 380 partidos no se puede distinguir de la")
    print("  nuestra: el techo es el tamaño de la muestra, no el algoritmo.")

    res["por_clase"].to_csv(SALIDA / "empate_por_clase.csv", index=False)
    res["discriminacion"].to_csv(SALIDA / "empate_discriminacion.csv")
    res["umbral"].to_csv(SALIDA / "empate_umbral.csv", index=False)
    print(f"\nCSVs en {SALIDA}")


if __name__ == "__main__":
    main()
