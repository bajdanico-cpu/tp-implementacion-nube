"""La grilla completa: cada modelo contra cada variante de datos de entrenamiento.

Las dos rondas anteriores dejaron un hueco: las ablaciones de datos se midieron con el
feature set viejo (143 columnas, sin Elo) y los modelos alternativos se midieron con el
set nuevo (159) **pero entrenando con todo**, incluida la temporada 2022-23 que la ablación
había señalado como perjudicial. Nunca se cruzaron.

Este módulo corre la grilla entera —modelo x variante de datos— sobre el feature set
actual, para que la comparación sea de una sola pieza.

    python -m training.compare_models

Las variantes de datos comparten una idea: los partidos excluidos salen como **objetivo de
entrenamiento** pero se conservan como **historia**. Las features de Gold ya están
calculadas sobre la secuencia completa, así que filtrar filas del train no le quita
información a ninguna otra.
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from common.config import CFG, PROJECT_ROOT
from common.logging_setup import get_logger, setup
from eda.baselines import CLASES_ORD
from features import spec
from training import betting, dataset, metrics, models, models_alt as ma
from training.device import resolve

log = get_logger(__name__)

SALIDA = PROJECT_ROOT / "training" / "output"

MODELOS = ("xgb_gbt", "xgb_rf", "hgb", "logreg", "poisson", "poisson_dc",
           "ordinal", "mlp")

VARIANTES_DATOS = {
    # nombre -> filtro sobre las filas de entrenamiento (None = todas)
    "todo": None,
    "sin_xg_falso": lambda d: d["xg_available"],
    "sin_2022_23": lambda d: d["season"] != "2022-23",
}


def _fit_predict(nombre: str, info, tr: pd.DataFrame, va: pd.DataFrame,
                 Xte: np.ndarray, features: list[str], n_seeds: int) -> np.ndarray:
    """Entrena una configuración y devuelve las probabilidades sobre el holdout.

    Protocolo común a todos: donde hay early stopping se usa la temporada de validación
    para fijar las rondas, después se refitea con train + validación, y se promedian
    semillas. Los que no lo soportan (logreg, ordinal, mlp y los dos
    poisson) se ajustan directo sobre train + validación.
    """
    full = pd.concat([tr, va])
    X_tr, y_tr = dataset.matriz(tr, features), dataset.codificar(tr["target_1x2"])
    X_va, y_va = dataset.matriz(va, features), dataset.codificar(va["target_1x2"])
    X_f, y_f = dataset.matriz(full, features), dataset.codificar(full["target_1x2"])

    if nombre in ("poisson", "poisson_dc"):
        # `poisson_dc` es el mismo modelo con la correccion de Dixon-Coles, que levanta la
        # suposicion de independencia en los marcadores bajos. Van los dos a la grilla
        # para que la comparacion sea contra si mismo y no contra otra familia.
        probas = []
        for i in range(n_seeds):
            m = ma.PoissonBivariado(device=info.used, seed=CFG.seed + i,
                                    dixon_coles=(nombre == "poisson_dc"))
            m.fit(X_f, full["home_goals"].to_numpy(), full["away_goals"].to_numpy())
            probas.append(m.predict_proba(Xte))
        return ma.ensamble(probas)

    if nombre == "ordinal":
        return ma.LogitOrdinal().fit(X_f, y_f).predict_proba(Xte)

    if nombre == "mlp":
        probas = [ma.mlp(seed=CFG.seed + i).fit(X_f, y_f).predict_proba(Xte)
                  for i in range(min(n_seeds, 3))]
        return ma.ensamble(probas)

    if nombre == "xgb_gbt":
        # Early stopping temporal contra la validación, nunca contra el holdout.
        m = models.construir(nombre, info)
        m.set_params(early_stopping_rounds=100)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        params = {"n_estimators": int(m.best_iteration) + 1}
    else:
        params = None

    probas = []
    for i in range(n_seeds):
        m = models.construir(nombre, info, seed=CFG.seed + i, params=params)
        m.fit(X_f, y_f)
        probas.append(m.predict_proba(Xte))
    return ma.ensamble(probas)


def correr(features: list[str] | None = None, n_seeds: int = 3) -> pd.DataFrame:
    features = features or spec.FEATURES
    info = resolve("auto")
    gold = dataset.cargar()

    val_season, test_season = CFG.valid_season, CFG.holdout_season
    train_seasons = [s for s in CFG.seasons_for_training() if s != val_season]
    tr_base = gold[gold["season"].isin(train_seasons)]
    va = gold[gold["season"] == val_season]
    te = gold[gold["season"] == test_season]
    Xte = dataset.matriz(te, features)
    y_te = te["target_1x2"].to_numpy()

    filas = []
    for var, filtro in VARIANTES_DATOS.items():
        tr = tr_base if filtro is None else tr_base[filtro(tr_base)]
        for nombre in MODELOS:
            log.info("%-10s | %-14s | train %d + val %d", nombre, var, len(tr), len(va))
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    P = _fit_predict(nombre, info, tr, va, Xte, features, n_seeds)
            except Exception as exc:  # noqa: BLE001
                log.error("  falló: %s", exc)
                filas.append({"modelo": nombre, "datos": var, "error": str(exc)[:120]})
                continue

            pred = np.array(CLASES_ORD)[P.argmax(1)]
            rep = metrics.reporte(y_te, pred, P)
            roi = betting.reporte(te, P)["modelo"]
            filas.append({
                "modelo": nombre, "datos": var,
                "n_train": len(tr) + len(va),
                "accuracy": rep["accuracy"],
                "ic_bajo": rep["accuracy_ic95"][0], "ic_alto": rep["accuracy_ic95"][1],
                "f1_macro": rep["f1_macro"], "f1_draw": rep["f1_draw"],
                "log_loss": rep["log_loss"],
                "empates_pred": int((P.argmax(1) == 1).sum()),
                "roi": roi.get("roi"), "n_apuestas": roi.get("n_apuestas"),
            })
    return pd.DataFrame(filas)


def main() -> None:
    ap = argparse.ArgumentParser(description="Grilla modelo x datos de entrenamiento.")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    setup(CFG.log_level, CFG.log_format)
    SALIDA.mkdir(parents=True, exist_ok=True)
    df = correr(n_seeds=args.seeds)
    df.to_csv(SALIDA / "comparacion_completa.csv", index=False)

    print("\n" + "=" * 96)
    print(f"GRILLA COMPLETA — {len(spec.FEATURES)} features ({spec.FEATURE_SET_VERSION}), "
          f"holdout 2025-26 (380 partidos)")
    print("Referencias: mercado acc 0,4947 / LL 1,0118  ·  prior de clase 0,4263 / 1,0845")
    print("=" * 96 + "\n")

    ok = df[df.get("accuracy").notna()] if "accuracy" in df else df
    print(ok.round(4).to_string(index=False))

    if not ok.empty:
        print("\n--- mejor por accuracy ---")
        print(ok.loc[ok["accuracy"].idxmax()].to_string())
        print("\n--- mejor por log-loss ---")
        print(ok.loc[ok["log_loss"].idxmin()].to_string())
        print("\n--- efecto de la variante de datos, promediado sobre modelos ---")
        print(ok.groupby("datos")[["accuracy", "f1_macro", "log_loss"]]
                .mean().round(4).to_string())

    print(f"\nCSV en {SALIDA / 'comparacion_completa.csv'}")


if __name__ == "__main__":
    main()
