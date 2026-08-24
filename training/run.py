"""CLI de entrenamiento.

    python -m training.run --model xgb_gbt                 # holdout + persistencia
    python -m training.run --model xgb_rf --device cpu
    python -m training.run --walk-forward                  # 38 folds, simula el ciclo
    python -m training.run --roi                           # simulación de apuestas
    python -m training.run --todos                         # los tres modelos, comparados
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from common.config import CFG, PROJECT_ROOT
from common.logging_setup import get_logger, setup
from features import spec
from training import betting, dataset, evaluate, models, registry
from training.device import resolve

log = get_logger(__name__)

SALIDA = PROJECT_ROOT / "training" / "output"


def _importancias(entrenados, features: list[str]) -> pd.DataFrame:
    """Importancia por ganancia, promediada entre semillas.

    Es entregable de la defensa por dos motivos: justifica el set `podado` con evidencia
    en vez de a mano, y responde el *"período de tiempo a definir"* del canvas mostrando
    si pesa más la ventana de 3 o la de 5.
    """
    total = None
    for m in entrenados:
        if not hasattr(m, "get_booster"):
            return pd.DataFrame()
        s = pd.Series(m.get_booster().get_score(importance_type="gain"))
        total = s if total is None else total.add(s, fill_value=0.0)
    if total is None:
        return pd.DataFrame()
    total = total / len(entrenados)
    idx = {f"f{i}": f for i, f in enumerate(features)}
    return (pd.DataFrame({"feature": [idx.get(k, k) for k in total.index],
                          "ganancia": total.to_numpy()})
            .sort_values("ganancia", ascending=False).reset_index(drop=True))


def correr_modelo(nombre: str, info, features: list[str], gold, guardar: bool) -> dict:
    res = evaluate.evaluar_holdout(nombre, info, features, gold)
    rep, sp = res["reporte"], res["split"]

    imp = _importancias(res["modelos"], features)
    roi = betting.reporte(sp.filas_test, res["proba"])

    _mostrar(nombre, rep, roi)

    if guardar:
        meta = {
            "feature_set_version": spec.FEATURE_SET_VERSION,
            "feature_names": features,
            "n_features": len(features),
            "classes_": dataset.CLASES,
            "hyperparams": (models.hiperparametros(nombre, info)
                            if nombre in ("xgb_gbt", "xgb_rf") else {"modelo": nombre}),
            "best_iteration": rep.get("best_iteration"),
            "n_seeds": CFG.n_seeds, "seed": CFG.seed,
            "device_requested": info.requested, "device_used": info.used,
            "gpu_name": info.gpu_name,
            "train_seasons": CFG.seasons_for_training(),
            "valid_season": CFG.valid_season, "holdout_season": CFG.holdout_season,
            "n_train": rep["n_train"], "n_test": rep["n"],
        }
        prior = CFG.gold_root / "prior_ascendidos.json"
        if prior.exists():
            meta["promoted_prior"] = json.loads(prior.read_text(encoding="utf-8"))
        ver = registry.guardar(nombre, res["modelos"], meta,
                               {"holdout": rep, "apuestas": roi})
        if not imp.empty:
            imp.to_csv(ver.ruta / "importancias.csv", index=False)

    SALIDA.mkdir(parents=True, exist_ok=True)
    if not imp.empty:
        imp.to_csv(SALIDA / f"importancias_{nombre}.csv", index=False)
    return {"nombre": nombre, "reporte": rep, "roi": roi, "importancias": imp}


def _mostrar(nombre: str, rep: dict, roi: dict) -> None:
    ic = rep["accuracy_ic95"]
    b = rep["baselines"]
    print(f"\n{'=' * 68}\n{nombre}  ({rep['n_train']} train -> {rep['n']} holdout)\n{'=' * 68}")
    print(f"  accuracy   {rep['accuracy']:.4f}  IC95 [{ic[0]:.3f}, {ic[1]:.3f}]")
    print(f"  f1 macro   {rep['f1_macro']:.4f}   precision {rep['precision_macro']:.4f}"
          f"   recall {rep['recall_macro']:.4f}")
    print(f"  log-loss   {rep['log_loss']:.4f}")
    print(f"  por clase  " + "  ".join(
        f"{c}: f1={rep[f'f1_{c}']:.3f}" for c in rep["clases"]))
    print("\n  matriz de confusión (filas = real, cols = predicho)")
    print(f"           {'  '.join(f'{c:>6s}' for c in rep['clases'])}")
    for c, fila in zip(rep["clases"], rep["matriz_confusion"]):
        print(f"    {c:>6s} {'  '.join(f'{v:6d}' for v in fila)}")

    print("\n  baselines sobre las mismas filas:")
    for k, v in b.items():
        if "accuracy" in v:
            ll = f"  log-loss {v['log_loss']:.4f}" if v.get("log_loss") else ""
            print(f"    {k:22s} accuracy {v['accuracy']:.4f}{ll}")
    ok = rep["criterio_bloque5"]["cumple"]
    print(f"\n  criterio del bloque 5 (ganarle al promedio del dataset): "
          f"{'CUMPLE' if ok else 'NO CUMPLE'}")

    m = roi["modelo"]
    print(f"\n  simulación de apuestas (umbral EV > {roi['umbral_ev']}, "
          f"overround medio {roi['overround_medio']:.3f}):")
    if m.get("n_apuestas"):
        print(f"    {m['n_apuestas']} apuestas | ROI {m['roi']:+.3f} | "
              f"acierto {m['tasa_acierto']:.3f} | cuota media {m['cuota_media']:.2f} | "
              f"drawdown máx {m['drawdown_maximo']:.1f}u")
        sl = roi["siempre_local"]
        print(f"    referencia 'siempre al local': ROI {sl['roi']:+.3f} "
              f"sobre {sl['n_apuestas']} apuestas")
    else:
        print(f"    {m.get('nota')}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Entrenamiento y evaluación.")
    ap.add_argument("--model", default="xgb_gbt", choices=models.MODELOS)
    ap.add_argument("--todos", action="store_true", help="corre los tres modelos")
    ap.add_argument("--device", default=None, choices=("auto", "cuda", "cpu"))
    ap.add_argument("--features", default="completo", choices=("completo", "podado"))
    ap.add_argument("--top-n", type=int, default=None, help="tamaño del set podado")
    ap.add_argument("--walk-forward", action="store_true")
    ap.add_argument("--no-guardar", action="store_true")
    args = ap.parse_args()

    setup(CFG.log_level, CFG.log_format)
    SALIDA.mkdir(parents=True, exist_ok=True)
    info = resolve(args.device)
    log.info("device: %s (%s)", info.used, info.reason)

    gold = dataset.cargar()
    features = spec.FEATURES
    if args.features == "podado":
        features = _podado(args.model, args.top_n or CFG.podado_top_n)

    nombres = list(models.MODELOS) if args.todos else [args.model]
    resultados = [correr_modelo(n, info, features, gold, not args.no_guardar)
                  for n in nombres]

    if args.todos:
        comp = pd.DataFrame([{
            "modelo": r["nombre"], "accuracy": r["reporte"]["accuracy"],
            "ic_bajo": r["reporte"]["accuracy_ic95"][0],
            "ic_alto": r["reporte"]["accuracy_ic95"][1],
            "f1_macro": r["reporte"]["f1_macro"], "log_loss": r["reporte"]["log_loss"],
            "roi": r["roi"]["modelo"].get("roi"),
        } for r in resultados])
        comp.to_csv(SALIDA / "comparacion_modelos.csv", index=False)
        print(f"\n{'=' * 68}\nCOMPARACIÓN\n{'=' * 68}")
        print(comp.round(4).to_string(index=False))

    if args.walk_forward:
        wf = evaluate.walk_forward(args.model, info, features, gold)
        wf.drop(columns=["aciertos", "fixture_ids"]).to_csv(
            SALIDA / f"walkforward_{args.model}.csv", index=False)
        wf.to_pickle(SALIDA / f"walkforward_{args.model}.pkl")
        res = evaluate.resumen_walk_forward(wf)
        print(f"\n{'=' * 68}\nWALK-FORWARD ({args.model})\n{'=' * 68}")
        for k, v in res.items():
            print(f"  {k:34s} {v:.4f}" if isinstance(v, float) else f"  {k:34s} {v}")


def _podado(modelo: str, top_n: int) -> list[str]:
    """El set podado sale de las importancias del set completo, no de elegir a mano."""
    ruta = SALIDA / f"importancias_{modelo}.csv"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No hay importancias para {modelo}. Corré primero:\n"
            f"  python -m training.run --model {modelo} --features completo")
    imp = pd.read_csv(ruta)
    elegidas = imp.head(top_n)["feature"].tolist()
    log.info("Set podado: %d de %d features, por ganancia", len(elegidas),
             len(spec.FEATURES))
    return elegidas


if __name__ == "__main__":
    main()
