"""Cuánto aporta cada bloque de features, medido y no supuesto.

    python -m training.ablacion
    python -m training.ablacion --device cpu

Los dos últimos bloques que se agregaron —**Otras competencias** (copas y Europa, 24
columnas) y **Opta** (56)— se construyeron sobre hipótesis razonables: que la carga de
partidos entre semana desgasta, y que la ubicación del remate dice algo que el xG agregado
no dice. Las hipótesis son buenas. Lo que hace este módulo es preguntar si aparecen en la
métrica, y dejar el número escrito aunque la respuesta sea que no.

Se compara siempre con el **modelo de evaluación** (`incluir_holdout=False`): entrena hasta
2024-25 y se mide contra 2025-26. El de producción incluye el holdout, así que sus números
sobre esa temporada no sirven para comparar nada.

⚠️ La lectura correcta de la tabla es el intervalo de confianza, no el punto. Con 380
partidos el error estándar de la accuracy ronda ±5 puntos: cuatro sets separados por menos
de un punto son **indistinguibles**, y decir "el mejor" sobre esa diferencia es leer ruido.
"""

from __future__ import annotations

import argparse

import pandas as pd

from common.config import CFG, PROJECT_ROOT
from common.logging_setup import get_logger, setup
from features import spec
from training import dataset, evaluate
from training.device import resolve

log = get_logger(__name__)

SALIDA = PROJECT_ROOT / "training" / "output"

# Los bloques que se ponen a prueba. Son los dos que llegaron con la API oficial; el resto
# del set es la "base" contra la que se los mide.
BLOQUES = ("Otras competencias", "Opta")


def sets_de_features() -> dict[str, list[str]]:
    """Los cuatro sets: base, base + cada bloque, y todo."""
    grupos = spec.grupos()
    faltantes = [b for b in BLOQUES if b not in grupos]
    if faltantes:
        raise KeyError(f"Bloques inexistentes en el spec: {faltantes}")

    comp = {f.nombre for f in grupos["Otras competencias"]}
    opta = {f.nombre for f in grupos["Opta"]}

    return {
        "base": [f for f in spec.FEATURES if f not in comp and f not in opta],
        "base + competencias": [f for f in spec.FEATURES if f not in opta],
        "base + Opta": [f for f in spec.FEATURES if f not in comp],
        "todo": list(spec.FEATURES),
    }


def correr(modelo: str | None = None, device: str | None = None) -> pd.DataFrame:
    modelo = modelo or CFG.modelo
    info = resolve(device)
    log.info("device: %s (%s)", info.used, info.reason)

    gold = dataset.cargar()
    filas = []
    for nombre, feats in sets_de_features().items():
        rep = evaluate.evaluar_holdout(modelo, info, feats, gold,
                                       incluir_holdout=False)["reporte"]
        filas.append({
            "set": nombre,
            "n_features": len(feats),
            "accuracy": rep["accuracy"],
            "ic_bajo": rep["accuracy_ic95"][0],
            "ic_alto": rep["accuracy_ic95"][1],
            "f1_macro": rep["f1_macro"],
            "log_loss": rep["log_loss"],
            "best_iteration": rep["best_iteration"],
        })
        log.info("%-22s n=%3d  acc=%.4f  f1=%.4f  LL=%.4f", nombre, len(feats),
                 rep["accuracy"], rep["f1_macro"], rep["log_loss"])

    return pd.DataFrame(filas)


def main() -> None:
    ap = argparse.ArgumentParser(description="Aporte de cada bloque de features.")
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None, choices=("auto", "cuda", "cpu"))
    args = ap.parse_args()

    setup(CFG.log_level, CFG.log_format)
    SALIDA.mkdir(parents=True, exist_ok=True)

    df = correr(args.model, args.device)
    df.to_csv(SALIDA / "ablacion_bloques.csv", index=False)

    print("\n" + "=" * 78)
    print(f"APORTE DE CADA BLOQUE — {args.model or CFG.modelo}, holdout 2025-26 (380)")
    print("Referencias: mercado 0,4947  ·  prior de clase 0,4263")
    print("=" * 78 + "\n")
    print(df.round(4).to_string(index=False))

    rango = df["accuracy"].max() - df["accuracy"].min()
    print(f"\n  diferencia entre el mejor y el peor set: {rango:.4f}")
    print("  error estandar de la accuracy con n=380  : ~0,0250 (IC95 ~ +-5 puntos)")
    if rango < 0.05:
        print("\n  -> los cuatro sets son INDISTINGUIBLES. Los bloques nuevos no")
        print("     empeoran, pero tampoco se puede afirmar que mejoren.")

    print(f"\nCSV en {SALIDA / 'ablacion_bloques.csv'}")


if __name__ == "__main__":
    main()
