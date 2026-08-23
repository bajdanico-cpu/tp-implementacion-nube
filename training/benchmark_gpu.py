"""¿Conviene la GPU a esta escala? Se mide, no se supone.

## Hipótesis pre-registrada

Escrita ANTES de correr el benchmark, para que contrastarla signifique algo:

> Con 1.140 filas x 143 features (650 KB, que entra entero en la caché L3 de 12 MB), el
> tiempo de pared en GPU está dominado por el overhead de lanzar kernels, no por el
> cómputo. Un fit hace ~25.000-50.000 lanzamientos, y en Windows/WDDM cada uno cuesta
> 10-20 µs: eso son 0,25-1,0 s de piso, contra 0,3-0,8 s que le lleva a la CPU hacer el
> trabajo completo. **Predicción: la GPU pierde por un factor de 3 a 8, y el punto de
> cruce está entre 10^5 y 3x10^5 filas.**

Contrastar esa predicción con la medición es el entregable. Si la hipótesis se cae, se
reporta que se cayó: un benchmark que sólo confirma lo que ya creías no es un experimento.

## Protocolo

- **Warmup obligatorio y descartado.** El primer fit en CUDA paga la inicialización del
  contexto (1-2 s). Meterla en el promedio infla el resultado; se reporta aparte.
- **Mediana y MAD, no media.** La primera corrida siempre es outlier.
- **Barrido de escala** replicando el Gold real con bootstrap + ruido: x1 (1.140), x10,
  x100 (114.000 — coincide a propósito con el tamaño de `fact_player_gw`, el dataset del
  otro proyecto sobre el mismo Silver) y x1.000.
- **Ambiente registrado** en el CSV: sin eso, otro con otra máquina no puede reproducirlo.

⚠️ La GTX 1650 mueve el display. En el barrido x1.000 el TDR de Windows (timeout de 2 s)
podría matar un kernel. Si pasa se documenta: es un riesgo real de GPU de laptop que en la
nube no existe, y refuerza el argumento de separar el entorno de desarrollo del de
entrenamiento.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

from common.config import CFG, PROJECT_ROOT
from common.logging_setup import get_logger, setup
from features import spec
from training import dataset, models
from training.device import DeviceInfo, _cpu_jobs, _nombre_gpu, resolve

log = get_logger(__name__)

SALIDA = PROJECT_ROOT / "training" / "output"
ESCALAS = (1, 10, 100, 1000)
REPETICIONES = 5

# Número FIJO de árboles para todas las mediciones. Los hiperparámetros de producción
# usan 2000 con early stopping, pero acá interesa aislar cómo escala el tiempo con el
# tamaño del dataset, no reproducir el entrenamiento real. Con 2000 x 3 clases cada fit a
# escala x1000 tardaría horas en CPU y el barrido dejaría de ser reproducible por un
# compañero. 200 alcanza de sobra para que la curva se vea, y mantiene la comparación
# justa: los dos devices construyen exactamente los mismos árboles.
N_ARBOLES = 200

HIPOTESIS = ("la GPU pierde por un factor de 3 a 8 a escala x1; "
             "el cruce cae entre 1e5 y 3e5 filas")


def _replicar(X: np.ndarray, y: np.ndarray, veces: int,
              seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Agranda el dataset con bootstrap + ruido, manteniendo la estructura de columnas."""
    if veces == 1:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(X), len(X) * veces)
    Xr = X[idx].copy()
    sigma = 0.05 * np.nan_to_num(np.nanstd(X, axis=0))
    Xr += rng.normal(0.0, 1.0, Xr.shape).astype(np.float32) * sigma.astype(np.float32)
    return Xr, y[idx]


def _medir(nombre: str, X, y, info: DeviceInfo, repeticiones: int) -> dict:
    """Warmup descartado, después `repeticiones` fits. Devuelve mediana y MAD."""
    fijos = {"n_estimators": N_ARBOLES} if nombre == "xgb_gbt" else {}

    t0 = time.perf_counter()
    models.construir(nombre, info, params=fijos).fit(X[:64], y[:64])
    warmup = time.perf_counter() - t0

    tiempos = []
    for i in range(repeticiones):
        m = models.construir(nombre, info, seed=CFG.seed + i, params=fijos)
        t = time.perf_counter()
        m.fit(X, y)
        tiempos.append(time.perf_counter() - t)

    arr = np.array(tiempos)
    med = float(np.median(arr))
    return {"n_arboles": N_ARBOLES, "fit_s_mediana": med,
            "fit_s_mad": float(np.median(np.abs(arr - med))),
            "fit_s_min": float(arr.min()),
            "warmup_s": float(warmup)}


def benchmark(nombre: str = "xgb_gbt", escalas=ESCALAS,
              repeticiones: int = REPETICIONES) -> pd.DataFrame:
    gold = dataset.cargar()
    X, y = dataset.train_completo(gold, spec.FEATURES)
    log.info("Base: %d filas x %d features (%.0f KB)", len(X), X.shape[1], X.nbytes / 1024)

    devices = []
    for d in ("cpu", "cuda"):
        try:
            devices.append(resolve(d))
        except RuntimeError as exc:
            log.warning("Se saltea %s: %s", d, exc)

    filas = []
    for veces in escalas:
        Xr, yr = _replicar(X, y, veces)
        for info in devices:
            log.info("midiendo %s escala x%d (%d filas)...", info.used, veces, len(Xr))
            try:
                r = _medir(nombre, Xr, yr, info, repeticiones)
            except Exception as exc:  # noqa: BLE001
                log.error("falló %s x%d: %s", info.used, veces, exc)
                filas.append({"device": info.used, "escala": veces, "filas": len(Xr),
                              "error": str(exc)[:200]})
                continue
            filas.append({"modelo": nombre, "device": info.used, "escala": veces,
                          "filas": len(Xr), "features": Xr.shape[1],
                          "celdas": len(Xr) * Xr.shape[1],
                          "mb": round(Xr.nbytes / 1024 ** 2, 2), **r})
    df = pd.DataFrame(filas)

    if {"device", "escala"} <= set(df.columns) and "fit_s_mediana" in df:
        piv = df.pivot_table(index="escala", columns="device", values="fit_s_mediana")
        if {"cpu", "cuda"} <= set(piv.columns):
            piv["speedup_gpu"] = piv["cpu"] / piv["cuda"]
            df = df.merge(piv[["speedup_gpu"]].reset_index(), on="escala", how="left")
    return df.assign(**_ambiente())


def _ambiente() -> dict:
    import xgboost as xgb

    return {"gpu": _nombre_gpu() or "—", "cpu_count": os.cpu_count(),
            "n_jobs_cpu": _cpu_jobs(), "xgboost": xgb.__version__,
            "use_cuda": bool(xgb.build_info().get("USE_CUDA"))}


def graficar(df: pd.DataFrame, ruta) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df[df.get("fit_s_mediana").notna()] if "fit_s_mediana" in df else df
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for dev, g in d.groupby("device"):
        g = g.sort_values("filas")
        ax.plot(g["filas"], g["fit_s_mediana"], marker="o", label=dev)
    ax.axvline(1140, ls=":", color="grey")
    ax.annotate("el tamaño real de este TP", (1140, ax.get_ylim()[1]),
                rotation=90, va="top", fontsize=8, color="grey")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("filas de entrenamiento"); ax.set_ylabel("segundos por fit (mediana)")
    ax.set_title("XGBoost: CPU vs GPU según el tamaño del dataset")
    ax.legend(); ax.grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(ruta, dpi=140); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark CPU vs GPU.")
    ap.add_argument("--modelo", default="xgb_gbt")
    ap.add_argument("--escalas", type=int, nargs="+", default=list(ESCALAS))
    ap.add_argument("--repeticiones", type=int, default=REPETICIONES)
    args = ap.parse_args()

    setup(CFG.log_level, CFG.log_format)
    SALIDA.mkdir(parents=True, exist_ok=True)
    df = benchmark(args.modelo, tuple(args.escalas), args.repeticiones)
    df.to_csv(SALIDA / "benchmark_gpu.csv", index=False)
    graficar(df, SALIDA / "benchmark_gpu.png")

    print(f"\nHipótesis pre-registrada: {HIPOTESIS}\n")
    cols = [c for c in ("device", "escala", "filas", "mb", "fit_s_mediana", "fit_s_mad",
                        "warmup_s", "speedup_gpu") if c in df.columns]
    print(df[cols].to_string(index=False))
    print(f"\nCSV y gráfico en {SALIDA}")


if __name__ == "__main__":
    main()
