"""Banco A: dos versiones de Gold, el mismo protocolo, y el control de semillas.

    python -m training.comparar_gold                          # archivada mas reciente vs vigente
    python -m training.comparar_gold --base 20260901T233127Z --seeds 5
    python -m training.comparar_gold --subgrupos ascendidos arranque

Es la herramienta del **Banco A** de `docs/PLAN-MEJORAS.md`, y está pensada para las cuatro
fases: una fase cambia cómo se construye Gold, y la pregunta es siempre la misma — *¿el
modelo entrenado con el Gold nuevo es mejor que el entrenado con el viejo, sobre exactamente
los mismos 380 partidos del holdout?*

## Por qué hace falta un módulo y no un script suelto

Tres disciplinas que el proyecto aprendió a la mala y que acá van por defecto:

**Control de semillas.** No se reporta un número, se reportan *k* corridas con media y
desvío. El umbral de empate parecía dar +0,0079 de accuracy y resultó ser ruido de semilla:
el delta se movía entre −0,005 y +0,026 sin que la regla cambiara.

**McNemar pareado.** Los dos modelos predicen los MISMOS partidos, así que comparar dos
accuracies sueltas desperdicia la mitad de la información. Sólo informan los partidos donde
discrepan.

**Subgrupos.** Un cambio dirigido —sembrar el rating de los ascendidos, por ejemplo— afecta
a 40 partidos de 380. Pedirle que mueva la métrica global es pedirle que 40 filas arrastren
a 380: aunque funcione perfecto, el efecto global queda debajo del ruido. El subgrupo es
donde vive la hipótesis y donde hay señal por fila.

## Lo que NO hace

No decide. Imprime los tres números y el criterio; adoptar o rechazar es una decisión que
se toma leyendo, y que se registra en `training/README.md` o en `attempts.jsonl`.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import CFG, PROJECT_ROOT
from common.logging_setup import get_logger, setup
from common.storage import versiones, versiones_root
from eda.baselines import CLASES_ORD
from features import spec
from training import compare_models as cm
from training import dataset, metrics
from training.device import resolve

log = get_logger(__name__)

SALIDA = PROJECT_ROOT / "training" / "output"
TABLA = "gold_tp_match"

# Los subgrupos donde vive la hipotesis de cada fase. Se calculan sobre las filas del
# holdout, con columnas que ya estan en Gold.
SUBGRUPOS = {
    "ascendidos": lambda d: d["local_es_ascendido"].fillna(False).astype(bool)
                            | d["visita_es_ascendido"].fillna(False).astype(bool),
    "arranque": lambda d: d["gameweek"] <= 5,
    "resto": lambda d: ~(d["local_es_ascendido"].fillna(False).astype(bool)
                         | d["visita_es_ascendido"].fillna(False).astype(bool))
                       & (d["gameweek"] > 5),
}


# --------------------------------------------------------------------------- #
# Cargar las dos versiones
# --------------------------------------------------------------------------- #

def gold_vigente() -> pd.DataFrame:
    return dataset.cargar()


def gold_archivado(stamp: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Una version archivada de Gold. Por defecto, la mas reciente."""
    vs = versiones("gold", TABLA)
    if not vs:
        raise FileNotFoundError(
            "No hay ninguna version archivada de Gold. "
            "Antes de cambiar Gold hay que correr: "
            'python -m common.versiones --snapshot "antes de fase N"')
    elegida = next((v for v in vs if v["stamp"] == stamp), None) if stamp else vs[-1]
    if elegida is None:
        raise ValueError(f"No hay version '{stamp}'. Disponibles: {[v['stamp'] for v in vs]}")
    ruta = versiones_root("gold", TABLA) / elegida["archivo"]
    return pd.read_parquet(ruta), elegida


# --------------------------------------------------------------------------- #
# Entrenar y predecir sobre una version
# --------------------------------------------------------------------------- #

def probabilidades(gold: pd.DataFrame, modelo: str, semilla: int,
                   features: list[str] | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    """Entrena con la variante de datos de produccion y predice el holdout.

    Mismo protocolo que `compare_models`: early stopping temporal contra la temporada de
    validacion, refit con train+validacion, y **nunca** el holdout adentro.
    """
    features = features or [c for c in spec.FEATURES if c in gold.columns]
    filtro = cm.VARIANTES_DATOS[CFG.datos_entrenamiento]
    info = resolve("auto")

    val_season, test_season = CFG.valid_season, CFG.holdout_season
    train_seasons = [s for s in CFG.seasons_for_training() if s != val_season]
    tr = gold[gold["season"].isin(train_seasons)]
    tr = tr if filtro is None else tr[filtro(tr)]
    va = gold[gold["season"] == val_season]
    te = gold[gold["season"] == test_season].sort_values(spec.CLAVE_PARTIDO)

    original = CFG.raw["training"].get("seed")
    try:
        CFG.raw["training"]["seed"] = semilla
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            P = cm._fit_predict(modelo, info, tr, va, dataset.matriz(te, features),
                                features, n_seeds=1)
    finally:
        CFG.raw["training"]["seed"] = original
    return P, te


# --------------------------------------------------------------------------- #
# La comparacion
# --------------------------------------------------------------------------- #

def _metricas(P: np.ndarray, y: np.ndarray) -> dict:
    pred = np.asarray(CLASES_ORD)[P.argmax(1)]
    rep = metrics.reporte(y, pred, P, con_ic=False)
    return {"rps": rep["rps"], "accuracy": rep["accuracy"],
            "log_loss": rep["log_loss"], "f1_macro": rep["f1_macro"],
            "aciertos": pred == y}


def comparar(base: pd.DataFrame, nuevo: pd.DataFrame, modelo: str | None = None,
             seeds: tuple[int, ...] = (42, 7, 123, 2024, 999)) -> dict:
    """Las dos versiones, semilla por semilla, sobre las mismas filas del holdout."""
    modelo = modelo or CFG.modelo
    # Las features comunes: si la fase agrego columnas, comparar sobre la interseccion
    # mediria el cambio de valores; sobre las propias de cada una, el cambio completo.
    # Se usa lo propio de cada Gold, que es lo que el modelo de esa version usaria.
    filas, aciertos = [], {}
    for semilla in seeds:
        for nombre, g in (("base", base), ("nuevo", nuevo)):
            P, te = probabilidades(g, modelo, semilla)
            y = te["target_1x2"].to_numpy()
            m = _metricas(P, y)
            aciertos[(nombre, semilla)] = m.pop("aciertos")
            filas.append({"version": nombre, "seed": semilla, **m})
            log.info("%-5s seed=%-5d RPS %.4f  acc %.4f  LL %.4f",
                     nombre, semilla, m["rps"], m["accuracy"], m["log_loss"])
        # El holdout es el mismo en las dos, asi que las filas se alinean.
    d = pd.DataFrame(filas)

    resumen = d.groupby("version")[["rps", "accuracy", "log_loss", "f1_macro"]].agg(
        ["mean", "std"]).round(4)

    mcn = []
    for semilla in seeds:
        mc = metrics.mcnemar(aciertos[("nuevo", semilla)], aciertos[("base", semilla)])
        mcn.append({"seed": semilla, **mc,
                    "delta_acc": float(aciertos[("nuevo", semilla)].mean()
                                       - aciertos[("base", semilla)].mean())})
    return {"por_semilla": d, "resumen": resumen, "mcnemar": pd.DataFrame(mcn),
            "aciertos": aciertos, "seeds": seeds}


def por_subgrupo(base: pd.DataFrame, nuevo: pd.DataFrame, res: dict,
                 cuales: list[str] | None = None) -> pd.DataFrame:
    """Accuracy y RPS de cada version dentro de cada subgrupo del holdout.

    Es donde una fase dirigida tiene que mostrar el efecto: sembrar el rating de los
    ascendidos toca 40 de 380 partidos, y exigirle que mueva la metrica global es exigirle
    que 40 filas arrastren a 380.
    """
    cuales = cuales or list(SUBGRUPOS)
    te = nuevo[nuevo["season"] == CFG.holdout_season].sort_values(spec.CLAVE_PARTIDO)
    filas = []
    for nombre in cuales:
        mask = SUBGRUPOS[nombre](te).to_numpy()
        if mask.sum() == 0:
            continue
        for version in ("base", "nuevo"):
            accs = [res["aciertos"][(version, s)][mask].mean() for s in res["seeds"]]
            filas.append({"subgrupo": nombre, "n": int(mask.sum()), "version": version,
                          "accuracy_media": float(np.mean(accs)),
                          "accuracy_desvio": float(np.std(accs, ddof=1))})
    return pd.DataFrame(filas)


def banco_b(base: pd.DataFrame, nuevo: pd.DataFrame, modelo: str | None = None,
            seeds: tuple[int, ...] = (42, 7, 123)) -> pd.DataFrame:
    """Banco B: el walk-forward, que reentrena en cada fecha, sobre las dos versiones.

    El holdout fijo mide un modelo entrenado una vez; acá se reentrena semanalmente, que es
    lo que hace el ciclo real. Un cambio puede ayudar en uno y no en el otro — por eso son
    dos bancos y no uno.
    """
    from training import evaluate

    modelo = modelo or CFG.modelo
    info = resolve("auto")
    filas = []
    original = CFG.raw["training"].get("seed")
    try:
        for semilla in seeds:
            CFG.raw["training"]["seed"] = semilla
            for nombre, g in (("base", base), ("nuevo", nuevo)):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    wf = evaluate.walk_forward(modelo, info, gold=g, guardar_proba=True)
                P = np.concatenate([np.asarray(p) for p in wf["proba"]])
                y = np.concatenate([np.asarray(v) for v in wf["y"]])
                m = _metricas(P, y)
                m.pop("aciertos")
                filas.append({"version": nombre, "seed": semilla, "fechas": len(wf), **m})
                log.info("BANCO B %-5s seed=%-5d RPS %.4f acc %.4f",
                         nombre, semilla, m["rps"], m["accuracy"])
    finally:
        CFG.raw["training"]["seed"] = original
    return pd.DataFrame(filas)


def main() -> None:
    ap = argparse.ArgumentParser(description="Banco A: dos versiones de Gold.")
    ap.add_argument("--banco-b", action="store_true",
                    help="corre tambien el walk-forward sobre las dos versiones")
    ap.add_argument("--base", default=None, help="stamp de la version archivada")
    ap.add_argument("--modelo", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=[42, 7, 123, 2024, 999])
    ap.add_argument("--subgrupos", nargs="*", default=None)
    args = ap.parse_args()
    setup(CFG.log_level, CFG.log_format)
    SALIDA.mkdir(parents=True, exist_ok=True)

    base, meta = gold_archivado(args.base)
    nuevo = gold_vigente()

    print(f"\n{'=' * 92}\nBANCO A — dos versiones de Gold, holdout {CFG.holdout_season}\n{'=' * 92}\n")
    print(f"  base   {meta['stamp']}  {meta['filas']} x {meta['columnas']}")
    print(f"         etiqueta: {meta.get('etiqueta') or '(sin etiqueta)'}")
    print(f"  nuevo  VIGENTE     {len(nuevo)} x {nuevo.shape[1]}")
    print(f"  modelo {args.modelo or CFG.modelo}   semillas {args.seeds}\n")

    res = comparar(base, nuevo, args.modelo, tuple(args.seeds))

    print(f"\n{'-' * 92}\nRESUMEN — media y desvio entre semillas\n{'-' * 92}\n")
    print(res["resumen"].to_string())

    r = res["por_semilla"].pivot(index="seed", columns="version", values="rps")
    r["delta_rps"] = r["nuevo"] - r["base"]
    a = res["por_semilla"].pivot(index="seed", columns="version", values="accuracy")
    r["delta_acc"] = a["nuevo"] - a["base"]
    print(f"\n{'-' * 92}\nDELTA POR SEMILLA  (RPS: menos es mejor)\n{'-' * 92}\n")
    print(r.round(4).to_string())
    print(f"\n  delta RPS medio {r['delta_rps'].mean():+.4f}  desvio {r['delta_rps'].std():.4f}"
          f"   rango [{r['delta_rps'].min():+.4f}, {r['delta_rps'].max():+.4f}]")
    print(f"  delta acc medio {r['delta_acc'].mean():+.4f}  desvio {r['delta_acc'].std():.4f}")
    cruza = r["delta_rps"].min() < 0 < r["delta_rps"].max()
    print(f"\n  CRITERIO: el delta de RPS {'CRUZA' if cruza else 'no cruza'} el cero entre "
          f"semillas -> {'no se distingue del ruido' if cruza else 'consistente en signo'}")

    print(f"\n{'-' * 92}\nMcNEMAR PAREADO — nuevo contra base, mismas filas\n{'-' * 92}\n")
    print(res["mcnemar"].round(4).to_string(index=False))

    sub = por_subgrupo(base, nuevo, res, args.subgrupos)
    if not sub.empty:
        print(f"\n{'-' * 92}\nPOR SUBGRUPO — donde vive la hipotesis\n{'-' * 92}\n")
        print(sub.round(4).to_string(index=False))

    if args.banco_b:
        print(f"\n{'-' * 92}\nBANCO B — walk-forward, reentrenando en cada fecha\n{'-' * 92}\n")
        b = banco_b(base, nuevo, args.modelo, tuple(args.seeds[:3]))
        print(b.round(4).to_string(index=False))
        p = b.pivot(index="seed", columns="version", values="rps")
        p["delta_rps"] = p["nuevo"] - p["base"]
        pa = b.pivot(index="seed", columns="version", values="accuracy")
        p["delta_acc"] = pa["nuevo"] - pa["base"]
        print(f"\n{p.round(4).to_string()}")
        print(f"\n  delta RPS medio {p['delta_rps'].mean():+.4f}   "
              f"delta acc medio {p['delta_acc'].mean():+.4f}")
        b.to_csv(SALIDA / "comparar_gold_bancoB.csv", index=False)

    res["por_semilla"].to_csv(SALIDA / "comparar_gold.csv", index=False)
    sub.to_csv(SALIDA / "comparar_gold_subgrupos.csv", index=False)
    print(f"\nCSVs en {SALIDA}\n")


if __name__ == "__main__":
    main()
