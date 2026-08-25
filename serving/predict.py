"""Predecir una fecha: el camino de producción.

    python -m serving.predict --gw 2                    # la fecha que viene
    python -m serving.predict --gw 1 --evaluar          # una ya jugada, contra el resultado

Es el primer eslabón de la Fase 6. Todavía no hay HTTP ni contenedor, pero la lógica que
va adentro del endpoint es ésta, y ya cumple las tres cosas que el bloque 8 del canvas
pide de una predicción de producción:

1. **Las features se calculan con el mismo código que el entrenamiento.**
   `features.gold_tp.construir(objetivos=...)` es literalmente la misma función; lo único
   que cambia es de dónde salen los objetivos. Dos implementaciones paralelas de las
   features es como se produce el train/serve skew.

2. **El orden de las columnas se valida contra el metadata del modelo.** XGBoost recibe un
   ndarray: si las columnas vienen en otro orden no se queja, predice cualquier cosa. Es
   un fallo silencioso, así que se chequea explícitamente.

3. **Cada predicción se registra** con el `fixture_id`, el momento en que se predijo, la
   versión del modelo y del feature set, y las tres probabilidades. Sin ese registro no hay
   monitoreo después: sólo un endpoint que responde.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import CFG, PROJECT_ROOT, utc_stamp
from common.logging_setup import get_logger, setup
from common.storage import read_table
from eda.baselines import CLASES_ORD
from features import gold_tp, spec
from training import betting, registry

log = get_logger(__name__)

PREDICCIONES = PROJECT_ROOT / "data" / "predicciones"


# ---------------------------------------------------------------------------
# Objetivos: los partidos a predecir
# ---------------------------------------------------------------------------

def objetivos_de_fecha(season: str, gameweek: int) -> pd.DataFrame:
    """Los fixtures de una gameweek, con su corte, listos para el ensamblado.

    El corte es el inicio de la fecha: `min(kickoff_time)` de esa gameweek. Todos los
    partidos de la fecha comparten corte, así que se predicen con la misma información.
    """
    fx = read_table("fact_fixture")
    d = fx[(fx["season"] == season) & (fx["gameweek"] == gameweek)]
    if d.empty:
        raise ValueError(f"No hay fixtures para {season} GW{gameweek} en fact_fixture.")

    obj = d[["season", "gameweek", "fixture_id", "kickoff_time",
             "home_short", "away_short"]].copy()
    obj["corte"] = obj["kickoff_time"].min()
    return obj.sort_values("kickoff_time").reset_index(drop=True)


# ---------------------------------------------------------------------------
# El modelo
# ---------------------------------------------------------------------------

def cargar_modelo(nombre: str | None = None, version: str | None = None):
    """Carga los boosters de la versión pedida (o la de producción) y su metadata.

    Devuelve `(boosters, metadata)`. Son varios porque el entrenamiento promedia semillas:
    la predicción tiene que promediar las mismas.
    """
    import xgboost as xgb

    nombre = nombre or CFG.modelo
    if version:
        ruta = registry.RAIZ / nombre / version
    else:
        v = registry.produccion(nombre)
        if v is not None:
            ruta = v.ruta
        else:
            dirs = sorted((registry.RAIZ / nombre).glob("2*"))
            if not dirs:
                raise FileNotFoundError(
                    f"No hay ningún modelo en models/{nombre}/. "
                    f"Corré: python -m training.run --model {nombre}")
            ruta = dirs[-1]
            log.warning("No hay PRODUCTION.json; se usa la última versión: %s", ruta.name)

    meta = json.loads((ruta / "metadata.json").read_text(encoding="utf-8"))
    archivos = sorted(ruta.glob("model*.ubj"))
    if not archivos:
        raise FileNotFoundError(f"No hay archivos .ubj en {ruta}")

    boosters = []
    for f in archivos:
        b = xgb.Booster()
        b.load_model(str(f))
        # Se sirve en CPU aunque se haya entrenado en GPU: el .ubj es portable y el
        # bloque 7 del canvas dice explícitamente que la inferencia va sin GPU.
        b.set_param({"device": "cpu"})
        boosters.append(b)

    log.info("Modelo %s versión %s — %d semillas, %d features",
             nombre, ruta.name, len(boosters), meta["n_features"])
    return boosters, meta


def _validar_features(meta: dict) -> list[str]:
    """El contrato con el entrenamiento. Un desajuste acá es un fallo silencioso."""
    esperadas = meta["feature_names"]
    if meta.get("feature_set_version") != spec.FEATURE_SET_VERSION:
        raise ValueError(
            f"El modelo se entrenó con el feature set {meta.get('feature_set_version')} "
            f"y el código está en {spec.FEATURE_SET_VERSION}. Reentrená antes de servir.")
    faltan = [c for c in esperadas if c not in spec.FEATURES]
    if faltan:
        raise ValueError(f"El spec ya no tiene {len(faltan)} features del modelo: {faltan[:5]}")
    return esperadas


def predecir_proba(boosters, X: np.ndarray) -> np.ndarray:
    """Probabilidades promediadas entre semillas, en el orden de CLASES_ORD."""
    import xgboost as xgb

    dm = xgb.DMatrix(X)
    P = np.mean([b.predict(dm) for b in boosters], axis=0)
    return P / P.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------------

def predecir(season: str, gameweek: int, nombre: str | None = None,
             version: str | None = None) -> pd.DataFrame:
    """Las tres probabilidades de cada partido de la fecha, más la decisión de apuesta."""
    boosters, meta = cargar_modelo(nombre, version)
    features = _validar_features(meta)

    obj = objetivos_de_fecha(season, gameweek)
    log.info("Prediciendo %s GW%d — %d partidos, corte %s",
             season, gameweek, len(obj), obj["corte"].iloc[0])

    feats = gold_tp.construir(objetivos=obj, con_target=False)
    feats = feats.sort_values("kickoff_time").reset_index(drop=True)

    X = feats[features].to_numpy(dtype=np.float32)
    P = predecir_proba(boosters, X)

    out = feats[["season", "gameweek", "fixture_id", "kickoff_time",
                 "home_short", "away_short"]].copy()
    for i, c in enumerate(CLASES_ORD):
        out[f"p_{c}"] = P[:, i]
    out["prediccion"] = np.array(CLASES_ORD)[P.argmax(1)]
    out["confianza"] = P.max(1)

    # Trazabilidad: sin esto no hay monitoreo, sólo un endpoint que responde.
    out["predicted_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out["model_name"] = meta["model_name"]
    out["model_version"] = meta["model_version"]
    out["feature_set_version"] = meta["feature_set_version"]

    # Auditoría anti-leakage: la historia usada es anterior al corte, también acá.
    for lado in spec.LADOS:
        out[f"hist_kickoff_{lado}"] = feats[f"hist_kickoff_{lado}"]
    _assert_sin_leakage(out, feats)
    return out


def _assert_sin_leakage(out: pd.DataFrame, feats: pd.DataFrame) -> None:
    corte = feats["corte"]
    for lado in spec.LADOS:
        hk = feats[f"hist_kickoff_{lado}"]
        malas = hk.notna() & (hk >= corte)
        if malas.any():
            raise AssertionError(
                f"{int(malas.sum())} predicciones usan historia posterior al corte ({lado}).")


def guardar(pred: pd.DataFrame) -> Path:
    PREDICCIONES.mkdir(parents=True, exist_ok=True)
    s, gw = pred["season"].iloc[0], int(pred["gameweek"].iloc[0])
    ruta = PREDICCIONES / f"{s}_GW{gw:02d}_{utc_stamp()}.parquet"
    pred.to_parquet(ruta, index=False)
    log.info("Predicción registrada en %s", ruta)
    return ruta


# ---------------------------------------------------------------------------
# Evaluación contra el resultado real
# ---------------------------------------------------------------------------

def evaluar(pred: pd.DataFrame) -> dict:
    """Compara contra el resultado real, si ya se jugó. Es el cierre del ciclo."""
    from training import metrics

    m = read_table("fact_match")
    real = m[["season", "home_short", "away_short", "target_1x2",
              "home_goals", "away_goals"]]
    d = pred.merge(real, on=["season", "home_short", "away_short"], how="left")
    jugados = d[d["target_1x2"].notna()]
    if jugados.empty:
        return {"nota": "todavía no hay resultados para esta fecha"}

    P = jugados[[f"p_{c}" for c in CLASES_ORD]].to_numpy()
    rep = metrics.reporte(jugados["target_1x2"].to_numpy(),
                          jugados["prediccion"].to_numpy(), P, con_ic=False)
    rep["acierta_siempre_local"] = float((jugados["target_1x2"] == "home").mean())
    rep["detalle"] = d
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description="Predice una fecha.")
    ap.add_argument("--season", default=CFG.current_season)
    ap.add_argument("--gw", type=int, required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--version", default=None)
    ap.add_argument("--evaluar", action="store_true",
                    help="compara contra el resultado real (si la fecha ya se jugó)")
    ap.add_argument("--no-guardar", action="store_true")
    args = ap.parse_args()

    setup(CFG.log_level, CFG.log_format)
    pred = predecir(args.season, args.gw, args.model, args.version)

    print(f"\n{'=' * 78}")
    print(f"{args.season} — FECHA {args.gw}    modelo {pred['model_name'].iloc[0]} "
          f"({pred['model_version'].iloc[0]})")
    print(f"{'=' * 78}\n")
    print(f"{'kickoff':<17}{'partido':<16}{'local':>7}{'empate':>8}{'visita':>8}"
          f"   predice")
    for r in pred.itertuples():
        partido = f"{r.home_short}-{r.away_short}"
        print(f"{str(r.kickoff_time)[:16]:<17}{partido:<16}"
              f"{r.p_home:>7.3f}{r.p_draw:>8.3f}{r.p_away:>8.3f}   {r.prediccion}")

    if not args.no_guardar:
        guardar(pred)

    if args.evaluar:
        ev = evaluar(pred)
        if "nota" in ev:
            print(f"\n{ev['nota']}")
            return
        print(f"\n{'-' * 78}\nCONTRA EL RESULTADO REAL\n{'-' * 78}\n")
        d = ev["detalle"]
        print(f"{'partido':<16}{'predijo':<10}{'p':>7}   {'real':<8}{'resultado':<10} ok")
        for r in d.itertuples():
            if pd.isna(r.target_1x2):
                continue
            marca = "OK" if r.prediccion == r.target_1x2 else "--"
            print(f"{r.home_short + '-' + r.away_short:<16}{r.prediccion:<10}"
                  f"{r.confianza:>7.3f}   {r.target_1x2:<8}"
                  f"{int(r.home_goals)}-{int(r.away_goals):<8} {marca}")
        print(f"\n  accuracy   {ev['accuracy']:.3f}  ({int(ev['accuracy'] * ev['n'])} de {ev['n']})")
        print(f"  log-loss   {ev['log_loss']:.4f}")
        print(f"  'siempre local' habria acertado: {ev['acierta_siempre_local']:.3f}")


if __name__ == "__main__":
    main()
