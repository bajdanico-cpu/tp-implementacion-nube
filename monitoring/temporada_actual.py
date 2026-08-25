"""Monitoreo de la temporada en curso: el segundo test, que crece cada fecha.

    python -m monitoring.temporada_actual

El holdout 2025-26 es una foto: 380 partidos fijos que ya se midieron mil veces. La
temporada en curso es otra cosa — **datos que el modelo no vio y que nadie miró todavía**,
que llegan de a diez por semana. Es la evaluación más honesta que existe, y la única que no
se puede sobreajustar mirándola.

Por eso 2026-27 no entra al entrenamiento aunque sus partidos jugados sí entren a Gold: se
usan como **historia** para las ventanas de las fechas siguientes, nunca como objetivo.

Lo que produce, que es el bloque 10 del canvas:

- Fecha a fecha: accuracy y log-loss del modelo contra los mismos baselines calculados
  **sobre las mismas filas**. La distinción importa: una caída del modelo acompañada de una
  caída del mercado es la liga siendo más impredecible, no el modelo degradándose.
- Acumulado de la temporada, con su intervalo de confianza.
- La serie de aciertos pareada que alimenta el McNemar de la regla de promoción.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.config import CFG, PROJECT_ROOT
from common.logging_setup import get_logger, setup
from common.storage import read_table
from eda.baselines import CLASES_ORD, baseline_prior_de_clase, baseline_siempre_local
from features import spec
from training import betting, dataset, metrics
from serving import predict as srv

log = get_logger(__name__)

SALIDA = PROJECT_ROOT / "monitoring" / "output"


def gameweeks_jugadas(season: str) -> list[int]:
    """Gameweeks con resultado ya cargado.

    ⚠️ NO se usa el flag `finished` de FPL. Medido el 24/08/2026: con los diez partidos de
    la fecha a 90 minutos y el marcador cargado, `finished` seguía en `False` — FPL lo
    activa recién cuando confirma los puntos de bonus, horas después. Detectar "fecha
    terminada" por ese flag dejaría al monitoreo ciego durante medio día.
    """
    m = read_table("fact_match")
    fx = read_table("fact_fixture")
    jugados = m[m["season"] == season][["season", "home_short", "away_short"]]
    if jugados.empty:
        return []
    programados = fx[fx["season"] == season]
    d = programados.merge(jugados, on=["season", "home_short", "away_short"])
    if d.empty:
        return []
    # Una fecha esta completa cuando TODOS sus partidos tienen resultado. Se reindexa
    # sobre las gameweeks programadas: las que no empezaron cuentan como cero jugados.
    total = programados.groupby("gameweek").size()
    con_resultado = d.groupby("gameweek").size().reindex(total.index, fill_value=0)
    completas = total.index[con_resultado == total]
    return sorted(int(g) for g in completas)


def evaluar_fecha(season: str, gameweek: int, gold: pd.DataFrame,
                  boosters, meta) -> dict | None:
    """Una fecha: predicción del modelo contra el resultado y contra los baselines."""
    obj = srv.objetivos_de_fecha(season, gameweek)
    from features import gold_tp

    feats = gold_tp.construir(objetivos=obj, con_target=False)
    X = feats[meta["feature_names"]].to_numpy(dtype=np.float32)
    P = srv.predecir_proba(boosters, X)

    real = read_table("fact_match")[["season", "home_short", "away_short", "target_1x2"]]
    d = feats[["season", "gameweek", "fixture_id", "home_short", "away_short"]].copy()
    d = d.merge(real, on=["season", "home_short", "away_short"], how="left")
    if d["target_1x2"].isna().any():
        return None

    y = d["target_1x2"].to_numpy()
    pred = np.array(CLASES_ORD)[P.argmax(1)]
    rep = metrics.reporte(y, pred, P, con_ic=False)

    # Los baselines, sobre LAS MISMAS filas. Comparar contra un baseline de otro período
    # confunde "el modelo empeoró" con "la liga se puso impredecible".
    hist = gold[gold["season"].isin(CFG.seasons_for_training())]
    local = baseline_siempre_local(d.assign(target_1x2=y))
    prior = baseline_prior_de_clase(hist, d.assign(target_1x2=y))

    fila = {
        "season": season, "gameweek": gameweek, "n": len(d),
        "accuracy": rep["accuracy"], "f1_macro": rep["f1_macro"],
        "log_loss": rep["log_loss"],
        "acc_siempre_local": local["accuracy"],
        "acc_prior": prior["accuracy"], "ll_prior": prior.get("log_loss"),
        "confianza_media": float(P.max(1).mean()),
        "aciertos": (pred == y).tolist(),
    }
    fila["gana_a_local"] = fila["accuracy"] > fila["acc_siempre_local"]
    fila["gana_al_prior"] = (fila["ll_prior"] is not None
                             and fila["log_loss"] < fila["ll_prior"])

    # Si hay cuotas, se agrega el resultado de apostar. En la temporada en curso puede que
    # todavía no estén: football-data publica el archivo de temporada con retraso.
    if all(c in feats.columns for c in spec.MERCADO):
        cols = spec.CLAVES + ["gameweek", "target_1x2"] + spec.MERCADO
        filas_ap = feats.assign(target_1x2=y)
        disponibles = [c for c in cols if c in filas_ap.columns]
        try:
            roi = betting.reporte(filas_ap[disponibles], P)["modelo"]
            fila["roi"] = roi.get("roi")
            fila["n_apuestas"] = roi.get("n_apuestas")
        except Exception:  # noqa: BLE001 — sin cuotas no se apuesta, y no es un error
            pass
    return fila


def correr(season: str | None = None) -> pd.DataFrame:
    season = season or CFG.current_season
    gws = gameweeks_jugadas(season)
    if not gws:
        log.warning("No hay ninguna fecha completa de %s todavía.", season)
        return pd.DataFrame()

    boosters, meta = srv.cargar_modelo()
    gold = dataset.cargar()
    log.info("Evaluando %s: %d fechas jugadas", season, len(gws))

    filas = [f for gw in gws
             if (f := evaluar_fecha(season, gw, gold, boosters, meta)) is not None]
    return pd.DataFrame(filas)


def resumen(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    aciertos = np.concatenate([np.asarray(a, dtype=bool) for a in df["aciertos"]])
    lo, hi = metrics.ic_bootstrap(aciertos.astype(int), np.ones(len(aciertos), dtype=int))
    return {
        "fechas": int(len(df)),
        "partidos": int(aciertos.size),
        "accuracy_acumulada": float(aciertos.mean()),
        "accuracy_ic95": [lo, hi],
        "log_loss_media": float(df["log_loss"].mean()),
        "accuracy_siempre_local": float(df["acc_siempre_local"].mean()),
        "pct_fechas_gana_a_local": float(df["gana_a_local"].mean()),
        "pct_fechas_gana_al_prior": float(df["gana_al_prior"].mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Monitoreo de la temporada en curso.")
    ap.add_argument("--season", default=None)
    args = ap.parse_args()

    setup(CFG.log_level, CFG.log_format)
    SALIDA.mkdir(parents=True, exist_ok=True)
    df = correr(args.season)
    if df.empty:
        print("Todavía no hay fechas completas para evaluar.")
        return

    season = df["season"].iloc[0]
    df.drop(columns="aciertos").to_csv(SALIDA / f"monitoreo_{season}.csv", index=False)

    print(f"\n{'=' * 78}")
    print(f"TEMPORADA EN CURSO — {season}   (el modelo NUNCA vio estos partidos)")
    print("=" * 78 + "\n")
    cols = ["gameweek", "n", "accuracy", "log_loss", "acc_siempre_local", "acc_prior",
            "confianza_media", "gana_a_local", "gana_al_prior"]
    print(df[[c for c in cols if c in df.columns]].round(3).to_string(index=False))

    r = resumen(df)
    print(f"\n{'-' * 78}\nACUMULADO\n{'-' * 78}")
    ic = r.pop("accuracy_ic95")
    for k, v in r.items():
        print(f"  {k:28s} {v:.4f}" if isinstance(v, float) else f"  {k:28s} {v}")
    print(f"  {'IC 95% de la accuracy':28s} [{ic[0]:.3f}, {ic[1]:.3f}]")
    if r["partidos"] < 100:
        print(f"\n  ⚠️  Con {r['partidos']} partidos el intervalo es enorme: "
              f"±{(ic[1] - ic[0]) / 2 * 100:.0f} puntos. Todavía no se puede concluir nada.")
    print(f"\nCSV en {SALIDA / f'monitoreo_{season}.csv'}")


if __name__ == "__main__":
    main()
