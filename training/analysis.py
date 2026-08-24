"""Dónde le gana el modelo a cada baseline, y dónde no.

Un número global ("accuracy 0,49") no dice si el modelo sirve: dice un promedio. Lo que
decide si vale la pena es **en qué situaciones** le gana al mercado y a los baselines, y si
esas situaciones se pueden identificar de antemano — porque si se pueden, se apuesta sólo
ahí.

    python -m training.analysis

Cortes que se miran:

- **Fecha a fecha** — cuántas gameweeks le gana a cada vara.
- **Por favoritismo del mercado** — ¿gana en los partidos parejos o en los cantados?
- **Por acuerdo con el mercado** — cuando el modelo dice algo distinto al mercado, ¿quién
  tiene razón? Es la pregunta que decide si el modelo aporta información propia.
- **Por clase** — dónde se pierde el log-loss.
- **El costo de subestimar el empate** — como las tres probabilidades suman 1, todo lo que
  se le quita al empate se les regala a local y visitante, e infla el valor esperado de
  cada apuesta. Se mide cuánto.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common.config import CFG, PROJECT_ROOT
from common.logging_setup import get_logger, setup
from eda.baselines import CLASES_ORD
from training import betting, dataset, evaluate, metrics
from training.device import resolve

log = get_logger(__name__)

SALIDA = PROJECT_ROOT / "training" / "output"
IDX_DRAW = CLASES_ORD.index("draw")


def _tabla(filas: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    """Un dataframe con todo lo necesario para cortar: modelo, mercado y resultado."""
    P_mkt = filas[["p_mercado_away", "p_mercado_draw", "p_mercado_home"]].to_numpy()
    y = filas["target_1x2"].to_numpy()
    i_real = np.array([CLASES_ORD.index(c) for c in y])

    d = pd.DataFrame({
        "season": filas["season"].to_numpy(),
        "gameweek": filas["gameweek"].to_numpy(),
        "fixture_id": filas["fixture_id"].to_numpy(),
        "real": y,
        "pred_modelo": np.array(CLASES_ORD)[proba.argmax(1)],
        "pred_mercado": np.array(CLASES_ORD)[P_mkt.argmax(1)],
        "p_modelo_real": proba[np.arange(len(y)), i_real],
        "p_mercado_real": P_mkt[np.arange(len(y)), i_real],
        "p_modelo_draw": proba[:, IDX_DRAW],
        "p_mercado_draw": P_mkt[:, IDX_DRAW],
        "favoritismo": P_mkt.max(1),
    })
    d["acierta_modelo"] = d["real"] == d["pred_modelo"]
    d["acierta_mercado"] = d["real"] == d["pred_mercado"]
    d["acierta_local"] = d["real"] == "home"
    d["ll_modelo"] = -np.log(np.clip(d["p_modelo_real"], 1e-12, 1))
    d["ll_mercado"] = -np.log(np.clip(d["p_mercado_real"], 1e-12, 1))
    d["coinciden"] = d["pred_modelo"] == d["pred_mercado"]
    return d


def por_fecha(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby("gameweek").agg(
        n=("real", "size"),
        acc_modelo=("acierta_modelo", "mean"),
        acc_mercado=("acierta_mercado", "mean"),
        acc_local=("acierta_local", "mean"),
        ll_modelo=("ll_modelo", "mean"),
        ll_mercado=("ll_mercado", "mean"),
    ).reset_index()
    g["gana_a_mercado"] = g["acc_modelo"] > g["acc_mercado"]
    g["gana_a_local"] = g["acc_modelo"] > g["acc_local"]
    g["gana_ll_mercado"] = g["ll_modelo"] < g["ll_mercado"]
    return g


def por_favoritismo(d: pd.DataFrame) -> pd.DataFrame:
    """¿El modelo aporta en los partidos parejos o en los cantados?"""
    d = d.copy()
    d["tramo"] = pd.cut(d["favoritismo"], [0, .40, .50, .60, 1.0],
                        labels=["muy parejo (<40%)", "parejo (40-50%)",
                                "favorito claro (50-60%)", "cantado (>60%)"])
    return (d.groupby("tramo", observed=True)
             .agg(n=("real", "size"),
                  acc_modelo=("acierta_modelo", "mean"),
                  acc_mercado=("acierta_mercado", "mean"),
                  ll_modelo=("ll_modelo", "mean"),
                  ll_mercado=("ll_mercado", "mean"))
             .assign(dif_acc=lambda x: x.acc_modelo - x.acc_mercado,
                     dif_ll=lambda x: x.ll_modelo - x.ll_mercado)
             .reset_index())


def por_acuerdo(d: pd.DataFrame) -> pd.DataFrame:
    """Cuando el modelo discrepa del mercado, ¿quién acierta?

    Es la pregunta que define si el modelo tiene información propia o sólo repite lo que
    el mercado ya sabe.
    """
    filas = []
    for coinciden, g in d.groupby("coinciden"):
        filas.append({
            "situacion": "coinciden" if coinciden else "DISCREPAN",
            "n": len(g),
            "acierta_modelo": g["acierta_modelo"].mean(),
            "acierta_mercado": g["acierta_mercado"].mean(),
            "ll_modelo": g["ll_modelo"].mean(),
            "ll_mercado": g["ll_mercado"].mean(),
        })
    return pd.DataFrame(filas)


def por_clase(d: pd.DataFrame) -> pd.DataFrame:
    """Dónde se pierde el log-loss: qué resultado real castiga más."""
    return (d.groupby("real")
             .agg(n=("real", "size"),
                  ll_modelo=("ll_modelo", "mean"),
                  ll_mercado=("ll_mercado", "mean"),
                  p_modelo=("p_modelo_real", "mean"),
                  p_mercado=("p_mercado_real", "mean"))
             .assign(dif_ll=lambda x: x.ll_modelo - x.ll_mercado)
             .reset_index())


def costo_del_empate(d: pd.DataFrame, filas: pd.DataFrame,
                     proba: np.ndarray) -> dict:
    """Cuánto cuesta subestimar el empate, en unidades de valor esperado.

    Las tres probabilidades suman 1: todo lo que se le quita al empate se reparte entre
    local y visitante. Con cuotas de ~2 a ~4, un punto de probabilidad de más se convierte
    en 2 a 4 puntos de EV inflado — y el umbral de apuesta es 5 puntos. O sea que un sesgo
    chico en el empate alcanza para disparar apuestas que no tenían valor.
    """
    sesgo = float(d["p_modelo_draw"].mean() - (d["real"] == "draw").mean())
    sesgo_mkt = float(d["p_mercado_draw"].mean() - (d["real"] == "draw").mean())

    # Corrección: se le devuelve al empate lo que le falta y se descuenta proporcional
    # del resto. Es la corrección mínima que no toca el orden relativo de local y visita.
    P = proba.copy()
    faltante = -sesgo
    P[:, IDX_DRAW] += faltante
    otros = [i for i in range(3) if i != IDX_DRAW]
    total_otros = P[:, otros].sum(axis=1, keepdims=True)
    P[:, otros] *= (1 - P[:, [IDX_DRAW]]) / total_otros
    P = np.clip(P, 1e-9, 1)
    P /= P.sum(axis=1, keepdims=True)

    return {
        "sesgo_empate_modelo": sesgo,
        "sesgo_empate_mercado": sesgo_mkt,
        "roi_original": betting.reporte(filas, proba)["modelo"].get("roi"),
        "n_apuestas_original": betting.reporte(filas, proba)["modelo"].get("n_apuestas"),
        "roi_corrigiendo_empate": betting.reporte(filas, P)["modelo"].get("roi"),
        "n_apuestas_corrigiendo": betting.reporte(filas, P)["modelo"].get("n_apuestas"),
        "log_loss_original": float(metrics.reporte(
            d["real"].to_numpy(), d["pred_modelo"].to_numpy(), proba,
            con_ic=False)["log_loss"]),
        "log_loss_corrigiendo": float(metrics.reporte(
            d["real"].to_numpy(), np.array(CLASES_ORD)[P.argmax(1)], P,
            con_ic=False)["log_loss"]),
    }


def apuestas_por_clase(filas: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    """A qué se apuesta, cuánto se acierta y cuánto se gana, desglosado."""
    ap = betting.decidir(filas, proba)
    if ap.empty:
        return pd.DataFrame()
    return (ap.groupby("apuesta")
              .agg(n=("ganancia", "size"), acierto=("acierta", "mean"),
                   cuota_media=("cuota", "mean"), ev_medio=("ev", "mean"),
                   ganancia=("ganancia", "sum"))
              .assign(roi=lambda x: x.ganancia / x.n)
              .reset_index())


def correr(modelo: str = "xgb_gbt") -> dict:
    info = resolve("auto")
    gold = dataset.cargar()
    res = evaluate.evaluar_holdout(modelo, info, gold=gold)
    filas, proba = res["split"].filas_test, res["proba"]
    d = _tabla(filas, proba)

    fechas = por_fecha(d)
    out = {
        "detalle": d,
        "por_fecha": fechas,
        "por_favoritismo": por_favoritismo(d),
        "por_acuerdo": por_acuerdo(d),
        "por_clase": por_clase(d),
        "costo_empate": costo_del_empate(d, filas, proba),
        "apuestas": apuestas_por_clase(filas, proba),
        "resumen_fechas": {
            "fechas": len(fechas),
            "gana_a_siempre_local": float(fechas["gana_a_local"].mean()),
            "gana_al_mercado_en_accuracy": float(fechas["gana_a_mercado"].mean()),
            "gana_al_mercado_en_logloss": float(fechas["gana_ll_mercado"].mean()),
        },
    }
    return out


def main() -> None:
    setup(CFG.log_level, CFG.log_format)
    SALIDA.mkdir(parents=True, exist_ok=True)
    r = correr()

    print("\n" + "=" * 78)
    print("DONDE LE GANA EL MODELO A CADA VARA — holdout 2025-26, 380 partidos")
    print("=" * 78)

    rf = r["resumen_fechas"]
    print(f"\nSobre {rf['fechas']} fechas:")
    print(f"  le gana a 'siempre local' en accuracy : {rf['gana_a_siempre_local']:.1%}")
    print(f"  le gana al mercado en accuracy        : {rf['gana_al_mercado_en_accuracy']:.1%}")
    print(f"  le gana al mercado en log-loss        : {rf['gana_al_mercado_en_logloss']:.1%}")

    print("\n--- Por favoritismo del mercado ---")
    print(r["por_favoritismo"].round(3).to_string(index=False))

    print("\n--- Cuando el modelo discrepa del mercado, quien acierta ---")
    print(r["por_acuerdo"].round(3).to_string(index=False))

    print("\n--- Por resultado real: donde se pierde el log-loss ---")
    print(r["por_clase"].round(3).to_string(index=False))

    print("\n--- El costo de subestimar el empate ---")
    for k, v in r["costo_empate"].items():
        print(f"  {k:26s} {v:.4f}" if isinstance(v, float) else f"  {k:26s} {v}")

    if not r["apuestas"].empty:
        print("\n--- Apuestas por clase ---")
        print(r["apuestas"].round(3).to_string(index=False))

    r["por_fecha"].to_csv(SALIDA / "analisis_por_fecha.csv", index=False)
    r["detalle"].to_csv(SALIDA / "analisis_detalle.csv", index=False)
    print(f"\nDetalle en {SALIDA}")


if __name__ == "__main__":
    main()
