"""Bloque 6 del canvas: "apostamos o no apostamos".

Acá —y sólo acá— entran las cuotas. **No son features del modelo**, y la razón es
estructural, no una precaución:

> Si el modelo usa las cuotas como feature, aprende a copiarlas. Entonces `p ≈ 1/cuota`,
> el valor esperado da ~0 por construcción y **el sistema nunca encontraría una apuesta
> con valor**. Detectar una discrepancia con el mercado exige que las dos estimaciones
> sean independientes.

A eso se suma que en producción no llegan a tiempo: `fixtures.csv` de football-data cubre
sólo los próximos 2-3 días, y el archivo de temporada se publica *después* de los partidos.

El valor esperado de apostar 1 unidad a un resultado con probabilidad `p` y cuota decimal
`c` (que devuelve `c` unidades incluyendo la apuesta):

    EV = p·(c − 1) − (1 − p)·1 = p·c − 1

Se apuesta sólo donde `EV > umbral`. El umbral no es cero: con `EV = 0` cualquier error de
estimación pone la apuesta en pérdida, así que se exige un margen.

**Lo que se espera, dicho de antemano:** el overround de las casas ronda el 5 %, o sea que
el juego tiene una comisión implícita de ese orden. Para que el ROI dé positivo, el modelo
tiene que estar mejor calibrado que el mercado por más que esa comisión. Lo probable es
ROI negativo — y ése es un resultado válido y reportable, no un fracaso del pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common.config import CFG
from eda.baselines import CLASES_ORD

CUOTAS = {"home": "odds_avg_close_home",
          "draw": "odds_avg_close_draw",
          "away": "odds_avg_close_away"}


def valor_esperado(proba: np.ndarray, cuotas: np.ndarray) -> np.ndarray:
    """EV por unidad apostada: p·c − 1. Cero exacto si la cuota es justa (c = 1/p)."""
    return proba * cuotas - 1.0


def decidir(filas: pd.DataFrame, proba: np.ndarray,
            umbral_ev: float | None = None) -> pd.DataFrame:
    """Una fila por apuesta candidata, con su EV. Sólo quedan las que superan el umbral.

    `proba` viene con las columnas en el orden de `CLASES_ORD`.
    """
    umbral = (CFG.training.get("apuestas", {}).get("umbral_ev", 0.05)
              if umbral_ev is None else umbral_ev)

    cuotas = filas[[CUOTAS[c] for c in CLASES_ORD]].to_numpy(dtype=float)
    ev = valor_esperado(proba, cuotas)

    cand = []
    for j, clase in enumerate(CLASES_ORD):
        d = filas[["season", "fixture_id", "gameweek", "target_1x2"]].copy()
        d["apuesta"] = clase
        d["p_modelo"] = proba[:, j]
        d["cuota"] = cuotas[:, j]
        d["ev"] = ev[:, j]
        cand.append(d)
    todas = pd.concat(cand, ignore_index=True)

    apostadas = todas[todas["ev"] > umbral].copy()
    apostadas["acierta"] = apostadas["apuesta"] == apostadas["target_1x2"]
    # Stake plano: 1 unidad por apuesta. Ganancia = cuota − 1 si acierta, −1 si no.
    apostadas["ganancia"] = np.where(apostadas["acierta"], apostadas["cuota"] - 1.0, -1.0)
    return apostadas.sort_values(["season", "gameweek", "fixture_id"]).reset_index(drop=True)


def simular(apuestas: pd.DataFrame) -> dict:
    """ROI, drawdown y tasa de acierto de una tanda de apuestas con stake plano."""
    if apuestas.empty:
        return {"n_apuestas": 0, "roi": None, "nota": "ninguna apuesta superó el umbral"}

    g = apuestas["ganancia"].to_numpy()
    acumulada = np.cumsum(g)
    pico = np.maximum.accumulate(np.concatenate([[0.0], acumulada]))[1:]
    out = {
        "n_apuestas": int(len(g)),
        "unidades_apostadas": float(len(g)),
        "ganancia_neta": float(g.sum()),
        "roi": float(g.sum() / len(g)),
        "tasa_acierto": float(apuestas["acierta"].mean()),
        "cuota_media": float(apuestas["cuota"].mean()),
        "drawdown_maximo": float(np.max(pico - acumulada)),
        "por_clase": apuestas.groupby("apuesta")["ganancia"].agg(["count", "sum"])
                             .to_dict("index"),
    }
    # La referencia trivial no tiene EV: no eligió nada, apuesta a todo.
    if "ev" in apuestas:
        out["ev_medio_estimado"] = float(apuestas["ev"].mean())
    return out


def referencia_siempre_local(filas: pd.DataFrame) -> dict:
    """Apostar al local en todos los partidos: la estrategia trivial contra la que medir."""
    d = filas.copy()
    d["cuota"] = d[CUOTAS["home"]]
    d["acierta"] = d["target_1x2"] == "home"
    d["ganancia"] = np.where(d["acierta"], d["cuota"] - 1.0, -1.0)
    return simular(d[["ganancia", "acierta", "cuota"]].assign(apuesta="home"))


def reporte(filas: pd.DataFrame, proba: np.ndarray,
            umbral_ev: float | None = None) -> dict:
    """La simulación completa, más la referencia trivial."""
    apuestas = decidir(filas, proba, umbral_ev)
    return {
        "modelo": simular(apuestas),
        "siempre_local": referencia_siempre_local(filas),
        "umbral_ev": (CFG.training.get("apuestas", {}).get("umbral_ev", 0.05)
                      if umbral_ev is None else umbral_ev),
        "overround_medio": float(
            (1 / filas[[CUOTAS[c] for c in CLASES_ORD]]).sum(axis=1).mean()),
    }
