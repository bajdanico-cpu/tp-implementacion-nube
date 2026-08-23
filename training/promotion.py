"""Cuándo un modelo nuevo reemplaza al que está en producción.

El canvas dice *"se compara en la siguiente fecha, si le gana al de producción se pasa a
producción"*. Tomado literalmente eso no se puede medir, y vale la pena decir por qué con
números:

| | |
|---|---|
| partidos por gameweek | **10** |
| error estándar de la accuracy con n=10 | **±15,7 puntos** |
| filas que agrega una semana sobre 1.140 de train | **+0,9 %** |

Comparar dos modelos sobre una sola fecha es tirar una moneda: promoverías al peor
aproximadamente la mitad de las veces. Y con +0,9 % de datos nuevos el candidato es casi
idéntico al que está en producción, así que lo que medís es ruido, no aprendizaje.

Por eso se separan las cadencias:

- **Reentrenar: semanal.** Es barato (segundos) y es lo que demuestra el loop cerrado.
- **Promover: cuando el test lo respalda**, sobre una ventana de varias fechas.

El test es **McNemar pareado**. Como los dos modelos predicen exactamente los mismos
partidos, la comparación es pareada y sólo aportan información los partidos donde
**discrepan**. Eso lo hace mucho más potente que comparar dos accuracies sueltas, que es
lo que haría falta si cada modelo hubiera visto partidos distintos.

> El disparador honesto no es el calendario, es el **cambio de temporada**: cada año
> cambian 3 de 20 equipos y rota ~40 % de los minutos. El reentreno semanal se mantiene
> porque es barato y porque demostrar el ciclo es el objetivo del TP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from common.config import CFG
from common.logging_setup import get_logger
from training import metrics

log = get_logger(__name__)


@dataclass(frozen=True)
class Decision:
    promover: bool
    motivo: str
    detalle: dict[str, Any]


def _cfg(clave: str, default):
    return CFG.training.get("promocion", {}).get(clave, default)


def decidir(aciertos_candidato: np.ndarray, aciertos_produccion: np.ndarray,
            acc_holdout_candidato: float | None = None,
            acc_holdout_produccion: float | None = None,
            alpha: float | None = None) -> Decision:
    """Las tres condiciones para promover.

    1. McNemar pareado con p < alpha sobre la ventana rodante.
    2. El candidato gana en los pares discordantes.
    3. No empeora en el holdout fijo — que actúa de red: evita promover un modelo que
       mejoró en las últimas fechas por sobreajustarse al período reciente.
    """
    alpha = _cfg("alpha", 0.05) if alpha is None else alpha
    mc = metrics.mcnemar(aciertos_candidato, aciertos_produccion)
    detalle = {"mcnemar": mc, "alpha": alpha,
               "n_ventana": int(len(aciertos_candidato)),
               "acc_candidato": float(np.mean(aciertos_candidato)),
               "acc_produccion": float(np.mean(aciertos_produccion))}

    if mc["n_discordantes"] == 0:
        return Decision(False, "los dos modelos predicen exactamente igual", detalle)
    if mc["gana"] != "a":
        return Decision(False, "el candidato no gana en los pares discordantes", detalle)
    if mc["p_valor"] >= alpha:
        return Decision(
            False,
            f"la diferencia no es significativa (p={mc['p_valor']:.3f} >= {alpha}); "
            f"con {mc['n_discordantes']} pares discordantes no alcanza para decidir",
            detalle)

    if (acc_holdout_candidato is not None and acc_holdout_produccion is not None
            and acc_holdout_candidato < acc_holdout_produccion):
        detalle["holdout"] = {"candidato": acc_holdout_candidato,
                              "produccion": acc_holdout_produccion}
        return Decision(
            False,
            "gana en la ventana reciente pero empeora en el holdout fijo: "
            "huele a sobreajuste al período reciente", detalle)

    return Decision(True, f"McNemar p={mc['p_valor']:.4f} a favor del candidato", detalle)


def ventana_de_aciertos(wf, n_fechas: int | None = None) -> np.ndarray:
    """Concatena los aciertos de las últimas N fechas de un walk-forward.

    Con `n_fechas=10` son ~100 partidos, que es la escala donde una diferencia real
    empieza a distinguirse del ruido.
    """
    n = _cfg("ventana_fechas", 10) if n_fechas is None else n_fechas
    ultimas = wf.sort_values("gameweek").tail(n)
    return np.concatenate([np.asarray(a, dtype=bool) for a in ultimas["aciertos"]])
