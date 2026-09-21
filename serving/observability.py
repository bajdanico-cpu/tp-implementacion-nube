"""Los eventos que emite la API: qué se loguea de cada request, y qué no.

El formato y el transporte viven en `common/logging_setup.py` —texto en local, una línea
de JSON en Cloud Run, para todo el proyecto y no sólo para la API—. Acá está lo propio del
serving: **qué campos tiene cada evento**.

Tres eventos, y el segundo es el que más importa:

    prediccion         una fecha servida, con su latencia
    prediccion_error   una que falló, con el status y el motivo
    health_degradado   el pulso dejó de estar en ok

Loguear sólo los éxitos es la forma más común de tener observabilidad que no sirve:
cuando algo anda mal no hay nada que mirar, que es justo cuando hace falta.

**Qué NO va.** Sólo agregados: cuántos partidos, cuántos de cada clase, la confianza
media, la latencia. Nunca las probabilidades por partido ni las features. Acá la regla es
barata de cumplir —son equipos de fútbol, no clientes— y por eso conviene dejarla escrita
y con un test que la vigile: el día que el caso tenga PII, el lugar donde se decide es
éste, y para entonces ya va a estar la costumbre.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pandas as pd

from common.logging_setup import evento, get_logger
from eda.baselines import CLASES_ORD

log = get_logger("serving.api")


class Cronometro:
    """Milisegundos desde que arrancó, leíbles **en cualquier momento**.

    Se calcula al leer `.ms` y no al salir del bloque, porque el caso que importa es el
    del `except`: ahí adentro el `finally` del context manager todavía no corrió, y una
    marca que se completa al final habría dado 0,0 ms en todos los errores.
    """

    __slots__ = ("_inicio", "_fin")

    def __init__(self) -> None:
        self._inicio = time.perf_counter()
        self._fin: float | None = None

    @property
    def ms(self) -> float:
        fin = self._fin if self._fin is not None else time.perf_counter()
        return round((fin - self._inicio) * 1000, 2)

    def parar(self) -> None:
        self._fin = time.perf_counter()


@contextmanager
def medir_latencia() -> Iterator[Cronometro]:
    """Mide el tiempo del bloque, **también si el bloque falla**.

    Con un `perf_counter()` suelto antes y después, una excepción se lleva puesta la
    medición — y las que interesan son justamente las que fallan.
    """
    c = Cronometro()
    try:
        yield c
    finally:
        c.parar()


def prediccion(season: str, gameweek: int, estado: str, origen: str,
               pred: pd.DataFrame, latencia_ms: float, **extra: Any) -> None:
    """Una fecha servida. Todo agregado: nada por partido."""
    evento(
        log, "prediccion",
        f"{season} GW{gameweek} [{estado}/{origen}] {len(pred)} partidos, {latencia_ms} ms",
        season=season, gameweek=gameweek, estado=estado, origen=origen,
        n_partidos=int(len(pred)),
        # Cuántas de cada clase anunció. Alcanza para ver de un vistazo si el modelo se
        # volvió monótono —todo local, por ejemplo— sin exponer una sola probabilidad.
        anunciadas={c: int((pred["prediccion"] == c).sum()) for c in CLASES_ORD},
        confianza_media=round(float(pred["confianza"].mean()), 4),
        model_version=pred["model_version"].iloc[0],
        feature_set_version=pred["feature_set_version"].iloc[0],
        latencia_ms=latencia_ms,
        **extra,
    )


def prediccion_error(season: str, gameweek: int, status: int, error: BaseException,
                     latencia_ms: float) -> None:
    """Una fecha que no se pudo servir. Es el evento que se consulta cuando algo anda mal."""
    evento(
        log, "prediccion_error",
        f"{season} GW{gameweek} -> {status}: {error}",
        # 5xx es culpa nuestra; 4xx es de quien pidió. Separarlos por severidad hace que
        # en la consola se puedan mirar sólo los que nos corresponden.
        nivel=logging.ERROR if status >= 500 else logging.WARNING,
        season=season, gameweek=gameweek, status=int(status),
        error_type=type(error).__name__, reason=str(error)[:400],
        latencia_ms=latencia_ms,
    )


def health_degradado(detalle: str | None) -> None:
    evento(log, "health_degradado", f"health degradado: {detalle}",
           nivel=logging.ERROR, reason=(detalle or "")[:400])
