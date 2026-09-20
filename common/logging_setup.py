"""Logging unificado para todo el pipeline.

Se configura una sola vez por proceso; `get_logger` es lo que usa el resto del código.

**Dos formatos, según dónde corra.** En local, una línea legible por humanos. En Cloud Run,
**una línea de JSON por evento**, que es lo que hace que Cloud Logging la parsee y la deje
consultable por campo:

    gcloud logging read 'jsonPayload.evento="prediccion" AND jsonPayload.latencia_ms>100'

Sin eso, un log es una tira de texto: sirve para leer de a uno y no para preguntarle nada.
El formato se elige solo —si hay `K_SERVICE`, la variable que Cloud Run siempre define,
va JSON— y se puede forzar con `TP_LOG_FORMAT=json|texto`.

**Qué se loguea y qué no.** La decisión y la métrica; nunca datos de quien consulta. Acá el
dominio lo hace fácil (son equipos de fútbol, no clientes), y justamente por eso conviene
dejar la regla escrita: el día que el caso tenga PII, el lugar donde se decide es éste.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone

_CONFIGURED = False

# Cloud Run mapea `severity` a su propio nivel; con "INFO"/"ERROR" alcanza.
_SEVERIDAD = {"WARNING": "WARNING", "ERROR": "ERROR", "CRITICAL": "CRITICAL",
              "DEBUG": "DEBUG", "INFO": "INFO"}

# Lo que trae un LogRecord de fábrica. Todo lo demás que aparezca en `record.__dict__`
# lo puso alguien con `extra=`, y es justamente lo que queremos en el JSON.
_ESTANDAR = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename", "module",
    "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created", "msecs",
    "relativeCreated", "thread", "threadName", "processName", "process", "taskName",
    "message", "asctime",
}


class FormatoJSON(logging.Formatter):
    """Una línea de JSON por evento, con los campos de `extra` al mismo nivel.

    Los campos van planos y no anidados bajo una clave: así se consultan directo como
    `jsonPayload.latencia_ms` en vez de tener que cavar.
    """

    def format(self, record: logging.LogRecord) -> str:
        salida = {
            "severity": _SEVERIDAD.get(record.levelname, "INFO"),
            "message": record.getMessage(),
            "logger": record.name,
            "time": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
        }
        for k, v in record.__dict__.items():
            if k not in _ESTANDAR and not k.startswith("_"):
                salida[k] = v
        if record.exc_info:
            salida["exception"] = self.formatException(record.exc_info)
        return json.dumps(salida, default=str, ensure_ascii=False)


def formato_elegido() -> str:
    """`json` o `texto`. En Cloud Run, JSON; en tu terminal, texto."""
    pedido = (os.getenv("TP_LOG_FORMAT") or "").strip().lower()
    if pedido in ("json", "texto"):
        return pedido
    return "json" if os.getenv("K_SERVICE") else "texto"


def setup(level: str | None = None, fmt: str | None = None) -> None:
    """Configura el root logger. Idempotente: llamarlo dos veces no duplica handlers."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stdout)
    if formato_elegido() == "json":
        handler.setFormatter(FormatoJSON())
    else:
        fmt = fmt or "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"
        handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))

    nivel = level or os.getenv("TP_LOG_LEVEL") or "INFO"
    root = logging.getLogger()
    root.setLevel(getattr(logging, str(nivel).upper(), logging.INFO))
    root.addHandler(handler)

    # urllib3 loguea cada reintento en DEBUG; a INFO ya es ruido.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    _CONFIGURED = True


def reset() -> None:
    """Deshace la configuración. Para tests que necesitan probar los dos formatos."""
    global _CONFIGURED
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    _CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Devuelve un logger, configurando el root si hace falta."""
    if not _CONFIGURED:
        setup()
    return logging.getLogger(name)


def evento(log: logging.Logger, nombre: str, mensaje: str, **campos) -> None:
    """Loguea un evento consultable por campo.

        evento(log, "prediccion", "GW5 servida", gameweek=5, latencia_ms=47.1)

    En local sale como texto legible; en Cloud Run, como `jsonPayload` con `evento`,
    `gameweek` y `latencia_ms` como campos propios.
    """
    log.info(mensaje, extra={"evento": nombre, **campos})
