"""Logging unificado para todo el pipeline.

Se configura una sola vez por proceso; `get_logger` es lo que usa el resto del código.

**Dos formatos, según dónde corra.** En local, una línea legible por humanos. En Cloud Run,
**una línea de JSON por evento**, que es lo que hace que Cloud Logging la parsee y la deje
consultable por campo:

    gcloud logging read 'jsonPayload.evento="prediccion" AND jsonPayload.latencia_ms>100'

Sin eso, un log es una tira de texto: sirve para leer de a uno y no para preguntarle nada.
El formato se elige solo —si el proceso corre en Cloud Run va JSON— y se puede forzar
con `TP_LOG_FORMAT=json|texto`.

**Qué se loguea y qué no.** La decisión y la métrica; nunca datos de quien consulta. Acá el
dominio lo hace fácil (son equipos de fútbol, no clientes), y justamente por eso conviene
dejar la regla escrita: el día que el caso tenga PII, el lugar donde se decide es éste.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
import urllib.request
from datetime import datetime, timezone

_CONFIGURED = False

# El request que se está atendiendo, para que TODA línea que se loguee mientras dura
# lleve con qué request va. Lo pone el middleware de `serving/main.py`; fuera de un
# request (el Job, un test) queda vacío y el formatter no agrega nada.
#
#   trace    -> `logging.googleapis.com/trace`: Cloud Logging agrupa bajo la línea del
#               request (la de `run.googleapis.com/requests`, con status y latencia del
#               borde) todas las de la app. Sin él, son tres entradas sueltas que sólo se
#               asocian por timestamp.
#   corrida  -> el id que manda `scripts/smoke_load.py` en `X-Corrida`. Es lo que permite
#               poner lado a lado la latencia medida en el cliente y la medida en el
#               servidor PARA LOS MISMOS requests.
_REQUEST: contextvars.ContextVar[dict | None] = contextvars.ContextVar("_REQUEST", default=None)

# `trace` necesita el proyecto ("projects/P/traces/T"). Cloud Run no define
# GOOGLE_CLOUD_PROJECT, así que si no vino por variable se le pregunta una vez al
# metadata server, que sólo existe adentro de GCP.
_PROYECTO: list[str | None] = []
_METADATA = "http://metadata.google.internal/computeMetadata/v1/project/project-id"

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
        req = _REQUEST.get()
        if req:
            if req.get("trace"):
                salida["logging.googleapis.com/trace"] = req["trace"]
            if req.get("corrida"):
                salida.setdefault("corrida", req["corrida"])
        if record.exc_info:
            salida["exception"] = self.formatException(record.exc_info)
        return json.dumps(salida, default=str, ensure_ascii=False)


# Cloud Run define variables DISTINTAS segun que corra: `K_SERVICE` en un Service,
# `CLOUD_RUN_JOB` en un Job. Mirar solo la primera dejaba al pipeline logueando en
# texto plano -- o sea sin `jsonPayload`, sin poder filtrar por paso ni por fecha--,
# y es justo la corrida sobre la que uno quiere preguntar cuando algo sale mal.
EN_CLOUD_RUN = ("K_SERVICE", "CLOUD_RUN_JOB")


def en_cloud_run() -> bool:
    """¿Este proceso corre en Cloud Run, sea Service o Job?"""
    return any(os.getenv(v) for v in EN_CLOUD_RUN)


def formato_elegido() -> str:
    """`json` o `texto`. En Cloud Run, JSON; en tu terminal, texto."""
    pedido = (os.getenv("TP_LOG_FORMAT") or "").strip().lower()
    if pedido in ("json", "texto"):
        return pedido
    return "json" if en_cloud_run() else "texto"


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


def proyecto_gcp() -> str | None:
    """El Project ID, para armar el `trace`. Se resuelve una vez por proceso."""
    if not _PROYECTO:
        p = os.getenv("TP_GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
        if not p and en_cloud_run():
            try:
                pedido = urllib.request.Request(_METADATA, headers={"Metadata-Flavor": "Google"})
                with urllib.request.urlopen(pedido, timeout=1) as r:
                    p = r.read().decode().strip() or None
            except Exception:  # noqa: BLE001 — sin proyecto, sin trace; el log sigue
                p = None
        _PROYECTO.append(p)
    return _PROYECTO[0]


def trace_de(cabeceras: dict[str, str]) -> str | None:
    """`projects/P/traces/T` a partir de las cabeceras que agrega Cloud Run.

    `X-Cloud-Trace-Context: TRACE/SPAN;o=1` es la de siempre; `traceparent`
    (`00-TRACE-SPAN-01`, W3C) es la que la reemplaza. Se acepta cualquiera.
    """
    trace_id = None
    xctc = cabeceras.get("x-cloud-trace-context")
    if xctc:
        trace_id = xctc.split("/", 1)[0].strip() or None
    elif cabeceras.get("traceparent"):
        partes = cabeceras["traceparent"].split("-")
        trace_id = partes[1] if len(partes) >= 3 else None
    proyecto = proyecto_gcp()
    if not trace_id or not proyecto:
        return None
    return f"projects/{proyecto}/traces/{trace_id}"


def abrir_request(trace: str | None, corrida: str | None) -> contextvars.Token:
    """Marca el request en curso. Devuelve el token para `cerrar_request`."""
    # La corrida viene de afuera: se recorta para que nadie meta un párrafo en cada línea.
    corrida = (corrida or "").strip()[:64] or None
    return _REQUEST.set({"trace": trace, "corrida": corrida})


def cerrar_request(token: contextvars.Token) -> None:
    _REQUEST.reset(token)


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


def evento(log: logging.Logger, nombre: str, mensaje: str,
           nivel: int = logging.INFO, **campos) -> None:
    """Loguea un evento consultable por campo.

        evento(log, "prediccion", "GW5 servida", gameweek=5, latencia_ms=47.1)

    En local sale como texto legible; en Cloud Run, como `jsonPayload` con `evento`,
    `gameweek` y `latencia_ms` como campos propios.

    `nivel` importa más de lo que parece: Cloud Logging lo traduce a `severity`, y es lo
    que permite filtrar los errores en la consola sin escribir una consulta.
    """
    log.log(nivel, mensaje, extra={"evento": nombre, **campos})
