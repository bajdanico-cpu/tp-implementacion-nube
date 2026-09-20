"""Eventos estructurados de la API: una línea JSON por evento, a stdout.

Cloud Run levanta stdout y, si la línea es JSON, Cloud Logging la guarda como `jsonPayload`
(consultable por campo). Se loguean decisiones y métricas, nunca las features del modelo.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator

logger = logging.getLogger("premier.api")

_configured = False


def configure_logging() -> None:
    """Deja el logger de la API emitiendo JSON crudo a stdout. Idempotente."""
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    _configured = True


@contextmanager
def measure_latency() -> Iterator[dict[str, float]]:
    """Mide el tiempo de pared del bloque; los ms quedan en el dict al salir."""
    holder: dict[str, float] = {"latency_ms": 0.0}
    start = time.perf_counter()
    try:
        yield holder
    finally:
        holder["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)


def log_event(event: str, **fields: Any) -> None:
    """Emite un evento estructurado. `event` nombra el tipo; el resto son campos libres."""
    logger.info(json.dumps({"event": event, **fields}, ensure_ascii=False))
