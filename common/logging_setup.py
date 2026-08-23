"""Logging unificado para todo el pipeline.

Se configura una sola vez por proceso; `get_logger` es lo que usa el resto del código.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def setup(level: str = "INFO", fmt: str | None = None) -> None:
    """Configura el root logger. Idempotente: llamarlo dos veces no duplica handlers."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    fmt = fmt or "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.addHandler(handler)

    # urllib3 loguea cada reintento en DEBUG; a INFO ya es ruido.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Devuelve un logger, configurando el root si hace falta."""
    if not _CONFIGURED:
        setup()
    return logging.getLogger(name)
