"""Descarga HTTP con reintentos, compartida por los ingestores de archivos planos
(vaastav y football-data). La API de FPL tiene su propio cliente en `fpl_client.py`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests

from common.logging_setup import get_logger

log = get_logger(__name__)

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


@dataclass
class FetchResult:
    """Resultado de una descarga. `ok=False` con `status=404` es un caso esperado
    (p.ej. `gws/` de una temporada que todavía no arrancó), no un fallo."""

    url: str
    status: int
    content: bytes | None
    ok: bool
    error: str | None = None
    # URL final tras seguir redirects. Importante: football-data redirige la
    # temporada que aún no empezó a OTRA división, y eso hay que detectarlo.
    final_url: str | None = None
    redirected: bool = False


def fetch(url: str, timeout: int = 60, max_retries: int = 3, backoff: float = 2.0) -> FetchResult:
    """GET con reintentos sobre 429/5xx. Los 4xx se devuelven sin reintentar."""
    last_error: str | None = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers={"User-Agent": _UA}, timeout=timeout)

            if resp.status_code == 200:
                return FetchResult(
                    url=url,
                    status=200,
                    content=resp.content,
                    ok=True,
                    final_url=resp.url,
                    redirected=len(resp.history) > 0,
                )

            # 3xx que requests no siguió solo. El caso real: football-data responde
            # 300 (Multiple Choices) para la temporada que todavía no arrancó, porque
            # el archivo no existe y Apache ofrece alternativas de otras divisiones.
            # Es un estado permanente, no transitorio: reintentar es tiempo perdido.
            if 300 <= resp.status_code < 400:
                return FetchResult(
                    url=url,
                    status=resp.status_code,
                    content=None,
                    ok=False,
                    error=f"HTTP {resp.status_code} (el recurso no está disponible)",
                    final_url=resp.url,
                    redirected=len(resp.history) > 0,
                )

            # 4xx que no sea 429: respuesta legítima del servidor, no se reintenta.
            if 400 <= resp.status_code < 500 and resp.status_code != 429:
                return FetchResult(
                    url=url,
                    status=resp.status_code,
                    content=None,
                    ok=False,
                    error=f"HTTP {resp.status_code}",
                    final_url=resp.url,
                    redirected=len(resp.history) > 0,
                )

            last_error = f"HTTP {resp.status_code}"

        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt < max_retries:
            wait = backoff**attempt
            log.warning("%s falló (%d/%d): %s. Reintento en %.1fs",
                        url, attempt, max_retries, last_error, wait)
            time.sleep(wait)

    return FetchResult(url=url, status=0, content=None, ok=False, error=last_error)
