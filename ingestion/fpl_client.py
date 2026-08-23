"""Cliente HTTP de la API oficial de Fantasy Premier League.

La API es pública y no pide autenticación, pero tiene dos particularidades:

1. **Exige User-Agent de browser.** Sin él responde 403.
2. **Tiene política de CORS**, así que no se puede llamar desde un frontend.
   Siempre se consume desde el servidor (o desde acá, en local).

Devuelve los bytes crudos además del JSON parseado: Bronze guarda el crudo tal
cual llegó, sin normalizar nada.
"""

from __future__ import annotations

import json
import time
from typing import Any

import requests

from common.config import CFG
from common.logging_setup import get_logger

log = get_logger(__name__)


class FPLClientError(RuntimeError):
    """Falla de red o de la API tras agotar los reintentos."""


class FPLClient:
    """Cliente con reintentos, backoff exponencial y pausa entre llamadas."""

    def __init__(self) -> None:
        cfg = CFG.fpl
        self.base_url: str = cfg["base_url"].rstrip("/")
        self.timeout: int = cfg["timeout_s"]
        self.max_retries: int = cfg["max_retries"]
        self.backoff_base: float = cfg["backoff_base_s"]
        self.sleep_between: float = cfg["sleep_between_calls_s"]

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": cfg["user_agent"],
                "Accept": "application/json",
            }
        )

    # ------------------------------------------------------------------ #

    def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> tuple[bytes, Any]:
        """GET con reintentos. Devuelve (bytes crudos, json parseado)."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)

                # 429 y 5xx son transitorios: se reintenta. 4xx (salvo 429) no.
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
                resp.raise_for_status()

                payload = json.loads(resp.content)
                time.sleep(self.sleep_between)
                return resp.content, payload

            except (requests.RequestException, json.JSONDecodeError) as exc:
                last_error = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)

                # Un 404 es una respuesta legítima de la API (p.ej. GW inexistente),
                # no un fallo transitorio: no tiene sentido reintentarlo.
                if status is not None and 400 <= status < 500 and status != 429:
                    raise FPLClientError(f"{url} -> HTTP {status}") from exc

                if attempt < self.max_retries:
                    wait = self.backoff_base**attempt
                    log.warning(
                        "%s falló (intento %d/%d): %s. Reintento en %.1fs",
                        endpoint, attempt, self.max_retries, exc, wait,
                    )
                    time.sleep(wait)

        raise FPLClientError(
            f"{url} falló tras {self.max_retries} intentos: {last_error}"
        ) from last_error

    # ------------------------------------------------------------------ #
    #  Endpoints
    # ------------------------------------------------------------------ #

    def bootstrap_static(self) -> tuple[bytes, dict[str, Any]]:
        """Equipos, jugadores y **deadlines de cada gameweek**.

        El `deadline_time` de cada evento es la pieza central del control
        anti-leakage: define el instante hasta el cual una feature puede mirar.
        """
        return self._get("bootstrap-static/")

    def fixtures(self, event: int | None = None, future: bool = False) -> tuple[bytes, list[Any]]:
        """Fixtures. Sin argumentos trae los 380 de la temporada.

        Cada objeto trae `team_h_score`/`team_a_score` (el target, una vez jugado)
        y `team_h_difficulty`/`team_a_difficulty` (el FDR de FPL, feature gratis).
        """
        params: dict[str, Any] = {}
        if event is not None:
            params["event"] = event
        if future:
            params["future"] = 1
        return self._get("fixtures/", params=params or None)

    def event_live(self, gameweek: int) -> tuple[bytes, dict[str, Any]]:
        """Stats en vivo de una gameweek: el ground truth automático.

        Antes de que se juegue la fecha devuelve la estructura con los elementos
        en cero — no es un error, es el estado esperado en pre-temporada.
        """
        return self._get(f"event/{gameweek}/live/")

    def element_summary(self, player_id: int) -> tuple[bytes, dict[str, Any]]:
        """Historial de un jugador.

        Ojo con el alcance: `history` es SÓLO la temporada actual fecha a fecha, y
        `history_past` trae una fila por temporada (totales), no el detalle por
        fecha. Para el histórico partido a partido no hay sustituto de vaastav.
        """
        return self._get(f"element-summary/{player_id}/")

    # ------------------------------------------------------------------ #

    def current_and_next_gw(self, bootstrap: dict[str, Any]) -> tuple[int | None, int | None]:
        """(gameweek en curso, próxima gameweek) según los flags de la API.

        En pre-temporada `is_current` no está en ningún evento y sólo hay `is_next`.
        """
        events = bootstrap.get("events", [])
        current = next((e["id"] for e in events if e.get("is_current")), None)
        nxt = next((e["id"] for e in events if e.get("is_next")), None)
        return current, nxt

    @staticmethod
    def deadlines(bootstrap: dict[str, Any]) -> dict[int, str]:
        """{gameweek: deadline_time ISO}. La referencia temporal del anti-leakage."""
        return {e["id"]: e["deadline_time"] for e in bootstrap.get("events", [])}
