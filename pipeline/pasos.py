"""Un runner de pasos, chico y sin dependencias.

No es un orquestador: es lo mínimo que hace falta para que una cadena de comandos deje de
vivir en un README y pase a ser código que se puede correr, retomar y auditar. La
diferencia práctica es que un paso puede declararse **opcional** —football-data se cae
cada tanto y sus cuotas no son features, así que no tiene por qué tirar la corrida
entera— y que cada corrida deja su registro.

`correr()` no llama a `sys.exit` nunca: devuelve un `Resultado`. Sólo `main()` traduce eso
a un código de salida. Es lo que permite que el mismo módulo sea un Cloud Run Job sin
tocar una línea.
"""

from __future__ import annotations

import json
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.config import CFG, utc_stamp
from common.logging_setup import get_logger

log = get_logger(__name__)

RUNS = CFG.data_root / "pipeline" / "runs"


@dataclass(frozen=True)
class Paso:
    nombre: str
    correr: Callable[[], Any]
    obligatorio: bool = True
    descripcion: str = ""


@dataclass
class Resultado:
    corrida: str
    pasos: list[dict] = field(default_factory=list)
    ok: bool = True

    @property
    def fallados(self) -> list[str]:
        return [p["paso"] for p in self.pasos if p["estado"] == "error"]

    def resumen(self) -> str:
        lineas = [f"corrida {self.corrida}"]
        for p in self.pasos:
            marca = {"ok": "ok  ", "error": "ERROR", "salteado": "--  "}[p["estado"]]
            detalle = f"  {p['error']}" if p["estado"] == "error" else ""
            lineas.append(f"  {marca} {p['paso']:16s} {p['segundos']:6.1f}s{detalle}")
        return "\n".join(lineas)


def _serializable(v: Any) -> Any:
    """Las métricas que devuelve cada paso, en algo que entre en un JSON."""
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, dict):
        return {str(k): _serializable(x) for k, x in list(v.items())[:20]}
    if hasattr(v, "shape"):
        return {"filas": int(v.shape[0]), "columnas": int(v.shape[1])} \
            if len(v.shape) == 2 else {"filas": int(v.shape[0])}
    if isinstance(v, Path):
        return str(v)
    return str(v)[:200]


def correr(pasos: Sequence[Paso], desde: str | None = None,
           solo: Sequence[str] = (), registrar: bool = True) -> Resultado:
    """Corre la cadena. Un paso obligatorio que falla la corta; uno opcional, no.

    `desde` retoma en ese paso y `solo` corre nada más que los nombrados. Los dos existen
    para el caso real: la ingesta salió bien, Gold falló por un dato incompleto, se
    arregla y no tiene sentido volver a bajar 27 MB.
    """
    nombres = [p.nombre for p in pasos]
    for n in (*( [desde] if desde else [] ), *solo):
        if n not in nombres:
            raise ValueError(f"paso desconocido: {n!r}. Hay: {nombres}")

    arrancar = nombres.index(desde) if desde else 0
    res = Resultado(corrida=utc_stamp())

    for i, paso in enumerate(pasos):
        saltear = i < arrancar or (solo and paso.nombre not in solo)
        if saltear:
            res.pasos.append({"paso": paso.nombre, "estado": "salteado", "segundos": 0.0})
            continue

        log.info("--- %s ---", paso.nombre)
        t0 = time.perf_counter()
        try:
            salida = paso.correr()
            res.pasos.append({"paso": paso.nombre, "estado": "ok",
                              "segundos": round(time.perf_counter() - t0, 1),
                              "salida": _serializable(salida)})
        except Exception as exc:  # noqa: BLE001 — se registra y se decide qué hacer
            res.pasos.append({"paso": paso.nombre, "estado": "error",
                              "segundos": round(time.perf_counter() - t0, 1),
                              "error": f"{type(exc).__name__}: {exc}",
                              "traceback": traceback.format_exc()[-2000:]})
            if paso.obligatorio:
                res.ok = False
                log.error("El paso obligatorio '%s' falló: %s", paso.nombre, exc)
                break
            log.warning("El paso opcional '%s' falló y la cadena sigue: %s",
                        paso.nombre, exc)

    if registrar:
        _registrar(res)
    return res


def _registrar(res: Resultado) -> Path:
    """Cada corrida deja su archivo. Es la evidencia de operación, no un log más."""
    RUNS.mkdir(parents=True, exist_ok=True)
    ruta = RUNS / f"{res.corrida}.json"
    ruta.write_text(json.dumps({
        "corrida": res.corrida,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ok": res.ok,
        "pasos": res.pasos,
    }, indent=2, default=str), encoding="utf-8")
    log.info("Corrida registrada en %s", ruta)
    return ruta
