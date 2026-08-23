"""Resolución de GPU/CPU con verificación real y fallback.

Chequear `xgboost.build_info()["USE_CUDA"]` NO alcanza: el wheel de PyPI para Windows
viene compilado con CUDA siempre, así que ese flag da `True` incluso en una máquina sin
GPU o con el driver roto. La única verificación honesta es **entrenar un árbol de verdad**
y ver si levanta excepción.

Política:

- `auto`  — intenta CUDA, cae a CPU con warning. Es el default.
- `cuda`  — explícito: si no hay GPU, LEVANTA. Explícito es explícito; si alguien pidió
            GPU y silenciosamente entrenó en CPU, el benchmark miente.
- `cpu`   — fuerza CPU.

Precedencia: `--device` > `TP_DEVICE` > `config.yaml → training.device`.

El `DeviceInfo` completo va al `metadata.json` del modelo: que `device_requested` difiera
de `device_used` es un evento que el TP quiere poder auditar después.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from common.config import CFG
from common.logging_setup import get_logger

log = get_logger(__name__)

VALIDOS = ("auto", "cuda", "cpu")


@dataclass(frozen=True)
class DeviceInfo:
    used: str
    requested: str
    reason: str
    gpu_name: str | None = None
    n_jobs: int | None = None
    xgb_build: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _nombre_gpu() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False)
        linea = out.stdout.strip().splitlines()
        return linea[0].strip() if linea else None
    except (OSError, subprocess.SubprocessError):
        return None


def _cpu_jobs() -> int:
    """La mitad de los hilos lógicos.

    En esta máquina son 12 lógicos sobre 6 físicos. Con datasets chicos, lanzar 12 hilos
    de OpenMP hace que XGBoost se pise a sí mismo: el costo de sincronización supera al
    trabajo por hilo.
    """
    return max(1, (os.cpu_count() or 2) // 2)


def _prueba_real(device: str) -> tuple[bool, str]:
    """Entrena un árbol de verdad. Es el único chequeo que no miente."""
    import xgboost as xgb

    X = np.random.default_rng(0).random((32, 4), dtype=np.float32)
    y = np.array([0, 1, 2] * 10 + [0, 1])
    try:
        xgb.XGBClassifier(device=device, tree_method="hist", n_estimators=1,
                          objective="multi:softprob", num_class=3,
                          verbosity=0).fit(X, y)
        return True, "fit de prueba OK"
    except Exception as exc:  # noqa: BLE001 — cualquier fallo de CUDA cuenta igual
        return False, f"{type(exc).__name__}: {str(exc)[:200]}"


def resolve(requested: str | None = None) -> DeviceInfo:
    """Decide el device y lo verifica de verdad."""
    import xgboost as xgb

    requested = (requested or CFG.device or "auto").lower()
    if requested not in VALIDOS:
        raise ValueError(f"device inválido: {requested!r}. Válidos: {VALIDOS}")

    build = {k: v for k, v in xgb.build_info().items()
             if k in ("USE_CUDA", "CUDA_VERSION", "USE_OPENMP")}
    build["xgboost"] = xgb.__version__

    if requested == "cpu":
        return DeviceInfo("cpu", requested, "pedido explícitamente",
                          n_jobs=_cpu_jobs(), xgb_build=build)

    if not build.get("USE_CUDA"):
        motivo = "el build de XGBoost no tiene CUDA"
        if requested == "cuda":
            raise RuntimeError(f"Se pidió device='cuda' pero {motivo}.")
        log.warning("Fallback a CPU: %s", motivo)
        return DeviceInfo("cpu", requested, motivo, n_jobs=_cpu_jobs(), xgb_build=build)

    ok, detalle = _prueba_real("cuda")
    if ok:
        return DeviceInfo("cuda", requested, detalle, gpu_name=_nombre_gpu(),
                          xgb_build=build)

    if requested == "cuda":
        raise RuntimeError(f"Se pidió device='cuda' pero falló el fit de prueba: {detalle}")
    log.warning("Fallback a CPU: %s", detalle)
    return DeviceInfo("cpu", requested, detalle, n_jobs=_cpu_jobs(), xgb_build=build)


def main() -> None:
    """`python -m training.device` — lo que corre un compañero para saber si tiene GPU."""
    from common.logging_setup import setup

    setup(CFG.log_level, CFG.log_format)
    info = resolve()
    print(f"device pedido    : {info.requested}")
    print(f"device usado     : {info.used}")
    print(f"motivo           : {info.reason}")
    print(f"GPU              : {info.gpu_name or '—'}")
    print(f"hilos de CPU     : {info.n_jobs or '—'}")
    print(f"build de xgboost : {info.xgb_build}")


if __name__ == "__main__":
    main()
