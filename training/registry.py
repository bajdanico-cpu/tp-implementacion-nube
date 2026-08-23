"""Persistencia versionada de modelos y la regla de promoción del bloque 9.

El canvas dice: *"Se compara en la siguiente fecha, si le gana al de producción se pasa a
producción."* Este módulo implementa la parte de guardar y promover; el criterio
estadístico vive en `training/promotion.py`.

Dos decisiones que importan:

**`.ubj` en vez de pickle.** El formato nativo de XGBoost sobrevive a upgrades de la
librería, es legible desde otros lenguajes y —clave para este TP— no arrastra el estado de
device: un modelo entrenado en GPU se carga y predice en CPU sin tocar nada, que es
exactamente el escenario de servirlo en Cloud Run sin GPU.

**`attempts.jsonl` guarda los intentos RECHAZADOS.** No es un detalle administrativo: un
pipeline que sólo registra lo que promovió no puede demostrar que sabe decir que no. La
mitad del valor del bloque 9 está en poder mostrar los candidatos que no pasaron.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.config import CFG, PROJECT_ROOT, utc_stamp
from common.logging_setup import get_logger

log = get_logger(__name__)

RAIZ = PROJECT_ROOT / "models"
PRODUCCION = "PRODUCTION.json"
INTENTOS = "attempts.jsonl"


@dataclass(frozen=True)
class Version:
    nombre: str
    version: str
    ruta: Path

    @property
    def modelo(self) -> Path:
        return self.ruta / "model.ubj"

    @property
    def metadata(self) -> Path:
        return self.ruta / "metadata.json"


def _git() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            r = subprocess.run(["git", *args], capture_output=True, text=True,
                               cwd=PROJECT_ROOT, timeout=10, check=False)
            return r.stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None

    return {"git_sha": run("rev-parse", "HEAD"),
            "git_dirty": bool(run("status", "--porcelain"))}


def versiones_librerias() -> dict[str, str]:
    import sys

    import numpy
    import pandas
    import sklearn
    import xgboost

    return {"python": sys.version.split()[0], "xgboost": xgboost.__version__,
            "scikit-learn": sklearn.__version__, "pandas": pandas.__version__,
            "numpy": numpy.__version__}


def guardar(nombre: str, modelos: list, metadata: dict[str, Any],
            metricas: dict[str, Any]) -> Version:
    """Guarda una corrida completa: los boosters de cada semilla, metadata y métricas."""
    stamp = utc_stamp()
    ruta = RAIZ / nombre / stamp
    ruta.mkdir(parents=True, exist_ok=True)

    for i, m in enumerate(modelos):
        destino = ruta / (f"model.ubj" if len(modelos) == 1 else f"model_seed{i}.ubj")
        booster = m.get_booster() if hasattr(m, "get_booster") else None
        if booster is not None:
            booster.save_model(str(destino))
        else:
            import joblib
            joblib.dump(m, ruta / f"model_seed{i}.joblib")

    meta = {**metadata, **_git(), "lib_versions": versiones_librerias(),
            "model_name": nombre, "model_version": stamp,
            "built_at": datetime.now(timezone.utc).isoformat()}
    (ruta / "metadata.json").write_text(json.dumps(meta, indent=2, default=str),
                                        encoding="utf-8")
    (ruta / "metrics.json").write_text(json.dumps(metricas, indent=2, default=str),
                                       encoding="utf-8")
    log.info("Modelo guardado: %s", ruta)
    return Version(nombre, stamp, ruta)


def produccion(nombre: str) -> Version | None:
    """La versión que está en producción, si hay alguna."""
    p = RAIZ / nombre / PRODUCCION
    if not p.exists():
        return None
    ver = json.loads(p.read_text(encoding="utf-8"))["version"]
    return Version(nombre, ver, RAIZ / nombre / ver)


def promover(v: Version, motivo: str) -> None:
    (RAIZ / v.nombre / PRODUCCION).write_text(
        json.dumps({"version": v.version, "motivo": motivo,
                    "promovido_at": datetime.now(timezone.utc).isoformat()}, indent=2),
        encoding="utf-8")
    log.info("PROMOVIDO %s -> %s (%s)", v.nombre, v.version, motivo)


def registrar_rechazo(nombre: str, version: str, motivo: str,
                      detalle: dict[str, Any] | None = None) -> None:
    """Un candidato que no pasó. Se guarda igual: es evidencia de que el control funciona."""
    (RAIZ / nombre).mkdir(parents=True, exist_ok=True)
    linea = {"version": version, "resultado": "rechazado", "motivo": motivo,
             "detalle": detalle or {},
             "at": datetime.now(timezone.utc).isoformat()}
    with (RAIZ / nombre / INTENTOS).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(linea, default=str) + "\n")
    log.info("RECHAZADO %s %s: %s", nombre, version, motivo)


def cargar_metadata(v: Version) -> dict[str, Any]:
    return json.loads(v.metadata.read_text(encoding="utf-8"))


def cargar_booster(ruta: Path, device: str = "cpu"):
    """Carga un `.ubj` y lo deja listo para predecir en el device pedido.

    Es el camino que valida el test `test_modelo_entrenado_en_gpu_predice_igual_en_cpu`:
    entrenar con GPU y servir sin ella es el escenario real del bloque 7.
    """
    import xgboost as xgb

    b = xgb.Booster()
    b.load_model(str(ruta))
    b.set_param({"device": device})
    return b
