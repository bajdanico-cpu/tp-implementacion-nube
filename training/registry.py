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

from common.config import CFG, utc_stamp
from common.logging_setup import get_logger
from common.storage import backend

log = get_logger(__name__)

# Todo el I/O de modelos pasa por el backend de storage, igual que Silver y Gold. Antes
# usaba `pathlib` directo, y por eso `models/` era lo unico que no podia salir de la
# imagen: el servicio sabia leer un bucket para el dato y no para el modelo.
RAIZ = CFG.models_root
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

    for i, m in enumerate(modelos):
        destino = ruta / ("model.ubj" if len(modelos) == 1 else f"model_seed{i}.ubj")
        booster = m.get_booster() if hasattr(m, "get_booster") else None
        if booster is not None:
            # `save_raw` en vez de `save_model(str(ruta))`: devuelve los bytes y evita
            # que XGBoost escriba en el filesystem por su cuenta, que es lo que impedia
            # guardar contra un bucket.
            backend().write_bytes(destino, bytes(booster.save_raw(raw_format="ubj")))
        else:
            import io

            import joblib
            buf = io.BytesIO()
            joblib.dump(m, buf)
            backend().write_bytes(ruta / f"model_seed{i}.joblib", buf.getvalue())

    meta = {**metadata, **_git(), "lib_versions": versiones_librerias(),
            "model_name": nombre, "model_version": stamp,
            "built_at": datetime.now(timezone.utc).isoformat()}
    _escribir_json(ruta / "metadata.json", meta)
    _escribir_json(ruta / "metrics.json", metricas)
    log.info("Modelo guardado: %s", ruta)
    return Version(nombre, stamp, ruta)


def _escribir_json(ruta: Path, contenido: Any) -> None:
    backend().write_bytes(ruta, json.dumps(contenido, indent=2,
                                           default=str).encode("utf-8"))


def _leer_json(ruta: Path) -> Any:
    return json.loads(backend().read_bytes(ruta).decode("utf-8"))


def produccion(nombre: str) -> Version | None:
    """La versión que está en producción, si hay alguna."""
    p = RAIZ / nombre / PRODUCCION
    if not backend().exists(p):
        return None
    ver = _leer_json(p)["version"]
    return Version(nombre, ver, RAIZ / nombre / ver)


def promover(v: Version, motivo: str) -> None:
    _escribir_json(RAIZ / v.nombre / PRODUCCION,
                   {"version": v.version, "motivo": motivo,
                    "promovido_at": datetime.now(timezone.utc).isoformat()})
    log.info("PROMOVIDO %s -> %s (%s)", v.nombre, v.version, motivo)


def registrar_rechazo(nombre: str, version: str, motivo: str,
                      detalle: dict[str, Any] | None = None) -> None:
    """Un candidato que no pasó. Se guarda igual: es evidencia de que el control funciona."""
    linea = {"version": version, "resultado": "rechazado", "motivo": motivo,
             "detalle": detalle or {},
             "at": datetime.now(timezone.utc).isoformat()}
    # Append leyendo y reescribiendo: en un bucket no existe abrir en modo "a". El
    # archivo es una linea por intento rechazado, asi que el costo es irrelevante.
    ruta = RAIZ / nombre / INTENTOS
    previo = backend().read_bytes(ruta) if backend().exists(ruta) else b""
    backend().write_bytes(ruta, previo + (json.dumps(linea, default=str) + "\n").encode("utf-8"))
    log.info("RECHAZADO %s %s: %s", nombre, version, motivo)


def cargar_metadata(v: Version) -> dict[str, Any]:
    return _leer_json(v.metadata)


def cargar_booster(ruta: Path, device: str = "cpu"):
    """Carga un `.ubj` y lo deja listo para predecir en el device pedido.

    Es el camino que valida el test `test_modelo_entrenado_en_gpu_predice_igual_en_cpu`:
    entrenar con GPU y servir sin ella es el escenario real del bloque 7.
    """
    import xgboost as xgb

    b = xgb.Booster()
    # `bytearray` y no la ruta: XGBoost acepta el modelo en memoria, asi que el `.ubj`
    # puede venir de un bucket sin pasar por un archivo temporal.
    b.load_model(bytearray(backend().read_bytes(ruta)))
    b.set_param({"device": device})
    return b


# ---------------------------------------------------------------------------
# Inventario y CLI de promoción
# ---------------------------------------------------------------------------

def versiones_disponibles(nombre: str) -> list[Version]:
    """Las versiones de un modelo en disco, de la más vieja a la más nueva.

    El orden lexicográfico de los stamps `YYYYMMDDTHHMMSSZ` coincide con el cronológico.
    """
    dirs = backend().list_dirs(RAIZ / nombre)
    return [Version(nombre, d.name, d) for d in dirs if d.name.startswith("2")]


def boosters_de(v: Version) -> list[Path]:
    """Los `.ubj` de una versión. Vacío si no tiene ninguno."""
    return backend().list_files(v.ruta, "model*.ubj")


def tiene_boosters(v: Version) -> bool:
    """¿Esta versión se puede SERVIR, o sólo sirve para trazar?

    Los binarios están en `.gitignore` y la trazabilidad no: `metadata.json`,
    `metrics.json` e `importancias.csv` sí van al repo. Entonces una versión puede llegar
    por `git checkout` completa de papeles y sin un solo booster. Distinguir las dos cosas
    es la diferencia entre un modelo servible y una carpeta que promete uno.
    """
    return bool(boosters_de(v))


def servibles(nombre: str) -> list[Version]:
    """Sólo las versiones que tienen los binarios, en el mismo orden."""
    return [v for v in versiones_disponibles(nombre) if tiene_boosters(v)]


def inventario(nombre: str) -> list[dict[str, Any]]:
    """Una fila por versión, con lo que hace falta para elegir cuál promover."""
    actual = produccion(nombre)
    filas = []
    for v in versiones_disponibles(nombre):
        meta = _leer_json(v.metadata) if backend().exists(v.metadata) else {}
        filas.append({
            "version": v.version,
            "produccion": actual is not None and actual.version == v.version,
            "boosters": len(boosters_de(v)),
            "n_features": meta.get("n_features"),
            "feature_set_version": meta.get("feature_set_version"),
            "incluye_holdout": meta.get("incluye_holdout"),
            "built_at": meta.get("built_at"),
        })
    return filas


def main(argv: list[str] | None = None) -> int:
    """Inventario y promoción desde la línea de comandos.

        python -m training.registry --listar
        python -m training.registry --promover 20260825T024144Z --motivo "..."

    Existe porque `promover()` estaba escrito desde el día uno y **no lo llamaba nadie**:
    `training/run.py` sólo guarda. Por eso nunca hubo `PRODUCTION.json`, y sin él
    `serving.predict` elegía "la última carpeta por nombre" — que en septiembre de 2026
    fue una que llegó por git sin binarios, y dejó todo `/predict` en 503.
    """
    import argparse

    from common.logging_setup import setup

    p = argparse.ArgumentParser(description="Inventario y promoción de modelos.")
    p.add_argument("--modelo", default=None, help=f"Por defecto, el de config: {CFG.modelo}")
    p.add_argument("--listar", action="store_true", help="Muestra las versiones en disco.")
    p.add_argument("--promover", metavar="VERSION", help="La versión que pasa a producción.")
    p.add_argument("--motivo", default=None, help="Por qué se promueve. Queda registrado.")
    p.add_argument("--log-level", default=None)
    args = p.parse_args(argv)

    setup(args.log_level)
    nombre = args.modelo or CFG.modelo

    if args.promover:
        v = Version(nombre, args.promover, RAIZ / nombre / args.promover)
        if not backend().exists(v.metadata):
            p.error(f"No existe models/{nombre}/{args.promover}/")
        if not tiene_boosters(v):
            p.error(f"models/{nombre}/{args.promover}/ no tiene archivos .ubj: "
                    f"se puede trazar pero no servir.")
        if not args.motivo:
            p.error("--promover necesita --motivo: la decisión se registra con su razón.")
        promover(v, args.motivo)

    if args.listar or not args.promover:
        filas = inventario(nombre)
        if not filas:
            print(f"No hay ninguna versión de {nombre} en models/.")
            return 1
        print(f"\nmodels/{nombre}/\n")
        print(f"  {'':2s} {'version':18s} {'ubj':>4s} {'feats':>6s} "
              f"{'holdout':>8s}  feature_set")
        for f in filas:
            marca = "->" if f["produccion"] else "  "
            aviso = "" if f["boosters"] else "   <- sin binarios, no se puede servir"
            print(f"  {marca:2s} {f['version']:18s} {f['boosters']:4d} "
                  f"{str(f['n_features'] or '-'):>6s} {str(f['incluye_holdout']):>8s}  "
                  f"{f['feature_set_version'] or '-'}{aviso}")
        actual = produccion(nombre)
        print(f"\n  produccion: {actual.version if actual else 'NINGUNA (sin PRODUCTION.json)'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
