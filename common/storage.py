"""Capa de storage intercambiable.

TODO el I/O del pipeline pasa por acá. Hoy el backend es `local` (parquet en disco);
cuando haya proyecto GCP se implementa `GCSBackend` / `BigQueryBackend` y el resto
del código no se toca — ni la ingesta, ni las transformaciones, ni los tests.

Ese es el punto: la lógica de negocio nunca sabe dónde viven los bytes.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

from common.config import CFG, utc_stamp
from common.logging_setup import get_logger

log = get_logger(__name__)


def sha256(data: bytes) -> str:
    """Hash del contenido crudo. Va al manifest para detectar si una fuente cambió."""
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- #
#  Interfaz
# --------------------------------------------------------------------------- #


class StorageBackend(ABC):
    """Contrato que tiene que cumplir cualquier backend (local, GCS, BigQuery)."""

    @abstractmethod
    def write_bytes(self, path: Path, data: bytes) -> None: ...

    @abstractmethod
    def read_bytes(self, path: Path) -> bytes: ...

    @abstractmethod
    def exists(self, path: Path) -> bool: ...

    @abstractmethod
    def write_dataframe(self, df: pd.DataFrame, path: Path) -> None: ...

    @abstractmethod
    def read_dataframe(self, path: Path) -> pd.DataFrame: ...

    @abstractmethod
    def list_dirs(self, path: Path) -> list[Path]: ...


class LocalBackend(StorageBackend):
    """Disco local. El modo de trabajo por defecto: corre en la PC, bajo demanda."""

    def write_bytes(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def read_bytes(self, path: Path) -> bytes:
        return path.read_bytes()

    def exists(self, path: Path) -> bool:
        return path.exists()

    def write_dataframe(self, df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")

    def read_dataframe(self, path: Path) -> pd.DataFrame:
        return pd.read_parquet(path, engine="pyarrow")

    def list_dirs(self, path: Path) -> list[Path]:
        if not path.exists():
            return []
        return sorted(p for p in path.iterdir() if p.is_dir())


class GCSBackend(StorageBackend):
    """Placeholder para cuando exista el proyecto GCP.

    Implementar acá y cambiar `storage.backend` en config.yaml a "gcs" es todo lo
    que hace falta para que el pipeline pase a la nube. Las rutas ya están armadas
    como claves de objeto (`bronze/fpl/2026-27/...`), así que se mapean directo.
    """

    _MSG = (
        "Backend GCS todavía no implementado. Para activarlo: descomentar "
        "google-cloud-storage en requirements.txt, completar storage.gcp en "
        "config.yaml e implementar estos métodos con google.cloud.storage.Client."
    )

    def write_bytes(self, path: Path, data: bytes) -> None:
        raise NotImplementedError(self._MSG)

    def read_bytes(self, path: Path) -> bytes:
        raise NotImplementedError(self._MSG)

    def exists(self, path: Path) -> bool:
        raise NotImplementedError(self._MSG)

    def write_dataframe(self, df: pd.DataFrame, path: Path) -> None:
        raise NotImplementedError(self._MSG)

    def read_dataframe(self, path: Path) -> pd.DataFrame:
        raise NotImplementedError(self._MSG)

    def list_dirs(self, path: Path) -> list[Path]:
        raise NotImplementedError(self._MSG)


_BACKENDS: dict[str, type[StorageBackend]] = {
    "local": LocalBackend,
    "gcs": GCSBackend,
}


def _make_backend() -> StorageBackend:
    name = CFG.backend
    if name not in _BACKENDS:
        raise ValueError(
            f"storage.backend='{name}' no reconocido. Opciones: {sorted(_BACKENDS)}"
        )
    return _BACKENDS[name]()


BACKEND: StorageBackend = _make_backend()


# --------------------------------------------------------------------------- #
#  API que usa el resto del pipeline
# --------------------------------------------------------------------------- #


def write_raw(
    source: str,
    season: str,
    dataset: str,
    filename: str,
    data: bytes,
    stamp: str | None = None,
) -> Path:
    """Escribe un artefacto crudo en Bronze, particionado por timestamp de ingesta.

    Nunca sobrescribe: cada corrida crea su propia carpeta `ingested_at=...`.
    Conservar el snapshot pre-deadline junto al post-partido es lo que permite
    demostrar después que las features no vieron el futuro.
    """
    stamp = stamp or utc_stamp()
    target = CFG.bronze_dir(source, season, dataset, stamp) / filename
    BACKEND.write_bytes(target, data)
    log.debug("bronze <- %s (%.1f KB)", target, len(data) / 1024)
    return target


def latest_snapshot(source: str, season: str, dataset: str) -> Path | None:
    """Carpeta `ingested_at=...` más reciente de un dataset, o None si nunca se bajó.

    Se usa para la caché de la ingesta (no re-descargar) y para que Silver lea
    siempre el snapshot vigente.
    """
    root = CFG.bronze_dataset_root(source, season, dataset)
    snapshots = BACKEND.list_dirs(root)
    return snapshots[-1] if snapshots else None


def snapshot_stamp(snapshot: Path) -> str:
    """El `stamp` de una carpeta `ingested_at=<stamp>`."""
    return snapshot.name.removeprefix("ingested_at=")


def snapshot_at_or_before(source: str, season: str, dataset: str,
                          stamp: str) -> Path | None:
    """El snapshot más reciente que NO sea posterior a `stamp`.

    Existe porque Bronze es append-only y fechado, y eso sólo sirve si algo lo
    aprovecha. El caso concreto: el `bootstrap` dice a qué club pertenece cada
    jugador **hoy**. Para atribuir las estadísticas de la fecha 1 hay que leer el
    bootstrap de *ese momento*, no el de ahora: si un jugador se transfirió después,
    el último bootstrap le adjudicaría sus goles al club nuevo.

    Los stamps son UTC en formato `YYYYMMDDTHHMMSSZ`, así que el orden lexicográfico
    es el cronológico.
    """
    root = CFG.bronze_dataset_root(source, season, dataset)
    previos = [d for d in BACKEND.list_dirs(root) if snapshot_stamp(d) <= stamp]
    return previos[-1] if previos else None


def read_raw_at(source: str, season: str, dataset: str, filename: str,
                stamp: str) -> bytes | None:
    """Como `read_raw`, pero del snapshot vigente al momento `stamp`."""
    snap = snapshot_at_or_before(source, season, dataset, stamp)
    if snap is None:
        return None
    path = snap / filename
    return BACKEND.read_bytes(path) if BACKEND.exists(path) else None


def read_raw(source: str, season: str, dataset: str, filename: str) -> bytes | None:
    """Lee un artefacto del snapshot más reciente. None si no existe."""
    snap = latest_snapshot(source, season, dataset)
    if snap is None:
        return None
    path = snap / filename
    return BACKEND.read_bytes(path) if BACKEND.exists(path) else None


def write_manifest(
    source: str, season: str, dataset: str, entries: list[dict[str, Any]], stamp: str
) -> Path:
    """Deja constancia de qué se bajó, de dónde y con qué resultado.

    Es la trazabilidad de la ingesta: sin esto, un dataset corrupto en Bronze no
    tiene forma de rastrearse hasta la corrida que lo produjo.
    """
    manifest = {
        "source": source,
        "season": season,
        "dataset": dataset,
        "ingested_at": stamp,
        "entries": entries,
    }
    payload = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8")
    return write_raw(source, season, dataset, "_manifest.json", payload, stamp=stamp)


def read_manifest(source: str, season: str, dataset: str) -> dict[str, Any] | None:
    """Manifest del snapshot más reciente, o None."""
    raw = read_raw(source, season, dataset, "_manifest.json")
    return json.loads(raw) if raw else None


# --------------------------------------------------------------------------- #
#  Versionado de Silver y Gold: nunca se pisa nada
# --------------------------------------------------------------------------- #
#
# Bronze es append-only desde el primer día (`ingested_at=<stamp>`) y los modelos están
# versionados por timestamp. Silver y Gold **no lo estaban**: `write_table` escribía
# `data/gold/gold_tp_match.parquet` encima del anterior. Como `data/` está en .gitignore,
# eso significaba que una corrida de `python -m features.gold_tp` destruía sin rastro el
# Gold con el que se entrenó el modelo que está en producción.
#
# Ahora, antes de escribir, la versión vigente se aparta:
#
#     data/gold/gold_tp_match.parquet                          <- la vigente
#     data/_versiones/gold/gold_tp_match/<stamp>.parquet       <- las anteriores
#     data/_versiones/gold/gold_tp_match/<stamp>.json          <- que era cada una
#
# El histórico vive FUERA de `data/silver` y `data/gold` a propósito: así las carpetas de
# capa siguen conteniendo exactamente una versión de cada tabla, y el lab de GCP sube lo
# vigente sin arrastrar el archivo histórico.
#
# El `stamp` sale del mtime del archivo que se aparta, no de "ahora": es cuándo se creó esa
# versión, que es el dato que sirve para cruzarla con el `built_at` de un modelo.

VERSIONES = "_versiones"
ETIQUETA_ENV = "TP_VERSION_LABEL"


def versiones_root(layer: str, name: str | None = None) -> Path:
    """Dónde vive el histórico de una capa (y opcionalmente de una tabla)."""
    root = CFG.data_root / VERSIONES / layer
    return root / name if name else root


def _forma_parquet(data: bytes) -> tuple[int | None, int | None]:
    """Filas y columnas de un parquet, leídas del footer. No materializa la tabla."""
    try:
        import io as _io

        import pyarrow.parquet as pq

        md = pq.ParquetFile(_io.BytesIO(data)).metadata
        return md.num_rows, md.num_columns
    except Exception:  # noqa: BLE001 — no es parquet, o no se puede leer: no es un error
        return None, None


def versiones(layer: str, name: str) -> list[dict[str, Any]]:
    """Los manifiestos de las versiones archivadas de una tabla, de vieja a nueva."""
    carpeta = versiones_root(layer, name)
    if not carpeta.exists():
        return []
    out = []
    for m in sorted(carpeta.glob("*.json")):
        try:
            out.append(json.loads(m.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            log.warning("manifiesto ilegible: %s", m)
    return out


def archivar(path: Path, layer: str, etiqueta: str | None = None) -> Path | None:
    """Aparta la versión vigente de `path` antes de que algo la pise.

    Devuelve la ruta donde quedó archivada, o `None` si no había nada que archivar o si
    ese mismo contenido ya está guardado.

    **Deduplica por contenido.** Si el archivo vigente es idéntico (mismo sha256) a alguna
    versión ya archivada, no se guarda de nuevo: reconstruir Gold sin cambiar nada no
    ensucia el histórico. Como mucho se archiva una versión redundante por corrida, que es
    el precio de no perder nada — y es el lado correcto del que equivocarse.
    """
    if not BACKEND.exists(path):
        return None

    data = BACKEND.read_bytes(path)
    h = sha256(data)
    nombre = path.stem

    if any(v.get("sha256") == h for v in versiones(layer, nombre)):
        log.debug("%s.%s ya estaba archivado con este contenido", layer, nombre)
        return None

    # El stamp es cuándo se creó ESA version, no ahora.
    from datetime import datetime, timezone

    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        mtime = datetime.now(timezone.utc)
    stamp = mtime.strftime("%Y%m%dT%H%M%SZ")

    destino = versiones_root(layer, nombre) / f"{stamp}{path.suffix}"
    if BACKEND.exists(destino):                       # dos versiones en el mismo segundo
        stamp = f"{stamp}_{h[:8]}"
        destino = versiones_root(layer, nombre) / f"{stamp}{path.suffix}"
    BACKEND.write_bytes(destino, data)

    import os

    filas, columnas = _forma_parquet(data)
    manifiesto = {
        "tabla": nombre, "layer": layer, "stamp": stamp,
        "archivo": destino.name,
        "creada_at": mtime.isoformat(timespec="seconds"),
        "archivada_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bytes": len(data), "sha256": h,
        "filas": filas, "columnas": columnas,
        "etiqueta": etiqueta or os.getenv(ETIQUETA_ENV) or "",
    }
    BACKEND.write_bytes(destino.with_suffix(".json"),
                        json.dumps(manifiesto, indent=2, ensure_ascii=False).encode("utf-8"))
    log.info("%s.%s: version anterior archivada en %s (%s filas)",
             layer, nombre, destino.name, f"{filas:,}" if filas else "?")
    return destino


def write_table(df: pd.DataFrame, name: str, layer: str = "silver") -> Path:
    """Persiste una tabla de Silver o Gold, **apartando antes la versión vigente**."""
    root = {"silver": CFG.silver_root, "gold": CFG.gold_root}.get(layer)
    if root is None:
        raise ValueError(f"layer='{layer}' inválido; usar 'silver' o 'gold'")
    path = root / f"{name}.parquet"
    archivar(path, layer)
    BACKEND.write_dataframe(df, path)
    log.info("%s.%s <- %s filas x %s cols", layer, name, f"{len(df):,}", len(df.columns))
    return path


def read_table(name: str, layer: str = "silver") -> pd.DataFrame:
    """Lee una tabla de Silver o Gold."""
    root = {"silver": CFG.silver_root, "gold": CFG.gold_root}.get(layer)
    if root is None:
        raise ValueError(f"layer='{layer}' inválido; usar 'silver' o 'gold'")
    path = root / f"{name}.parquet"
    if not BACKEND.exists(path):
        raise FileNotFoundError(
            f"No existe {layer}.{name} en {path}. "
            f"¿Corriste `python -m transform.silver`?"
        )
    return BACKEND.read_dataframe(path)


def table_exists(name: str, layer: str = "silver") -> bool:
    root = {"silver": CFG.silver_root, "gold": CFG.gold_root}[layer]
    return BACKEND.exists(root / f"{name}.parquet")
