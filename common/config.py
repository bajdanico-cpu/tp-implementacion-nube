"""Carga y validación de `config.yaml`, más los helpers de temporada.

Las dos fuentes nombran las temporadas distinto y esa diferencia se resuelve
UNA sola vez, acá:

    vaastav / FPL     -> "2022-23"   (formato canónico del proyecto)
    football-data.co.uk -> "2223"

Uso:
    from common.config import CFG
    CFG.seasons_to_ingest()      -> ["2022-23", ..., "2026-27"]
    CFG.bronze_dir("fpl", "2026-27", "bootstrap")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# La raíz del proyecto es el directorio que contiene este paquete.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def season_start_year(season: str) -> int:
    """"2022-23" -> 2022."""
    return int(season.split("-")[0])


def season_from_start_year(year: int) -> str:
    """2022 -> "2022-23"."""
    return f"{year}-{str(year + 1)[-2:]}"


def season_to_football_data(season: str) -> str:
    """"2022-23" -> "2223", que es como football-data.co.uk arma sus URLs."""
    start = season_start_year(season)
    return f"{str(start)[-2:]}{str(start + 1)[-2:]}"


def utc_stamp() -> str:
    """Timestamp de ingesta, usado para particionar Bronze. Formato apto para nombre de carpeta."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any] = field(repr=False)

    # ---------- temporadas ----------

    @property
    def ingest_from(self) -> str:
        return self.raw["seasons"]["ingest_from"]

    @property
    def current_season(self) -> str:
        return self.raw["seasons"]["current"]

    @property
    def train_from(self) -> str:
        return self.raw["seasons"]["train_from"]

    @property
    def holdout_season(self) -> str:
        return self.raw["seasons"]["holdout"]

    def seasons_to_ingest(self) -> list[str]:
        """Todas las temporadas entre `ingest_from` y `current`, inclusive."""
        start = season_start_year(self.ingest_from)
        end = season_start_year(self.current_season)
        return [season_from_start_year(y) for y in range(start, end + 1)]

    def seasons_for_training(self) -> list[str]:
        """Temporadas de entrenamiento: desde `train_from`, excluyendo holdout y la actual.

        La temporada actual se excluye porque está incompleta, y el holdout porque
        es el test temporal.
        """
        excluded = {self.holdout_season, self.current_season}
        start = season_start_year(self.train_from)
        end = season_start_year(self.current_season)
        return [
            s
            for y in range(start, end + 1)
            if (s := season_from_start_year(y)) not in excluded
        ]

    # ---------- fuentes ----------

    @property
    def fpl(self) -> dict[str, Any]:
        return self.raw["sources"]["fpl_api"]

    @property
    def vaastav(self) -> dict[str, Any]:
        return self.raw["sources"]["vaastav"]

    @property
    def football_data(self) -> dict[str, Any]:
        return self.raw["sources"]["football_data"]

    # ---------- storage ----------

    @property
    def backend(self) -> str:
        return self.raw["storage"]["backend"]

    @property
    def data_root(self) -> Path:
        """Raíz de datos. Absoluta, para que el pipeline corra desde cualquier cwd."""
        root = Path(self.raw["storage"]["root"])
        return root if root.is_absolute() else PROJECT_ROOT / root

    def bronze_dir(self, source: str, season: str, dataset: str, stamp: str | None = None) -> Path:
        """`data/bronze/{source}/{season}/{dataset}/ingested_at=<stamp>/`.

        El timestamp en la ruta es lo que hace Bronze append-only: el snapshot
        pre-deadline y el post-partido conviven en vez de pisarse. Esa es la
        defensa auditable contra el leakage.
        """
        stamp = stamp or utc_stamp()
        base = self.data_root / self.raw["storage"]["bronze_dir"]
        return base / source / season / dataset / f"ingested_at={stamp}"

    def bronze_dataset_root(self, source: str, season: str, dataset: str) -> Path:
        """Carpeta que agrupa todos los snapshots de un dataset (sin el timestamp)."""
        base = self.data_root / self.raw["storage"]["bronze_dir"]
        return base / source / season / dataset

    @property
    def silver_root(self) -> Path:
        return self.data_root / self.raw["storage"]["silver_dir"]

    @property
    def gold_root(self) -> Path:
        return self.data_root / self.raw["storage"]["gold_dir"]

    # ---------- features ----------

    @property
    def rolling_windows(self) -> list[int]:
        return self.raw["features"]["rolling_windows"]

    @property
    def banned_columns(self) -> list[str]:
        """Columnas prohibidas en Gold por riesgo de leakage. Los tests las verifican."""
        return self.raw["features"]["banned_columns"]

    @property
    def xg_min_gameweek(self) -> dict[str, int]:
        """Primera gameweek con xG real, por temporada.

        En 2022-23 el xG viene hardcodeado en 0,0 para los 20 equipos hasta la GW15.
        Dejarlo como cero le enseña al modelo un artefacto de calendario; se enmascara
        a NaN y se marca con la flag `xg_available`.
        """
        return self.raw["features"].get("xg_min_gameweek", {})

    @property
    def podado_top_n(self) -> int:
        return int(self.raw["features"].get("podado_top_n", 40))

    # ---------- training ----------

    @property
    def training(self) -> dict[str, Any]:
        return self.raw.get("training", {})

    @property
    def device(self) -> str:
        """`auto` | `cuda` | `cpu`. TP_DEVICE tiene precedencia sobre config.yaml."""
        return os.getenv("TP_DEVICE", self.training.get("device", "auto"))

    @property
    def modelo(self) -> str:
        """El modelo de produccion, elegido por medicion. Ver config.yaml."""
        return self.training.get("modelo", "xgb_gbt")

    @property
    def datos_entrenamiento(self) -> str:
        """Que filas se usan como objetivo de entrenamiento (las demas siguen de historia)."""
        return self.training.get("datos_entrenamiento", "todo")

    @property
    def incluir_holdout(self) -> bool:
        """Si el modelo de produccion entrena tambien con el holdout. Ver config.yaml."""
        return bool(self.training.get("incluir_holdout", False))

    def seasons_a_entrenar(self, incluir_holdout: bool | None = None) -> list[str]:
        """Temporadas que entran como objetivo de entrenamiento.

        Con `incluir_holdout` se suma la temporada reservada. NUNCA se incluye la
        temporada en curso: sus partidos entran a Gold para servir de historia, pero
        usarlos como objetivo destruiria el unico test honesto que queda.
        """
        s = list(self.seasons_for_training())
        usar = self.incluir_holdout if incluir_holdout is None else incluir_holdout
        if usar and self.holdout_season not in s:
            s.append(self.holdout_season)
        return s

    @property
    def valid_season(self) -> str:
        """Temporada de early stopping. Sale del train, nunca es el holdout."""
        return self.training.get("valid_season", "2024-25")

    @property
    def decision(self) -> dict[str, Any]:
        """La regla que va de las tres probabilidades a una clase, y sus candidatas.

        Vive en config y no en el codigo porque es una decision de NEGOCIO (cuanto cuesta
        perderse un empate contra cuanto cuesta anunciar uno que no fue), no de modelado:
        los boosters no cambian. Ver `serving/decision.py`.
        """
        return self.training.get("decision", {}) or {}

    @property
    def seed(self) -> int:
        return int(self.training.get("seed", 42))

    @property
    def n_seeds(self) -> int:
        return int(self.training.get("n_seeds", 5))

    def train_only_seasons(self) -> list[str]:
        """Temporadas de entrenamiento SIN la de validación de early stopping."""
        return [s for s in self.seasons_for_training() if s != self.valid_season]

    # ---------- logging ----------

    @property
    def log_level(self) -> str:
        return os.getenv("TP_LOG_LEVEL", self.raw["logging"]["level"])

    @property
    def log_format(self) -> str:
        return self.raw["logging"]["format"]


def load(path: Path | None = None) -> Config:
    """Lee config.yaml y valida lo mínimo indispensable."""
    path = path or CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"No se encontró config.yaml en {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    for key in ("project", "seasons", "sources", "storage", "features", "logging"):
        if key not in raw:
            raise ValueError(f"config.yaml: falta la sección obligatoria '{key}'")

    cfg = Config(raw=raw)

    if season_start_year(cfg.ingest_from) > season_start_year(cfg.current_season):
        raise ValueError(
            f"seasons.ingest_from ({cfg.ingest_from}) es posterior a "
            f"seasons.current ({cfg.current_season})"
        )
    if season_start_year(cfg.train_from) < season_start_year(cfg.ingest_from):
        raise ValueError(
            f"seasons.train_from ({cfg.train_from}) es anterior a lo que se ingesta "
            f"({cfg.ingest_from}): habría que re-ingestar para entrenar con esa ventana"
        )

    return cfg


# Instancia única, importable desde cualquier módulo.
CFG = load()


if __name__ == "__main__":
    from common.logging_setup import setup

    setup(CFG.log_level, CFG.log_format)

    print(f"Proyecto            : {CFG.raw['project']['name']}  (target={CFG.raw['project']['target']})")
    print(f"Raíz del proyecto   : {PROJECT_ROOT}")
    print(f"Raíz de datos       : {CFG.data_root}")
    print(f"Backend de storage  : {CFG.backend}")
    print()
    print(f"Temporadas a ingestar : {CFG.seasons_to_ingest()}")
    print(f"Temporadas de train   : {CFG.seasons_for_training()}")
    print(f"Holdout temporal      : {CFG.holdout_season}")
    print(f"Temporada actual      : {CFG.current_season}")
    print()
    print("Mapeo de nombres de temporada (canónico -> football-data):")
    for s in CFG.seasons_to_ingest():
        print(f"  {s}  ->  {season_to_football_data(s)}")
    print()
    print(f"Ventanas rolling    : {CFG.rolling_windows}")
    print(f"Columnas prohibidas : {CFG.banned_columns}")
