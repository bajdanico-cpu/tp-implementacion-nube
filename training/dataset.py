"""Carga de Gold y armado del split temporal.

**Split temporal, nunca aleatorio.** Un split aleatorio pone partidos de mayo en el train
y de agosto en el test: el modelo ve el futuro y la métrica miente.

    train : 2022-23, 2023-24, 2024-25   (1.140 partidos)
    holdout : 2025-26                   (  380 partidos)

Dentro del train se reserva `training.valid_season` (2024-25) para el early stopping.
**Nunca el holdout**: usarlo para decidir cuándo parar es la forma sutil de contaminarlo,
porque el número de rondas pasa a estar elegido con información del test.

Las matrices salen como `np.float32` en el orden fijado por `features/spec.py`, y ese
orden se persiste en el `metadata.json`: si el serving arma las columnas en otro orden,
XGBoost no se queja y devuelve basura silenciosamente.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from common.config import CFG
from common.logging_setup import get_logger
from common.storage import read_table
from eda.baselines import CLASES_ORD
from features import spec

log = get_logger(__name__)

TABLA = "gold_tp_match"

# XGBoost exige etiquetas 0..n-1. Se codifica con el MISMO orden que `CLASES_ORD`
# (['away','draw','home'], que es el orden lexicográfico que asume `sklearn.log_loss`).
# Así las columnas de `predict_proba` quedan alineadas con las etiquetas por
# construcción, y no por una convención que alguien tenga que recordar.
CLASES = list(CLASES_ORD)
_A_INDICE = {c: i for i, c in enumerate(CLASES)}


def codificar(y) -> np.ndarray:
    """['home','draw',...] -> [2,1,...] segun CLASES_ORD."""
    s = pd.Series(y)
    desconocidas = set(s.unique()) - set(CLASES)
    if desconocidas:
        raise ValueError(f"Etiquetas fuera de {CLASES}: {sorted(desconocidas)}")
    return s.map(_A_INDICE).to_numpy(dtype=np.int64)


def decodificar(idx) -> np.ndarray:
    """[2,1,...] -> ['home','draw',...]"""
    return np.asarray(CLASES, dtype=object)[np.asarray(idx, dtype=int)]


@dataclass(frozen=True)
class Split:
    """Un corte temporal, con las matrices ya armadas."""

    X_train: np.ndarray
    y_train: np.ndarray                # codificada: 0=away, 1=draw, 2=home
    X_valid: np.ndarray | None
    y_valid: np.ndarray | None
    X_test: np.ndarray
    y_test: np.ndarray                 # codificada
    y_test_txt: np.ndarray             # las etiquetas legibles, para el reporte
    features: list[str]
    filas_test: pd.DataFrame          # claves + cuotas, para la simulación de ROI
    temporadas_train: list[str]
    temporada_valid: str | None
    temporada_test: str


def cargar() -> pd.DataFrame:
    return read_table(TABLA, layer="gold")


# Filtros de las filas que se usan como OBJETIVO de entrenamiento. Lo importante: las
# filas excluidas siguen contando como HISTORIA para las ventanas de los partidos
# posteriores, porque las features de Gold ya vienen calculadas sobre la secuencia
# completa. Filtrar el train no le quita informacion a ninguna otra fila.
FILTROS_TRAIN = {
    "todo": None,
    # El xG de 2022-23 viene hardcodeado en cero hasta la GW15: esas filas ensenian un
    # artefacto del calendario de publicacion de FPL. Sacarlas mejora 5 de 7 modelos.
    "sin_xg_falso": lambda d: d["xg_available"],
    # La temporada entera. Ganaba con el feature set viejo; con Elo la ventaja se diluye.
    "sin_2022_23": lambda d: d["season"] != "2022-23",
}


def filtrar_train(df: pd.DataFrame, variante: str | None = None) -> pd.DataFrame:
    variante = variante or CFG.datos_entrenamiento
    if variante not in FILTROS_TRAIN:
        raise ValueError(f"variante desconocida: {variante!r}. "
                         f"Validas: {list(FILTROS_TRAIN)}")
    f = FILTROS_TRAIN[variante]
    if f is None:
        return df
    out = df[f(df)]
    log.info("Datos de entrenamiento '%s': %d -> %d filas", variante, len(df), len(out))
    return out


def matriz(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """DataFrame -> ndarray float32, en el orden EXACTO del spec.

    Se usa ndarray y no DataFrame a propósito: fija el contrato en el orden de columnas,
    que es lo que después valida el serving contra `feature_names` del metadata.
    """
    faltan = [c for c in features if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan {len(faltan)} features en el dataframe: {faltan[:8]}")
    return df[features].to_numpy(dtype=np.float32)


def preparar(gold: pd.DataFrame | None = None,
             features: list[str] | None = None,
             con_validacion: bool = True,
             datos: str | None = None) -> Split:
    """Arma el split temporal del canvas."""
    gold = cargar() if gold is None else gold
    features = features or spec.FEATURES

    test_season = CFG.holdout_season
    train_seasons = CFG.seasons_for_training()
    valid_season = CFG.valid_season if con_validacion else None

    if valid_season and valid_season not in train_seasons:
        raise ValueError(
            f"La temporada de validación {valid_season} no está en el train "
            f"{train_seasons}. Si estuviera en el holdout, el early stopping lo "
            f"contaminaría.")

    solo_train = [s for s in train_seasons if s != valid_season]
    tr = filtrar_train(gold[gold["season"].isin(solo_train)], datos)
    va = gold[gold["season"] == valid_season] if valid_season else None
    te = gold[gold["season"] == test_season]

    if te.empty:
        raise ValueError(f"El holdout {test_season} no tiene filas en Gold.")

    log.info("Split temporal | train %s: %d | valid %s: %d | holdout %s: %d",
             solo_train, len(tr), valid_season, 0 if va is None else len(va),
             test_season, len(te))

    cols_test = (spec.CLAVES + ["gameweek", "target_1x2", "home_goals", "away_goals"]
                 + spec.MERCADO)
    return Split(
        X_train=matriz(tr, features), y_train=codificar(tr["target_1x2"]),
        X_valid=None if va is None else matriz(va, features),
        y_valid=None if va is None else codificar(va["target_1x2"]),
        X_test=matriz(te, features), y_test=codificar(te["target_1x2"]),
        y_test_txt=te["target_1x2"].to_numpy(),
        features=list(features),
        filas_test=te[[c for c in cols_test if c in te.columns]].reset_index(drop=True),
        temporadas_train=solo_train, temporada_valid=valid_season,
        temporada_test=test_season,
    )


def train_completo(gold: pd.DataFrame, features: list[str],
                   datos: str | None = None,
                   incluir_holdout: bool | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Todas las filas de entrenamiento, para el refit final con las rondas ya fijadas.

    Con `incluir_holdout` se suma la temporada reservada. Es lo que hace el modelo de
    PRODUCCION: el holdout ya cumplio su funcion de elegir, y con 1.004 filas sumar 380
    -- ademas las mas recientes -- es un 38% mas de datos.

    El modelo de EVALUACION no lo incluye, porque sus numeros sobre 2025-26 tienen que
    seguir significando algo.
    """
    temporadas = CFG.seasons_a_entrenar(incluir_holdout)
    d = filtrar_train(gold[gold["season"].isin(temporadas)], datos)
    log.info("Refit final sobre %s: %d filas", temporadas, len(d))
    return matriz(d, features), codificar(d["target_1x2"])
