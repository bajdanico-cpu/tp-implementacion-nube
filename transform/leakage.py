"""Control de leakage temporal — la regla que sostiene la validez del trabajo.

La regla de oro
---------------
    El snapshot de features usado para predecir la fecha N tiene que ser
    reproducible usando EXCLUSIVAMENTE datos anteriores al deadline de la fecha N.

Por qué importa acá más que en otros dominios
---------------------------------------------
Los puntos y las estadísticas de una fecha se conocen *después* de que se jugó.
Si una feature de la fecha N usa datos de la fecha N, el modelo está mirando el
resultado que tiene que predecir. La accuracy se dispara, todo parece andar
bárbaro, y el trabajo no vale nada. Es un error que no da síntomas: hay que
buscarlo a propósito.

Este módulo no es sólo para los tests. Gold lo importa y lo ejecuta como parte de
la construcción de features, así que el pipeline falla en el momento de generar
el dato malo y no tres capas después.
"""

from __future__ import annotations

import pandas as pd

from common.config import CFG
from common.logging_setup import get_logger

log = get_logger(__name__)


class LeakageError(AssertionError):
    """Se detectó información posterior al deadline dentro de las features."""


def gameweek_deadlines(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Deadline por (season, gameweek), desde `fact_fixture`.

    Todos los fixtures de una gameweek comparten deadline; se toma el mínimo por
    las dudas, que es el criterio conservador.
    """
    return (
        fixtures.groupby(["season", "gameweek"], as_index=False)["deadline_time"]
        .min()
        .rename(columns={"deadline_time": "deadline"})
    )


def find_violations(
    sources: pd.DataFrame,
    fixtures: pd.DataFrame,
    target_season: str,
    target_gameweek: int,
    ts_col: str = "kickoff_time",
) -> pd.DataFrame:
    """Filas de `sources` que NO podrían haberse conocido al predecir esa fecha.

    Devuelve las filas infractoras (vacío si está todo bien), para poder mirarlas
    en vez de sólo saber que algo falló.
    """
    deadlines = gameweek_deadlines(fixtures)
    match = deadlines[
        (deadlines["season"] == target_season)
        & (deadlines["gameweek"] == target_gameweek)
    ]
    if match.empty:
        raise ValueError(
            f"No hay deadline para {target_season} GW{target_gameweek} en fact_fixture."
        )

    deadline = match["deadline"].iloc[0]
    ts = pd.to_datetime(sources[ts_col], utc=True, errors="coerce")

    # Estrictamente anterior: un dato con timestamp EXACTAMENTE igual al deadline
    # ya no estaba disponible para decidir.
    violating = ts >= deadline
    return sources[violating.fillna(False)]


def assert_sources_before_deadline(
    sources: pd.DataFrame,
    fixtures: pd.DataFrame,
    target_season: str,
    target_gameweek: int,
    ts_col: str = "kickoff_time",
    context: str = "",
) -> None:
    """Versión que falla. Es la que llama Gold al construir features."""
    bad = find_violations(sources, fixtures, target_season, target_gameweek, ts_col)
    if not bad.empty:
        deadline = gameweek_deadlines(fixtures).query(
            "season == @target_season and gameweek == @target_gameweek"
        )["deadline"].iloc[0]
        muestra = bad[[c for c in (ts_col, "season", "gameweek") if c in bad.columns]].head(5)
        raise LeakageError(
            f"LEAKAGE{' en ' + context if context else ''}: "
            f"{len(bad)} filas fuente con {ts_col} >= deadline de "
            f"{target_season} GW{target_gameweek} ({deadline}).\n"
            f"Muestra:\n{muestra.to_string(index=False)}"
        )


def assert_no_banned_columns(features: pd.DataFrame, context: str = "") -> None:
    """Verifica que no se colaron columnas prohibidas por config.

    Cubre el caso `xP` y el de columnas derivadas del resultado del propio partido
    (`total_points`, `bps`, `bonus`, los goles). Cualquiera de ellas en una tabla de
    features es leakage directo.
    """
    banned = set(CFG.banned_columns)
    found = sorted(banned & set(features.columns))
    if found:
        raise LeakageError(
            f"Columnas prohibidas{' en ' + context if context else ''}: {found}. "
            f"Están en features.banned_columns de config.yaml porque sólo se "
            f"conocen DESPUÉS de que se jugó el partido."
        )


def audit_all_gameweeks(
    sources: pd.DataFrame,
    fixtures: pd.DataFrame,
    ts_col: str = "kickoff_time",
) -> pd.DataFrame:
    """Barre todas las gameweeks y reporta cuántas filas fuente quedarían del lado
    correcto del deadline en cada una.

    No falla: sirve para ver el panorama y para la defensa del trabajo.
    """
    deadlines = gameweek_deadlines(fixtures)
    ts = pd.to_datetime(sources[ts_col], utc=True, errors="coerce")

    rows = []
    for _, dl in deadlines.iterrows():
        misma_temporada = sources["season"] == dl["season"] if "season" in sources.columns else True
        disponibles = int((ts < dl["deadline"]).sum())
        rows.append({
            "season": dl["season"],
            "gameweek": dl["gameweek"],
            "deadline": dl["deadline"],
            "filas_disponibles": disponibles,
            "filas_posteriores": int(len(sources) - disponibles),
            "_": misma_temporada if isinstance(misma_temporada, bool) else None,
        })
    return pd.DataFrame(rows).drop(columns=["_"])
