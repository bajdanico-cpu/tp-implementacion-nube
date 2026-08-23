"""Fixtures compartidas de los tests.

Las tablas Silver se cargan una sola vez por sesión. Si no están construidas, los
tests que las necesitan se saltean con un mensaje que dice qué correr, en vez de
fallar con un FileNotFoundError críptico.
"""

from __future__ import annotations

import pandas as pd
import pytest

from common.storage import read_table, table_exists

SILVER_TABLES = ("dim_team", "fact_match", "fact_fixture", "fact_player_gw")


def _load(name: str) -> pd.DataFrame:
    if not table_exists(name):
        pytest.skip(
            f"Falta la tabla silver.{name}. "
            f"Corré `python -m ingestion.run` y después `python -m transform.silver`."
        )
    return read_table(name)


@pytest.fixture(scope="session")
def dim_team() -> pd.DataFrame:
    return _load("dim_team")


@pytest.fixture(scope="session")
def fact_match() -> pd.DataFrame:
    return _load("fact_match")


@pytest.fixture(scope="session")
def fact_fixture() -> pd.DataFrame:
    return _load("fact_fixture")


@pytest.fixture(scope="session")
def fact_player_gw() -> pd.DataFrame:
    return _load("fact_player_gw")


@pytest.fixture(scope="session")
def closed_seasons(fact_match: pd.DataFrame) -> list[str]:
    """Temporadas con los 380 partidos jugados. Sobre éstas se exige todo."""
    counts = fact_match.groupby("season").size()
    return sorted(counts[counts == 380].index)


@pytest.fixture(scope="session")
def gold_tp() -> pd.DataFrame:
    """La tabla Gold. Se salta con instrucciones si todavía no se construyó."""
    if not table_exists("gold_tp_match", layer="gold"):
        pytest.skip("Falta gold.gold_tp_match. Corré `python -m features.gold_tp`.")
    return read_table("gold_tp_match", layer="gold")
