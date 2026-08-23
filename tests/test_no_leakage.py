"""El test que sostiene la validez del trabajo.

Verifica la regla de oro: ninguna feature usada para predecir la fecha N puede
provenir de un instante posterior al deadline de la fecha N.

Incluye una **prueba de fuego**: se inyecta a propósito una fila que viola la regla
y se comprueba que el control la detecta. Un test anti-leakage que nunca se vio
fallar no prueba nada — sólo demuestra que la condición no se está evaluando.
"""

from __future__ import annotations

import pandas as pd
import pytest

from common.config import CFG
from transform import leakage


# --------------------------------------------------------------------------- #
#  La regla, sobre datos reales
# --------------------------------------------------------------------------- #


def test_deadline_anterior_a_todos_los_partidos_de_su_fecha(fact_fixture):
    """El deadline de una gameweek tiene que ser anterior a todos sus partidos.

    Si esto no se cumple, el ancla temporal está mal y todo lo demás sobra.
    """
    df = fact_fixture.dropna(subset=["deadline_time", "kickoff_time"])
    violaciones = df[df["deadline_time"] >= df["kickoff_time"]]

    assert violaciones.empty, (
        f"{len(violaciones)} fixtures arrancan antes o justo en su propio deadline:\n"
        f"{violaciones[['season', 'gameweek', 'deadline_time', 'kickoff_time']].head().to_string()}"
    )


def test_stats_de_jugador_son_posteriores_al_deadline_de_su_fecha(fact_player_gw, fact_fixture):
    """Las stats de la fecha N ocurren DESPUÉS del deadline de la fecha N.

    Es la razón de ser de todo esto: por eso no se pueden usar como feature de esa
    misma fecha, sólo de las siguientes.
    """
    deadlines = leakage.gameweek_deadlines(fact_fixture)
    df = fact_player_gw.merge(deadlines, on=["season", "gameweek"], how="inner")
    df = df.dropna(subset=["kickoff_time", "deadline"])

    anteriores = df[df["kickoff_time"] < df["deadline"]]
    assert anteriores.empty, (
        f"{len(anteriores)} filas de jugador tienen kickoff ANTES del deadline de su "
        f"propia fecha. Eso indica que la gameweek está mal asignada."
    )


@pytest.mark.parametrize("gameweek", [5, 15, 30])
def test_ventana_historica_respeta_el_deadline(fact_player_gw, fact_fixture, gameweek):
    """Simula la construcción de features de una fecha.

    Toma como fuente sólo las fechas anteriores — que es como tiene que trabajar
    Gold — y verifica que ninguna quede del lado equivocado del deadline.
    """
    season = CFG.holdout_season
    fuentes = fact_player_gw[
        (fact_player_gw["season"] == season) & (fact_player_gw["gameweek"] < gameweek)
    ]
    if fuentes.empty:
        pytest.skip(f"Sin datos para {season} antes de la GW{gameweek}")

    leakage.assert_sources_before_deadline(
        fuentes, fact_fixture, season, gameweek,
        context=f"features de {season} GW{gameweek}",
    )


# --------------------------------------------------------------------------- #
#  Prueba de fuego: el control tiene que FALLAR cuando corresponde
# --------------------------------------------------------------------------- #


def test_el_control_detecta_una_violacion_inyectada(fact_player_gw, fact_fixture):
    """Se cuela una fila de la MISMA fecha que se quiere predecir.

    Es exactamente el error que arruina el trabajo: usar las stats de la fecha N
    para predecir la fecha N. El control tiene que verlo.
    """
    season = CFG.holdout_season
    gameweek = 10

    limpias = fact_player_gw[
        (fact_player_gw["season"] == season) & (fact_player_gw["gameweek"] < gameweek)
    ]
    contaminada = fact_player_gw[
        (fact_player_gw["season"] == season) & (fact_player_gw["gameweek"] == gameweek)
    ].head(1)

    if limpias.empty or contaminada.empty:
        pytest.skip(f"Sin datos suficientes para {season} GW{gameweek}")

    envenenadas = pd.concat([limpias, contaminada], ignore_index=True)

    with pytest.raises(leakage.LeakageError, match="LEAKAGE"):
        leakage.assert_sources_before_deadline(
            envenenadas, fact_fixture, season, gameweek,
        )


def test_el_control_no_da_falsos_positivos(fact_player_gw, fact_fixture):
    """El mismo conjunto, sin la fila contaminada, tiene que pasar."""
    season = CFG.holdout_season
    gameweek = 10

    limpias = fact_player_gw[
        (fact_player_gw["season"] == season) & (fact_player_gw["gameweek"] < gameweek)
    ]
    if limpias.empty:
        pytest.skip(f"Sin datos para {season} antes de la GW{gameweek}")

    leakage.assert_sources_before_deadline(limpias, fact_fixture, season, gameweek)


# --------------------------------------------------------------------------- #
#  Columnas prohibidas
# --------------------------------------------------------------------------- #


def test_xp_no_esta_en_silver(fact_player_gw):
    """Regresión sobre `xP`.

    vaastav lo scrapea del campo `ep_this` DESPUÉS de que termina la fecha, y su
    propia documentación advierte que puede reflejar información post-partido en
    vez de la predicción que los managers veían antes del deadline. Se descarta en
    Silver justamente para que no exista la tentación.
    """
    assert "xP" not in fact_player_gw.columns, (
        "`xP` volvió a aparecer en fact_player_gw. Está en DROPPED_LEAKY de "
        "transform/silver.py por riesgo de leakage documentado."
    )


def test_el_control_de_columnas_prohibidas_funciona():
    """La lista de config tiene que hacer fallar a una tabla que las contenga."""
    tabla_mala = pd.DataFrame({"season": ["2025-26"], "xP": [4.2]})

    with pytest.raises(leakage.LeakageError, match="prohibidas"):
        leakage.assert_no_banned_columns(tabla_mala, context="tabla de prueba")


def test_una_tabla_limpia_pasa_el_control():
    tabla_ok = pd.DataFrame({"season": ["2025-26"], "puntos_ultimas_3": [7.0]})
    leakage.assert_no_banned_columns(tabla_ok)


def test_config_declara_xp_como_prohibida():
    """Si alguien saca `xP` de config.yaml, el control se vuelve decorativo."""
    assert "xP" in CFG.banned_columns
