"""Contratos de las tablas Silver.

Fijan lo que las capas de arriba pueden dar por sentado: qué columnas existen, qué
valores admiten y dónde no puede haber nulos. Si una fuente cambia de formato en
mitad de la temporada, esto lo detecta antes de que el modelo entrene con basura.
"""

from __future__ import annotations

import pandas as pd
import pytest

from common.config import CFG

PARTIDOS_POR_TEMPORADA = 380


# --------------------------------------------------------------------------- #
#  fact_match
# --------------------------------------------------------------------------- #

COLUMNAS_MATCH = [
    "season", "match_date", "home_short", "away_short", "home_code", "away_code",
    "home_goals", "away_goals", "result", "target_1x2",
    "home_shots", "away_shots", "home_shots_target", "away_shots_target",
    "home_corners", "away_corners",
    "odds_b365_close_home", "odds_b365_close_draw", "odds_b365_close_away",
    "odds_avg_close_home", "odds_avg_close_draw", "odds_avg_close_away",
]


@pytest.mark.parametrize("col", COLUMNAS_MATCH)
def test_fact_match_tiene_las_columnas_del_contrato(fact_match, col):
    assert col in fact_match.columns


def test_fact_match_tiene_380_partidos_por_temporada_cerrada(fact_match, closed_seasons):
    assert closed_seasons, "No hay ninguna temporada completa en fact_match"
    for season in closed_seasons:
        n = (fact_match["season"] == season).sum()
        assert n == PARTIDOS_POR_TEMPORADA, f"{season} tiene {n} partidos, no {PARTIDOS_POR_TEMPORADA}"


def test_fact_match_sin_nulos_en_las_claves(fact_match):
    claves = ["season", "match_date", "home_short", "away_short"]
    nulos = fact_match[claves].isna().sum()
    assert nulos.sum() == 0, f"Nulos en columnas clave:\n{nulos[nulos > 0].to_string()}"


def test_fact_match_no_tiene_partidos_duplicados(fact_match):
    dup = fact_match[fact_match.duplicated(["season", "home_short", "away_short"], keep=False)]
    assert dup.empty, (
        f"{len(dup)} partidos duplicados por (temporada, local, visitante). "
        f"En una liga de ida y vuelta cada par juega una sola vez en cada cancha."
    )


def test_el_target_solo_toma_tres_valores(fact_match):
    assert set(fact_match["target_1x2"].dropna()) <= {"home", "draw", "away"}


def test_el_target_es_coherente_con_los_goles(fact_match):
    """El 1X2 tiene que derivarse de los goles, no contradecirlos."""
    df = fact_match.dropna(subset=["home_goals", "away_goals", "target_1x2"])
    esperado = pd.Series("draw", index=df.index)
    esperado[df["home_goals"] > df["away_goals"]] = "home"
    esperado[df["home_goals"] < df["away_goals"]] = "away"

    malas = df[esperado != df["target_1x2"]]
    assert malas.empty, f"{len(malas)} partidos con target incoherente con el marcador"


def test_un_equipo_no_juega_contra_si_mismo(fact_match):
    assert (fact_match["home_short"] == fact_match["away_short"]).sum() == 0


def test_las_cuotas_de_cierre_son_mayores_a_uno(fact_match):
    """Una cuota decimal <= 1 es imposible: implicaría probabilidad > 100%."""
    for col in ["odds_avg_close_home", "odds_avg_close_draw", "odds_avg_close_away"]:
        vals = fact_match[col].dropna()
        if vals.empty:
            continue
        assert (vals > 1.0).all(), f"{col} tiene valores <= 1"


def test_las_cuotas_de_cierre_tienen_cobertura_casi_total(fact_match, closed_seasons):
    """Sin cuotas no hay baseline duro. Se tolera algún faltante, no un agujero."""
    for season in closed_seasons:
        sub = fact_match[fact_match["season"] == season]
        cobertura = sub["odds_avg_close_home"].notna().mean()
        assert cobertura > 0.95, f"{season}: sólo {cobertura:.1%} de partidos con cuota de cierre"


# --------------------------------------------------------------------------- #
#  fact_fixture
# --------------------------------------------------------------------------- #

COLUMNAS_FIXTURE = [
    "season", "gameweek", "fixture_id", "match_date", "kickoff_time",
    "deadline_time", "deadline_source", "home_short", "away_short",
    "team_h_difficulty", "team_a_difficulty",
]


@pytest.mark.parametrize("col", COLUMNAS_FIXTURE)
def test_fact_fixture_tiene_las_columnas_del_contrato(fact_fixture, col):
    assert col in fact_fixture.columns


def test_cada_temporada_tiene_380_fixtures(fact_fixture):
    """20 equipos, ida y vuelta. Ése es el invariante, no el número de gameweeks."""
    n = fact_fixture.groupby("season").size()
    malas = n[n != PARTIDOS_POR_TEMPORADA]
    assert malas.empty, f"Temporadas sin 380 fixtures:\n{malas.to_string()}"


def test_las_gameweeks_estan_en_el_rango_1_38(fact_fixture):
    gws = fact_fixture["gameweek"].dropna()
    assert gws.between(1, 38).all(), f"gameweeks fuera de rango: {sorted(set(gws))}"


def test_las_gameweeks_no_tienen_todas_la_misma_cantidad_de_partidos(fact_fixture):
    """Documenta un hecho del dominio que condiciona el feature engineering.

    Una gameweek NO equivale a "una fecha de 10 partidos". En 2022-23 la GW7 se
    canceló entera por la muerte de Isabel II y sus partidos se reprogramaron a
    otras fechas: hay gameweeks con 7 partidos y otras con 16, y la GW7 no existe.

    Consecuencia práctica: las ventanas rolling tienen que calcularse sobre los
    **últimos N partidos de cada equipo**, no sobre las últimas N gameweeks. Si este
    test dejara de pasar, se podría simplificar; mientras pase, no.
    """
    tamanos = fact_fixture.groupby(["season", "gameweek"]).size()
    assert tamanos.nunique() > 1, (
        "Todas las gameweeks tienen la misma cantidad de partidos. Revisar si se "
        "puede simplificar el cálculo de ventanas rolling en Gold."
    )


def test_el_deadline_nunca_es_nulo(fact_fixture):
    """Sin deadline no hay control anti-leakage posible para esa fecha."""
    nulos = fact_fixture["deadline_time"].isna().sum()
    assert nulos == 0, f"{nulos} fixtures sin deadline"


def test_todos_los_fixtures_de_una_fecha_comparten_deadline(fact_fixture):
    distintos = fact_fixture.groupby(["season", "gameweek"])["deadline_time"].nunique()
    malas = distintos[distintos > 1]
    assert malas.empty, f"Gameweeks con más de un deadline:\n{malas.to_string()}"


def test_el_fdr_esta_entre_1_y_5(fact_fixture):
    """El Fixture Difficulty Rating de FPL es una escala de 1 a 5."""
    for col in ["team_h_difficulty", "team_a_difficulty"]:
        vals = fact_fixture[col].dropna()
        assert vals.between(1, 5).all(), f"{col} fuera del rango 1-5"


def test_la_temporada_actual_usa_el_deadline_real_de_la_api(fact_fixture):
    """Para la temporada en curso no se deriva nada: el deadline sale de la API."""
    actual = fact_fixture[fact_fixture["season"] == CFG.current_season]
    if actual.empty:
        pytest.skip("La temporada actual todavía no está en fact_fixture")
    assert (actual["deadline_source"] == "api").all()


# --------------------------------------------------------------------------- #
#  fact_player_gw
# --------------------------------------------------------------------------- #

COLUMNAS_PLAYER = [
    "season", "gameweek", "player_name", "position", "team_short", "opponent_short",
    "was_home", "kickoff_time", "minutes", "goals_scored", "assists",
    "expected_goals", "expected_assists", "expected_goals_conceded", "total_points",
]


@pytest.mark.parametrize("col", COLUMNAS_PLAYER)
def test_fact_player_gw_tiene_las_columnas_del_contrato(fact_player_gw, col):
    assert col in fact_player_gw.columns


def test_las_posiciones_son_las_cuatro_conocidas(fact_player_gw):
    """Sólo jugadores.

    vaastav trae también filas con `position == 'AM'` (Assistant Manager): son los
    DT, que FPL hizo elegibles con un chip vigente sólo entre la GW23 de 2024-25 y
    el final de esa temporada. Tienen 0 minutos y puntúan por resultado del equipo,
    así que se excluyen en Silver para que no contaminen los agregados por posición.
    """
    assert set(fact_player_gw["position"].dropna()) <= {"GK", "DEF", "MID", "FWD"}


def test_los_directores_tecnicos_quedaron_fuera(fact_player_gw):
    assert (fact_player_gw["position"] == "AM").sum() == 0


def test_los_minutos_estan_en_un_rango_plausible(fact_player_gw):
    """0 a 120 cubre partido completo más alargue."""
    m = fact_player_gw["minutes"].dropna()
    assert m.between(0, 120).all(), f"minutos fuera de rango: {m.min()} a {m.max()}"


def test_sin_nulos_en_las_claves_de_jugador(fact_player_gw):
    claves = ["season", "gameweek", "team_short", "opponent_short"]
    nulos = fact_player_gw[claves].isna().sum()
    assert nulos.sum() == 0, f"Nulos en claves:\n{nulos[nulos > 0].to_string()}"


def test_un_equipo_no_es_su_propio_rival(fact_player_gw):
    assert (fact_player_gw["team_short"] == fact_player_gw["opponent_short"]).sum() == 0


def test_el_xg_existe_en_toda_la_ventana_ingestada(fact_player_gw):
    """La ventana arranca en 2022-23 justamente porque antes no hay xG.

    Si esto falla, o se amplió `ingest_from` sin querer o la fuente cambió.
    """
    cobertura = fact_player_gw.groupby("season")["expected_goals"].apply(lambda s: s.notna().mean())
    malas = cobertura[cobertura < 0.99]
    assert malas.empty, (
        f"Temporadas con xG incompleto:\n{malas.to_string()}\n"
        f"La ventana de ingesta empieza en {CFG.ingest_from} porque antes no existe."
    )


# --------------------------------------------------------------------------- #
#  Coherencia entre tablas
# --------------------------------------------------------------------------- #


def test_los_equipos_de_fact_match_existen_en_dim_team(fact_match, dim_team):
    validos = set(zip(dim_team["season"], dim_team["short_name"]))
    locales = set(zip(fact_match["season"], fact_match["home_short"]))
    visitantes = set(zip(fact_match["season"], fact_match["away_short"]))

    huerfanos = (locales | visitantes) - validos
    assert not huerfanos, f"Equipos de fact_match que no están en dim_team: {sorted(huerfanos)}"


def test_los_equipos_de_fact_player_gw_existen_en_dim_team(fact_player_gw, dim_team):
    validos = set(zip(dim_team["season"], dim_team["short_name"]))
    presentes = set(zip(fact_player_gw["season"], fact_player_gw["team_short"]))

    huerfanos = presentes - validos
    assert not huerfanos, f"Equipos de fact_player_gw que no están en dim_team: {sorted(huerfanos)}"


def test_cada_partido_de_football_data_cruza_con_un_fixture_de_fpl(fact_match, fact_fixture, closed_seasons):
    """El cruce entre fuentes tiene que ser total en las temporadas cerradas.

    Es la verificación de punta a punta del mapeo de equipos y del parseo de fechas.
    """
    clave = ["season", "match_date", "home_short", "away_short"]
    for season in closed_seasons:
        m = fact_match[fact_match["season"] == season][clave]
        f = fact_fixture[fact_fixture["season"] == season][clave]
        cruzados = m.merge(f, on=clave, how="inner")
        assert len(cruzados) == len(m), (
            f"{season}: sólo cruzaron {len(cruzados)} de {len(m)} partidos entre "
            f"football-data y FPL."
        )
