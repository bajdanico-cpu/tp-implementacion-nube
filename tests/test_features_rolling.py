"""Unitarios de las ventanas, sobre datos sintéticos armados a mano.

No dependen de Silver: corren siempre, incluso en una máquina que no ingestó nada. Cada
uno fija una decisión de diseño que costó plata descubrir, y varios son **pruebas de
fuego**: construyen explícitamente el caso que la implementación ingenua rompe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features import player_agg, team_form as tf


def _largo(filas: list[dict]) -> pd.DataFrame:
    """Tabla larga mínima: equipo, kickoff y las estadísticas que se prueban."""
    d = pd.DataFrame(filas)
    d["kickoff_time"] = pd.to_datetime(d["kickoff_time"], utc=True)
    for c in ("gf", "gc"):
        if c not in d:
            d[c] = 0
    return d


def _objetivos(filas: list[dict]) -> pd.DataFrame:
    d = pd.DataFrame(filas)
    d["corte"] = pd.to_datetime(d["corte"], utc=True)
    return d


# ---------------------------------------------------------------------------
# La ventana sólo mira hacia atrás
# ---------------------------------------------------------------------------

def test_la_ventana_solo_usa_partidos_anteriores_al_corte():
    """El valor pegado tiene que ser la media de los partidos PREVIOS, sin el propio."""
    largo = _largo([
        {"season": "S", "team_short": "AAA", "kickoff_time": "2024-01-01", "pts": 3},
        {"season": "S", "team_short": "AAA", "kickoff_time": "2024-01-08", "pts": 0},
        {"season": "S", "team_short": "AAA", "kickoff_time": "2024-01-15", "pts": 3},
    ])
    hist = tf._rolling(largo, ["team_short"], ["pts"], 3, "u3")

    obj = _objetivos([{"team_short": "AAA", "corte": "2024-01-15"}])
    res = tf.pegar_asof(obj, hist, ["team_short"], ["pts_u3"])

    # Sólo los dos primeros partidos son anteriores al corte: (3 + 0) / 2 = 1.5
    assert res["pts_u3"].iloc[0] == pytest.approx(1.5)


def test_un_partido_que_arranca_justo_en_el_corte_no_entra():
    """`allow_exact_matches=False`: estrictamente anterior, igual que transform.leakage."""
    largo = _largo([
        {"season": "S", "team_short": "AAA", "kickoff_time": "2024-01-01", "pts": 3},
        {"season": "S", "team_short": "AAA", "kickoff_time": "2024-01-08", "pts": 0},
    ])
    hist = tf._rolling(largo, ["team_short"], ["pts"], 3, "u3")
    obj = _objetivos([{"team_short": "AAA", "corte": "2024-01-08"}])
    res = tf.pegar_asof(obj, hist, ["team_short"], ["pts_u3"])
    assert res["pts_u3"].iloc[0] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# PRUEBA DE FUEGO: las dobles fechas
# ---------------------------------------------------------------------------

def test_los_dos_partidos_de_una_doble_fecha_comparten_features():
    """Hay 85 pares (temporada, gameweek, equipo) con dos partidos en la misma fecha.

    Los dos comparten el corte, así que TIENEN que compartir el vector de features: se
    predicen en el mismo momento con la misma información. Un `shift(1)` le daría al
    segundo el resultado del primero, que se jugó después del corte.
    """
    largo = _largo([
        {"season": "S", "team_short": "AAA", "kickoff_time": "2024-01-01", "pts": 3},
        {"season": "S", "team_short": "AAA", "kickoff_time": "2024-01-08", "pts": 0},
        # los dos de la doble fecha:
        {"season": "S", "team_short": "AAA", "kickoff_time": "2024-01-16", "pts": 3},
        {"season": "S", "team_short": "AAA", "kickoff_time": "2024-01-19", "pts": 1},
    ])
    hist = tf._rolling(largo, ["team_short"], ["pts"], 3, "u3")

    corte = "2024-01-16"  # inicio de la fecha = kickoff del primero de los dos
    obj = _objetivos([{"team_short": "AAA", "corte": corte},
                      {"team_short": "AAA", "corte": corte}])
    res = tf.pegar_asof(obj, hist, ["team_short"], ["pts_u3"])

    assert res["pts_u3"].nunique() == 1, "los dos partidos deberían ver lo mismo"
    assert res["pts_u3"].iloc[0] == pytest.approx(1.5)  # (3 + 0) / 2


def test_shift_ingenuo_habria_filtrado_en_la_doble_fecha():
    """Deja constancia de que el problema era real, no teórico.

    Se calcula a mano lo que daría `shift(1).rolling(3)` sobre el mismo frame y se
    verifica que difiere: el segundo partido de la doble fecha vería el resultado del
    primero.
    """
    largo = _largo([
        {"season": "S", "team_short": "AAA", "kickoff_time": "2024-01-01", "pts": 3},
        {"season": "S", "team_short": "AAA", "kickoff_time": "2024-01-08", "pts": 0},
        {"season": "S", "team_short": "AAA", "kickoff_time": "2024-01-16", "pts": 3},
        {"season": "S", "team_short": "AAA", "kickoff_time": "2024-01-19", "pts": 1},
    ]).sort_values("kickoff_time")

    con_shift = largo["pts"].shift(1).rolling(3, min_periods=1).mean().to_numpy()

    # El cuarto partido (el segundo de la doble fecha) incluiría al tercero, que se jugó
    # DESPUÉS del corte de esa fecha.
    assert con_shift[3] == pytest.approx((3 + 0 + 3) / 3)
    assert con_shift[3] != pytest.approx(1.5), "shift no filtra el partido del mismo corte"


# ---------------------------------------------------------------------------
# Bordes de temporada
# ---------------------------------------------------------------------------

def test_la_ventana_cruzada_tiene_dato_en_la_primera_fecha_y_la_intra_temporada_no():
    """Las dos señales del arranque de temporada, que es el 2,6 % de los partidos."""
    largo = _largo([
        {"season": "S1", "team_short": "AAA", "kickoff_time": "2024-05-01", "pts": 3},
        {"season": "S2", "team_short": "AAA", "kickoff_time": "2024-08-10", "pts": 0},
    ])
    cruzada = tf._rolling(largo, ["team_short"], ["pts"], 3, "u3")
    intra = tf._rolling(largo, ["season", "team_short"], ["pts"], 3, "u3")

    obj = _objetivos([{"season": "S2", "team_short": "AAA", "corte": "2024-08-10"}])
    r_cruz = tf.pegar_asof(obj, cruzada, ["team_short"], ["pts_u3"])
    r_intra = tf.pegar_asof(obj, intra, ["season", "team_short"], ["pts_u3"])

    assert r_cruz["pts_u3"].iloc[0] == pytest.approx(3.0), "la cruzada ve la temporada previa"
    assert pd.isna(r_intra["pts_u3"].iloc[0]), "la intra-temporada arranca en blanco"


# ---------------------------------------------------------------------------
# Agregación de jugadores
# ---------------------------------------------------------------------------

def _players(filas: list[dict]) -> pd.DataFrame:
    d = pd.DataFrame(filas)
    for c in ("expected_goals", "expected_assists", "total_points", "minutes",
              "saves", "goals_conceded"):
        if c not in d:
            d[c] = 0
    if "gameweek" not in d:
        d["gameweek"] = 20
    if "player_name" not in d:
        d["player_name"] = [f"J{i}" for i in range(len(d))]
    return d


def test_el_xgc_es_el_xg_del_rival_no_la_suma_de_la_plantilla():
    """`sum(expected_goals_conceded)` se cuenta una vez por jugador e infla x11.

    Medido sobre 2024-25: la suma da media 15,75 contra 1,47 goles concedidos reales.
    El xGC correcto es el xG del rival en el mismo fixture (media 1,44).
    """
    players = _players([
        {"season": "S", "fixture_id": 1, "team_short": "AAA", "position": "FWD",
         "expected_goals": 0.8, "minutes": 90, "total_points": 5},
        {"season": "S", "fixture_id": 1, "team_short": "AAA", "position": "MID",
         "expected_goals": 0.4, "minutes": 90, "total_points": 3},
        {"season": "S", "fixture_id": 1, "team_short": "BBB", "position": "FWD",
         "expected_goals": 0.3, "minutes": 90, "total_points": 2},
    ])
    out = player_agg.team_stats_by_fixture(players).set_index("team_short")

    assert out.loc["AAA", "xg"] == pytest.approx(1.2)
    assert out.loc["AAA", "xgc"] == pytest.approx(0.3), "el xGC de AAA es el xG de BBB"
    assert out.loc["BBB", "xgc"] == pytest.approx(1.2)
    assert out.loc["AAA", "xgc"] < 6, "muy por debajo del ~15 que daría la suma ingenua"


def test_los_puntos_por_linea_suman_el_total_del_equipo():
    """El pedido explícito del bloque 4: puntaje de defensa, mediocampo y delantera."""
    players = _players([
        {"season": "S", "fixture_id": 1, "team_short": "AAA", "position": "GK",
         "total_points": 6, "minutes": 90},
        {"season": "S", "fixture_id": 1, "team_short": "AAA", "position": "DEF",
         "total_points": 12, "minutes": 90},
        {"season": "S", "fixture_id": 1, "team_short": "AAA", "position": "MID",
         "total_points": 9, "minutes": 90},
        {"season": "S", "fixture_id": 1, "team_short": "AAA", "position": "FWD",
         "total_points": 4, "minutes": 90},
        {"season": "S", "fixture_id": 1, "team_short": "BBB", "position": "MID",
         "total_points": 1, "minutes": 90},
    ])
    out = player_agg.team_stats_by_fixture(players).set_index("team_short")
    fila = out.loc["AAA"]
    assert (fila["pts_arq"], fila["pts_def"], fila["pts_med"], fila["pts_del"]) == (6, 12, 9, 4)
    assert fila[["pts_arq", "pts_def", "pts_med", "pts_del"]].sum() == 31


def test_los_puntos_de_un_jugador_transferido_quedan_en_su_equipo_de_entonces():
    """La pregunta de los pases, respondida por el orden de las operaciones.

    Se agrega POR FIXTURE antes de calcular ventanas, así que cada jugador queda
    atribuido al equipo con el que efectivamente jugó ese día. Sus puntos de antes del
    pase se quedan en el equipo viejo para siempre.
    """
    jugador = "Fulano de Tal"
    players = _players([
        {"season": "S", "fixture_id": 1, "team_short": "AAA", "position": "FWD",
         "player_name": jugador, "total_points": 10, "minutes": 90},
        {"season": "S", "fixture_id": 2, "team_short": "BBB", "position": "FWD",
         "player_name": jugador, "total_points": 8, "minutes": 90},
        {"season": "S", "fixture_id": 1, "team_short": "ZZZ", "position": "FWD",
         "player_name": "Otro", "total_points": 1, "minutes": 90},
        {"season": "S", "fixture_id": 2, "team_short": "ZZZ", "position": "FWD",
         "player_name": "Otro", "total_points": 1, "minutes": 90},
    ])
    out = player_agg.team_stats_by_fixture(players)
    aaa = out[(out.team_short == "AAA") & (out.fixture_id == 1)]["pts_del"].iloc[0]
    bbb = out[(out.team_short == "BBB") & (out.fixture_id == 2)]["pts_del"].iloc[0]

    assert aaa == 10, "sus puntos de antes del pase quedan en el equipo viejo"
    assert bbb == 8, "los de después suman al nuevo"
    assert (out.team_short == "AAA").sum() == 1, "no aparece en el equipo nuevo hacia atrás"


def test_el_xg_falso_de_2022_23_queda_en_nulo_y_no_en_cero():
    """El xG de 2022-23 viene hardcodeado en 0,0 hasta la GW15 para los 20 equipos.

    Si entra como cero, el modelo aprende que "xG bajo" y "arranque de temporada" van
    juntos: un artefacto del calendario de publicación de FPL, no una propiedad del fútbol.
    """
    players = _players([
        {"season": "2022-23", "gameweek": 5, "fixture_id": 1, "team_short": "AAA",
         "position": "FWD", "expected_goals": 0.0, "minutes": 90},
        {"season": "2022-23", "gameweek": 5, "fixture_id": 1, "team_short": "BBB",
         "position": "FWD", "expected_goals": 0.0, "minutes": 90},
        {"season": "2022-23", "gameweek": 20, "fixture_id": 2, "team_short": "AAA",
         "position": "FWD", "expected_goals": 1.1, "minutes": 90},
        {"season": "2022-23", "gameweek": 20, "fixture_id": 2, "team_short": "BBB",
         "position": "FWD", "expected_goals": 0.7, "minutes": 90},
    ])
    out = player_agg.team_stats_by_fixture(players)
    temprano = out[out.gameweek == 5]
    tarde = out[out.gameweek == 20]

    assert temprano["xg"].isna().all(), "el cero falso tiene que quedar en NaN"
    assert (~temprano["xg_available"]).all()
    assert tarde["xg"].notna().all() and tarde["xg_available"].all()


def test_la_tasa_de_atajadas_es_el_equivalente_defensivo_de_xg_por_tiro():
    """atajadas / (atajadas + goles) = que proporcion de los remates al arco se detiene.

    Separa "concede poco" de "concede mucho pero lo atajan". Con 3 atajadas y 1 gol, de
    los 4 remates al arco se detuvieron 3: 0,75.
    """
    players = _players([
        {"season": "S", "fixture_id": 1, "team_short": "AAA", "position": "GK",
         "saves": 3, "goals_conceded": 1, "minutes": 90},
        {"season": "S", "fixture_id": 1, "team_short": "AAA", "position": "DEF",
         "saves": 0, "goals_conceded": 1, "minutes": 90},
        {"season": "S", "fixture_id": 1, "team_short": "BBB", "position": "GK",
         "saves": 1, "goals_conceded": 3, "minutes": 90},
    ])
    out = player_agg.team_stats_by_fixture(players).set_index("team_short")
    assert out.loc["AAA", "tasa_atajadas"] == pytest.approx(3 / 4)
    assert out.loc["BBB", "tasa_atajadas"] == pytest.approx(1 / 4)


def test_sin_remates_al_arco_la_tasa_de_atajadas_es_nula_y_no_cero():
    """Cero atajadas y cero goles no es "atajo el 0%": es que no hubo remates al arco.

    Un cero ahi le ensenaria al modelo que el arquero fallo todo, cuando en realidad no
    tuvo que intervenir. XGBoost maneja el NaN nativamente.
    """
    players = _players([
        {"season": "S", "fixture_id": 1, "team_short": "AAA", "position": "GK",
         "saves": 0, "goals_conceded": 0, "minutes": 90},
        {"season": "S", "fixture_id": 1, "team_short": "BBB", "position": "GK",
         "saves": 2, "goals_conceded": 0, "minutes": 90},
    ])
    out = player_agg.team_stats_by_fixture(players).set_index("team_short")
    assert pd.isna(out.loc["AAA", "tasa_atajadas"])
    assert out.loc["BBB", "tasa_atajadas"] == pytest.approx(1.0)
