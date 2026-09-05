"""Tests de los pi-ratings (Fase 2).

La prueba de fuego es `gamma = 1`: con ese valor los dos ratings tienen que **colapsar en
uno solo**, porque todo lo aprendido de local se transfiere a visitante. Es la hipótesis
nula de la fase — si el sistema no colapsa cuando debe, entonces la "separación local /
visitante" que la fase vende no es lo que el código está calculando.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features import pi_ratings as pr


def _largo(partidos: list[tuple]) -> pd.DataFrame:
    """`(season, fixture_id, fecha, local, visita, gl, gv)` -> tabla larga de dos filas."""
    filas = []
    for season, fid, fecha, loc, vis, gl, gv in partidos:
        k = pd.Timestamp(fecha, tz="UTC")
        filas.append({"season": season, "fixture_id": fid, "kickoff_time": k,
                      "team_short": loc, "rival_short": vis, "gf": gl, "gc": gv,
                      "es_local": True})
        filas.append({"season": season, "fixture_id": fid, "kickoff_time": k,
                      "team_short": vis, "rival_short": loc, "gf": gv, "gc": gl,
                      "es_local": False})
    return pd.DataFrame(filas)


def _ida_y_vuelta(n: int = 20) -> pd.DataFrame:
    """`A` golea de local y pierde de visitante. El caso donde la localia importa."""
    p = []
    for i in range(n):
        p.append(("2022-23", 2 * i, f"2022-09-{i % 28 + 1:02d}", "A", "B", 3, 0))
        p.append(("2022-23", 2 * i + 1, f"2022-10-{i % 28 + 1:02d}", "B", "A", 3, 0))
    return _largo(p)


# ---------------------------------------------------------------------------
# La conversión rating -> goles
# ---------------------------------------------------------------------------

def test_a_goles_es_impar_y_vale_cero_en_cero():
    assert pr.a_goles(0.0) == pytest.approx(0.0)
    assert pr.a_goles(1.5) == pytest.approx(-pr.a_goles(-1.5))


def test_a_goles_es_monotona_creciente():
    r = np.linspace(-4, 4, 200)
    assert np.all(np.diff(pr.a_goles(r)) > 0)


def test_a_goles_se_estira_en_los_extremos():
    """Cerca de 0 casi lineal, y despues se abre: es como se comportan las diferencias
    de goles reales, donde pasar de +2 a +3 es mucho mas raro que de 0 a +1."""
    assert pr.a_goles(2.0) - pr.a_goles(1.0) > pr.a_goles(1.0) - pr.a_goles(0.0)


# ---------------------------------------------------------------------------
# PRUEBA DE FUEGO: gamma
# ---------------------------------------------------------------------------

def test_prueba_de_fuego_gamma_1_colapsa_los_dos_ratings_en_uno():
    """La hipotesis nula de la fase.

    Con `gamma = 1` todo lo que se aprende de un equipo jugando de local se transfiere
    entero a su rating de visitante, asi que los dos numeros tienen que ser identicos y el
    sistema deja de distinguir localia. Si esto falla, lo que el codigo calcula no es lo
    que la fase dice que calcula.
    """
    r = pr.calcular(_ida_y_vuelta(), lam=0.2, gamma=1.0)
    assert np.allclose(r["pi_home"], r["pi_away"])
    assert np.allclose(r["pi_ventaja"], 0.0)


def test_gamma_0_deja_los_dos_ratings_completamente_separados():
    """El otro extremo: dos equipos distintos que comparten nombre.

    `A` solo juega de local contra `B`, asi que con gamma=0 el rating de visitante de `A`
    nunca se entera de nada y queda en el valor inicial.
    """
    largo = _largo([("2022-23", i, f"2022-09-{i + 1:02d}", "A", "B", 3, 0)
                    for i in range(10)])
    r = pr.calcular(largo, lam=0.2, gamma=0.0)
    a = r[r.team_short == "A"].sort_values("kickoff_time").iloc[-1]
    assert a["pi_home"] > 0, "gano diez veces de local: su rating de local tiene que subir"
    assert a["pi_away"] == pytest.approx(pr.INICIAL), "de visitante nunca jugo"


def test_gamma_intermedio_deja_los_dos_ratings_distintos_pero_correlacionados():
    r = pr.calcular(_ida_y_vuelta(), lam=0.2, gamma=0.5)
    a = r[r.team_short == "A"].sort_values("kickoff_time").iloc[-1]
    assert a["pi_home"] > a["pi_away"], "golea de local y pierde de visitante"
    assert a["pi_ventaja"] == pytest.approx(a["pi_home"] - a["pi_away"])


# ---------------------------------------------------------------------------
# La dinámica del rating
# ---------------------------------------------------------------------------

def test_lo_que_gana_uno_lo_pierde_el_otro():
    """Suma cero en el ajuste directo: el rating es relativo, no absoluto."""
    largo = _largo([("2022-23", 1, "2022-09-01", "A", "B", 2, 0)])
    r = pr.calcular(largo, lam=0.2, gamma=0.5).set_index("team_short")
    assert r.loc["A", "pi_home"] == pytest.approx(-r.loc["B", "pi_away"])


def test_ganar_por_menos_de_lo_esperado_BAJA_el_rating():
    """La diferencia con el Elo, y la razon de ser de aprender de la diferencia de goles.

    `A` viene goleando 5-0, asi que el sistema espera una goleada. Ganar 1-0 es peor que lo
    esperado y tiene que **bajarle** el rating, aunque haya ganado. Un Elo clasico, que
    actualiza con 1/0,5/0, se lo subiria.
    """
    base = [("2022-23", i, f"2022-09-{i + 1:02d}", "A", "B", 5, 0) for i in range(12)]
    r_antes = pr.calcular(_largo(base), lam=0.2, gamma=0.5)
    antes = r_antes[r_antes.team_short == "A"].sort_values("kickoff_time").iloc[-1]["pi_home"]

    r_desp = pr.calcular(_largo(base + [("2022-23", 99, "2022-10-01", "A", "B", 1, 0)]),
                         lam=0.2, gamma=0.5)
    despues = r_desp[r_desp.team_short == "A"].sort_values("kickoff_time").iloc[-1]["pi_home"]
    assert despues < antes, "gano, pero por mucho menos de lo esperado"


def test_una_goleada_no_mueve_el_rating_proporcionalmente_al_marcador():
    """El error va atenuado por logaritmo, igual que el margen en el Elo: un 6-0 no puede
    valer seis veces un 1-0 o el rating lo domina un resultado suelto."""
    uno = pr.calcular(_largo([("2022-23", 1, "2022-09-01", "A", "B", 1, 0)]), lam=0.2)
    seis = pr.calcular(_largo([("2022-23", 1, "2022-09-01", "A", "B", 6, 0)]), lam=0.2)
    m1 = uno.set_index("team_short").loc["A", "pi_home"]
    m6 = seis.set_index("team_short").loc["A", "pi_home"]
    assert m6 > m1
    assert m6 < 6 * m1, "sin atenuar, un 6-0 valdria seis veces un 1-0"


def test_el_fixture_id_es_entero():
    r = pr.calcular(_ida_y_vuelta(2))
    assert r["fixture_id"].dtype == np.dtype("int64")


def test_emite_dos_filas_por_partido():
    r = pr.calcular(_ida_y_vuelta(3))
    assert len(r) == 2 * 6
    assert set(r["team_short"]) == {"A", "B"}


# ---------------------------------------------------------------------------
# La feature de partido: cruza los lados, no los repite
# ---------------------------------------------------------------------------

def test_gd_esperado_cruza_local_de_uno_con_visitante_del_otro():
    """El `dif_` automatico de Gold resta la MISMA columna de los dos lados. Esta feature
    existe porque hace falta lo otro: `f(local_pi_home) - f(visita_pi_away)`."""
    local_home = pd.Series([1.0, 0.0, -2.0])
    visita_away = pd.Series([0.0, 1.0, 1.0])
    got = pr.gd_esperado(local_home, visita_away)
    esperado = pr.a_goles(local_home.to_numpy()) - pr.a_goles(visita_away.to_numpy())
    assert np.allclose(got.to_numpy(), esperado)
    assert got.iloc[0] > 0 and got.iloc[1] < 0


def test_gd_esperado_es_cero_entre_iguales():
    assert pr.gd_esperado(pd.Series([1.7]), pd.Series([1.7])).iloc[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# El ajuste no puede tocar el holdout
# ---------------------------------------------------------------------------

def test_el_ajuste_solo_promedia_el_error_de_las_temporadas_pedidas():
    largo = _largo([("2022-23", 1, "2022-09-01", "A", "B", 3, 0),
                    ("2025-26", 2, "2025-09-01", "A", "B", 0, 3)])
    res = pr.ajustar(largo, temporadas_fit=["2022-23"])
    assert res["grilla"]["n"].unique().tolist() == [1], "solo un partido es de 2022-23"
    assert "lambda" in res and "gamma" in res


def test_la_grilla_incluye_gamma_1_a_proposito():
    """Es la hipotesis nula: si separar local y visitante no sirviera, el ajuste elegiria
    gamma=1 y el sistema volveria a ser un Elo. Sacarla de la grilla seria asumir la
    conclusion."""
    assert 1.0 in pr.GRILLA_GAMMA
