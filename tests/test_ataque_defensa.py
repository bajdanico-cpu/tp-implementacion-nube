"""Tests de los ratings de ataque y defensa (Fase 3).

Las dos pruebas de fuego:

**El crédito se reparte.** Si el local mete más goles de los esperados, tiene que subir *su*
ataque y bajar *la defensa del rival* — las dos cosas. Un rating escalar como el Elo no
puede hacer eso, y es la razón entera de tener dos números.

**Los dos ratings tienen que poder moverse en direcciones opuestas.** Un equipo que golea y
recibe goles (el 4-3 crónico) debe terminar con ataque alto y defensa baja. Si el sistema no
los separa, es un Elo con dos nombres.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features import ataque_defensa as af


def _largo(partidos: list[tuple]) -> pd.DataFrame:
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


def _serie(gl: int, gv: int, n: int = 15) -> pd.DataFrame:
    return _largo([("2022-23", i, f"2022-{9 + i // 28:02d}-{i % 28 + 1:02d}",
                    "A", "B", gl, gv) for i in range(n)])


# ---------------------------------------------------------------------------
# PRUEBA DE FUEGO: el crédito se reparte entre los dos equipos
# ---------------------------------------------------------------------------

def test_prueba_de_fuego_un_gol_de_mas_sube_el_ataque_de_uno_y_baja_la_defensa_del_otro():
    """La razón de ser de tener dos ratings.

    `A` mete mucho más de lo esperado: sube SU ataque **y** baja LA DEFENSA de `B`. Un
    escalar sólo podría mover un número.
    """
    r = af.calcular(_largo([("2022-23", 1, "2022-09-01", "A", "B", 5, 0)]), k=0.05)
    d = r.set_index("team_short")
    assert d.loc["A", "af_ataque"] > 0, "A ataco mejor de lo esperado"
    assert d.loc["B", "af_defensa"] < 0, "B defendio peor de lo esperado"
    # Y del otro lado del marcador, lo simetrico: B no metio nada.
    assert d.loc["B", "af_ataque"] < 0
    assert d.loc["A", "af_defensa"] > 0


def test_prueba_de_fuego_los_dos_ratings_pueden_ir_en_direcciones_opuestas():
    """El 4-3 crónico: ataque alto y defensa baja **a la vez**.

    Si el sistema no pudiera separarlos sería un Elo con dos nombres, y el 4-3 y el 1-0 le
    darían lo mismo.

    Hacen falta dos condiciones que no son obvias y que costaron dos intentos:

    1. **Una liga, no un par.** `mu` es el log del promedio de goles **de los datos que se
       recorren**: si todos los partidos son 4-3, el 4-3 *es* el promedio y los ratings se
       quedan en cero. Son relativos a la muestra, no absolutos.
    2. **Los rivales tienen que jugar entre ellos.** Si `B` sólo juega contra `A`, el
       sistema no puede saber si `A` defiende mal o `B` ataca bien: no hay con qué
       comparar. La identificación viene de que `B`, `C` y `D` metan 3 contra `A` y 1
       entre ellos — esa diferencia sólo se explica por la defensa de `A`.
    """
    otros = ("B", "C", "D")
    p, fid = [], 0
    for i in range(6):
        for o in otros:                      # A golea y recibe contra todos
            p.append(("2022-23", fid, f"2022-09-{i % 28 + 1:02d}", "A", o, 4, 3)); fid += 1
            p.append(("2022-23", fid, f"2022-10-{i % 28 + 1:02d}", o, "A", 4, 3)); fid += 1
        for x, y in ((a, b) for a in otros for b in otros if a != b):
            p.append(("2022-23", fid, f"2022-11-{i % 28 + 1:02d}", x, y, 1, 1)); fid += 1

    r = af.calcular(_largo(p), k=0.05)
    a = r[r.team_short == "A"].sort_values("kickoff_time").iloc[-1]
    assert a["af_ataque"] > 0, "mete 3,5 por partido cuando la liga promedia 2,25"
    assert a["af_defensa"] < 0, "y recibe 3,5, que es igual de anormal para el otro lado"


def test_un_equipo_solido_termina_con_los_dos_ratings_altos():
    r = af.calcular(_serie(2, 0), k=0.05)
    a = r[r.team_short == "A"].sort_values("kickoff_time").iloc[-1]
    assert a["af_ataque"] > 0 and a["af_defensa"] > 0


# ---------------------------------------------------------------------------
# Los goles esperados
# ---------------------------------------------------------------------------

def test_con_los_ratings_en_cero_el_sistema_predice_el_promedio():
    """El punto de partida honesto para un equipo del que no se sabe nada."""
    mu = float(np.log(1.45))
    lam_l, lam_v = af.lambdas(0.0, 0.0, 0.0, 0.0, mu, ventaja=0.0)
    assert lam_l == pytest.approx(1.45)
    assert lam_v == pytest.approx(1.45)


def test_la_ventaja_de_local_solo_sube_los_goles_del_local():
    mu = float(np.log(1.45))
    lam_l, lam_v = af.lambdas(0.0, 0.0, 0.0, 0.0, mu, ventaja=0.18)
    assert lam_l > 1.45
    assert lam_v == pytest.approx(1.45)


def test_los_goles_esperados_son_siempre_positivos():
    """La forma exponencial lo garantiza por construccion. Un lambda negativo no significa
    nada en un modelo de conteos."""
    for a in (-3.0, 0.0, 3.0):
        for d in (-3.0, 0.0, 3.0):
            lam_l, lam_v = af.lambdas(a, d, -a, -d, float(np.log(1.45)), 0.18)
            assert lam_l > 0 and lam_v > 0


def test_el_tope_evita_que_la_exponencial_explote():
    """Sin techo, seis goleadas seguidas disparan el rating y exp(mu+3) da 29 goles."""
    r = af.calcular(_serie(9, 0, n=40), k=0.08)
    assert r["af_ataque"].abs().max() <= af.TOPE + 1e-9
    assert r["af_defensa"].abs().max() <= af.TOPE + 1e-9


# ---------------------------------------------------------------------------
# La feature de partido
# ---------------------------------------------------------------------------

def test_lambda_total_no_es_lo_mismo_que_ser_parejos():
    """La feature que apunta al empate.

    Dos partidos igual de PAREJOS —diferencia esperada cero— pero uno entre equipos
    goleadores y otro entre equipos aburridos. `af_lambda_dif` no los distingue;
    `af_lambda_total` sí, y ahí es donde puede vivir señal sobre el empate.

    Se usa `lambdas()` con la ventaja de local en cero para aislar la propiedad: con la
    ventaja puesta, dos equipos identicos NO dan diferencia cero, y esa asimetria taparia
    lo que el test quiere mostrar.
    """
    mu = float(np.log(1.45))
    goleadores = af.lambdas(0.6, -0.6, 0.6, -0.6, mu, ventaja=0.0)
    aburridos = af.lambdas(-0.6, 0.6, -0.6, 0.6, mu, ventaja=0.0)

    assert goleadores[0] - goleadores[1] == pytest.approx(0.0, abs=1e-9)
    assert aburridos[0] - aburridos[1] == pytest.approx(0.0, abs=1e-9)
    assert sum(goleadores) > sum(aburridos) * 3, "los goleadores esperan MUCHOS mas goles"


def test_partido_cruza_ataque_de_uno_con_defensa_del_otro():
    g = pd.DataFrame({"local_af_ataque": [0.5], "local_af_defensa": [0.0],
                      "visita_af_ataque": [0.0], "visita_af_defensa": [0.3]})
    out = af.partido(g, mu=0.0)
    assert out["af_lambda_local"].iloc[0] == pytest.approx(np.exp(0.5 - 0.3 + af.CFG.af_ventaja))
    assert out["af_lambda_visita"].iloc[0] == pytest.approx(np.exp(0.0 - 0.0))


def test_si_faltan_las_columnas_base_las_de_partido_quedan_en_NaN():
    out = af.partido(pd.DataFrame({"otra": [1.0]}))
    for c in af.COLUMNAS_PARTIDO:
        assert c in out.columns and out[c].isna().all()


# ---------------------------------------------------------------------------
# El ajuste
# ---------------------------------------------------------------------------

def test_el_ajuste_solo_usa_las_temporadas_pedidas():
    largo = _largo([("2022-23", 1, "2022-09-01", "A", "B", 3, 0),
                    ("2025-26", 2, "2025-09-01", "A", "B", 0, 3)])
    res = af.ajustar(largo, temporadas_fit=["2022-23"])
    assert res["grilla"]["n"].unique().tolist() == [1]


def test_el_mu_es_el_mismo_para_calcular_y_para_partido():
    """Si cada uno usara el suyo, las lambdas de las features quedarian en otra escala que
    la que calibro los ratings."""
    largo = _serie(2, 1)
    assert af.mu_de(largo) == pytest.approx(float(np.log(1.5)))
