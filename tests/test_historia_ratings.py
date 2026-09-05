"""Tests de la Fase 1: la historia profunda que siembra el rating.

Tres cosas que tienen que ser ciertas y que, si fallan, fallan **en silencio**:

1. La historia empuja el rating pero **no produce features**. Si un partido de 2005 se
   colara como fila de Gold, el modelo entrenaría con datos sin xG ni FPL.
2. El E0 de la ventana **no se cuenta dos veces**. La historia se ingesta entera, así que
   incluye las temporadas que también trae `largo`.
3. El puente de nombres **encuentra a los ascendidos**. Es el bug que ya se cometió una vez:
   football-data dice `Coventry`/`Hull`/`Ipswich` y el registro `Coventry City`/`Hull
   City`/`Ipswich Town`, y con un match exacto los tres quedaban sin una sola fila —
   justo los tres equipos que motivan toda la fase.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features import elo


# ---------------------------------------------------------------------------
# Datos sintéticos
# ---------------------------------------------------------------------------

def _largo(partidos: list[tuple]) -> pd.DataFrame:
    """`(season, fixture_id, fecha, local, visita, gl, gv)` -> la tabla larga de dos filas."""
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


def _historia(partidos: list[tuple]) -> pd.DataFrame:
    """`(season, division, fecha, local, visita, gl, gv)` con clave canónica ya resuelta."""
    return pd.DataFrame([
        {"season": s, "division": d, "match_date": pd.Timestamp(f),
         "home_key": loc.lower(), "away_key": vis.lower(),
         "home_short": loc, "away_short": vis,
         "home_goals": gl, "away_goals": gv}
        for s, d, f, loc, vis, gl, gv in partidos])


# ---------------------------------------------------------------------------
# 1. La historia es insumo, no salida
# ---------------------------------------------------------------------------

def test_los_partidos_de_historia_no_producen_filas():
    largo = _largo([("2022-23", 1, "2022-08-06", "ARS", "CHE", 2, 0)])
    hist = _historia([("2010-11", "E0", "2010-08-14", "ARS", "CHE", 1, 1),
                      ("2010-11", "E1", "2010-08-14", "LEE", "NFO", 3, 0)])
    e = elo.calcular(largo, hist)
    assert len(e) == 2, "un partido de ventana -> dos filas (local y visita), nada mas"
    assert set(e["team_short"]) == {"ARS", "CHE"}
    assert e["fixture_id"].tolist() == [1, 1]


def test_pero_si_mueven_el_rating():
    """Mismo partido de ventana, con y sin historia previa: el Elo tiene que diferir."""
    largo = _largo([("2022-23", 1, "2022-08-06", "ARS", "CHE", 1, 0)])
    hist = _historia([("2021-22", "E0", f"2021-09-{d:02d}", "ARS", "CHE", 3, 0)
                      for d in range(1, 20)])
    sin = elo.calcular(largo).set_index("team_short")["elo"]
    con = elo.calcular(largo, hist).set_index("team_short")["elo"]
    assert con["ARS"] > sin["ARS"], "19 goleadas previas tienen que dejar a ARS mas arriba"
    assert con["CHE"] < sin["CHE"]


def test_el_fixture_id_vuelve_a_entero():
    """El concat con la historia (fixture_id nulo) promueve la columna a object, y
    `gold_tp` cruza por ella. Si queda object, el merge no encuentra nada."""
    largo = _largo([("2022-23", 7, "2022-08-06", "ARS", "CHE", 1, 0)])
    hist = _historia([("2010-11", "E0", "2010-08-14", "ARS", "CHE", 1, 1)])
    e = elo.calcular(largo, hist)
    assert e["fixture_id"].dtype == np.dtype("int64")


# ---------------------------------------------------------------------------
# 2. PRUEBA DE FUEGO: no contar dos veces
# ---------------------------------------------------------------------------

def test_prueba_de_fuego_el_e0_de_la_ventana_no_se_cuenta_dos_veces():
    """La historia se ingesta entera e incluye el E0 de las temporadas de la ventana.

    Si esas filas no se descartaran, cada partido de Premier movería el rating dos veces y
    el Elo quedaría con el doble de amplitud — un error que ninguna métrica delataría.
    """
    largo = _largo([("2022-23", 1, "2022-08-06", "ARS", "CHE", 3, 0)])
    duplicado = _historia([("2022-23", "E0", "2022-08-06", "ARS", "CHE", 3, 0)])

    solo = elo.calcular(largo, _historia([])).set_index("team_short")["elo"]
    con_dup = elo.calcular(largo, duplicado).set_index("team_short")["elo"]
    assert con_dup["ARS"] == pytest.approx(solo["ARS"]), "el E0 de la ventana entro dos veces"


def test_el_e1_de_la_ventana_si_cuenta():
    """Un descendido que sigue jugando en el Championship tiene que seguir actualizando su
    rating: si vuelve a subir, no puede volver con el rating congelado del descenso."""
    largo = _largo([("2022-23", 1, "2022-08-06", "ARS", "CHE", 1, 0),
                    ("2023-24", 2, "2023-08-12", "ARS", "LEE", 1, 0)])
    hist = _historia([("2022-23", "E1", f"2022-10-{d:02d}", "LEE", "NFO", 4, 0)
                      for d in range(1, 15)])
    sin = elo.calcular(largo).set_index(["team_short", "fixture_id"])["elo"]
    con = elo.calcular(largo, hist).set_index(["team_short", "fixture_id"])["elo"]
    assert con[("LEE", 2)] != sin[("LEE", 2)]


# ---------------------------------------------------------------------------
# 3. La regresión de entre-temporadas
# ---------------------------------------------------------------------------

def test_sin_historia_la_regresion_sigue_apuntando_a_1500():
    """El comportamiento previo a la Fase 1 tiene que quedar intacto: el A/B compara el
    sembrado, no un cambio colateral de la regresion."""
    rating = {"ARS": 1700.0, "CHE": 1300.0}
    elo._regresar(rating, "2023-24", {}, con_historia=False)
    assert rating["ARS"] == pytest.approx(1700 - 200 * elo.REGRESION_TEMPORADA)
    assert rating["CHE"] == pytest.approx(1300 + 200 * elo.REGRESION_TEMPORADA)


def test_con_historia_la_regresion_apunta_a_la_media_de_la_division():
    """La pieza que deja convivir tres divisiones en un mismo Elo.

    Si el Championship regresara a 1500 como la Premier, la diferencia de nivel entre las
    dos —que tardó veinte años de ascensos y descensos en construirse— se borraría cada
    agosto, y un ascendido volvería a entrar con un rating generico.
    """
    rating = {"ARS": 2100.0, "MCI": 1900.0, "LEE": 1500.0, "NFO": 1300.0}
    planteles = {("2023-24", "E0"): {"ARS", "MCI"}, ("2023-24", "E1"): {"LEE", "NFO"}}
    elo._regresar(rating, "2023-24", planteles, con_historia=True)

    # Cada uno hacia la media de SU division (2000 y 1400), no hacia 1500.
    assert rating["ARS"] == pytest.approx(2100 + (2000 - 2100) * elo.REGRESION_TEMPORADA)
    assert rating["LEE"] == pytest.approx(1500 + (1400 - 1500) * elo.REGRESION_TEMPORADA)
    # Y la media de cada division se conserva: la regresion no mueve el centro de masa.
    assert (rating["ARS"] + rating["MCI"]) / 2 == pytest.approx(2000)
    assert (rating["LEE"] + rating["NFO"]) / 2 == pytest.approx(1400)


def test_la_separacion_entre_divisiones_emerge_de_los_ascensos():
    """Sin offset puesto a mano: la unica via es que los equipos se muevan de division.

    Un equipo que domina el E1 sube y, si en el E0 le sigue yendo bien, arrastra rating
    hacia arriba; el que baja hace lo contrario. Con suficientes temporadas los dos grupos
    se separan solos.
    """
    partidos = []
    # Diez temporadas: `fuerte` gana todo en E1, sube, y en E0 tambien gana.
    for i, season in enumerate([f"20{y:02d}-{y + 1:02d}" for y in range(10, 20)]):
        div = "E1" if i < 5 else "E0"
        rival = "debil_e1" if div == "E1" else "debil_e0"
        for d in range(1, 20):
            partidos.append((season, div, f"20{10 + i}-10-{d:02d}", "fuerte", rival, 3, 0))
        # Un partido entre los dos debiles, para que cada division tenga su propio pool.
        partidos.append((season, "E1", f"20{10 + i}-11-01", "debil_e1", "otro_e1", 1, 1))
        partidos.append((season, "E0", f"20{10 + i}-11-01", "debil_e0", "otro_e0", 1, 1))

    largo = _largo([("2022-23", 1, "2022-08-06", "fuerte", "debil_e0", 1, 0)])
    e = elo.calcular(largo, _historia(partidos))
    r = e.set_index("team_short")["elo"]
    assert r["fuerte"] > r["debil_e0"]


# ---------------------------------------------------------------------------
# 4. PRUEBA DE FUEGO: el puente de nombres encuentra a los ascendidos
# ---------------------------------------------------------------------------

def test_prueba_de_fuego_el_puente_resuelve_los_sufijos_genericos():
    """El bug real, con los tres equipos reales.

    football-data los llama `Coventry`, `Hull` e `Ipswich`; el registro canonico
    `Coventry City`, `Hull City` e `Ipswich Town`. Con un match exacto sobre el nombre
    normalizado los tres quedaban con CERO filas de historia — y son precisamente los
    ascendidos, o sea el problema que la fase vino a resolver.
    """
    from transform import historia

    registry = pd.DataFrame([
        {"team_code": 1, "short_name": "COV", "team_name": "Coventry City"},
        {"team_code": 2, "short_name": "HUL", "team_name": "Hull City"},
        {"team_code": 3, "short_name": "IPS", "team_name": "Ipswich Town"},
        {"team_code": 4, "short_name": "MUN", "team_name": "Man Utd"},
    ])
    import unittest.mock as mock
    with mock.patch.object(historia.team_mapping, "build_registry", return_value=registry):
        puente = historia._puente(["Coventry", "Hull", "Ipswich", "Man United",
                                   "Accrington", "Rochdale"])

    assert puente["Coventry"] == "COV"
    assert puente["Hull"] == "HUL"
    assert puente["Ipswich"] == "IPS"
    assert puente["Man United"] == "MUN"
    # Y los clubes que nunca pasaron por la ventana simplemente no estan: no es un error.
    assert "Accrington" not in puente and "Rochdale" not in puente


def test_el_control_de_equipos_sin_historia_los_detecta():
    from transform import historia

    d = pd.DataFrame({"home_short": ["ARS", None], "away_short": ["CHE", None]})
    assert historia.equipos_sin_historia(d, ["ARS", "CHE", "COV"]) == ["COV"]
    assert historia.equipos_sin_historia(d, ["ARS", "CHE"]) == []
