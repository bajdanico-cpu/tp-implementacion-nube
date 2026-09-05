"""Tests del valor de plantel (Fase 5) — la primera fuente con información de afuera.

Las dos que importan:

**Anti-leakage.** Una valuación publicada DESPUÉS del corte no puede entrar. Es una fuente
nueva y el proyecto ya sabe cómo se cuelan estas cosas (`xP`, las cuotas de cierre): el
control va en el código que genera el dato, no en un test que corre después — pero el test
también está, con el caso construido a mano.

**El club sale de la valuación, no del jugador.** `players.current_club_id` dice dónde está
hoy; usarlo atribuiría todo el pasado al club equivocado. Cada valuación trae su propio
club, y el test verifica que un pase mueve el valor de un plantel al otro en la fecha justa.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features import valores as fval


def _valores(filas: list[tuple]) -> pd.DataFrame:
    """`(player_id, equipo, linea, desde, hasta, valor)`."""
    return pd.DataFrame([
        {"player_id": p, "team_short": e, "linea": l,
         "desde": pd.Timestamp(d), "hasta": pd.Timestamp(h), "valor_eur": float(v)}
        for p, e, l, d, h, v in filas])


def _obj(cortes: list[tuple]) -> pd.DataFrame:
    """`(season, gameweek, fixture_id, lado, equipo, corte)`."""
    return pd.DataFrame([
        {"season": s, "gameweek": gw, "fixture_id": f, "lado": lado,
         "team_short": e, "corte": pd.Timestamp(c, tz="UTC")}
        for s, gw, f, lado, e, c in cortes])


# ---------------------------------------------------------------------------
# PRUEBA DE FUEGO: el corte manda
# ---------------------------------------------------------------------------

def test_prueba_de_fuego_una_valuacion_posterior_al_corte_no_entra():
    """El caso construido a mano: al jugador lo revaluaron DESPUES del partido.

    Si el intervalo se eligiera por cercania en vez de por contencion, el valor de octubre
    entraria en un partido de septiembre y la feature sabria el futuro.
    """
    v = _valores([
        (1, "ARS", "del", "2024-06-01", "2024-10-01", 50e6),    # vigente al corte
        (1, "ARS", "del", "2024-10-01", "2099-12-31", 90e6),    # posterior: NO
    ])
    obj = _obj([("2024-25", 5, 1, "local", "ARS", "2024-09-15")])
    out = fval.construir(obj, v)
    assert out["valor_plantel"].iloc[0] == pytest.approx(50e6)


def test_el_intervalo_se_elige_por_contencion_no_por_cercania():
    v = _valores([(1, "ARS", "del", "2024-01-01", "2024-02-01", 10e6),
                  (1, "ARS", "del", "2024-02-01", "2024-12-01", 20e6)])
    obj = _obj([("2024-25", 1, 1, "local", "ARS", "2024-11-30")])
    # El corte esta MUCHO mas cerca del final del segundo intervalo que de su inicio.
    assert fval.construir(obj, v)["valor_plantel"].iloc[0] == pytest.approx(20e6)


def test_el_pase_mueve_el_valor_de_un_plantel_al_otro():
    """El club sale de la valuacion. Antes del pase suma en ARS, despues en CHE."""
    v = _valores([(1, "ARS", "del", "2024-01-01", "2024-07-01", 40e6),
                  (1, "CHE", "del", "2024-07-01", "2099-12-31", 45e6)])
    antes = fval.construir(_obj([("2024-25", 1, 1, "local", "ARS", "2024-05-01"),
                                 ("2024-25", 1, 1, "visita", "CHE", "2024-05-01")]), v)
    despues = fval.construir(_obj([("2024-25", 9, 2, "local", "ARS", "2024-09-01"),
                                   ("2024-25", 9, 2, "visita", "CHE", "2024-09-01")]), v)
    assert antes.set_index("lado").loc["local", "valor_plantel"] == pytest.approx(40e6)
    assert pd.isna(antes.set_index("lado").loc["visita", "valor_plantel"])
    assert pd.isna(despues.set_index("lado").loc["local", "valor_plantel"])
    assert despues.set_index("lado").loc["visita", "valor_plantel"] == pytest.approx(45e6)


# ---------------------------------------------------------------------------
# Las agregaciones
# ---------------------------------------------------------------------------

def test_el_valor_por_linea_suma_el_total():
    v = _valores([(1, "ARS", "arq", "2024-01-01", "2099-12-31", 10e6),
                  (2, "ARS", "def", "2024-01-01", "2099-12-31", 20e6),
                  (3, "ARS", "med", "2024-01-01", "2099-12-31", 30e6),
                  (4, "ARS", "del", "2024-01-01", "2099-12-31", 40e6)])
    r = fval.construir(_obj([("2024-25", 1, 1, "local", "ARS", "2024-06-01")]), v).iloc[0]
    assert r["valor_plantel"] == pytest.approx(100e6)
    assert sum(r[f"valor_{l}"] for l in fval.LINEAS) == pytest.approx(100e6)
    assert r["valor_n"] == 4


def test_el_top11_toma_los_once_mas_caros():
    v = _valores([(i, "ARS", "med", "2024-01-01", "2099-12-31", (i + 1) * 1e6)
                  for i in range(20)])
    r = fval.construir(_obj([("2024-25", 1, 1, "local", "ARS", "2024-06-01")]), v).iloc[0]
    esperado = sum(range(10, 21)) * 1e6      # los valores 10..20 millones
    assert r["valor_top11"] == pytest.approx(esperado)
    assert r["valor_top11"] < r["valor_plantel"]


def test_el_valor_relativo_suma_uno_entre_los_equipos_del_corte():
    """La normalizacion que hace la feature comparable entre temporadas: el valor nominal
    de los planteles sube todos los años y el crudo deja al arbol reconocer la temporada."""
    v = _valores([(1, "ARS", "del", "2024-01-01", "2099-12-31", 75e6),
                  (2, "CHE", "del", "2024-01-01", "2099-12-31", 25e6)])
    out = fval.construir(_obj([("2024-25", 1, 1, "local", "ARS", "2024-06-01"),
                               ("2024-25", 1, 1, "visita", "CHE", "2024-06-01")]), v)
    assert out["valor_rel"].sum() == pytest.approx(1.0)
    assert out.set_index("lado").loc["local", "valor_rel"] == pytest.approx(0.75)


def test_un_equipo_sin_valuaciones_vigentes_queda_en_nan_y_no_en_cero():
    """Cero significa "vale cero", NaN significa "no sabemos". XGBoost aprende una
    direccion para el faltante; un cero inventado le enseña algo falso."""
    v = _valores([(1, "ARS", "del", "2024-01-01", "2099-12-31", 50e6)])
    out = fval.construir(_obj([("2024-25", 1, 1, "local", "ARS", "2024-06-01"),
                               ("2024-25", 1, 1, "visita", "COV", "2024-06-01")]), v)
    assert pd.isna(out.set_index("lado").loc["visita", "valor_plantel"])


def test_la_linea_faltante_suma_cero_pero_el_equipo_existe():
    """Distinto del caso anterior: el equipo TIENE plantel, lo que no tiene es arqueros
    valuados. Ahi el cero es correcto: la linea aporta cero al total."""
    v = _valores([(1, "ARS", "del", "2024-01-01", "2099-12-31", 50e6)])
    r = fval.construir(_obj([("2024-25", 1, 1, "local", "ARS", "2024-06-01")]), v).iloc[0]
    assert r["valor_arq"] == 0.0
    assert r["valor_del"] == pytest.approx(50e6)


def test_falla_claro_si_ningun_corte_cae_en_un_intervalo():
    v = _valores([(1, "ARS", "del", "2024-01-01", "2024-02-01", 50e6)])
    obj = _obj([("2024-25", 1, 1, "local", "ARS", "2030-06-01")])
    with pytest.raises(ValueError, match="Ningun corte"):
        fval.construir(obj, v)


# ---------------------------------------------------------------------------
# El mapeo de clubes, donde ya nos comimos un bug
# ---------------------------------------------------------------------------

def test_los_alias_de_transfermarkt_cubren_los_nombres_oficiales():
    """Once de los veintisiete equipos no resuelven sin el mapa: Transfermarkt usa el
    nombre oficial completo y la normalizacion no puede con `Manchester` vs `Man` ni con
    `Hotspur`."""
    from transform import team_mapping as tm

    for nombre, corto in (("Manchester City", "MCI"), ("Manchester United", "MUN"),
                          ("Tottenham Hotspur", "TOT"), ("Brighton & Hove Albion", "BHA"),
                          ("Wolverhampton Wanderers", "WOL"), ("AFC Bournemouth", "BOU")):
        assert tm.TM_ALIASES[nombre] == corto


def test_los_filiales_no_estan_en_el_mapa_de_alias():
    """`Manchester City U21`, `... Reserves` y sobre todo `Newcastle United Jets` --que es
    un club AUSTRALIANO-- no pueden entrar al plantel. El cruce va por nombre exacto."""
    from transform import team_mapping as tm

    for impostor in ("Manchester City U21", "Manchester United Reserves",
                     "Newcastle United Jets", "Tottenham Hotspur U23"):
        assert impostor not in tm.TM_ALIASES
