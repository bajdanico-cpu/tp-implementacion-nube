"""Cuándo una fecha se puede predecir, y cuándo su fila deja de moverse.

Todo sintético y sin Silver: el calendario es lógica de fechas pura y se puede probar
sobre casos que en los datos reales aparecen una vez por temporada, o ninguna — una fecha
en curso, un partido postergado, el final de la temporada.

Son los bordes que decidían mal antes de que este módulo existiera: pedir la fecha 7
estando en la 5 devolvía diez predicciones calculadas con la información de la 4, sin que
nada lo dijera.
"""

from __future__ import annotations

import pandas as pd
import pytest

from features import calendario as cal

SEASON = "2099-00"
EQUIPOS = [("AAA", "BBB"), ("CCC", "DDD"), ("EEE", "FFF")]


def _fixtures(n_gw: int = 4, por_gw: int = 3) -> pd.DataFrame:
    """Un calendario de juguete: una fecha por semana, todas el mismo día."""
    filas, fid = [], 0
    for gw in range(1, n_gw + 1):
        dia = pd.Timestamp("2099-08-01", tz="UTC") + pd.Timedelta(days=7 * (gw - 1))
        for i, (loc, vis) in enumerate(EQUIPOS[:por_gw]):
            fid += 1
            ko = dia + pd.Timedelta(hours=12 + i)
            filas.append({"season": SEASON, "gameweek": gw, "fixture_id": fid,
                          "match_date": ko.normalize().tz_localize(None),
                          "kickoff_time": ko, "home_short": loc, "away_short": vis})
    return pd.DataFrame(filas)


def _matches(fixtures: pd.DataFrame, hasta_gw: int = 0,
             saltear: set[int] | None = None) -> pd.DataFrame:
    """Los resultados de las fechas 1..`hasta_gw`, menos los fixture_id de `saltear`."""
    saltear = saltear or set()
    d = fixtures[(fixtures["gameweek"] <= hasta_gw)
                 & (~fixtures["fixture_id"].isin(saltear))]
    return d[cal.CLAVE_FIXTURE].copy()


AHORA = pd.Timestamp("2099-08-15 10:00", tz="UTC")   # entre la GW3 y la GW4


# ---------------------------------------------------------------------------
# Definitividad
# ---------------------------------------------------------------------------

def test_una_fecha_sin_partidos_previos_pendientes_es_definitiva():
    fx = _fixtures()
    e = cal.estado(fx, _matches(fx, hasta_gw=2), SEASON)
    assert bool(e.loc[e["gameweek"] == 3, "definitiva"].iloc[0])


def test_un_partido_previo_sin_jugar_la_vuelve_no_definitiva():
    """Un solo partido de la fecha anterior sin ingestar alcanza.

    No es quisquilloso: ese partido entra en las ventanas rolling de los dos equipos que
    lo jugaron, así que su ausencia cambia las features de la fecha siguiente.
    """
    fx = _fixtures()
    fm = _matches(fx, hasta_gw=2, saltear={4})          # falta un partido de la GW2
    e = cal.estado(fx, fm, SEASON, ahora=AHORA)
    assert not bool(e.loc[e["gameweek"] == 3, "definitiva"].iloc[0])
    # Y la próxima pasa a ser la 2, que es la que todavía tiene algo por jugar — no la 3,
    # que ya no se puede calcular bien.
    assert cal.proxima_predecible(fx, fm, SEASON, ahora=AHORA) == 2


def test_una_fecha_en_curso_sigue_siendo_definitiva():
    """El borde que se resuelve solo, y por eso conviene tener un test que lo fije.

    Los partidos de la propia fecha tienen kickoff >= corte, así que no entran en el
    predicado de definitividad. Una fecha empezada mantiene su fila: los que ya se
    jugaron entran con target y los que faltan como inferencia, compartiendo corte.
    """
    fx = _fixtures()
    fm = _matches(fx, hasta_gw=2)
    fm = pd.concat([fm, fx[fx["fixture_id"] == 7][cal.CLAVE_FIXTURE]])   # 1 de 3 de la GW3
    e = cal.estado(fx, fm, SEASON)
    fila = e[e["gameweek"] == 3].iloc[0]
    assert fila["situacion"] == "en_curso"
    assert bool(fila["definitiva"])
    assert cal.proxima_predecible(fx, fm, SEASON, ahora=AHORA) == 3


def test_la_gw1_de_una_temporada_nueva_es_definitiva():
    fx = _fixtures()
    e = cal.estado(fx, _matches(fx, hasta_gw=0), SEASON)
    fila = e[e["gameweek"] == 1].iloc[0]
    assert fila["n_previos"] == 0 and bool(fila["definitiva"])


# ---------------------------------------------------------------------------
# Cuál es la próxima
# ---------------------------------------------------------------------------

def test_la_proxima_es_la_primera_con_partidos_sin_jugar():
    fx = _fixtures()
    fm = _matches(fx, hasta_gw=2)
    assert cal.proxima_predecible(fx, fm, SEASON, ahora=AHORA) == 3


def test_un_partido_postergado_no_traba_la_proxima_fecha():
    """Sin la gracia, un partido suspendido congela la predicción para siempre."""
    fx = _fixtures()
    # Se jugó todo hasta la GW3 salvo un partido de la GW1, suspendido hace semanas.
    fm = _matches(fx, hasta_gw=3, saltear={1})
    assert cal.proxima_predecible(fx, fm, SEASON, ahora=AHORA) == 4


def test_sin_fixtures_pendientes_no_hay_proxima_fecha():
    """Fin de temporada: se contesta `None`, no se explota ni se inventa una fecha."""
    fx = _fixtures()
    fm = _matches(fx, hasta_gw=4)
    assert cal.proxima_predecible(fx, fm, SEASON, ahora=AHORA) is None
    assert cal.objetivos_inferencia(fx, fm, season=SEASON, ahora=AHORA).empty


def test_una_temporada_que_no_existe_no_rompe():
    fx = _fixtures()
    assert cal.estado(fx, _matches(fx, 2), "1800-01").empty
    assert cal.proxima_predecible(fx, _matches(fx, 2), "1800-01") is None


# ---------------------------------------------------------------------------
# Opta: la degradación que no falla sola
# ---------------------------------------------------------------------------

def _comp_y_stats(fixtures: pd.DataFrame, hasta_gw: int, faltan: int = 0):
    """`fact_match_comp` y `fact_opta_stats` de juguete, con `faltan` partidos sin stats."""
    d = fixtures[fixtures["gameweek"] <= hasta_gw].reset_index(drop=True)
    comp = pd.DataFrame({
        "season": d["season"], "fixture_pl_id": d["fixture_id"],
        "team_id_pl": d["fixture_id"] * 10, "es_premier": True,
        "kickoff_time": d["kickoff_time"], "terminado": True,
    })
    stats = comp[["season", "fixture_pl_id", "team_id_pl"]].iloc[faltan:].copy()
    return comp, stats


def test_sin_opta_de_la_fecha_anterior_la_proxima_no_es_definitiva():
    """`features/opta.py` rueda sobre fixtures jugados y NO jugados, con min_periods=1.

    Si falta la ingesta de un partido anterior al corte, el merge_asof no falla: devuelve
    un promedio calculado sobre menos partidos. Es el tipo de degradación que sólo se ve
    comparando dos corridas, así que se chequea antes.
    """
    fx = _fixtures()
    fm = _matches(fx, hasta_gw=2)
    comp, stats = _comp_y_stats(fx, hasta_gw=2, faltan=1)
    assert cal.proxima_predecible(fx, fm, SEASON, ahora=AHORA,
                                  comp=comp, stats=stats) is None


def test_con_opta_completa_la_proxima_es_definitiva():
    fx = _fixtures()
    fm = _matches(fx, hasta_gw=2)
    comp, stats = _comp_y_stats(fx, hasta_gw=2, faltan=0)
    assert cal.proxima_predecible(fx, fm, SEASON, ahora=AHORA,
                                  comp=comp, stats=stats) == 3


def test_sin_opta_ingestada_la_cobertura_no_bloquea():
    """Gold funciona sin Opta; la cobertura sólo puede bloquear si hay algo que cubrir."""
    fx = _fixtures()
    assert cal.cobertura_opta(None, None, SEASON, AHORA) == 1.0


# ---------------------------------------------------------------------------
# Los objetivos que van a Gold
# ---------------------------------------------------------------------------

def test_los_objetivos_tienen_la_forma_que_espera_gold():
    fx = _fixtures()
    obj = cal.objetivos_inferencia(fx, _matches(fx, 2), season=SEASON, ahora=AHORA)
    assert list(obj.columns) == cal.OBJETIVO_COLS
    assert len(obj) == 3
    assert (obj["gameweek"] == 3).all()


def test_todos_los_objetivos_comparten_el_corte_de_su_fecha():
    """El corte es una propiedad del calendario, no del reloj ni del partido."""
    fx = _fixtures()
    obj = cal.objetivos_inferencia(fx, _matches(fx, 2), season=SEASON, ahora=AHORA)
    esperado = fx.loc[fx["gameweek"] == 3, "kickoff_time"].min()
    assert obj["corte"].nunique() == 1
    assert obj["corte"].iloc[0] == esperado


def test_de_una_fecha_en_curso_solo_van_los_partidos_sin_jugar():
    fx = _fixtures()
    fm = pd.concat([_matches(fx, hasta_gw=2), fx[fx["fixture_id"] == 7][cal.CLAVE_FIXTURE]])
    obj = cal.objetivos_inferencia(fx, fm, season=SEASON, ahora=AHORA)
    assert len(obj) == 2
    assert 7 not in set(obj["fixture_id"])


# ---------------------------------------------------------------------------
# Contra los datos reales
# ---------------------------------------------------------------------------

def test_la_proxima_real_no_tiene_ningun_partido_jugado(fact_fixture, fact_match):
    """Sobre Silver de verdad: si la próxima tuviera partidos jugados, predeciríamos
    algo que ya pasó."""
    from common.config import CFG

    gw = cal.proxima_predecible(fact_fixture, fact_match, CFG.current_season)
    if gw is None:
        pytest.skip("No hay fecha predecible en la temporada en curso.")
    e = cal.estado(fact_fixture, fact_match, CFG.current_season)
    fila = e[e["gameweek"] == gw].iloc[0]
    assert fila["n_jugados"] < fila["n"]
    assert bool(fila["definitiva"])
