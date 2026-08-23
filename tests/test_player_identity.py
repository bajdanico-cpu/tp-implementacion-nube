"""La identidad de un jugador entre temporadas.

Existe por un hallazgo que casi arruina el cálculo de continuidad de plantel: **FPL
reasigna los ids de jugador todos los años**, igual que hace con los ids de equipo.
Usando `fpl_player_id` como clave, la continuidad de plantel entre temporadas daba 9,4 %
—un número absurdo para la Premier— en vez del 61,3 % real.

Estos tests fijan el hecho medido para que nadie use el id como clave sin enterarse. Es
**bloqueante para el Gold-FPL** del otro proyecto, que sí sigue jugadores individuales.
"""

from __future__ import annotations

import pytest


def _mapa_por_temporada(players, clave: str):
    """Para cada valor de la clave, qué nombres distintos tuvo según la temporada."""
    d = players[players["minutes"] > 0]
    return (d.groupby([clave, "season"])["player_name"]
             .agg(lambda s: s.mode().iloc[0])
             .unstack())


def test_el_id_de_jugador_de_fpl_no_es_estable_entre_temporadas(fact_player_gw):
    """Medido: el 90 % de los ids apunta a un futbolista distinto según el año.

    El id 1 es Cédric Soares en 2022-23 y David Raya en 2025-26.
    """
    chk = _mapa_por_temporada(fact_player_gw, "fpl_player_id")
    inconsistentes = chk.apply(lambda r: r.dropna().nunique() > 1, axis=1)

    assert inconsistentes.mean() > 0.5, (
        "Si este test empieza a fallar, FPL cambió su política de ids: reevaluar si "
        "`fpl_player_id` ya sirve como clave entre temporadas.")


def test_player_name_si_identifica_al_jugador_entre_temporadas(fact_player_gw):
    """El nombre es la clave correcta: un nombre, un futbolista."""
    d = fact_player_gw[fact_player_gw["minutes"] > 0]
    por_nombre = d.groupby("player_name")["team_short"].nunique()

    # Los pases existen, pero son una minoría: ~14 % cambió de equipo en la ventana.
    assert por_nombre.mean() < 1.5
    assert (por_nombre == 1).mean() > 0.75


def test_la_continuidad_de_plantel_es_creible_con_el_nombre(fact_player_gw):
    """Con `player_name` da 57-66 %; con el id daba 9 %, que es imposible en la Premier."""
    d = fact_player_gw[fact_player_gw["minutes"] > 0]
    mins = d.groupby(["season", "team_short", "player_name"], as_index=False)["minutes"].sum()
    temporadas = sorted(mins["season"].unique())

    for previa, actual in zip(temporadas, temporadas[1:]):
        antes = set(map(tuple, mins[mins.season == previa][["team_short", "player_name"]].values))
        cur = mins[mins.season == actual].copy()
        cur["sigue"] = [(t, n) in antes for t, n in cur[["team_short", "player_name"]].values]
        share = cur[cur.sigue]["minutes"].sum() / cur["minutes"].sum()
        assert 0.4 < share < 0.9, (
            f"{previa} -> {actual}: continuidad {share:.1%}, fuera del rango plausible")


def test_el_id_de_equipo_tampoco_sirve(fact_player_gw):
    """El mismo problema que ya está documentado para los equipos; se deja fijado acá."""
    chk = (fact_player_gw.groupby(["opponent_team", "season"])["opponent_short"]
                         .agg(lambda s: s.mode().iloc[0]).unstack())
    inconsistentes = chk.apply(lambda r: r.dropna().nunique() > 1, axis=1)
    assert inconsistentes.any(), "el id numérico de equipo debería ser inestable también"
