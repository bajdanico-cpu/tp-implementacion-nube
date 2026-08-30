"""Snapshots de `/event/{GW}/live/` al esquema de `fact_player_gw`.

**Por qué existe este módulo.** El histórico jugador-fecha viene de vaastav, pero vaastav
no sirve en producción: medido sobre 2025-26, tocó `merged_gw.csv` doce veces en toda la
temporada, con un gap mediano de 10 días y máximo de 96. Al día siguiente de la primera
fecha de 2026-27 seguía devolviendo 404.

Sin este módulo, las doce features derivadas de jugadores —xG, xGC, puntajes por línea,
continuidad de plantel, concentración de minutos— llegan **vacías a producción**, que es
exactamente lo que pasó al predecir la fecha 2.

**Y no hace falta esperar a vaastav.** Verificado sobre la GW1 de 2026-27: el endpoint
`/event/{GW}/live/` devuelve `expected_goals`, `expected_assists`,
`expected_goals_conceded` y `total_points` por jugador. La suma de `expected_goals` de la
fecha dio **30,83 contra 30 goles reales**: es xG de verdad, no un placeholder.

Con esto el pipeline **se auto-alimenta el histórico** y vaastav queda sólo para el arranque
en frío de las temporadas viejas — que es lo que el canvas había previsto.

⚠️ El `id` de FPL no sirve como clave entre temporadas (se reasigna todos los años, y el
90 % apunta a otro futbolista). Dentro de una misma temporada sí es válido, que es como se
usa acá: para cruzar el snapshot en vivo con el `bootstrap` del mismo momento. La clave
entre temporadas sigue siendo `player_name`.

⚠️ **El bootstrap que se lee es el CONTEMPORÁNEO de cada fecha, no el último.** El
bootstrap dice a qué club pertenece cada jugador *hoy*; leer el último le atribuiría las
estadísticas de la fecha 1 al club al que el jugador se fue después. Medido sobre la GW1 de
2026-27: 5 de 610 filas quedaban mal atribuidas —Baleba jugó en el Brighton y aparecía como
del United, tres jugadores del City aparecían en Tottenham y Newcastle— y eso ensucia la
historia de los dos clubes, porque las features agregan por `(temporada, fixture, equipo)`.

Se arregla leyendo el snapshot de `bootstrap` vigente al momento del snapshot de
`event_live` de esa fecha. **Bronze es append-only y fechado justamente para esto**: el dato
correcto ya estaba guardado, sólo había que pedir el de la fecha correcta en vez del último.
"""

from __future__ import annotations

import json

import pandas as pd

from common.config import CFG
from common.logging_setup import get_logger
from common.storage import (latest_snapshot, read_raw, read_raw_at,
                            snapshot_stamp)

log = get_logger(__name__)

SOURCE = "fpl"

# `element_type` de FPL -> la posición como la nombra vaastav, para que las dos fuentes
# produzcan exactamente el mismo esquema.
POSICIONES = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# Estadísticas del jugador que trae el endpoint en vivo y que necesita `fact_player_gw`.
STATS = [
    "minutes", "starts", "goals_scored", "assists", "clean_sheets", "goals_conceded",
    "own_goals", "penalties_saved", "penalties_missed", "yellow_cards", "red_cards",
    "saves", "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "influence", "creativity", "threat", "ict_index",
    "total_points", "bonus", "bps",
]


def _cargar(season: str, nombre: str, archivo: str, stamp: str | None = None):
    """El snapshot vigente al momento `stamp`; si no se pasa, el más reciente."""
    crudo = (read_raw(SOURCE, season, nombre, archivo) if stamp is None
             else read_raw_at(SOURCE, season, nombre, archivo, stamp))
    return json.loads(crudo) if crudo else None


def gameweeks_disponibles(season: str) -> list[int]:
    """Qué gameweeks tienen snapshot de `event_live` en Bronze."""
    raiz = CFG.data_root / CFG.raw["storage"]["bronze_dir"] / SOURCE / season
    if not raiz.exists():
        return []
    gws = []
    for d in raiz.iterdir():
        if d.is_dir() and d.name.startswith("event_live_gw"):
            try:
                gw = int(d.name.removeprefix("event_live_gw"))
            except ValueError:
                continue
            if latest_snapshot(SOURCE, season, d.name) is not None:
                gws.append(gw)
    return sorted(gws)


def _dim_jugadores(bootstrap: dict, dim: pd.DataFrame, season: str) -> pd.DataFrame:
    """id de FPL -> nombre, posición y equipo, dentro de ESTA temporada."""
    ids = dim[dim["season"] == season].set_index("fpl_team_id")["short_name"]
    filas = []
    for e in bootstrap["elements"]:
        filas.append({
            "fpl_player_id": e["id"],
            # vaastav usa "Nombre Apellido"; se replica para que la clave entre
            # temporadas (player_name) sea comparable con el histórico.
            "player_name": f"{e['first_name']} {e['second_name']}".strip(),
            "position": POSICIONES.get(e["element_type"]),
            "team_short": ids.get(e["team"]),
        })
    return pd.DataFrame(filas)


def _dim_fixtures(fixtures: list, dim: pd.DataFrame, season: str) -> pd.DataFrame:
    """fixture -> los dos equipos y el kickoff, para saber contra quién jugó cada uno."""
    ids = dim[dim["season"] == season].set_index("fpl_team_id")["short_name"]
    filas = []
    for f in fixtures:
        if f.get("event") is None:
            continue
        filas.append({"fixture_id": f["id"], "gameweek": f["event"],
                      "kickoff_time": f.get("kickoff_time"),
                      "home_short": ids.get(f["team_h"]),
                      "away_short": ids.get(f["team_a"]),
                      "team_h_score": f.get("team_h_score"),
                      "team_a_score": f.get("team_a_score")})
    return pd.DataFrame(filas)


def build(season: str, dim: pd.DataFrame) -> pd.DataFrame | None:
    """Todas las gameweeks con snapshot, en el esquema de `fact_player_gw`."""
    gws = gameweeks_disponibles(season)
    if not gws:
        return None

    fx = _cargar(season, "fixtures", "fixtures.json")
    if fx is None:
        log.warning("[%s] hay event_live pero falta fixtures", season)
        return None
    fixtures = _dim_fixtures(fx, dim, season)

    frames = []
    for gw in gws:
        live = _cargar(season, f"event_live_gw{gw}", f"live_gw{gw}.json")
        if not live or not live.get("elements"):
            log.info("[%s] GW%d: snapshot vacío (normal en pretemporada)", season, gw)
            continue

        # El equipo de cada jugador sale del bootstrap VIGENTE cuando se tomó este
        # snapshot. Con el ultimo, un jugador transferido despues arrastraria sus
        # estadisticas al club nuevo. Ver el aviso del encabezado.
        snap = latest_snapshot(SOURCE, season, f"event_live_gw{gw}")
        boot = _cargar(season, "bootstrap", "bootstrap_static.json",
                       stamp=snapshot_stamp(snap) if snap else None)
        if boot is None:
            log.warning("[%s] GW%d: sin bootstrap contemporaneo, se omite", season, gw)
            continue
        jug = _dim_jugadores(boot, dim, season)

        filas = []
        for e in live["elements"]:
            s = e["stats"]
            # Un jugador puede aparecer en varios fixtures de la misma gameweek en una
            # doble fecha: `explain` dice en cuál(es) jugó.
            fixture_ids = [x["fixture"] for x in e.get("explain", [])] or [None]
            for fid in fixture_ids:
                filas.append({"fpl_player_id": e["id"], "fixture_id": fid,
                              **{c: s.get(c) for c in STATS}})
        d = pd.DataFrame(filas)
        if d.empty:
            continue

        # En una doble fecha `explain` reparte las estadísticas por fixture, pero el
        # bloque `stats` viene agregado de toda la gameweek. Se queda una sola fila por
        # (jugador, gameweek) para no duplicar los totales.
        d = d.drop_duplicates(subset=["fpl_player_id"], keep="first")
        d["gameweek"] = gw
        # El merge con `jug` va ACA, dentro del loop, porque `jug` ahora es distinto
        # para cada fecha: es el plantel tal como estaba en ese momento.
        d = d.merge(jug, on="fpl_player_id", how="left")
        frames.append(d)

    if not frames:
        return None

    fact = pd.concat(frames, ignore_index=True)

    # La API devuelve los xG como STRING ("0.00"), mientras que vaastav los da como
    # float. Sin coaccionar, el parquet falla al escribir y —peor— si llegara a pasar,
    # las dos fuentes tendrian tipos distintos para la misma columna.
    for c in STATS:
        fact[c] = pd.to_numeric(fact[c], errors="coerce")

    fact = fact.merge(fixtures.drop(columns="gameweek"), on="fixture_id", how="left")

    fact["season"] = season
    fact["was_home"] = fact["team_short"] == fact["home_short"]
    fact["opponent_short"] = fact["away_short"].where(fact["was_home"], fact["home_short"])
    fact["kickoff_time"] = pd.to_datetime(fact["kickoff_time"], utc=True, errors="coerce")
    fact["team"] = fact["team_short"]
    fact["round"] = fact["gameweek"]

    sin_equipo = fact["team_short"].isna()
    if sin_equipo.any():
        log.warning("[%s] %d filas sin equipo mapeado, se descartan",
                    season, int(sin_equipo.sum()))
        fact = fact[~sin_equipo]

    # Un jugador transferido A MITAD DE FECHA queda inclasificable: el bloque `explain`
    # del endpoint lo asocia al fixture de su club anterior, y el bootstrap de ese mismo
    # momento ya lo tiene en el nuevo. No hay forma de saber de qué lado estaba, así que
    # `was_home` y el rival saldrían inventados.
    #
    # Se descartan. Son casos de cero minutos --el jugador no llegó a jugar-- así que no
    # se pierde nada; lo que se evita es meter una fila en la historia de un club que no
    # jugó ese partido. Verificado: 1 fila de 114.503.
    ajeno = ~((fact["team_short"] == fact["home_short"]) |
              (fact["team_short"] == fact["away_short"]))
    if ajeno.any():
        detalle = fact.loc[ajeno, ["player_name", "team_short", "gameweek"]]
        log.warning("[%s] %d filas de jugadores transferidos a mitad de fecha "
                    "(su club no jugó ese partido), se descartan: %s",
                    season, int(ajeno.sum()),
                    detalle.to_dict("records")[:5])
        fact = fact[~ajeno]

    fact = fact[fact["position"].notna()]
    con_min = int((fact["minutes"] > 0).sum())
    log.info("[%s] event_live -> %d filas de jugador en %d gameweeks (%d con minutos)",
             season, len(fact), len(gws), con_min)
    return fact.drop(columns=["home_short", "away_short"])
