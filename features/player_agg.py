"""De `fact_player_gw` (jugador x fecha) a estadísticas de equipo x partido.

Este módulo es el que resuelve la pregunta de los pases: **agrega por fixture ANTES de
calcular cualquier ventana**. Como el paso de agregación es por partido, cada jugador
queda atribuido al equipo con el que efectivamente jugó ese día. Si un delantero pasa de
A a B en enero, sus goles de agosto quedan para siempre en el historial de A y los de
febrero suman al de B: ni doble conteo ni puntos huérfanos. Un jugador nuevo suma a su
equipo desde su primer partido, sin necesitar historial propio.

Dos salidas, porque son dos cosas distintas:

- `team_stats_by_fixture` — las estadísticas que se agregan dentro de UN partido
  (xG, xA, puntos por línea). Grano equipo x fixture, 3.040 filas.
- `plantel_por_ventana` — las que necesitan mirar VARIOS partidos a la vez
  (continuidad de plantel, concentración de minutos). No se pueden derivar promediando
  una columna por partido, porque hay que seguir a los jugadores entre partidos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common.config import CFG
from common.logging_setup import get_logger

log = get_logger(__name__)

VENTANA_PLANTEL = 5

# Las cuatro líneas de FPL. Los `AM` (que son DT, del chip Assistant Manager) ya los
# excluye Silver, pero se filtra explícito por si eso cambia.
POSICIONES = {"GK": "arq", "DEF": "def", "MID": "med", "FWD": "del"}

# Columnas que produce team_stats_by_fixture, además de las claves.
STATS_JUGADOR = ["xg", "xa", "xgc", "pts_arq", "pts_def", "pts_med", "pts_del",
                 "n_jugadores", "atajadas", "tasa_atajadas"]


def team_stats_by_fixture(players: pd.DataFrame) -> pd.DataFrame:
    """Agrega jugadores a equipo x fixture. Devuelve 3.040 filas para la ventana.

    El xGC merece explicación: NO es `sum(expected_goals_conceded)`. Ese campo mide el
    xG concedido por el equipo *mientras cada jugador estuvo en cancha*, así que sumarlo
    lo multiplica por la cantidad de jugadores del plantel. Medido sobre 2024-25, la suma
    da media 15,75 contra 1,47 goles concedidos reales: inflado x11. El xGC correcto es
    **el xG del rival en el mismo fixture** (media 1,44, que sí calza).
    """
    p = players[players["position"].isin(POSICIONES)].copy()

    claves = ["season", "gameweek", "fixture_id", "team_short"]
    base = p.groupby(claves, as_index=False).agg(
        xg=("expected_goals", "sum"),
        xa=("expected_assists", "sum"),
        n_jugadores=("minutes", lambda s: int((s > 0).sum())),
        atajadas=("saves", "sum"),
        gc_equipo=("goals_conceded", "max"),
    )

    # TASA DE ATAJADAS: el equivalente defensivo de `xg_por_tiro`.
    #
    # `xg_por_tiro` mide la CALIDAD de lo que generas; esto mide cuanto de lo que te
    # llega termina adentro. saves / (saves + goles) es la proporcion de remates al arco
    # que el arquero detiene: separa "concede poco" de "concede mucho pero lo atajan".
    #
    # Son cosas distintas y envejecen distinto: conceder pocos remates es estructural del
    # equipo y persiste; una tasa de atajadas alta es en buena parte varianza del arquero
    # y revierte a la media. El modelo puede aprender a descontarla.
    tiros_al_arco = base["atajadas"] + base["gc_equipo"]
    base["tasa_atajadas"] = base["atajadas"] / tiros_al_arco.replace(0, np.nan)
    base = base.drop(columns="gc_equipo")

    # Puntos por línea: el pedido explícito del bloque 4 del canvas.
    #
    # `total_points` está en banned_columns, y esto no lo viola: la prohibición es sobre
    # la columna cruda del partido a predecir, donde es el resultado. Acá se agrega por
    # línea sobre partidos que después sólo se leen HACIA ATRÁS (merge_asof), así que es
    # una feature de forma. Los nombres finales (`pts_def`) tampoco colisionan.
    por_linea = (
        p.groupby(claves + ["position"], as_index=False)["total_points"].sum()
         .pivot(index=claves, columns="position", values="total_points")
         .fillna(0.0)
         .rename(columns={k: f"pts_{v}" for k, v in POSICIONES.items()})
         .reset_index()
    )
    for col in (f"pts_{v}" for v in POSICIONES.values()):
        if col not in por_linea:
            por_linea[col] = 0.0

    out = base.merge(por_linea, on=claves, how="left", validate="one_to_one")

    # xGC = el xG del rival en el mismo fixture.
    rival = out[["season", "fixture_id", "team_short", "xg"]].rename(
        columns={"team_short": "rival_short", "xg": "xgc"})
    out = out.merge(rival, on=["season", "fixture_id"], how="left")
    out = out[out["team_short"] != out["rival_short"]].drop(columns="rival_short")

    out = _enmascarar_xg_falso(out)
    return out.reset_index(drop=True)


def _enmascarar_xg_falso(df: pd.DataFrame) -> pd.DataFrame:
    """El xG de 2022-23 viene HARDCODEADO EN CERO hasta la GW15.

    No es dato faltante: viene 0,0 para los 20 equipos en las GW 1-6 y 8-15 (la 7 no
    existe). Es el 37,9 % de esa temporada. Dejarlo como cero le enseña al modelo que
    "xG bajo" y "arranque de temporada" van juntos, que es un artefacto del calendario de
    publicación de FPL y no una propiedad del fútbol.

    Se enmascara a NaN — XGBoost aprende una dirección por defecto para el faltante — y
    se deja `xg_available` para que el modelo pueda distinguir "no hay dato" de "poco xG".
    """
    df = df.copy()
    df["xg_available"] = True
    for season, min_gw in CFG.xg_min_gameweek.items():
        mask = (df["season"] == season) & (df["gameweek"] < min_gw)
        if not mask.any():
            continue
        df.loc[mask, ["xg", "xa", "xgc"]] = np.nan
        df.loc[mask, "xg_available"] = False
        log.info("xG enmascarado a NaN: %s GW<%d (%d filas equipo-partido)",
                 season, min_gw, int(mask.sum()))
    return df


def plantel_por_ventana(players: pd.DataFrame, orden: pd.DataFrame,
                        ventana: int = VENTANA_PLANTEL) -> pd.DataFrame:
    """Continuidad de plantel y concentración de minutos, sobre los últimos N partidos.

    Estas dos no se pueden calcular promediando una columna por partido: hay que seguir a
    los jugadores *entre* partidos. Por eso van aparte.

    - `continuidad_plantel` — qué proporción de los minutos de la ventana la jugaron
      futbolistas que también jugaron el partido más reciente de la ventana. Se desploma
      cuando el plantel se renovó, que es justo cuando la forma pasada deja de representar
      al equipo de hoy. Medido: entre temporadas rota ~40 % de los minutos.
    - `mins_hhi` — índice de Herfindahl del reparto de minutos. Alto = once fijo,
      bajo = mucha rotación. Es el proxy que reemplaza al grupo "disponibilidad" del
      canvas, que no se puede construir porque la API de FPL no sirve el `status`
      histórico.

    `orden` trae, por (season, fixture_id, team_short), el índice `k` del partido dentro
    de la secuencia cronológica del equipo. Cruza temporadas a propósito.

    ⚠️ La clave de jugador es `player_name`, NUNCA `fpl_player_id`: FPL reasigna los ids
    cada temporada y el 90 % apunta a un futbolista distinto según el año.
    """
    p = players.loc[players["minutes"] > 0,
                    ["season", "fixture_id", "team_short", "player_name", "minutes"]]
    pm = p.merge(orden, on=["season", "fixture_id", "team_short"], validate="many_to_one")

    # Cada partido contribuye a las `ventana` ventanas que lo contienen. Replicar con un
    # offset es equivalente al rolling y queda totalmente vectorizado.
    partes = []
    for off in range(ventana):
        q = pm[["team_short", "k", "player_name", "minutes"]].copy()
        q["k_obj"] = q["k"] + off
        partes.append(q)
    w = pd.concat(partes, ignore_index=True)
    w = w[w["k_obj"] <= w.groupby("team_short")["k"].transform("max")]

    # Minutos de cada jugador dentro de cada ventana.
    porjug = w.groupby(["team_short", "k_obj", "player_name"], as_index=False)["minutes"].sum()
    tot = porjug.groupby(["team_short", "k_obj"], as_index=False)["minutes"].sum() \
                .rename(columns={"minutes": "mins_total"})
    porjug = porjug.merge(tot, on=["team_short", "k_obj"], validate="many_to_one")
    porjug["share"] = porjug["minutes"] / porjug["mins_total"]

    hhi = porjug.assign(sq=lambda d: d["share"] ** 2) \
                .groupby(["team_short", "k_obj"], as_index=False)["sq"].sum() \
                .rename(columns={"sq": "mins_hhi"})

    # Continuidad: los que jugaron el partido MÁS RECIENTE de la ventana (k == k_obj).
    ultimos = pm[["team_short", "k", "player_name"]].rename(columns={"k": "k_obj"})
    ultimos = ultimos.drop_duplicates().assign(en_ultimo=True)
    porjug = porjug.merge(ultimos, on=["team_short", "k_obj", "player_name"], how="left")
    porjug["en_ultimo"] = porjug["en_ultimo"].fillna(False)
    cont = (porjug[porjug["en_ultimo"]]
            .groupby(["team_short", "k_obj"], as_index=False)["share"].sum()
            .rename(columns={"share": "continuidad_plantel"}))

    out = (orden.merge(hhi, left_on=["team_short", "k"], right_on=["team_short", "k_obj"],
                       how="left")
                .drop(columns="k_obj")
                .merge(cont, left_on=["team_short", "k"], right_on=["team_short", "k_obj"],
                       how="left")
                .drop(columns="k_obj"))
    return out[["season", "fixture_id", "team_short", "mins_hhi", "continuidad_plantel"]]


def report() -> None:
    """Chequeo manual: `python -m features.player_agg`."""
    from common.storage import read_table

    players = read_table("fact_player_gw")
    stats = team_stats_by_fixture(players)
    print(f"team_stats_by_fixture: {len(stats)} filas (esperado 3.040)")
    print(stats[["xg", "xgc", "pts_arq", "pts_def", "pts_med", "pts_del", "n_jugadores"]]
          .describe().round(2).T.to_string())
    print()
    print("xg_available por temporada:")
    print(stats.groupby("season")["xg_available"].mean().round(3).to_string())


if __name__ == "__main__":
    from common.logging_setup import setup

    setup(CFG.log_level, CFG.log_format)
    report()
