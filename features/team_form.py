"""Tabla larga equipo x partido, ventanas rodantes y el corte anti-leakage.

Acá vive la regla que sostiene todo el proyecto:

    corte(partido) = min(kickoff_time) de todos los partidos de su (temporada, gameweek)

Toda feature usa **únicamente partidos terminados antes del corte**. El ancla es el inicio
de la fecha, así que todos los partidos de una misma gameweek se predicen con la misma
información.

El mecanismo es `merge_asof`, no `shift(1)`. La diferencia importa:

    shift cuenta PARTIDOS.  merge_asof cuenta TIEMPO.

Hay 85 pares (temporada, gameweek, equipo) donde el equipo juega dos veces en la misma
fecha. Con `shift(1)`, el segundo partido usaría el resultado del primero — que se jugó
DESPUÉS del corte de esa fecha. Es leakage silencioso en ~5,6 % de las filas. Con
`merge_asof` anclado al corte, los dos partidos comparten el mismo vector de features, que
es lo correcto: se predicen en el mismo momento con la misma información. De paso resuelve
sin código especial los partidos reprogramados de la GW7 de 2022-23, que no existe.

La estrategia es en dos pasos:

1. Se calcula la ventana INCLUSIVA (sin shift) y se la tagea con el `kickoff_time` del
   partido que la produjo.
2. `merge_asof(..., direction="backward", allow_exact_matches=False)` elige, para cada
   objetivo, el último estado conocido ESTRICTAMENTE anterior a su corte.

`allow_exact_matches=False` es el mismo "estrictamente anterior" que ya usa
`transform/leakage.py` (`violating = ts >= deadline`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common.config import CFG
from common.logging_setup import get_logger
from features import player_agg, spec

log = get_logger(__name__)

# Estadísticas que salen de fact_match (las de fact_player_gw las trae player_agg).
STATS_PARTIDO = ["pts", "gf", "gc", "tiros", "tiros_arco", "corners", "faltas", "tarjetas"]

BASE = [s.nombre for s in spec.BASE_STATS]

CLAVE_LARGO = ["season", "fixture_id", "team_short"]

# Techo de los contadores acumulados, para que no crezcan sin limite con el tiempo.
# 38 = una temporada completa: a partir de ahi "cuanta historia tiene" ya no discrimina.
TECHO_N_HIST = 38


# ---------------------------------------------------------------------------
# 1 · El corte
# ---------------------------------------------------------------------------

def cortes_por_fecha(fixtures: pd.DataFrame) -> pd.DataFrame:
    """(season, gameweek) -> inicio de la fecha = min(kickoff_time) de esa gameweek."""
    out = (fixtures.groupby(["season", "gameweek"], as_index=False)["kickoff_time"]
                   .min().rename(columns={"kickoff_time": "corte"}))
    return out


# ---------------------------------------------------------------------------
# 2 · La tabla larga: un equipo, un partido, una fila
# ---------------------------------------------------------------------------

def construir_largo(matches: pd.DataFrame, fixtures: pd.DataFrame,
                    stats_jug: pd.DataFrame) -> pd.DataFrame:
    """3.040 filas = 4 temporadas x 380 partidos x 2 equipos.

    Es la forma que hace triviales las ventanas y evita duplicar la lógica para el local
    y el visitante: cada equipo tiene su propia fila, con sus propios números.
    """
    fx = fixtures[["season", "match_date", "home_short", "away_short", "gameweek",
                   "fixture_id", "kickoff_time", "team_h_difficulty", "team_a_difficulty"]]
    m = matches.merge(fx, on=["season", "match_date", "home_short", "away_short"],
                      how="inner", validate="one_to_one")
    if len(m) != len(matches):
        raise ValueError(f"El cruce match-fixture perdió filas: {len(matches)} -> {len(m)}")

    lados = []
    for lado, yo, rival in (("local", "home", "away"), ("visita", "away", "home")):
        d = pd.DataFrame({
            "season": m["season"],
            "gameweek": m["gameweek"],
            "fixture_id": m["fixture_id"],
            "kickoff_time": m["kickoff_time"],
            "team_short": m[f"{yo}_short"],
            "rival_short": m[f"{rival}_short"],
            "es_local": lado == "local",
            "gf": m[f"{yo}_goals"],
            "gc": m[f"{rival}_goals"],
            "tiros": m[f"{yo}_shots"],
            "tiros_arco": m[f"{yo}_shots_target"],
            "corners": m[f"{yo}_corners"],
            "faltas": m[f"{yo}_fouls"],
            "tarjetas": m[f"{yo}_yellows"] + 2 * m[f"{yo}_reds"],
        })
        lados.append(d)
    largo = pd.concat(lados, ignore_index=True)

    largo["pts"] = np.where(largo["gf"] > largo["gc"], 3,
                            np.where(largo["gf"] == largo["gc"], 1, 0))
    largo["dg"] = largo["gf"] - largo["gc"]

    cols_jug = ["season", "fixture_id", "team_short", "xg", "xa", "xgc", "n_jugadores",
                "atajadas", "tasa_atajadas",
                "xg_available"] + [f"pts_{v}" for v in player_agg.POSICIONES.values()]
    largo = largo.merge(stats_jug[cols_jug], on=CLAVE_LARGO, how="left",
                        validate="one_to_one")

    largo = largo.sort_values(["team_short", "kickoff_time"]).reset_index(drop=True)
    # Índice cronológico del partido dentro de la secuencia del equipo. Cruza temporadas
    # a propósito: lo usa la ventana de plantel.
    largo["k"] = largo.groupby("team_short").cumcount()
    return largo


# ---------------------------------------------------------------------------
# 3 · Las ventanas, calculadas INCLUSIVAS y tageadas por kickoff
# ---------------------------------------------------------------------------

def _rolling(df: pd.DataFrame, por: list[str], stats: list[str], n: int,
             sufijo: str) -> pd.DataFrame:
    """Media de los últimos n partidos, INCLUYENDO el propio. El shift lo hace el asof."""
    d = df.sort_values(por + ["kickoff_time"]).copy()
    g = d.groupby(por, sort=False)
    out = {c: g[c].transform(lambda s: s.rolling(n, min_periods=1).mean()) for c in stats}
    res = pd.DataFrame(out).rename(columns={c: f"{c}_{sufijo}" for c in stats})
    res.insert(0, "hist_kickoff", d["kickoff_time"].to_numpy())
    for col in reversed(por):
        res.insert(0, col, d[col].to_numpy())
    return res


def historia_general(largo: pd.DataFrame) -> pd.DataFrame:
    """Ventanas u3 y u5 sobre las 16 base. CRUZAN el borde de temporada.

    Cruzar es deliberado: en la fecha 1 de una temporada, la alternativa es no tener nada.
    Su contrapunto es `historia_temporada`, que sí corta — el modelo elige entre las dos,
    y `continuidad_plantel` le dice cuánto vale la historia vieja.
    """
    partes = []
    for n in CFG.rolling_windows:
        partes.append(_rolling(largo, ["team_short"], BASE, n, f"u{n}"))
    out = partes[0]
    for extra in partes[1:]:
        out = pd.concat([out, extra.drop(columns=["team_short", "hist_kickoff"])], axis=1)

    d = largo.sort_values(["team_short", "kickoff_time"])
    n = d.groupby("team_short").cumcount().to_numpy() + 1
    # TECHO. `n_hist` es un contador acumulado: crece para siempre a medida que el
    # dataset envejece. En el train llegaba a 113 y en la fecha 2 de 2026-27 ya vale 153,
    # un valor que el modelo nunca vio -- y que va a ser peor cada semana. Es train/serve
    # skew silencioso, del tipo que no rompe nada y sólo degrada.
    #
    # Lo que la feature aporta es "cuanta historia tiene este equipo", y eso satura: con
    # una temporada completa ya se sabe todo lo que hay que saber. Mas alla del techo el
    # numero es una marca temporal disfrazada de feature.
    out["n_hist"] = np.minimum(n, TECHO_N_HIST)
    return out


def historia_temporada(largo: pd.DataFrame) -> pd.DataFrame:
    """Ventana u5 restringida a la temporada actual. NaN al arranque, a propósito."""
    return _rolling(largo, ["season", "team_short"], list(spec.STATS_TEMP), 5, "u5_temp")


def historia_condicion(largo: pd.DataFrame) -> pd.DataFrame:
    """Ventana u5 restringida a la condición: el local sólo de local, y viceversa."""
    return _rolling(largo, ["team_short", "es_local"], list(spec.STATS_COND), 5, "cond_u5")


def historia_campeonato(largo: pd.DataFrame) -> pd.DataFrame:
    """Acumulado de la temporada: expanding() en vez de rolling(), mismo anclaje."""
    d = largo.sort_values(["season", "team_short", "kickoff_time"]).copy()
    g = d.groupby(["season", "team_short"], sort=False)
    out = pd.DataFrame({
        "season": d["season"].to_numpy(),
        "team_short": d["team_short"].to_numpy(),
        "hist_kickoff": d["kickoff_time"].to_numpy(),
        "pts_camp": g["pts"].cumsum().to_numpy(),
        "gf_camp": g["gf"].cumsum().to_numpy(),
        "gc_camp": g["gc"].cumsum().to_numpy(),
    })
    out["pj_camp"] = g.cumcount().to_numpy() + 1
    out["dg_camp"] = out["gf_camp"] - out["gc_camp"]
    out["ppp_camp"] = out["pts_camp"] / out["pj_camp"]
    return out


# ---------------------------------------------------------------------------
# 4 · El pegado anti-leakage
# ---------------------------------------------------------------------------

def pegar_asof(objetivos: pd.DataFrame, hist: pd.DataFrame, por: list[str],
               cols: list[str]) -> pd.DataFrame:
    """Para cada objetivo, el último estado conocido ESTRICTAMENTE anterior a su corte.

    `allow_exact_matches=False` implementa el "estrictamente anterior". Es lo que hace que
    un partido no se vea a sí mismo, y que los dos partidos de una doble fecha compartan
    features en lugar de que el segundo espíe al primero.
    """
    izq = objetivos.sort_values("corte").reset_index(drop=True)
    der = hist[por + ["hist_kickoff"] + cols].sort_values("hist_kickoff").reset_index(drop=True)
    return pd.merge_asof(izq, der, left_on="corte", right_on="hist_kickoff", by=por,
                         direction="backward", allow_exact_matches=False)


def tabla_de_posiciones(camp: pd.DataFrame, cortes: pd.DataFrame,
                        equipos: pd.DataFrame) -> pd.DataFrame:
    """Posición en la tabla de cada equipo, en cada corte.

    Hay que rankear los 20 equipos SIMULTÁNEAMENTE en cada corte, así que no alcanza con
    una ventana por equipo: se arma la grilla (temporada, corte) x equipos y se hace el
    mismo asof. El desempate es (pts, dg, gf), el criterio real de la Premier.
    """
    grilla = cortes.merge(equipos, on="season", how="inner")
    filas = pegar_asof(grilla, camp, ["season", "team_short"],
                       ["pts_camp", "dg_camp", "gf_camp"])
    for c in ("pts_camp", "dg_camp", "gf_camp"):
        filas[c] = filas[c].fillna(0.0)
    filas["pos_tabla_camp"] = (
        filas.sort_values(["pts_camp", "dg_camp", "gf_camp"], ascending=False)
             .groupby(["season", "corte"]).cumcount() + 1
    )
    return filas[["season", "gameweek", "team_short", "pos_tabla_camp"]]
