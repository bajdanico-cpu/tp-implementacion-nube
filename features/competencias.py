"""Features de carga y forma que miran TODAS las competencias, no sólo la Premier.

Hasta ahora `partidos_7d/14d/21d` contaban únicamente partidos de liga, y por eso dieron
un resultado nulo: un equipo en semifinales de dos copas figuraba igual de descansado que
uno que sólo juega el fin de semana. Medido sobre 2025-26, los equipos de Premier sumaron
**255 apariciones en copas y Europa contra 760 de liga**: un 34 % de carga invisible.

## Todo es retrospectivo, y eso no es una limitación menor

Cada feature de este módulo se calcula **sólo con partidos ya jugados** antes del corte.
Ninguna mira el calendario futuro, y hay una razón medida para eso.

El calendario de copa se publica **ronda por ronda**: el sorteo ocurre al terminar la
anterior, y las rondas están espaciadas 20-67 días. Al 25/08/2026 la EFL Cup tenía sus 60
fixtures de primera ronda con **dos días** de anticipación, contra los 278 días de la
Premier. Una feature del tipo *"juega copa la semana que viene"* estaría **siempre completa
en entrenamiento** —porque el pasado se conoce entero— y **faltaría en producción** durante
los días entre el fin de una ronda y el sorteo de la siguiente. Sería train/serve skew, y
lo peor es que no se puede ni simular honestamente: la API no expone cuándo se publicó cada
fixture, así que no hay forma de reconstruir qué se sabía en cada momento.

Lo bueno es que la señal de fatiga **no necesita ver el futuro**. Los partidos de copa que
cansan a un equipo para la fecha del sábado son los del martes, y para cuando llega el corte
del sábado ya se jugaron. La ventana retrospectiva los captura enteros.

## Tres bloques

- **Congestión real** — partidos jugados en los últimos 7, 14 y 21 días contando todo.
- **Recorrido en copas** — cuántos lleva acumulados y hasta qué instancia llegó. Es
  *"seguir en semis"* convertido en número, y sólo crece si el equipo avanza.
- **Forma con todo vs sólo liga** — las mismas medias de puntos y goles, calculadas sobre
  todas las competencias. La comparación contra la versión de liga es informativa por sí
  misma: un equipo que rinde distinto en copa que en liga está rotando.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common.logging_setup import get_logger

log = get_logger(__name__)

VENTANAS_DIAS = (7, 14, 21)
VENTANA_FORMA = 5

COLUMNAS = [
    "partidos_todo_7d", "partidos_todo_14d", "partidos_todo_21d",
    "partidos_copa_7d", "partidos_copa_14d",
    "copas_acumuladas", "europa_acumuladas", "importancia_max",
    "dias_desde_ultimo_todo",
    "pts_todo_u5", "gf_todo_u5", "gc_todo_u5",
]


def _puntos(gf, gc):
    if pd.isna(gf) or pd.isna(gc):
        return np.nan
    return 3.0 if gf > gc else (1.0 if gf == gc else 0.0)


def preparar(comp: pd.DataFrame) -> pd.DataFrame:
    """Sólo partidos TERMINADOS, con puntos y goles resueltos. Es la historia utilizable."""
    d = comp[comp["terminado"]].copy()

    # El rival de la misma fila da los goles en contra.
    rival = d[["fixture_pl_id", "team_short", "gf_comp"]].rename(
        columns={"team_short": "otro", "gf_comp": "gc_comp"})
    d = d.merge(rival, on="fixture_pl_id", how="left")
    d = d[d["team_short"] != d["otro"]].drop(columns="otro")

    # En los partidos de copa contra equipos de otras divisiones no hay fila del rival,
    # así que el gol en contra queda nulo. Se recupera del propio fixture.
    falta = d["gc_comp"].isna()
    if falta.any():
        d = d.drop(columns="gc_comp")
        d = d.merge(_goles_rival(comp), on=["fixture_pl_id", "team_short"], how="left")

    d["pts_comp"] = [_puntos(a, b) for a, b in zip(d["gf_comp"], d["gc_comp"])]
    return d.sort_values(["team_short", "kickoff_time"]).reset_index(drop=True)


def _goles_rival(comp: pd.DataFrame) -> pd.DataFrame:
    """Goles en contra tomando el otro lado del fixture, incluso si no está en la tabla."""
    por_fixture = comp.groupby("fixture_pl_id")["gf_comp"].apply(list)
    filas = []
    for fid, grupo in comp.groupby("fixture_pl_id"):
        goles = por_fixture.loc[fid]
        total = sum(g for g in goles if pd.notna(g))
        for r in grupo.itertuples():
            propio = r.gf_comp
            gc = (total - propio) if (pd.notna(propio) and len(goles) > 1) else np.nan
            filas.append({"fixture_pl_id": fid, "team_short": r.team_short, "gc_comp": gc})
    return pd.DataFrame(filas).drop_duplicates(["fixture_pl_id", "team_short"])


def construir(hist: pd.DataFrame, objetivos: pd.DataFrame) -> pd.DataFrame:
    """Las features de cada objetivo, con su corte.

    `objetivos` trae `team_short` y `corte`. Para cada uno se miran los partidos del equipo
    con `kickoff_time < corte`, en todas las competencias.
    """
    obj = objetivos.reset_index(drop=True).copy()
    salida = {c: np.full(len(obj), np.nan) for c in COLUMNAS}

    por_equipo = {t: g.reset_index(drop=True) for t, g in hist.groupby("team_short")}

    for i, r in enumerate(obj.itertuples()):
        g = por_equipo.get(r.team_short)
        if g is None or g.empty:
            continue
        corte = r.corte
        # Estrictamente anterior al corte, igual que el merge_asof del resto.
        prev = g[g["kickoff_time"] < corte]
        if prev.empty:
            continue

        for dias in VENTANAS_DIAS:
            desde = corte - pd.Timedelta(days=dias)
            ventana = prev[prev["kickoff_time"] >= desde]
            salida[f"partidos_todo_{dias}d"][i] = len(ventana)
            if dias in (7, 14):
                salida[f"partidos_copa_{dias}d"][i] = int((~ventana["es_premier"]).sum())

        # Recorrido en copas, dentro de la temporada del objetivo.
        temporada = prev[prev["season"] == r.season]
        no_liga = temporada[~temporada["es_premier"]]
        salida["copas_acumuladas"][i] = len(no_liga[no_liga["competencia"].isin(
            ("eflcup", "facup"))])
        salida["europa_acumuladas"][i] = len(no_liga[no_liga["competencia"].isin(
            ("champions", "europa"))])
        imp = no_liga["importancia_ronda"].dropna()
        salida["importancia_max"][i] = imp.max() if len(imp) else 0.0

        salida["dias_desde_ultimo_todo"][i] = (
            (corte - prev["kickoff_time"].iloc[-1]).total_seconds() / 86400)

        # Forma sobre TODAS las competencias.
        u5 = prev.tail(VENTANA_FORMA)
        for col, src in (("pts_todo_u5", "pts_comp"), ("gf_todo_u5", "gf_comp"),
                         ("gc_todo_u5", "gc_comp")):
            v = u5[src].dropna()
            if len(v):
                salida[col][i] = v.mean()

    out = pd.DataFrame(salida)
    for c in ("partidos_todo_7d", "partidos_todo_14d", "partidos_todo_21d",
              "partidos_copa_7d", "partidos_copa_14d", "copas_acumuladas",
              "europa_acumuladas"):
        out[c] = out[c].fillna(0.0)
    return out
