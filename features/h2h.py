"""Historial entre los dos equipos del partido, en dos variantes.

El head-to-head NO es simétrico, así que se explicita la perspectiva. Y los dos `pts` no
son deducibles uno del otro: un empate reparte 1-1 (suma 2) y una victoria 3-0 (suma 3),
así que saber los puntos del local no determina los del visitante.

Dos variantes:

- `h2h_*`      — todos los enfrentamientos previos, en cualquier condición.
                 Medido: media 2,83 antecedentes en la ventana, 5,15 en el holdout;
                 sólo el 5 % de los partidos del holdout no tiene ninguno.
- `h2h_cond_*` — sólo aquellos en que ESTE mismo equipo fue local contra ESTE mismo
                 rival. Codifica "en este estadio, contra éste, históricamente me va
                 así". Flaco por construcción: con 4 temporadas el máximo posible es 3
                 (media 1,16; 2,33 en el holdout, con 11 % sin antecedentes).

Los contadores `h2h_n` y `h2h_cond_n` van como features para que el modelo sepa cuánta
historia hay detrás de cada media, en vez de tratar igual a un promedio de 1 partido y
a uno de 7.
"""

from __future__ import annotations

import pandas as pd

from features.spec import CLAVE_PARTIDO as CLAVE
from features.team_form import pegar_asof

STATS = ["pts", "gf", "gc"]


def vista_dirigida(largo: pd.DataFrame) -> pd.DataFrame:
    """Una fila por (equipo, rival, partido), con los números desde la óptica del equipo."""
    return largo[["team_short", "rival_short", "kickoff_time", "es_local",
                  "pts", "gf", "gc"]].copy()


def _acumulado(dirigida: pd.DataFrame, solo_local: bool | None) -> pd.DataFrame:
    """Media expandida INCLUSIVA de los enfrentamientos, tageada por kickoff.

    Igual que las ventanas de forma: se calcula inclusiva y el "correrse" lo hace después
    el `merge_asof`, que es lo que garantiza que un partido no se vea a sí mismo.
    """
    d = dirigida
    if solo_local is not None:
        d = d[d["es_local"] == solo_local]
    d = d.sort_values(["team_short", "rival_short", "kickoff_time"]).copy()
    g = d.groupby(["team_short", "rival_short"], sort=False)

    out = pd.DataFrame({
        "team_short": d["team_short"].to_numpy(),
        "rival_short": d["rival_short"].to_numpy(),
        "hist_kickoff": d["kickoff_time"].to_numpy(),
    })
    for c in STATS:
        out[c] = g[c].transform(lambda s: s.expanding().mean()).to_numpy()
    out["n"] = g.cumcount().to_numpy() + 1
    return out


def construir(largo: pd.DataFrame, objetivos: pd.DataFrame) -> pd.DataFrame:
    """Las 10 columnas de head-to-head, para cada partido.

    `objetivos` trae una fila por partido con `fixture_id`, `home_short`, `away_short` y
    `corte`. Se hacen cuatro pegados asof: la óptica del local y la del visitante, cada
    una en las dos variantes.
    """
    dirigida = vista_dirigida(largo)
    res = objetivos[CLAVE].copy()

    for prefijo, solo_local_del_local in (("h2h", None), ("h2h_cond", True)):
        # Óptica del local: sus enfrentamientos previos contra este rival.
        acum = _acumulado(dirigida, solo_local_del_local)
        izq = objetivos.rename(columns={"home_short": "team_short",
                                        "away_short": "rival_short"})
        loc = pegar_asof(izq, acum, ["team_short", "rival_short"], STATS + ["n"])
        loc = loc[CLAVE + STATS + ["n"]].rename(columns={
            "pts": f"{prefijo}_pts_local", "gf": f"{prefijo}_gf_local",
            "gc": f"{prefijo}_gc_local", "n": f"{prefijo}_n"})

        # Óptica del visitante, sobre EXACTAMENTE los mismos partidos: si la variante
        # restringe al local de hoy jugando de local, la contraparte es el visitante de
        # hoy jugando de visitante.
        solo_local_del_visitante = None if solo_local_del_local is None else False
        acum_v = _acumulado(dirigida, solo_local_del_visitante)
        izq_v = objetivos.rename(columns={"away_short": "team_short",
                                          "home_short": "rival_short"})
        vis = pegar_asof(izq_v, acum_v, ["team_short", "rival_short"], ["pts"])
        vis = vis[CLAVE + ["pts"]].rename(
            columns={"pts": f"{prefijo}_pts_visita"})

        # Siempre por clave, nunca por posición: pegar_asof reordena por `corte`.
        res = res.merge(loc, on=CLAVE, validate="one_to_one")
        res = res.merge(vis, on=CLAVE, validate="one_to_one")

    for c in ("h2h_n", "h2h_cond_n"):
        res[c] = res[c].fillna(0).astype("int64")
    return res
