"""Elo y otras features de estado que las medias móviles no capturan.

Las ventanas rodantes tienen un problema: **tratan igual a todos los rivales**. Ganarle al
último con un 2-0 pesa lo mismo que ganarle al primero. El Elo resuelve exactamente eso —
cada resultado vale según contra quién fue— y por eso es la feature clásica del fútbol.

Todas se calculan de forma secuencial sobre los partidos ya jugados y se tagean con el
`kickoff_time` del partido que las produjo, para que el `merge_asof` del corte las trate
igual que a cualquier otra ventana. Nada de esto mira hacia adelante.

Lo que se agrega:

- `elo` — rating Elo, con ventaja de localía y regresión a la media entre temporadas.
- `elo_dif` — la diferencia entre los dos, que es lo que el Elo realmente predice.
- `xg_diff_u5` — goles menos xG en los últimos 5. Mide **suerte de definición**, y es
  fuertemente reversible a la media: un equipo que viene convirtiendo por encima de su xG
  tiende a bajar. Es información que ni `gf_u5` ni `xg_u5` capturan por separado.
- `tiros_conc_u5` — tiros que le conceden al equipo. Las ventanas de `tiros` miden lo que
  el equipo genera; esto mide lo que regala, que es otra cosa.
- `partidos_14d` — partidos jugados en los últimos 14 días: congestión de calendario.
- `racha` — puntos de los últimos 3 partidos menos el promedio de la temporada; captura si
  el equipo está por encima o por debajo de su nivel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Parámetros del Elo. K controla cuánto se mueve el rating por partido: 20 es el valor
# habitual para fútbol de clubes (más alto lo hace ruidoso, más bajo lo hace lento).
K = 20.0
# Ventaja de localía en puntos de Elo. ~65 equivale a la ventaja histórica de la Premier.
VENTAJA_LOCAL = 65.0
INICIAL = 1500.0
# Al empezar una temporada los ratings se acercan a la media: los planteles cambian y el
# ~40 % de los minutos rota. Sin esto, un equipo descendido arrastraría su rating viejo.
REGRESION_TEMPORADA = 0.25


def _esperado(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def calcular(largo: pd.DataFrame) -> pd.DataFrame:
    """Elo por equipo, DESPUÉS de cada partido, tageado con su kickoff.

    Se recorre en orden cronológico, que es la única forma de calcular un Elo. Devuelve el
    rating posterior a cada partido: el `merge_asof` del corte se encarga de que un partido
    nunca vea el suyo propio.
    """
    partidos = (largo[largo["es_local"]]
                [["season", "fixture_id", "kickoff_time", "team_short", "rival_short",
                  "gf", "gc"]]
                .sort_values("kickoff_time")
                .reset_index(drop=True))

    rating: dict[str, float] = {}
    temporada_previa: str | None = None
    filas = []

    for r in partidos.itertuples(index=False):
        if r.season != temporada_previa:
            # Regresión a la media al cambiar de temporada.
            for eq in rating:
                rating[eq] += (INICIAL - rating[eq]) * REGRESION_TEMPORADA
            temporada_previa = r.season

        loc = rating.setdefault(r.team_short, INICIAL)
        vis = rating.setdefault(r.rival_short, INICIAL)

        esp_loc = _esperado(loc + VENTAJA_LOCAL, vis)
        real_loc = 1.0 if r.gf > r.gc else (0.5 if r.gf == r.gc else 0.0)

        # El margen de victoria amplifica el ajuste, atenuado por logaritmo para que una
        # goleada no distorsione el rating.
        margen = 1.0 + np.log1p(abs(r.gf - r.gc))
        delta = K * margen * (real_loc - esp_loc)

        rating[r.team_short] = loc + delta
        rating[r.rival_short] = vis - delta

        # SORPRESA: cuanto se aparto el resultado de lo que el Elo esperaba, antes de
        # actualizarlo. |real - esperado| en [0, 1]. Es la version legitima de "que tan
        # impredecible es este equipo": no usa las predicciones del modelo -- eso seria
        # un bucle de realimentacion, y ademas imposible de calcular en entrenamiento sin
        # leakage -- sino la expectativa del Elo, que sale solo de resultados pasados.
        sorpresa = abs(real_loc - esp_loc)

        filas.append({"season": r.season, "fixture_id": r.fixture_id,
                      "team_short": r.team_short, "kickoff_time": r.kickoff_time,
                      "elo": rating[r.team_short], "sorpresa": sorpresa,
                      "elo_esperado": esp_loc})
        filas.append({"season": r.season, "fixture_id": r.fixture_id,
                      "team_short": r.rival_short, "kickoff_time": r.kickoff_time,
                      "elo": rating[r.rival_short], "sorpresa": sorpresa,
                      "elo_esperado": 1.0 - esp_loc})

    return pd.DataFrame(filas)


def deltas_elo(e: pd.DataFrame) -> pd.DataFrame:
    """Cuanto GANO o PERDIO de rating el equipo en sus ultimos N partidos.

    Es informacion distinta del nivel y distinta de `racha`:

    - `elo` dice **donde esta** el equipo.
    - `racha` compara los puntos de los ultimos 3 contra su propio promedio, pero trata
      igual todos los rivales: ganarle al ultimo pesa lo mismo que ganarle al primero.
    - `elo_delta_uN` dice **hacia donde va, ponderado por contra quien**. Un equipo que
      suma 6 puntos contra dos rivales de arriba gana mucho mas rating que otro que suma
      los mismos 6 contra dos de abajo.

    La combinacion `elo` + `elo_delta` le permite al modelo distinguir cuatro situaciones
    que hoy se le mezclan: grande en alza, grande en caida, chico en alza y chico en caida.
    Un equipo de Elo bajo que viene subiendo fuerte es justamente el caso que las medias
    moviles no capturan.
    """
    d = e.sort_values(["team_short", "kickoff_time"]).copy()
    g = d.groupby("team_short", sort=False)["elo"]
    for n in (3, 5, 10):
        # El Elo de ahora menos el de hace n partidos. `shift` cuenta partidos del equipo,
        # que es lo correcto: no hay riesgo de leakage porque el merge_asof posterior se
        # encarga de que un partido no vea el suyo propio.
        d[f"elo_delta_u{n}"] = d["elo"] - g.shift(n)
    cols = [f"elo_delta_u{n}" for n in (3, 5, 10)]
    return d[["season", "fixture_id", "team_short"] + cols]


def ventanas_sorpresa(e: pd.DataFrame) -> pd.DataFrame:
    """Media movil de la sorpresa: que tan impredecible viene siendo el equipo.

    Un equipo con `sorpresa_u5` alta viene dando resultados que su Elo no anticipaba --
    para bien o para mal. Es informacion sobre la CONFIABILIDAD de la prediccion, no
    sobre su direccion, y es justo lo que faltaba: el modelo estaba sobreconfiado en la
    franja media (decia 0,477 y acertaba 0,394).
    """
    d = e.sort_values(["team_short", "kickoff_time"]).copy()
    g = d.groupby("team_short", sort=False)
    d["sorpresa_u5"] = g["sorpresa"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    d["sorpresa_u10"] = g["sorpresa"].transform(lambda s: s.rolling(10, min_periods=1).mean())
    return d[["season", "fixture_id", "team_short", "sorpresa_u5", "sorpresa_u10"]]


def extras(largo: pd.DataFrame) -> pd.DataFrame:
    """Las demás features de estado, calculadas inclusivas y tageadas por kickoff."""
    d = largo.sort_values(["team_short", "kickoff_time"]).copy()

    # Tiros concedidos: los del rival en ese mismo partido.
    rival = largo[["season", "fixture_id", "team_short", "tiros", "tiros_arco"]].rename(
        columns={"team_short": "rival_short", "tiros": "tiros_conc",
                 "tiros_arco": "tiros_arco_conc"})
    d = d.merge(rival, on=["season", "fixture_id", "rival_short"], how="left")

    # Sobre/sub-rendimiento respecto del xG: suerte de definición, reversible a la media.
    d["xg_diff"] = d["gf"] - d["xg"]
    d["xgc_diff"] = d["gc"] - d["xgc"]

    g = d.groupby("team_short", sort=False)
    for c in ("tiros_conc", "tiros_arco_conc", "xg_diff", "xgc_diff"):
        d[f"{c}_u5"] = g[c].transform(lambda s: s.rolling(5, min_periods=1).mean())

    # Congestión de calendario, en tres ventanas. Cada una capta algo distinto:
    #
    #   7d   "jugo entre semana". Es lo mas cerca que se puede estar de detectar un
    #        compromiso de copa o de Europa sin una fuente externa de esos calendarios.
    #   14d  la carga de las ultimas dos semanas.
    #   21d  la acumulada: no es lo mismo un pico aislado que tres semanas seguidas.
    #
    # ⚠️ Sólo cuenta partidos de PREMIER. Un equipo que sigue en Champions y en la Copa
    # de la Liga juega mucho mas de lo que estas columnas ven. La fuente que lo
    # resolveria (openfootball) no publica 2026-27, asi que usarla seria entrenar con
    # una feature que en produccion vale NaN. Queda anotado como deuda.
    for dias in (7, 14, 21):
        d[f"partidos_{dias}d"] = [
            int(((d.loc[d["team_short"] == t, "kickoff_time"] > k - pd.Timedelta(days=dias))
                 & (d.loc[d["team_short"] == t, "kickoff_time"] <= k)).sum())
            for t, k in zip(d["team_short"], d["kickoff_time"])
        ]

    # Racha: puntos de los últimos 3 contra el promedio de lo que va de temporada.
    d["pts_u3_tmp"] = g["pts"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    d["pts_exp_tmp"] = (d.groupby(["season", "team_short"], sort=False)["pts"]
                         .transform(lambda s: s.expanding().mean()))
    d["racha"] = d["pts_u3_tmp"] - d["pts_exp_tmp"]

    cols = ["tiros_conc_u5", "tiros_arco_conc_u5", "xg_diff_u5", "xgc_diff_u5",
            "partidos_7d", "partidos_14d", "partidos_21d", "racha"]
    return d[["season", "fixture_id", "team_short", "kickoff_time"] + cols]


COLUMNAS = ["elo", "elo_delta_u3", "elo_delta_u5", "elo_delta_u10",
            "tiros_conc_u5", "tiros_arco_conc_u5", "xg_diff_u5", "xgc_diff_u5",
            "partidos_7d", "partidos_14d", "partidos_21d", "racha",
            "sorpresa_u5", "sorpresa_u10"]


def construir(largo: pd.DataFrame) -> pd.DataFrame:
    """Todas las features de este módulo, listas para el `merge_asof`."""
    e = calcular(largo)
    s = ventanas_sorpresa(e)
    dl = deltas_elo(e)
    x = extras(largo)
    out = (e.drop(columns=["sorpresa", "elo_esperado"])
            .merge(dl, on=["season", "fixture_id", "team_short"], validate="one_to_one")
            .merge(s, on=["season", "fixture_id", "team_short"], validate="one_to_one")
            .merge(x, on=["season", "fixture_id", "team_short", "kickoff_time"],
                   how="outer", validate="one_to_one"))
    return out.rename(columns={"kickoff_time": "hist_kickoff"})
