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


def _secuencia(largo: pd.DataFrame, historia: pd.DataFrame | None) -> pd.DataFrame:
    """Todos los partidos que mueven el rating, en orden, con una clave de equipo única.

    Dos fuentes con contratos distintos:

    - `largo` — la ventana ingestada. Tiene `fixture_id` y `short_name` canónico, y es la
      **autoritativa**: son las filas que después salen como features.
    - `historia` — `fact_match_historico`, veinte años de E0/E1/E2. No tiene fixture_id ni
      Gold detrás: sólo empuja el rating.

    Dos cuidados que importan:

    1. **No contar dos veces.** La historia también trae el E0 de las temporadas de la
       ventana, porque se ingesta entera y sola. Esas filas se descartan: para esas fechas
       manda `largo`.
    2. **Una sola clave.** La historia usa el nombre de football-data normalizado y la
       ventana usa `short_name`. Se unifica prefiriendo `short_name` donde exista, así un
       equipo de Premier es `ARS` en las dos fuentes y uno del League One es `accrington`.
    """
    v = (largo[largo["es_local"]]
         [["season", "fixture_id", "kickoff_time", "team_short", "rival_short", "gf", "gc"]]
         .rename(columns={"team_short": "key_local", "rival_short": "key_visita"})
         .copy())
    v["fecha"] = v["kickoff_time"]
    v["division"] = "E0"
    v["es_ventana"] = True

    if historia is None or historia.empty:
        return v.sort_values(["fecha", "fixture_id"]).reset_index(drop=True)

    ventana_seasons = set(largo["season"].unique())
    h = historia[~((historia["division"] == "E0")
                   & (historia["season"].isin(ventana_seasons)))].copy()

    h["key_local"] = h["home_short"].fillna(h["home_key"])
    h["key_visita"] = h["away_short"].fillna(h["away_key"])
    h["fecha"] = pd.to_datetime(h["match_date"], utc=True)
    h = h.rename(columns={"home_goals": "gf", "away_goals": "gc"})
    h["fixture_id"] = pd.NA
    h["kickoff_time"] = h["fecha"]
    h["es_ventana"] = False

    cols = ["season", "division", "fecha", "kickoff_time", "fixture_id",
            "key_local", "key_visita", "gf", "gc", "es_ventana"]
    seq = pd.concat([v[cols], h[cols]], ignore_index=True)
    # Desempate estable: dentro del mismo instante, primero la historia y despues la
    # ventana, y adentro por division. Sin un orden fijo el Elo no seria reproducible.
    return (seq.sort_values(["fecha", "es_ventana", "division", "key_local"],
                            kind="mergesort")
               .reset_index(drop=True))


def _planteles(seq: pd.DataFrame) -> dict[tuple[str, str], set[str]]:
    """Qué equipos juegan cada (temporada, división).

    Se usa para regresar a la media **de la división**, y no es leakage: en qué categoría
    juega cada club se sabe antes de que ruede la pelota.
    """
    out: dict[tuple[str, str], set[str]] = {}
    for (s, d), g in seq.groupby(["season", "division"], sort=False):
        out[(s, d)] = set(g["key_local"]) | set(g["key_visita"])
    return out


def calcular(largo: pd.DataFrame, historia: pd.DataFrame | None = None) -> pd.DataFrame:
    """Elo por equipo, DESPUÉS de cada partido, tageado con su kickoff.

    Se recorre en orden cronológico, que es la única forma de calcular un Elo. Devuelve el
    rating posterior a cada partido **de la ventana**: el `merge_asof` del corte se encarga
    de que un partido nunca vea el suyo propio.

    ## Con `historia`, dos cosas cambian

    **El rating llega sembrado.** Sin historia, los 20 equipos arrancan en 1500 en la GW1 de
    2022-23 y el Elo tarda media temporada en significar algo — con `dif_elo` siendo la
    feature más importante del modelo, esa media temporada de train es ruido. Y un ascendido
    entra siempre en 1500, que es el fallo medido en la GW1 de 2026-27.

    **La regresión de entre-temporadas apunta a la media de la DIVISIÓN, no a 1500.** Es
    obligatorio si hay varias divisiones: el Elo es de suma cero *dentro* de cada grupo, así
    que la diferencia de nivel entre la Premier y el Championship sólo existe porque los
    ascensos y descensos la construyen a lo largo de los años. Regresar todo a 1500 cada
    agosto la borraría justo antes de usarla.

    Sin historia se mantiene el comportamiento anterior al pie de la letra —regresión a
    1500— para que la comparación A/B mida el cambio y no un efecto colateral.
    """
    seq = _secuencia(largo, historia)
    con_historia = historia is not None and not historia.empty
    planteles = _planteles(seq) if con_historia else {}

    rating: dict[str, float] = {}
    temporada_previa: str | None = None
    filas = []

    for r in seq.itertuples(index=False):
        if r.season != temporada_previa:
            _regresar(rating, r.season, planteles, con_historia)
            temporada_previa = r.season

        loc = rating.setdefault(r.key_local, INICIAL)
        vis = rating.setdefault(r.key_visita, INICIAL)

        esp_loc = _esperado(loc + VENTAJA_LOCAL, vis)
        real_loc = 1.0 if r.gf > r.gc else (0.5 if r.gf == r.gc else 0.0)

        # El margen de victoria amplifica el ajuste, atenuado por logaritmo para que una
        # goleada no distorsione el rating.
        margen = 1.0 + np.log1p(abs(r.gf - r.gc))
        delta = K * margen * (real_loc - esp_loc)

        rating[r.key_local] = loc + delta
        rating[r.key_visita] = vis - delta

        # Los partidos de historia mueven el rating y no producen features: no tienen
        # fixture_id ni fila en Gold. Son insumo, no salida.
        if not r.es_ventana:
            continue

        # SORPRESA: cuanto se aparto el resultado de lo que el Elo esperaba, antes de
        # actualizarlo. |real - esperado| en [0, 1]. Es la version legitima de "que tan
        # impredecible es este equipo": no usa las predicciones del modelo -- eso seria
        # un bucle de realimentacion, y ademas imposible de calcular en entrenamiento sin
        # leakage -- sino la expectativa del Elo, que sale solo de resultados pasados.
        sorpresa = abs(real_loc - esp_loc)

        filas.append({"season": r.season, "fixture_id": r.fixture_id,
                      "team_short": r.key_local, "kickoff_time": r.kickoff_time,
                      "elo": rating[r.key_local], "sorpresa": sorpresa,
                      "elo_esperado": esp_loc})
        filas.append({"season": r.season, "fixture_id": r.fixture_id,
                      "team_short": r.key_visita, "kickoff_time": r.kickoff_time,
                      "elo": rating[r.key_visita], "sorpresa": sorpresa,
                      "elo_esperado": 1.0 - esp_loc})

    out = pd.DataFrame(filas)
    # `fixture_id` vuelve a entero: el concat con la historia (que lo tiene nulo) lo
    # promueve a object, y los merges de `gold_tp` cruzan por esta columna.
    out["fixture_id"] = out["fixture_id"].astype("int64")
    return out


def _regresar(rating: dict[str, float], season: str,
              planteles: dict[tuple[str, str], set[str]], con_historia: bool) -> None:
    """Regresion a la media al empezar una temporada. Los planteles rotan ~40%.

    **Con historia el objetivo es la media de la DIVISION**, no 1500. Es la pieza que hace
    que las tres divisiones convivan en un mismo Elo: la diferencia de nivel entre la
    Premier y el Championship no esta puesta a mano, la construyen los ascensos y descensos
    a lo largo de veinte años. Regresar todo a 1500 cada agosto la borraria justo antes de
    usarla, y el rating de un ascendido volveria a ser el generico que se queria evitar.

    Sin historia se regresa a 1500, que es el comportamiento anterior, para que el A/B de
    la Fase 1 mida el sembrado y no un cambio colateral de la regresion.
    """
    if not con_historia:
        for eq in rating:
            rating[eq] += (INICIAL - rating[eq]) * REGRESION_TEMPORADA
        return

    for (s, _division), equipos in planteles.items():
        if s != season:
            continue
        presentes = [e for e in equipos if e in rating]
        if not presentes:
            continue
        media = sum(rating[e] for e in presentes) / len(presentes)
        for e in presentes:
            rating[e] += (media - rating[e]) * REGRESION_TEMPORADA


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

    # CALIDAD del xG, no cantidad. 2,0 de xG en 3 ocasiones claras y 2,0 en 20 remates de
    # afuera son cosas distintas y predicen distinto: el primero es un equipo que genera
    # situaciones, el segundo uno que patea de desesperado. El xG agregado no las separa;
    # dividirlo por los tiros si.
    #
    # Es la aproximacion disponible al xG a nivel TIRO que daria Understat. No es lo
    # mismo -- el promedio no distingue "una clarisima y muchas malas" de "todas
    # regulares" -- pero es gratis y sale de datos que ya estan.
    d["xg_por_tiro"] = d["xg"] / d["tiros"].replace(0, np.nan)
    d["xgc_por_tiro"] = d["xgc"] / d["tiros_conc"].replace(0, np.nan)
    # Proporcion de tiros que van al arco: puntería y seleccion de remate.
    d["prop_tiros_arco"] = d["tiros_arco"] / d["tiros"].replace(0, np.nan)
    d["prop_tiros_arco_conc"] = d["tiros_arco_conc"] / d["tiros_conc"].replace(0, np.nan)

    g = d.groupby("team_short", sort=False)
    for c in ("tiros_conc", "tiros_arco_conc", "xg_diff", "xgc_diff",
              "xg_por_tiro", "xgc_por_tiro", "prop_tiros_arco", "prop_tiros_arco_conc"):
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
            "xg_por_tiro_u5", "xgc_por_tiro_u5",
            "prop_tiros_arco_u5", "prop_tiros_arco_conc_u5",
            "partidos_7d", "partidos_14d", "partidos_21d", "racha"]
    return d[["season", "fixture_id", "team_short", "kickoff_time"] + cols]


COLUMNAS = ["elo", "elo_delta_u3", "elo_delta_u5", "elo_delta_u10",
            "tiros_conc_u5", "tiros_arco_conc_u5", "xg_diff_u5", "xgc_diff_u5",
            "xg_por_tiro_u5", "xgc_por_tiro_u5",
            "prop_tiros_arco_u5", "prop_tiros_arco_conc_u5",
            "partidos_7d", "partidos_14d", "partidos_21d", "racha",
            "sorpresa_u5", "sorpresa_u10"]


def construir(largo: pd.DataFrame,
              historia: pd.DataFrame | None = None) -> pd.DataFrame:
    """Todas las features de este módulo, listas para el `merge_asof`.

    `historia` es opcional a propósito: si `fact_match_historico` no está construido, el
    módulo se comporta exactamente como antes de la Fase 1. Sembrar el rating no puede ser
    un requisito para que el pipeline corra.
    """
    e = calcular(largo, historia)
    s = ventanas_sorpresa(e)
    dl = deltas_elo(e)
    x = extras(largo)
    out = (e.drop(columns=["sorpresa", "elo_esperado"])
            .merge(dl, on=["season", "fixture_id", "team_short"], validate="one_to_one")
            .merge(s, on=["season", "fixture_id", "team_short"], validate="one_to_one")
            .merge(x, on=["season", "fixture_id", "team_short", "kickoff_time"],
                   how="outer", validate="one_to_one"))
    return out.rename(columns={"kickoff_time": "hist_kickoff"})
