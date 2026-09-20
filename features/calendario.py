"""Qué fecha se puede predecir, y cuándo su fila de Gold deja de moverse.

Existe por una pregunta que el proyecto no sabía contestar: *si hoy se está jugando la
fecha 5 y alguien pide la 7, ¿qué se devuelve?* Medido el 20/09/2026, la respuesta era
"diez predicciones normales, calculadas con la misma información que la fecha 5" — porque
las features salen de un `merge_asof` hacia atrás y no había nada más nuevo que mirar. El
`dias_descanso` daba 35 en vez de 7, y nada en la respuesta lo decía.

La observación que ordena todo el módulo:

    las features de la fecha N dependen SÓLO de la historia anterior a corte(N)

Entonces, en cuanto se jugó y se ingestó todo lo anterior a ese corte, **la fila de la
fecha N es final: no se puede mover más**. Eso convierte "¿ya se puede predecir?" en un
predicado chequeable en vez de una regla de dedo, y es lo que habilita materializar la
fila en Gold y servirla con un lookup.

**"Jugado" es estar en `fact_match`, no el flag `finished` de FPL.** Dos razones. La
primera está medida y documentada desde agosto: con los diez partidos a 90 minutos y el
marcador cargado, `finished` seguía en `False` porque FPL lo activa al confirmar los
bonus, horas después. La segunda es más de fondo: `team_form.construir_largo` hace un
inner join entre `fact_match` y `fact_fixture`, así que estar en `fact_match` es
*exactamente* la condición para aportar historia. Usando la misma definición, este módulo
no puede desincronizarse de lo que las features van a leer.
"""

from __future__ import annotations

import pandas as pd

from common.config import CFG
from common.logging_setup import get_logger
from features import team_form as tf

log = get_logger(__name__)

# La clave del cruce fixture <-> resultado. Es la misma que usa `construir_largo`, con
# `validate="one_to_one"`: si acá se usara otra, "jugado" podría diferir de "aporta
# historia" y nadie se enteraría.
CLAVE_FIXTURE = ["season", "match_date", "home_short", "away_short"]

# Un partido postergado sin fecha nueva dejaría su gameweek incompleta para siempre, y
# con ella trabada la próxima predecible. Pasado este margen desde el corte de su fecha,
# se lo deja de esperar (con aviso). Cuando se juegue, entra a Gold por la rama histórica
# como cualquier otro, con el corte de su gameweek nominal.
GRACIA_POSTERGADO_D = 7

# La forma exacta que `gold_tp._objetivos` produce para los partidos jugados. Las dos
# ramas tienen que ser indistinguibles aguas abajo.
OBJETIVO_COLS = ["season", "gameweek", "fixture_id", "kickoff_time",
                 "home_short", "away_short", "corte"]

COLS_ESTADO = ["season", "gameweek", "corte", "n", "n_jugados", "n_pendientes",
               "n_previos", "n_previos_jugados", "n_previos_pendientes",
               "opta_completa", "definitiva", "situacion"]


def _ahora(ahora: pd.Timestamp | None = None) -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC") if ahora is None else pd.Timestamp(ahora)


def jugados(fixtures: pd.DataFrame, matches: pd.DataFrame) -> pd.Series:
    """Bool por fixture: ¿ya tiene resultado en `fact_match`?

    Devuelve una Series alineada con el índice de `fixtures`.
    """
    if fixtures.empty:
        return pd.Series(dtype=bool, index=fixtures.index)
    if matches.empty:
        return pd.Series(False, index=fixtures.index)

    m = matches[CLAVE_FIXTURE].drop_duplicates()
    marca = fixtures[CLAVE_FIXTURE].merge(m, on=CLAVE_FIXTURE, how="left", indicator=True)
    return pd.Series((marca["_merge"] == "both").to_numpy(), index=fixtures.index)


def cobertura_opta(comp: pd.DataFrame | None, stats: pd.DataFrame | None,
                   season: str, hasta: pd.Timestamp) -> float:
    """Proporción de equipo-partido de Premier con kickoff < `hasta` que tiene Opta.

    Importa para la definitividad por un detalle de `features/opta.py`: la historia de
    Opta se arma sobre TODOS los fixtures de Premier, jugados o no, y el rolling corre
    sobre esa serie con `min_periods=1`. Si a un partido anterior al corte le falta la
    ingesta de Opta, el `merge_asof` no falla: devuelve un promedio calculado sobre menos
    partidos. Es degradación silenciosa, del tipo que sólo se ve comparando dos corridas.

    Sin Opta ingestada, devuelve 1.0: el bloque entero se omite en Gold y no hay nada que
    pueda degradarse a medias.
    """
    if comp is None or stats is None or comp.empty:
        return 1.0

    clave = ["season", "fixture_pl_id", "team_id_pl"]
    if not set(clave).issubset(comp.columns) or not set(clave).issubset(stats.columns):
        return 1.0

    previos = comp.loc[comp["es_premier"]
                       & (comp["season"] == season)
                       & (comp["kickoff_time"] < hasta), clave]
    if previos.empty:
        return 1.0

    tiene = previos.merge(stats[clave].drop_duplicates(), on=clave,
                          how="left", indicator=True)
    return float((tiene["_merge"] == "both").mean())


def estado(fixtures: pd.DataFrame, matches: pd.DataFrame,
           season: str | None = None,
           comp: pd.DataFrame | None = None,
           stats: pd.DataFrame | None = None,
           ahora: pd.Timestamp | None = None) -> pd.DataFrame:
    """Una fila por gameweek de la temporada, con todo lo que define su situación.

    `definitiva` es la columna que importa: dice si la fila de Gold de esa fecha, en caso
    de construirse ahora, ya no podría cambiar.
    """
    season = season or CFG.current_season
    fx = fixtures[fixtures["season"] == season].copy()
    if fx.empty:
        return pd.DataFrame(columns=COLS_ESTADO)

    fx["jugado"] = jugados(fx, matches)
    cortes = tf.cortes_por_fecha(fx)

    # Un partido cuya fecha arrancó hace más de `GRACIA_POSTERGADO_D` y sigue sin
    # resultado se da por postergado. Dejar de esperarlo tiene que valer en TODAS partes:
    # si sólo se lo salteara al elegir la próxima fecha, seguiría contando como "previo
    # pendiente" y bloquearía la definitividad de todo el resto de la temporada. O se lo
    # espera, o no; a medias es no poder predecir nunca más.
    limite = _ahora(ahora) - pd.Timedelta(days=GRACIA_POSTERGADO_D)
    corte_de = dict(zip(cortes["gameweek"], cortes["corte"]))
    fx["postergado"] = ~fx["jugado"] & (fx["gameweek"].map(corte_de) < limite)
    for _, f in fx[fx["postergado"]].iterrows():
        log.warning("%s GW%d %s-%s sigue sin resultado y su fecha arrancó el %s: "
                    "se lo da por postergado y deja de trabar la predicción.",
                    season, f["gameweek"], f["home_short"], f["away_short"],
                    pd.Timestamp(corte_de[f["gameweek"]]).date())

    filas = []
    for gw, corte in zip(cortes["gameweek"], cortes["corte"]):
        de_la_fecha = fx[fx["gameweek"] == gw]
        # Los partidos de la PROPIA fecha nunca entran acá: por definición del corte,
        # todos tienen kickoff >= corte. Por eso una fecha empezada sigue siendo
        # definitiva, que es lo correcto — sus partidos ya jugados y los que faltan
        # comparten corte y comparten features de equipo, igual que en las 85 dobles
        # fechas que el proyecto ya maneja.
        previos = fx[fx["kickoff_time"] < corte]
        n_previos = len(previos)
        n_previos_jugados = int(previos["jugado"].sum())
        n_previos_pendientes = int((~previos["jugado"] & ~previos["postergado"]).sum())
        estructural = n_previos_pendientes == 0

        opta_completa = True
        if estructural and comp is not None:
            opta_completa = cobertura_opta(comp, stats, season, corte) >= 1.0

        n = len(de_la_fecha)
        n_jugados = int(de_la_fecha["jugado"].sum())
        # `n_pendientes` cuenta lo que todavía se puede predecir. Un partido postergado
        # no está jugado, pero tampoco se espera: no cuenta.
        n_pendientes = int((~de_la_fecha["jugado"] & ~de_la_fecha["postergado"]).sum())
        situacion = ("completa" if n_jugados == n
                     else "en_curso" if n_jugados > 0
                     else "pendiente")

        filas.append({
            "season": season, "gameweek": int(gw), "corte": corte,
            "n": n, "n_jugados": n_jugados, "n_pendientes": n_pendientes,
            "n_previos": n_previos, "n_previos_jugados": n_previos_jugados,
            "n_previos_pendientes": n_previos_pendientes,
            "opta_completa": opta_completa,
            "definitiva": estructural and opta_completa,
            "situacion": situacion,
        })

    return pd.DataFrame(filas, columns=COLS_ESTADO).sort_values("gameweek").reset_index(drop=True)


def gameweeks_completas(fixtures: pd.DataFrame, matches: pd.DataFrame,
                        season: str | None = None) -> list[int]:
    """Gameweeks donde TODOS los partidos tienen resultado.

    Vive acá y no en `monitoring/` para que haya una sola definición de "fecha jugada" en
    el repo: dos definiciones es como se produce una discrepancia que nadie ve.
    """
    e = estado(fixtures, matches, season)
    if e.empty:
        return []
    return sorted(int(g) for g in e.loc[e["situacion"] == "completa", "gameweek"])


def proxima_predecible(fixtures: pd.DataFrame, matches: pd.DataFrame,
                       season: str | None = None,
                       ahora: pd.Timestamp | None = None,
                       comp: pd.DataFrame | None = None,
                       stats: pd.DataFrame | None = None) -> int | None:
    """La gameweek más chica que tiene partidos sin jugar y cuya fila ya es definitiva.

    `None` cuando no queda ninguna: fin de temporada, o la próxima todavía depende de
    partidos que no se jugaron.
    """
    e = estado(fixtures, matches, season, comp, stats, ahora)
    if e.empty:
        return None

    # Candidata = le queda algo por jugar que todavía se espera. Los postergados ya
    # quedaron descontados en `estado`.
    pendientes = e[e["n_pendientes"] > 0]
    if pendientes.empty:
        return None

    definitivas = pendientes[pendientes["definitiva"]]
    if definitivas.empty:
        if not pendientes.empty:
            f = pendientes.iloc[0]
            log.info("La próxima fecha pendiente (GW%d) todavía no es definitiva: "
                     "%d de %d partidos anteriores a su corte sin jugar%s.",
                     f["gameweek"], f["n_previos_pendientes"], f["n_previos"],
                     "" if f["opta_completa"] else ", y falta Opta")
        return None

    return int(definitivas["gameweek"].iloc[0])


def objetivos_inferencia(fixtures: pd.DataFrame, matches: pd.DataFrame,
                         cortes: pd.DataFrame | None = None,
                         season: str | None = None,
                         gameweek: int | None = None,
                         ahora: pd.Timestamp | None = None,
                         comp: pd.DataFrame | None = None,
                         stats: pd.DataFrame | None = None) -> pd.DataFrame:
    """Los fixtures SIN JUGAR de la próxima fecha, con la forma de `gold_tp._objetivos`.

    DataFrame vacío —pero con las columnas correctas— si no hay fecha predecible. Que sea
    vacío y no `None` es a propósito: `construir` lo concatena sin ramas.
    """
    season = season or CFG.current_season
    if gameweek is None:
        gameweek = proxima_predecible(fixtures, matches, season, ahora, comp, stats)
    if gameweek is None:
        return pd.DataFrame(columns=OBJETIVO_COLS)

    fx = fixtures[(fixtures["season"] == season)
                  & (fixtures["gameweek"] == gameweek)].copy()
    if fx.empty:
        return pd.DataFrame(columns=OBJETIVO_COLS)

    sin_jugar = fx[~jugados(fx, matches)]
    if sin_jugar.empty:
        return pd.DataFrame(columns=OBJETIVO_COLS)

    if cortes is None:
        cortes = tf.cortes_por_fecha(fixtures[fixtures["season"] == season])

    obj = sin_jugar[["season", "gameweek", "fixture_id", "kickoff_time",
                     "home_short", "away_short"]].merge(
        cortes, on=["season", "gameweek"], how="left", validate="many_to_one")
    if obj["corte"].isna().any():
        raise ValueError(f"La fecha {season} GW{gameweek} no tiene corte en fact_fixture.")

    log.info("Objetivos de inferencia: %s GW%d, %d partido(s) sin jugar, corte %s",
             season, gameweek, len(obj), obj["corte"].iloc[0])
    return obj[OBJETIVO_COLS].sort_values("kickoff_time").reset_index(drop=True)
