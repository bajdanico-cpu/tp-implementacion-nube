"""Equipos ascendidos: cómo detectarlos y qué ponerles donde no hay historia.

Dos cosas que hay que hacer bien:

**1 · `dim_team.promoted` NO significa "ascendido".** Significa "primera temporada dentro
de la ventana ingestada". Marca 1 equipo en 2024-25 cuando en realidad ascendieron 3: los
otros dos ya habían aparecido antes en la ventana. Usarlo como flag de ascenso mete ruido
justo en los equipos sobre los que menos se sabe. Se deriva a mano como *"no estaba en la
temporada anterior"*, que da BUR/LUT/SHU (23-24), IPS/LEI/SOU (24-25), BUR/LEE/SUN (25-26)
y COV/HUL/IPS (26-27) — los tres de cada año, que es lo correcto.

**2 · El prior se ajusta en train y se CONGELA.** Si se recalculara en serving sobre datos
nuevos habría train/serve skew: el modelo habría aprendido con un valor y recibiría otro.
Es el mismo error que el canvas denuncia en `strength_*`, en otra forma. Por eso
`ajustar_prior` recibe explícitamente las temporadas de train y el resultado va al
`metadata.json` del modelo.

El prior se aplica **sólo** donde el equipo es ascendido y no tiene ninguna historia. El
resto de los NaN se deja como NaN a propósito: XGBoost y Random Forest aprenden una
dirección por defecto para el faltante, y "este equipo no tiene historia" es información
real que imputar borraría.
"""

from __future__ import annotations

import pandas as pd

from common.logging_setup import get_logger
from features import spec

log = get_logger(__name__)

# Sobre cuántos partidos de un ascendido se calcula el prior. Cinco es la ventana larga:
# es el período en que un recién llegado todavía no tiene historia propia suficiente.
PARTIDOS_PRIOR = 5

STATS_PRIOR = [s.nombre for s in spec.BASE_STATS] + ["dg"]


def flags_ascendido(dim: pd.DataFrame) -> pd.DataFrame:
    """(season, short_name) -> es_ascendido, derivado de la presencia temporada a temporada.

    La primera temporada de la ventana devuelve NaN, no False: no hay con qué saberlo, y
    decir "no ascendió" sería inventar. El NaN es honesto y los árboles lo manejan.
    """
    presencia = dim[["season", "short_name"]].drop_duplicates()
    temporadas = sorted(presencia["season"].unique())
    anterior = {s: temporadas[i - 1] if i else None
                for i, s in enumerate(temporadas)}

    por_temporada = {s: set(g["short_name"])
                     for s, g in presencia.groupby("season")}

    filas = []
    for _, r in presencia.iterrows():
        prev = anterior[r["season"]]
        val = pd.NA if prev is None else (r["short_name"] not in por_temporada[prev])
        filas.append({"season": r["season"], "team_short": r["short_name"],
                      "es_ascendido": val})
    out = pd.DataFrame(filas)
    out["es_ascendido"] = out["es_ascendido"].astype("boolean")
    return out


def ajustar_prior(largo: pd.DataFrame, flags: pd.DataFrame,
                  temporadas_train: list[str]) -> dict[str, float]:
    """Media de cada estadística base en los primeros partidos de los ascendidos.

    SÓLO sobre las temporadas de train. El resultado se congela en el metadata del modelo
    y se reusa en serving sin recalcular.
    """
    d = largo.merge(flags, on=["season", "team_short"], how="left")
    d = d[d["season"].isin(temporadas_train) & (d["es_ascendido"] == True)]  # noqa: E712
    d = d.sort_values(["season", "team_short", "kickoff_time"])
    primeros = d.groupby(["season", "team_short"]).head(PARTIDOS_PRIOR)

    prior = {c: float(primeros[c].mean()) for c in STATS_PRIOR if c in primeros}
    equipos = primeros.groupby(["season", "team_short"]).ngroups
    log.info("Prior de ascendidos ajustado sobre %d equipos-temporada (%d partidos) de %s",
             equipos, len(primeros), ", ".join(temporadas_train))
    return prior


def aplicar_prior(gold: pd.DataFrame, prior: dict[str, float]) -> pd.DataFrame:
    """Rellena las ventanas SÓLO de los ascendidos sin historia.

    Un ascendido puede no tener ningún partido de Premier en la ventana ingestada —
    Coventry y Hull en 2026-27 son el caso extremo, con cero. Dejarlos en NaN es una
    opción; darles el promedio histórico de los ascendidos es mejor porque es información
    real ("a los recién llegados les va así"), y `es_ascendido` más `n_hist` le permiten
    al modelo descontar esa señal.

    **Las ventanas intra-temporada (`_u5_temp`) quedan afuera del relleno.** En la fecha 1
    están vacías para los veinte equipos, no sólo para los ascendidos: es un estado
    compartido y bien definido ("todavía no se jugó nada esta temporada"), y rellenar sólo
    a algunos lo convertiría en una señal inconsistente. El prior cubre las ventanas que
    cruzan temporadas, que son las que a un recién ascendido le quedarían vacías para
    siempre.
    """
    out = gold.copy()
    rellenadas = 0
    for lado in spec.LADOS:
        sin_hist = out[f"{lado}_n_hist"].isna() | (out[f"{lado}_n_hist"] == 0)
        objetivo = sin_hist & (out[f"{lado}_es_ascendido"] == True)  # noqa: E712
        if not objetivo.any():
            continue
        for col in spec.FEATURES:
            if not col.startswith(f"{lado}_"):
                continue
            base = _stat_de(col, lado)
            if base is None or base not in prior:
                continue
            faltan = objetivo & out[col].isna()
            if faltan.any():
                out.loc[faltan, col] = prior[base]
                rellenadas += int(faltan.sum())
    if rellenadas:
        log.info("Prior de ascendidos aplicado a %d celdas", rellenadas)
    return out


# Sufijos de ventana que el prior SÍ rellena: son los que cruzan temporadas y quedarían
# vacíos para siempre en un equipo sin historia de Premier. `_u5_temp` no está.
SUFIJOS_RELLENABLES = ("_cond_u5", "_u3", "_u5")


def _stat_de(col: str, lado: str) -> str | None:
    """`local_pts_def_u5` -> `pts_def`. None si la columna no se debe rellenar."""
    resto = col[len(lado) + 1:]
    if resto.endswith("_u5_temp"):
        return None
    for sufijo in SUFIJOS_RELLENABLES:
        if resto.endswith(sufijo):
            return resto[: -len(sufijo)]
    return None
