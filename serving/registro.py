"""El registro de predicciones: qué anunció el sistema, y cuándo.

Es el artefacto que hace que el ciclo cerrado signifique algo. Una predicción tiene valor
como evidencia sólo si quedó escrita **antes** de que se jugara la fecha; recalcularla
después da otro número, porque cambió el modelo, el feature set, o simplemente el dato de
Silver. Medido: para la GW2 de 2026-27, la predicción del 24/08 y la re-corrida del 30/08
difieren hasta en la clase anunciada (BOU-EVE pasó de `away` a `home`).

Por eso este módulo existe aparte de `predict.py`: el registro no es un detalle de
implementación de la predicción, es la mitad del experimento.

**Append-only, como Bronze.** Nada se pisa: `guardar` nunca sobreescribe, y el sufijo
`_N` desempata dos registros del mismo segundo.

**Pero append-only no es acumular cualquier cosa.** Una predicción vale como evidencia
si la emitió el modelo que está en producción y llegó antes del corte; lo demás es otra
cosa. El registro juntó 18 parquets para cinco fechas: seis versiones distintas de una
tarde de reentrenos —modelos que ya no existen en `models/`, o sea predicciones que
nadie puede auditar— y re-corridas posteriores al partido, que son reconstrucciones.
Dos herramientas, para los dos momentos:

  * `guardar(si_existe="saltar")` evita que vuelva a crecer: si la predicción es
    idéntica a la última de esa fecha, no escribe.
  * `scripts.depurar_registro` limpia lo que ya se acumuló, dejando por fecha la que
    gobierna. No borra: mueve a `data/_registro_desarrollo/`.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import CFG, utc_stamp
from common.logging_setup import get_logger
from common.storage import backend

log = get_logger(__name__)

# Cuelga de `CFG.data_root` y no de `PROJECT_ROOT`: con backend GCS o con `TP_DATA_ROOT`
# apuntando a otro lado, el registro tiene que viajar con el resto del dato.
#
# Y TODO el I/O de este modulo pasa por `backend()`, igual que Silver y Gold. Esta
# ruta no es un archivo: es una clave. Con `pathlib` directo el modulo funcionaba en
# local y en Cloud Run devolvia el registro vacio -- `PREDICCIONES.exists()` da False
# adentro del contenedor, donde no hay `data/` -- asi que toda fecha jugada contestaba
# "ya se jugo y no quedo ninguna prediccion". El servicio arrancaba igual y /health
# decia ok, porque Gold si venia del bucket: fallaba solo la mitad que nadie mira.
PREDICCIONES = CFG.data_root / "predicciones"

# El sufijo `_N` opcional desempata dos registros del MISMO segundo. `utc_stamp()` tiene
# resolucion de segundos, asi que sin el sufijo el segundo archivo pisaba al primero --
# en un modulo cuya promesa es que nada se pisa nunca.
NOMBRE = re.compile(
    r"^(?P<season>\d{4}-\d{2})_GW(?P<gw>\d{2})_(?P<stamp>\d{8}T\d{6}Z)"
    r"(?:_(?P<n>\d+))?\.parquet$")

PROBS = ["p_home", "p_draw", "p_away"]


def listar(season: str | None = None, gameweek: int | None = None) -> pd.DataFrame:
    """Una fila por parquet registrado: season, gameweek, stamp, ruta."""
    filas = []
    for p in backend().list_files(PREDICCIONES, "*.parquet"):
        m = NOMBRE.match(p.name)
        if not m:
            log.warning("Archivo con nombre inesperado en el registro: %s", p.name)
            continue
        filas.append({"season": m["season"], "gameweek": int(m["gw"]),
                      "stamp": m["stamp"], "orden": int(m["n"] or 0), "ruta": p})

    d = pd.DataFrame(filas, columns=["season", "gameweek", "stamp", "orden", "ruta"])
    if season is not None:
        d = d[d["season"] == season]
    if gameweek is not None:
        d = d[d["gameweek"] == int(gameweek)]
    return (d.sort_values(["season", "gameweek", "stamp", "orden"])
             .drop(columns="orden").reset_index(drop=True))


def _corte(pred: pd.DataFrame) -> pd.Timestamp:
    """El inicio de la fecha, derivado del propio registro.

    Es la misma definición que usa Gold (`min(kickoff_time)` de la gameweek), así que no
    hace falta leer Silver para saber si una predicción llegó a tiempo. Importa: el
    servicio no puede leer Silver.
    """
    return pd.Timestamp(pred["kickoff_time"].min())


def congelada(season: str, gameweek: int) -> pd.DataFrame | None:
    """La predicción que VALE para esa fecha, o None si no hay ninguna registrada.

    Hay varios parquets por fecha. La que vale es **la última emitida antes del corte**:
    una predicción emitida con la fecha ya empezada no es una predicción, es una
    reconstrucción. Cuando no existe ninguna anterior al corte —pasa con la GW1 de
    2026-27, que se jugó antes de que el sistema empezara a registrar— se devuelve la más
    temprana disponible, marcada como tal en `registro_pre_deadline`. Mentir por omisión
    sería peor que devolverla con la etiqueta puesta.
    """
    archivos = listar(season, gameweek)
    if archivos.empty:
        return None

    candidatas = []
    for ruta in archivos["ruta"]:
        d = backend().read_dataframe(ruta)
        if d.empty:
            continue
        emitida = pd.Timestamp(d["predicted_at"].iloc[0])
        candidatas.append((emitida, ruta, d))

    if not candidatas:
        return None

    candidatas.sort(key=lambda t: t[0])
    corte = _corte(candidatas[0][2])
    a_tiempo = [c for c in candidatas if c[0] < corte]

    if a_tiempo:
        emitida, ruta, d = a_tiempo[-1]
        pre = True
    else:
        emitida, ruta, d = candidatas[0]
        pre = False
        log.warning("%s GW%d no tiene ninguna predicción anterior al corte (%s): "
                    "se devuelve la más temprana (%s), que es una reconstrucción.",
                    season, gameweek, corte, emitida)

    d = d.copy()
    d["registro_pre_deadline"] = pre
    d["registro_archivo"] = ruta.name
    return d.sort_values("kickoff_time").reset_index(drop=True)


def _es_repetida(pred: pd.DataFrame, previa: pd.DataFrame) -> bool:
    """¿Esta predicción dice exactamente lo mismo que la última registrada?"""
    for col in ("model_version", "feature_set_version"):
        if col not in previa.columns or previa[col].iloc[0] != pred[col].iloc[0]:
            return False
    if len(previa) != len(pred):
        return False

    k = ["season", "gameweek", "fixture_id"]
    a = pred.sort_values(k).reset_index(drop=True)
    b = previa.sort_values(k).reset_index(drop=True)
    if not a[k].equals(b[k]):
        return False
    return bool(np.allclose(a[PROBS].to_numpy(float), b[PROBS].to_numpy(float), atol=1e-12))


def guardar(pred: pd.DataFrame, si_existe: str = "saltar") -> Path | None:
    """Registra la predicción. Devuelve la ruta, o None si no hizo falta escribir.

    `si_existe="saltar"` (el default) no escribe cuando la última registrada de esa fecha
    dice exactamente lo mismo: mismo modelo, mismo feature set, mismas probabilidades.
    Sin eso, cada corrida del pipeline sumaba un archivo idéntico —la GW2 juntó siete— y
    el registro dejaba de servir para saber qué se anunció y cuándo.

    Con `si_existe="siempre"` escribe igual. No hay opción de sobrescribir: el registro es
    append-only, como Bronze, y por la misma razón.
    """
    if si_existe not in ("saltar", "siempre"):
        raise ValueError(f"si_existe inválido: {si_existe!r}")
    if pred.empty:
        raise ValueError("No se registra una predicción vacía.")

    s = pred["season"].iloc[0]
    gw = int(pred["gameweek"].iloc[0])

    if si_existe == "saltar":
        previas = listar(s, gw)
        if not previas.empty:
            ultima = backend().read_dataframe(previas["ruta"].iloc[-1])
            if _es_repetida(pred, ultima):
                log.info("La predicción de %s GW%d es idéntica a la registrada en %s: "
                         "no se escribe de nuevo.", s, gw, previas["stamp"].iloc[-1])
                return None

    base = f"{s}_GW{gw:02d}_{utc_stamp()}"
    ruta = PREDICCIONES / f"{base}.parquet"
    # Append-only de verdad: si ya existe ese nombre (dos registros dentro del mismo
    # segundo) se desempata con un sufijo en vez de sobrescribir.
    i = 0
    while backend().exists(ruta):
        i += 1
        ruta = PREDICCIONES / f"{base}_{i}.parquet"
    backend().write_dataframe(pred, ruta)
    log.info("Predicción registrada en %s", ruta)
    return ruta


def con_resultado(pred: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """Le pega a una predicción registrada el resultado real, donde ya lo haya.

    El resultado sale de **Gold**, no de `fact_match`: el servicio no lee Silver, y Gold
    ya trae `target_1x2` y los goles de todo partido jugado. El cruce va por
    `(season, fixture_id)`, que es la clave del partido en todo el proyecto.

    Queda en NaN lo de los partidos que todavía no terminaron — el estado normal de una
    fecha en curso, entre el pitazo final y la ingesta del resultado.
    """
    cols = ["season", "fixture_id", "target_1x2", "home_goals", "away_goals",
            "odds_avg_close_home", "odds_avg_close_draw", "odds_avg_close_away"]
    real = gold[[c for c in cols if c in gold.columns]].rename(columns={
        "odds_avg_close_home": "cuota_home",
        "odds_avg_close_draw": "cuota_draw",
        "odds_avg_close_away": "cuota_away"})
    d = pred.merge(real, on=["season", "fixture_id"], how="left")
    d["acierto"] = np.where(d["target_1x2"].notna(),
                            d["prediccion"] == d["target_1x2"], None)
    return d
