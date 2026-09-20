"""Contratos Pydantic de la API. Solo forma HTTP: la lógica vive en `serving/predict.py`.

Una fecha puede estar en tres estados, y la respuesta dice en cuál está en vez de
disimularlo:

    jugada / en_curso  -> la predicción CONGELADA del registro, más el resultado real
    proxima            -> se predice en vivo contra Gold
    (no preparada)     -> 409, diciendo cuál es la próxima predecible

Esa distinción es el punto. Antes cualquier fecha devolvía 200 con diez predicciones
indistinguibles entre sí: pedir la 7 estando en la 5 daba números calculados con la
información de la 4, y pedir la 2 daba una reconstrucción que no coincidía con lo que el
sistema había anunciado en su momento. `origen` es lo que separa una cosa de la otra.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Estado = Literal["jugada", "en_curso", "proxima"]
Origen = Literal["registro", "en_vivo"]


class HealthResponse(BaseModel):
    """El pulso: está arriba, con el modelo cargado y con un Gold utilizable."""

    status: str
    model_name: str | None = None
    model_version: str | None = None
    feature_set_version: str | None = None
    n_seeds: int | None = None
    backend: str | None = None
    gold_built_at: str | None = None
    gold_filas: int | None = None
    season_actual: str | None = None
    proxima_predecible: int | None = None
    detail: str | None = None


class MatchPrediction(BaseModel):
    fixture_id: int
    home_short: str
    away_short: str
    kickoff_time: str
    p_home: float
    p_draw: float
    p_away: float
    prediccion: str
    confianza: float
    # Sólo cuando el partido ya terminó. Una fecha puede estar jugada y tener partidos
    # sin resultado todavía: pasa cada fin de semana entre el pitazo final y la ingesta.
    target_1x2: str | None = None
    home_goals: int | None = None
    away_goals: int | None = None
    acierto: bool | None = None
    # Cuánto pagaba cada resultado. Son las cuotas de CIERRE de football-data, que se
    # publican después del partido: por eso sólo aparecen en fechas ya jugadas, y por eso
    # tampoco pueden ser features del modelo — en producción no llegan a tiempo.
    cuota_home: float | None = None
    cuota_draw: float | None = None
    cuota_away: float | None = None


class GameweekPredictionResponse(BaseModel):
    season: str
    gameweek: int
    count: int
    estado: Estado
    origen: Origen
    corte: str
    model_name: str
    model_version: str
    feature_set_version: str
    predicted_at: str
    # Cuándo se construyó la fila de features que produjo esto. Es la trazabilidad que
    # faltaba: sin ella, dos respuestas idénticas pueden venir de dos Golds distintos.
    gold_built_at: str | None = None
    # False cuando la predicción del registro se emitió DESPUÉS del inicio de la fecha.
    # No es una predicción, es una reconstrucción, y se dice.
    pre_deadline: bool | None = None
    n_con_resultado: int = 0
    accuracy: float | None = None
    predictions: list[MatchPrediction]


class FechaNoPreparadaResponse(BaseModel):
    """El cuerpo del 409. Estructurado, para que el cliente pueda reaccionar."""

    detail: str
    season: str
    gameweek: int
    proxima_predecible: int | None = None


class FechaCalendario(BaseModel):
    gameweek: int
    corte: str
    n_partidos: int
    estado: Estado | Literal["no_preparada"]
    tiene_registro: bool
    n_con_resultado: int = 0
    accuracy: float | None = None


class CalendarioResponse(BaseModel):
    """El índice de la temporada: qué se puede pedir y qué hay de cada fecha.

    Lista **las 38 fechas**, no sólo las que tienen features. Antes sólo salían las que
    estaban en Gold —hoy, la 1 a la 5— y el resto simplemente no existía en la pantalla:
    no se veía que la 18 no se puede predecir, se veía que la 18 no estaba. Un límite
    invisible no se puede entender ni defender.
    """

    season: str
    proxima_predecible: int | None
    gold_built_at: str | None = None
    hasta_gameweek: int | None = None
    fechas: list[FechaCalendario]


class Tarea(BaseModel):
    """Una actualización del pipeline pedida desde la API."""

    id: str
    estado: str            # lanzada | corriendo | ok | error
    motivo: str
    detalle: str = ""
    destino: str           # local | cloud-run-job
    referencia: str | None = None
    segundos: float = 0.0


class Diagnostico(BaseModel):
    """Hasta dónde llega Gold, y si conviene actualizar."""

    hace_falta: bool
    motivo: str
    proxima_predecible: int | None = None
    corte_proxima: str | None = None
    gold_built_at: str | None = None
    ultimo_partido_en_gold: str | None = None
    en_curso: Tarea | None = None
    puede_disparar: bool = True
