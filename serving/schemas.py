"""Contratos Pydantic de la API. Solo forma HTTP: la lógica vive en `serving/predict.py`.

Se exponen los campos "oficiales" de cada predicción (probabilidades + la regla de
producción). Las columnas de reglas candidatas (`serving/decision.py`) y las de auditoría
anti-leakage (`hist_kickoff_*`) no forman parte de este contrato.
"""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_name: str | None = None
    model_version: str | None = None
    feature_set_version: str | None = None
    n_seeds: int | None = None
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


class GameweekPredictionResponse(BaseModel):
    season: str
    gameweek: int
    count: int
    model_name: str
    model_version: str
    feature_set_version: str
    predicted_at: str
    predictions: list[MatchPrediction]
