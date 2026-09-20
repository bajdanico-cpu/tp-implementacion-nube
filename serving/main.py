"""API HTTP sobre `serving/predict.py`.

    uvicorn serving.main:app --reload --port 8080

Capa fina a propósito: toda la lógica (features, validación de columnas, control
anti-leakage, decisión) ya vive en `serving/predict.py` y `serving/decision.py`, con sus
propios tests. Acá solo se traduce eso a HTTP: contrato, códigos de error y un evento
estructurado por request (`serving/observability.py`).

No se llama a `predict.guardar()` desde acá: un GET es una lectura y no debería tener el
efecto secundario de escribir un parquet nuevo cada vez que alguien lo pide. El registro
para el ciclo cerrado (predicción -> resultado -> métricas) es responsabilidad del job
batch pre-deadline (`CONTEXTO-TP-PREMIER-ML.md` §6), no de este endpoint.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from eda.baselines import CLASES_ORD
from serving import predict
from serving.observability import configure_logging, log_event, measure_latency
from serving.schemas import GameweekPredictionResponse, HealthResponse, MatchPrediction

app = FastAPI(
    title="TP Premier ML — API",
    version="0.1.0",
    description="Predicción 1X2 de la fecha de Premier League pedida.",
)

configure_logging()


def _status_de(error: Exception) -> int:
    if isinstance(error, ValueError):
        return 404
    if isinstance(error, FileNotFoundError):
        return 503
    return 500


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        _, meta = predict.cargar_modelo()
        return HealthResponse(
            status="ok",
            model_name=meta["model_name"],
            model_version=meta["model_version"],
            feature_set_version=meta["feature_set_version"],
            n_seeds=meta.get("n_seeds"),
        )
    except Exception as exc:  # noqa: BLE001 — health nunca tira, informa degraded
        log_event("health_degraded", error_type=type(exc).__name__, reason=str(exc))
        return HealthResponse(status="degraded", detail=str(exc))


@app.get("/predict/{season}/{gameweek}", response_model=GameweekPredictionResponse)
def predict_gameweek(season: str, gameweek: int) -> GameweekPredictionResponse:
    error: Exception | None = None
    with measure_latency() as timer:
        try:
            pred = predict.predecir(season, gameweek)
        except (ValueError, FileNotFoundError, AssertionError) as exc:
            error = exc

    if error is not None:
        status = _status_de(error)
        log_event(
            "prediction_error",
            season=season,
            gameweek=gameweek,
            status=status,
            error_type=type(error).__name__,
            reason=str(error),
            latency_ms=timer["latency_ms"],
        )
        raise HTTPException(status_code=status, detail=str(error)) from error

    predictions = [
        MatchPrediction(
            fixture_id=int(row.fixture_id),
            home_short=row.home_short,
            away_short=row.away_short,
            kickoff_time=row.kickoff_time.isoformat(),
            p_home=float(row.p_home),
            p_draw=float(row.p_draw),
            p_away=float(row.p_away),
            prediccion=row.prediccion,
            confianza=float(row.confianza),
        )
        for row in pred.itertuples()
    ]
    log_event(
        "prediction",
        season=season,
        gameweek=gameweek,
        count=len(predictions),
        model_name=pred["model_name"].iloc[0],
        model_version=pred["model_version"].iloc[0],
        feature_set_version=pred["feature_set_version"].iloc[0],
        predicciones={c: int((pred["prediccion"] == c).sum()) for c in CLASES_ORD},
        confianza_media=round(float(pred["confianza"].mean()), 4),
        latency_ms=timer["latency_ms"],
    )
    return GameweekPredictionResponse(
        season=season,
        gameweek=gameweek,
        count=len(predictions),
        model_name=pred["model_name"].iloc[0],
        model_version=pred["model_version"].iloc[0],
        feature_set_version=pred["feature_set_version"].iloc[0],
        predicted_at=pred["predicted_at"].iloc[0],
        predictions=predictions,
    )
