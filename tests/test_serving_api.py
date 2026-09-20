"""Prueba corta de la API: confirma que el contrato HTTP funciona de punta a punta.

Corre contra los artefactos reales del repo (Silver + modelo versionado), no contra un
backend mockeado: si faltan, se salta con instrucciones de qué correr antes, igual que el
resto de `tests/` (ver `conftest.py`).
"""

from __future__ import annotations

import json
import logging
import time

import pytest
from fastapi.testclient import TestClient

from common.config import CFG
from serving import observability
from serving.main import app
from training import registry


@pytest.fixture
def eventos():
    """Captura las líneas que emite el logger de eventos y las devuelve ya parseadas."""
    observability.configure_logging()
    mensajes: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda record: mensajes.append(record.getMessage())
    observability.logger.addHandler(handler)
    yield lambda: [json.loads(m) for m in mensajes]
    observability.logger.removeHandler(handler)


def test_health_reports_modelo_cargado():
    if not (registry.RAIZ / CFG.modelo).exists():
        pytest.skip(f"No hay modelo en models/{CFG.modelo}/. Corré: python -m training.run")
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_devuelve_partidos_de_una_fecha_real(fact_fixture):
    if not (registry.RAIZ / CFG.modelo).exists():
        pytest.skip(f"No hay modelo en models/{CFG.modelo}/. Corré: python -m training.run")

    de_la_temporada = fact_fixture[fact_fixture["season"] == CFG.current_season]
    if de_la_temporada.empty:
        pytest.skip(f"No hay fixtures de {CFG.current_season} en fact_fixture.")
    gameweek = int(de_la_temporada["gameweek"].min())

    client = TestClient(app)
    response = client.get(f"/predict/{CFG.current_season}/{gameweek}")
    assert response.status_code == 200

    body = response.json()
    assert body["count"] > 0
    assert body["count"] == len(body["predictions"])
    for match in body["predictions"]:
        assert match["prediccion"] in ("home", "draw", "away")
        total = match["p_home"] + match["p_draw"] + match["p_away"]
        assert abs(total - 1.0) < 1e-6


def test_predict_fecha_inexistente_da_404():
    client = TestClient(app)
    response = client.get(f"/predict/{CFG.current_season}/9999")
    assert response.status_code == 404


def test_log_event_es_una_linea_json_valida(eventos):
    observability.log_event("prediction", latency_ms=1.2, decision="monitor")
    (evento,) = eventos()
    assert evento == {"event": "prediction", "latency_ms": 1.2, "decision": "monitor"}


def test_measure_latency_registra_aunque_el_bloque_falle():
    with pytest.raises(RuntimeError):
        with observability.measure_latency() as timer:
            time.sleep(0.01)
            raise RuntimeError("falla")
    assert timer["latency_ms"] >= 5


def test_predict_emite_evento_prediction(fact_fixture, eventos):
    if not (registry.RAIZ / CFG.modelo).exists():
        pytest.skip(f"No hay modelo en models/{CFG.modelo}/. Corré: python -m training.run")

    de_la_temporada = fact_fixture[fact_fixture["season"] == CFG.current_season]
    if de_la_temporada.empty:
        pytest.skip(f"No hay fixtures de {CFG.current_season} en fact_fixture.")
    gameweek = int(de_la_temporada["gameweek"].min())

    response = TestClient(app).get(f"/predict/{CFG.current_season}/{gameweek}")
    assert response.status_code == 200

    (evento,) = [e for e in eventos() if e["event"] == "prediction"]
    assert evento["season"] == CFG.current_season
    assert evento["gameweek"] == gameweek
    assert evento["count"] == response.json()["count"]
    assert evento["latency_ms"] > 0
    assert sum(evento["predicciones"].values()) == evento["count"]
    for campo in ("model_version", "feature_set_version", "confianza_media"):
        assert campo in evento
    # Solo agregados: nada por partido ni features del modelo.
    for prohibido in ("p_home", "p_draw", "p_away", "predictions", "fixture_id"):
        assert prohibido not in evento


def test_predict_fecha_inexistente_loguea_el_error(eventos):
    response = TestClient(app).get(f"/predict/{CFG.current_season}/9999")
    assert response.status_code == 404

    (evento,) = [e for e in eventos() if e["event"] == "prediction_error"]
    assert evento["status"] == 404
    assert evento["gameweek"] == 9999
    assert evento["error_type"] == "ValueError"
    assert evento["latency_ms"] > 0
