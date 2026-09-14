"""Prueba corta de la API: confirma que el contrato HTTP funciona de punta a punta.

Corre contra los artefactos reales del repo (Silver + modelo versionado), no contra un
backend mockeado: si faltan, se salta con instrucciones de qué correr antes, igual que el
resto de `tests/` (ver `conftest.py`).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from common.config import CFG
from serving.main import app
from training import registry


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
