"""Logs del lado del servidor, su correlación con el cliente, y los pins del rollback.

Tres cosas que se rompen sin que nada falle a la vista:

- que cada línea de log lleve el `trace` del request: sin él, en Cloud Logging la línea de
  Cloud Run y el evento de la app son entradas sueltas;
- que `smoke_load` no caiga en silencio en el servicio de otro proyecto (pasó: los logs
  "no aparecían" porque el tráfico iba a otro lado);
- que una revisión de Cloud Run pueda fijar modelo y temporada, que es lo que hace que un
  rollback por tráfico vuelva atrás algo más que el código.
"""

from __future__ import annotations

import io
import json
import logging

import pytest
from fastapi.testclient import TestClient

from common import logging_setup as ls
from common.config import CFG


@pytest.fixture
def proyecto(monkeypatch):
    monkeypatch.setenv("TP_GCP_PROJECT", "proyecto-test")
    monkeypatch.setattr(ls, "_PROYECTO", [])
    yield "proyecto-test"


# ---------------------------------------------------------------------------
# trace y corrida
# ---------------------------------------------------------------------------

def test_trace_desde_la_cabecera_de_cloud_run(proyecto):
    t = ls.trace_de({"x-cloud-trace-context": "abc123/456;o=1"})
    assert t == "projects/proyecto-test/traces/abc123"


def test_trace_desde_traceparent_w3c(proyecto):
    t = ls.trace_de({"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"})
    assert t == "projects/proyecto-test/traces/4bf92f3577b34da6a3ce929d0e0e4736"


def test_sin_proyecto_no_hay_trace(monkeypatch):
    """Sin proyecto el campo quedaría mal formado y Cloud Logging lo ignoraría."""
    for v in ("TP_GCP_PROJECT", "GOOGLE_CLOUD_PROJECT", "K_SERVICE", "CLOUD_RUN_JOB"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(ls, "_PROYECTO", [])
    assert ls.trace_de({"x-cloud-trace-context": "abc/1"}) is None


def _formatear(record_extra: dict) -> dict:
    rec = logging.LogRecord("serving.api", logging.INFO, __file__, 1, "hola", (), None)
    for k, v in record_extra.items():
        setattr(rec, k, v)
    return json.loads(ls.FormatoJSON().format(rec))


def test_el_json_lleva_trace_y_corrida_mientras_dura_el_request():
    token = ls.abrir_request("projects/p/traces/t1", "smoke-1")
    try:
        dentro = _formatear({"evento": "prediccion"})
    finally:
        ls.cerrar_request(token)
    fuera = _formatear({"evento": "prediccion"})

    assert dentro["logging.googleapis.com/trace"] == "projects/p/traces/t1"
    assert dentro["corrida"] == "smoke-1"
    # Fuera del request (el Job, un test) no se inventa nada.
    assert "logging.googleapis.com/trace" not in fuera and "corrida" not in fuera


def test_la_corrida_que_viene_de_afuera_se_recorta():
    token = ls.abrir_request(None, "x" * 500)
    try:
        assert len(_formatear({})["corrida"]) == 64
    finally:
        ls.cerrar_request(token)


def test_el_evento_de_la_api_sale_con_el_trace_del_request(proyecto):
    """De punta a punta: la cabecera que pone Cloud Run llega al evento `prediccion_error`.

    Se pide una fecha imposible para no depender de que haya Gold: con o sin Gold, el
    request falla (409 o 503) y deja su evento.
    """
    from serving import main
    from serving import observability as obs

    salida = io.StringIO()
    h = logging.StreamHandler(salida)
    h.setFormatter(ls.FormatoJSON())
    obs.log.addHandler(h)
    main.ESTADO = main.Estado()
    try:
        with TestClient(main.app) as c:
            r = c.get(f"/predict/{CFG.current_season}/999",
                      headers={"X-Cloud-Trace-Context": "cafe0001/1;o=1",
                               "X-Corrida": "smoke-test"})
    finally:
        obs.log.removeHandler(h)
        main.ESTADO = main.Estado()

    assert r.status_code in (409, 503)
    eventos = [json.loads(x) for x in salida.getvalue().splitlines() if x.strip()]
    (ev,) = [e for e in eventos if e.get("evento") == "prediccion_error"]
    assert ev["logging.googleapis.com/trace"] == "projects/proyecto-test/traces/cafe0001"
    assert ev["corrida"] == "smoke-test"


# ---------------------------------------------------------------------------
# smoke_load: nunca a otro proyecto en silencio
# ---------------------------------------------------------------------------

def test_smoke_load_prioriza_url_despues_env_despues_gcloud():
    from scripts import smoke_load as sl

    nunca = lambda: pytest.fail("no debía preguntarle a gcloud")  # noqa: E731
    assert sl.resolver_url("http://a", {"SERVICE_URL": "http://b"}, nunca) == ("http://a", "--url")
    assert sl.resolver_url(None, {"SERVICE_URL": "http://b"}, nunca) == ("http://b", "SERVICE_URL")
    url, origen = sl.resolver_url(None, {}, lambda: "https://mio.run.app")
    assert url == "https://mio.run.app" and "gcloud" in origen


def test_smoke_load_sin_url_corta_en_vez_de_ir_a_otro_servicio():
    from scripts import smoke_load as sl

    with pytest.raises(SystemExit, match="SERVICE_URL"):
        sl.resolver_url(None, {}, lambda: None)
    assert not hasattr(sl, "DEFAULT_URL")


# ---------------------------------------------------------------------------
# logs_servidor: lo que el servidor sabe de esos requests
# ---------------------------------------------------------------------------

def _borde(lat: str, status: int, rev: str = "premier-ml-api-00002-abc") -> dict:
    return {"logName": "projects/p/logs/run.googleapis.com%2Frequests",
            "httpRequest": {"latency": lat, "status": status},
            "resource": {"labels": {"revision_name": rev}}}


def _app(evento: str, **campos) -> dict:
    return {"logName": "projects/p/logs/run.googleapis.com%2Fstdout",
            "jsonPayload": {"evento": evento, **campos},
            "resource": {"labels": {"revision_name": "premier-ml-api-00002-abc"}}}


def test_logs_servidor_separa_borde_y_app():
    from scripts import logs_servidor as lsv

    entradas = [
        _borde("0.261346s", 200), _borde("0.249s", 409), _borde("0.250s", 409),
        _app("prediccion", latencia_ms=94.3, estado="proxima", origen="en_vivo",
             model_version="20260825T024144Z"),
        _app("prediccion_error", latencia_ms=3.7, status=409, error_type="FechaNoPreparada"),
        _app("prediccion_error", latencia_ms=3.9, status=409, error_type="FechaNoPreparada"),
        # la línea de "Prediciendo ..." no tiene `evento`: no cuenta como request
        {"logName": "x%2Fstdout", "jsonPayload": {"message": "Prediciendo"}},
    ]
    r = lsv.resumir(entradas)

    assert r["n_borde"] == 3
    assert r["status_borde"] == {200: 1, 409: 2}
    assert r["lat_borde"][0] == pytest.approx(261.346)
    assert sorted(r["lat_app"]) == [3.7, 3.9, 94.3]
    assert r["por_estado"] == {"proxima/en_vivo": [94.3]}
    assert r["errores"] == {(409, "FechaNoPreparada"): 2}
    assert r["revisiones"] == {"premier-ml-api-00002-abc": 3}
    assert r["modelos"] == {"20260825T024144Z": 1}


def test_logs_servidor_filtra_la_corrida_por_los_dos_lados():
    from scripts import logs_servidor as lsv

    f = lsv.filtro("premier-ml-api", "smoke-1")
    assert 'jsonPayload.corrida="smoke-1"' in f          # la app
    assert 'httpRequest.userAgent:"smoke_load/smoke-1"' in f   # Cloud Run


def test_latencia_de_borde_mal_formada_no_rompe():
    from scripts import logs_servidor as lsv

    assert lsv.latencia_borde_ms({"httpRequest": {"latency": "abc"}}) is None
    assert lsv.latencia_borde_ms({}) is None


# ---------------------------------------------------------------------------
# resumen_actualizacion: del botón al Gold nuevo
# ---------------------------------------------------------------------------

def test_resumen_toma_el_ultimo_disparo_y_su_fin():
    from scripts import resumen_actualizacion as ra

    ev = [
        {"timestamp": "2026-09-28T20:00:00Z", "jsonPayload": {"evento": "pipeline_disparo", "tarea": "A"}},
        {"timestamp": "2026-09-28T22:00:00Z", "jsonPayload": {"evento": "pipeline_disparo", "tarea": "B"}},
        {"timestamp": "2026-09-28T20:07:00Z", "jsonPayload": {"evento": "pipeline_fin", "tarea": "A"}},
        {"timestamp": "2026-09-28T22:06:00Z", "jsonPayload": {"evento": "pipeline_fin", "tarea": "B",
                                                              "estado": "ok"}},
    ]
    disparo, fin = ra.resumir_eventos(ev)
    assert disparo["jsonPayload"]["tarea"] == "B"
    assert fin["jsonPayload"]["tarea"] == "B"
    assert ra.resumir_eventos([]) == (None, None)


def test_resumen_de_la_corrida_marca_el_paso_que_fallo():
    from scripts import resumen_actualizacion as ra

    lineas = ra.resumir_corrida({
        "corrida": "20260928T220000Z", "at": "2026-09-28T22:06:00+00:00", "ok": False,
        "pasos": [
            {"paso": "bronze_fpl", "estado": "ok", "segundos": 3.1, "salida": None},
            {"paso": "gold", "estado": "ok", "segundos": 110.4,
             "salida": {"filas": 1580, "columnas": 301}},
            {"paso": "bronze_opta", "estado": "error", "segundos": 2.0,
             "error": "HTTPError: 503 Service Unavailable"},
        ]})
    texto = "\n".join(lineas)
    assert "FALLÓ" in texto and "1580 filas" in texto and "HTTPError: 503" in texto
    assert lineas[-1].split()[-1] == "115.5"


# ---------------------------------------------------------------------------
# Rollback: lo que una revisión puede fijar
# ---------------------------------------------------------------------------

def test_tp_model_version_fija_el_modelo_de_la_revision(monkeypatch):
    """La revisión pide una versión que no existe: tiene que fallar CON ESE nombre, no
    caer en silencio al PRODUCTION.json del bucket."""
    from serving import predict

    monkeypatch.setenv("TP_MODEL_VERSION", "19990101T000000Z")
    with pytest.raises(Exception, match="19990101T000000Z"):
        predict.cargar_modelo()


def test_tp_season_fija_la_temporada_de_la_revision(monkeypatch):
    monkeypatch.delenv("TP_SEASON", raising=False)
    de_config = CFG.current_season
    monkeypatch.setenv("TP_SEASON", "2027-28")
    assert CFG.current_season == "2027-28"
    monkeypatch.delenv("TP_SEASON")
    assert CFG.current_season == de_config
