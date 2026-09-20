"""Disparar el pipeline desde la API: autorización, concurrencia y diagnóstico.

Ningún test de acá corre el pipeline de verdad —tarda minutos y escribe Gold—, así que el
lanzamiento se sustituye. Lo que se prueba es la **política**: quién puede pedirlo, qué
pasa si ya hay uno corriendo, y si el sistema sabe cuándo hace falta.
"""

from __future__ import annotations

import pandas as pd
import pytest

from common.config import CFG
from serving import tareas


@pytest.fixture(autouse=True)
def limpio(monkeypatch):
    """Cada test arranca sin tareas y fuera de la nube."""
    tareas._TAREAS.clear()
    monkeypatch.delenv("K_SERVICE", raising=False)
    monkeypatch.delenv("TP_ADMIN_TOKEN", raising=False)
    yield
    tareas._TAREAS.clear()


@pytest.fixture
def sin_lanzar(monkeypatch):
    """Sustituye el lanzamiento: la tarea queda 'corriendo' sin que nada corra."""
    def falso(t):
        t.estado, t.destino, t.referencia = "corriendo", "falso", "0"

    monkeypatch.setattr(tareas, "_lanzar_subproceso", falso)
    monkeypatch.setattr(tareas, "_lanzar_job", falso)


# ---------------------------------------------------------------------------
# Autorización
# ---------------------------------------------------------------------------

def test_en_local_se_puede_disparar_sin_token():
    tareas.autorizar(None)          # no levanta


def test_en_la_nube_sin_token_configurado_el_endpoint_esta_apagado(monkeypatch):
    """Apagado es mejor que abierto: un POST anónimo que gasta cómputo es un problema."""
    monkeypatch.setenv("K_SERVICE", "premier-ml-api")
    with pytest.raises(tareas.NoAutorizado, match="apagado"):
        tareas.autorizar("lo-que-sea")


def test_en_la_nube_hace_falta_el_token_correcto(monkeypatch):
    monkeypatch.setenv("K_SERVICE", "premier-ml-api")
    monkeypatch.setenv("TP_ADMIN_TOKEN", "secreto")

    with pytest.raises(tareas.NoAutorizado, match="X-Admin-Token"):
        tareas.autorizar(None)
    with pytest.raises(tareas.NoAutorizado, match="X-Admin-Token"):
        tareas.autorizar("otro")
    tareas.autorizar("secreto")     # no levanta


# ---------------------------------------------------------------------------
# Una sola corrida a la vez
# ---------------------------------------------------------------------------

def test_no_se_pueden_disparar_dos_a_la_vez(sin_lanzar):
    """Dos pipelines simultáneos se pisan escribiendo Gold."""
    primera = tareas.disparar("prueba")
    assert primera.estado == "corriendo"

    with pytest.raises(tareas.YaHayUna) as exc:
        tareas.disparar("otra")
    assert exc.value.tarea.id == primera.id


def test_cuando_termina_se_puede_volver_a_disparar(sin_lanzar):
    import time

    t = tareas.disparar("prueba")
    t.estado, t.terminada_at = "ok", time.time()
    assert tareas.en_curso() is None
    assert tareas.disparar("otra").id != t.id


def test_un_error_al_lanzar_queda_registrado_y_no_bloquea(monkeypatch):
    def explota(t):
        raise RuntimeError("no hay proyecto configurado")

    monkeypatch.setattr(tareas, "_lanzar_subproceso", explota)
    t = tareas.disparar("prueba")
    assert t.estado == "error" and "proyecto" in t.detalle
    assert tareas.en_curso() is None          # no queda trabada


# ---------------------------------------------------------------------------
# ¿Hace falta actualizar?
# ---------------------------------------------------------------------------

def _gold(corte, ultimo_jugado, gw=5):
    """Gold mínimo: una fecha de inferencia y un partido jugado."""
    return pd.DataFrame({
        "season": [CFG.current_season] * 2,
        "gameweek": [gw - 1, gw],
        "split": ["actual", "inferencia"],
        "target_1x2": ["home", None],
        "kickoff_time": [pd.Timestamp(ultimo_jugado), pd.Timestamp(corte)],
        "corte": [pd.Timestamp(ultimo_jugado), pd.Timestamp(corte)],
        "gold_built_at": ["20260101T000000Z"] * 2,
    })


def test_si_el_corte_todavia_no_llego_no_hace_falta():
    manana = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=2)
    ayer = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)
    d = tareas.diagnostico(_gold(corte=manana, ultimo_jugado=ayer))
    assert d["hace_falta"] is False
    assert "es en" in d["motivo"]


def test_si_el_corte_ya_paso_hace_falta_aunque_gold_no_lo_sepa():
    """El caso real del 20/09/2026.

    La GW5 se jugó el 18, pero `fact_match` todavía no la tenía, así que Gold no mostraba
    ningún partido posterior al corte. Mirando sólo el artefacto el sistema contestaba
    'está todo bien' mientras servía la predicción de una fecha ya terminada.
    """
    hace_dos_dias = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=2)
    hace_una_semana = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)
    d = tareas.diagnostico(_gold(corte=hace_dos_dias, ultimo_jugado=hace_una_semana))
    assert d["hace_falta"] is True
    assert "arrancó hace" in d["motivo"]


def test_si_hay_partidos_posteriores_al_corte_la_fila_esta_vencida():
    hace_tres = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=3)
    ayer = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)
    d = tareas.diagnostico(_gold(corte=hace_tres, ultimo_jugado=ayer))
    assert d["hace_falta"] is True
    assert "vencida" in d["motivo"]


def test_sin_gold_siempre_hace_falta():
    assert tareas.diagnostico(None)["hace_falta"] is True
    assert tareas.diagnostico(pd.DataFrame())["hace_falta"] is True


def test_sin_fecha_de_inferencia_hace_falta():
    d = _gold(corte=pd.Timestamp.now(tz="UTC"), ultimo_jugado=pd.Timestamp.now(tz="UTC"))
    d = d[d["split"] != "inferencia"]
    assert tareas.diagnostico(d)["hace_falta"] is True
