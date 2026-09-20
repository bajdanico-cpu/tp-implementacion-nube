"""El runner de pasos, con pasos falsos: rápido y sin tocar datos.

Lo que se prueba acá es la política, no los pasos. Que un paso opcional no tire la
corrida, que `--desde` retome sin rehacer la ingesta, y que no se emita una predicción
con la fecha ya empezada. Los pasos reales tienen sus propios tests.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pipeline import pre_deadline
from pipeline.pasos import Paso, correr


def _paso(nombre, corridos, falla=False, obligatorio=True):
    def _correr():
        corridos.append(nombre)
        if falla:
            raise RuntimeError(f"{nombre} explotó")
        return {"filas": 1}

    return Paso(nombre, _correr, obligatorio=obligatorio)


def test_la_cadena_corre_todos_los_pasos_en_orden():
    hechos = []
    res = correr([_paso("a", hechos), _paso("b", hechos), _paso("c", hechos)],
                 registrar=False)
    assert res.ok and hechos == ["a", "b", "c"]


def test_un_paso_obligatorio_que_falla_corta_la_cadena():
    """Seguir después de que Silver falle produce un Gold con datos viejos, en silencio."""
    hechos = []
    res = correr([_paso("a", hechos), _paso("b", hechos, falla=True), _paso("c", hechos)],
                 registrar=False)
    assert not res.ok
    assert hechos == ["a", "b"]           # "c" no llegó a correr
    assert res.fallados == ["b"]


def test_un_paso_opcional_que_falla_no_corta():
    """football-data devuelve 503 cada tanto y sus cuotas no son features."""
    hechos = []
    res = correr([_paso("a", hechos), _paso("fd", hechos, falla=True, obligatorio=False),
                  _paso("c", hechos)], registrar=False)
    assert res.ok
    assert hechos == ["a", "fd", "c"]
    assert res.fallados == ["fd"]


def test_desde_saltea_los_pasos_anteriores():
    """El caso real: la ingesta salió bien y no tiene sentido re-bajar 27 MB."""
    hechos = []
    res = correr([_paso("a", hechos), _paso("b", hechos), _paso("c", hechos)],
                 desde="b", registrar=False)
    assert res.ok and hechos == ["b", "c"]
    assert [p["estado"] for p in res.pasos] == ["salteado", "ok", "ok"]


def test_solo_corre_los_pasos_nombrados():
    hechos = []
    res = correr([_paso("a", hechos), _paso("b", hechos), _paso("c", hechos)],
                 solo=("a", "c"), registrar=False)
    assert res.ok and hechos == ["a", "c"]


def test_un_paso_desconocido_falla_temprano():
    hechos = []
    with pytest.raises(ValueError, match="paso desconocido"):
        correr([_paso("a", hechos)], desde="zzz", registrar=False)


def test_cada_corrida_deja_su_registro(tmp_path, monkeypatch):
    """La evidencia de operación no puede depender de que alguien mire la consola."""
    import json

    from pipeline import pasos as mod

    monkeypatch.setattr(mod, "RUNS", tmp_path)
    hechos = []
    res = correr([_paso("a", hechos), _paso("b", hechos, falla=True)], registrar=True)

    archivos = list(tmp_path.glob("*.json"))
    assert len(archivos) == 1
    guardado = json.loads(archivos[0].read_text(encoding="utf-8"))
    assert guardado["ok"] is False
    assert guardado["pasos"][1]["error"].startswith("RuntimeError")
    assert "traceback" in guardado["pasos"][1]
    assert guardado["corrida"] == res.corrida


# ---------------------------------------------------------------------------
# El guard que distingue una predicción de una reconstrucción
# ---------------------------------------------------------------------------

def _fixtures_falsos(corte: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame({
        "season": ["2099-00"], "gameweek": [7], "kickoff_time": [corte],
        "home_short": ["AAA"], "away_short": ["BBB"],
    })


def test_no_predice_despues_del_corte_sin_forzar(monkeypatch):
    ayer = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)
    monkeypatch.setattr("common.storage.read_table", lambda *_a, **_k: _fixtures_falsos(ayer))

    with pytest.raises(pre_deadline.FechaYaArranco, match="reconstrucción"):
        pre_deadline._guard_pre_deadline("2099-00", 7, forzar=False)


def test_con_forzar_predice_igual(monkeypatch):
    ayer = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)
    monkeypatch.setattr("common.storage.read_table", lambda *_a, **_k: _fixtures_falsos(ayer))
    pre_deadline._guard_pre_deadline("2099-00", 7, forzar=True)   # no levanta


def test_antes_del_corte_no_hay_guard(monkeypatch):
    manana = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)
    monkeypatch.setattr("common.storage.read_table", lambda *_a, **_k: _fixtures_falsos(manana))
    pre_deadline._guard_pre_deadline("2099-00", 7, forzar=False)  # no levanta


def test_los_pasos_del_pre_deadline_estan_en_el_orden_correcto():
    """`competencias` depende de `dim_team`, así que Silver tiene que ir antes."""
    nombres = [p.nombre for p in pre_deadline._pasos("2099-00", 1, True, False, False)]
    assert nombres.index("silver") < nombres.index("competencias")
    assert nombres.index("silver") < nombres.index("opta")
    assert nombres.index("gold") < nombres.index("predecir")
    assert all(nombres.index(b) < nombres.index("silver")
               for b in ("bronze_fpl", "bronze_vaastav", "bronze_opta"))


def test_solo_football_data_es_opcional():
    """Las cuotas no son features. Todo lo demás sí alimenta a Gold."""
    pasos = pre_deadline._pasos("2099-00", 1, True, False, False)
    opcionales = [p.nombre for p in pasos if not p.obligatorio]
    assert opcionales == ["bronze_fd"]


# ---------------------------------------------------------------------------
# Cuando la fuente se atrasa
# ---------------------------------------------------------------------------

def _predecir_de(gameweek, monkeypatch, corte):
    """El paso `predecir` del pipeline, con el calendario y Silver sustituidos."""
    monkeypatch.setattr("common.storage.read_table", lambda *_a, **_k: _fixtures_falsos(corte))
    monkeypatch.setattr(pre_deadline, "_fecha_objetivo", lambda gw, s: gw or 7)
    pasos = pre_deadline._pasos("2099-00", gameweek, dry_run=True,
                                forzar=False, force_ingesta=False)
    return next(p for p in pasos if p.nombre == "predecir").correr


def test_si_la_proxima_ya_arranco_el_pipeline_avisa_y_sigue(monkeypatch):
    """El caso del 20/09/2026, y es el normal: la fecha se jugó y la fuente se atrasó.

    football-data sube el CSV de la temporada con retraso, así que Gold no puede avanzar
    y la próxima predecible sigue siendo una fecha que ya empezó. La ingesta igual sirvió
    --dejó Bronze y Silver más frescos-- y tirar toda la corrida abajo por eso convierte
    una espera normal en un error rojo que nadie sabe cómo resolver.
    """
    ayer = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)
    salida = _predecir_de(None, monkeypatch, ayer)()

    assert salida["registrado"] is False
    assert "todavía" in salida["motivo"]


def test_pedir_explicitamente_una_fecha_pasada_si_es_un_error(monkeypatch):
    """La misma situación, otra intención: si la pediste vos, te lo decimos."""
    ayer = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)
    with pytest.raises(pre_deadline.FechaYaArranco):
        _predecir_de(7, monkeypatch, ayer)()
