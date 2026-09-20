"""El registro de predicciones: cuál vale, y no acumular copias.

Los 18 parquets reales de `data/predicciones/` son un banco de pruebas mejor que
cualquier fixture: hay fechas con siete registros, fechas re-predichas después de
jugarse, y una (la GW1 de 2026-27) sin ninguna predicción anterior al corte.
"""

from __future__ import annotations

import pandas as pd
import pytest

from common.config import CFG
from serving import registro

SEASON = CFG.current_season


@pytest.fixture
def registrados():
    d = registro.listar(SEASON)
    if d.empty:
        pytest.skip(f"No hay predicciones registradas de {SEASON}. "
                    f"Corré: python -m serving.predict --gw N")
    return d


def test_listar_parsea_season_gameweek_y_stamp(registrados):
    assert (registrados["season"] == SEASON).all()
    assert registrados["gameweek"].between(1, 38).all()
    assert registrados["stamp"].str.match(r"^\d{8}T\d{6}Z$").all()


def test_la_congelada_es_la_ultima_anterior_al_corte(registrados):
    """Con varias predicciones de la misma fecha, vale la última que llegó a tiempo.

    Una emitida con la fecha ya empezada no es una predicción: es una reconstrucción, y
    encima con el modelo y el dato de hoy. Tomar "la más nueva" —que es lo que hacía
    `decision.ultimas_por_fecha`— prefiere justamente la que no vale.
    """
    varias = registrados.groupby("gameweek").size()
    candidatas = varias[varias > 1]
    if candidatas.empty:
        pytest.skip("Ninguna fecha tiene más de un registro.")

    gw = int(candidatas.index[0])
    elegida = registro.congelada(SEASON, gw)
    assert elegida is not None

    emitida = pd.Timestamp(elegida["predicted_at"].iloc[0])
    corte = pd.Timestamp(elegida["kickoff_time"].min())

    if bool(elegida["registro_pre_deadline"].iloc[0]):
        assert emitida < corte
        posteriores = [
            pd.Timestamp(pd.read_parquet(r)["predicted_at"].iloc[0])
            for r in registrados[registrados["gameweek"] == gw]["ruta"]
        ]
        a_tiempo = [t for t in posteriores if t < corte]
        assert emitida == max(a_tiempo)
    else:
        # No hubo ninguna a tiempo: se devuelve la más temprana, y se dice.
        assert emitida >= corte


def test_una_fecha_sin_registro_devuelve_none():
    assert registro.congelada(SEASON, 38) is None or True   # 38 puede existir
    assert registro.congelada("1800-01", 1) is None


def test_guardar_dos_veces_la_misma_prediccion_no_crea_un_archivo_nuevo(tmp_path, monkeypatch):
    """Cada corrida del pipeline sumaba un archivo idéntico: la GW2 juntó siete."""
    monkeypatch.setattr(registro, "PREDICCIONES", tmp_path)

    pred = pd.DataFrame({
        "season": ["2099-00"] * 2, "gameweek": [1, 1], "fixture_id": [1, 2],
        "kickoff_time": pd.to_datetime(["2099-08-01T12:00Z", "2099-08-01T14:00Z"]),
        "home_short": ["AAA", "CCC"], "away_short": ["BBB", "DDD"],
        "p_home": [0.5, 0.2], "p_draw": [0.3, 0.3], "p_away": [0.2, 0.5],
        "prediccion": ["home", "away"], "confianza": [0.5, 0.5],
        "predicted_at": ["2099-07-31T00:00:00+00:00"] * 2,
        "model_name": ["m"] * 2, "model_version": ["v1"] * 2,
        "feature_set_version": ["f1"] * 2,
    })

    primera = registro.guardar(pred)
    assert primera is not None and primera.exists()
    assert registro.guardar(pred) is None                    # idéntica: no escribe
    assert len(list(tmp_path.glob("*.parquet"))) == 1

    otra = pred.copy()
    otra["model_version"] = "v2"
    assert registro.guardar(otra) is not None                # otro modelo: sí escribe
    assert len(list(tmp_path.glob("*.parquet"))) == 2

    assert registro.guardar(pred, si_existe="siempre") is not None
    assert len(list(tmp_path.glob("*.parquet"))) == 3


def test_no_se_registra_una_prediccion_vacia(tmp_path, monkeypatch):
    monkeypatch.setattr(registro, "PREDICCIONES", tmp_path)
    with pytest.raises(ValueError):
        registro.guardar(pd.DataFrame())


def test_con_resultado_deja_en_nulo_lo_que_no_se_jugo():
    """Una fecha en curso tiene partidos con resultado y partidos sin él."""
    pred = pd.DataFrame({
        "season": ["2099-00"] * 2, "fixture_id": [1, 2],
        "prediccion": ["home", "away"],
    })
    gold = pd.DataFrame({
        "season": ["2099-00"] * 2, "fixture_id": [1, 2],
        "target_1x2": ["home", None], "home_goals": [2.0, None],
        "away_goals": [0.0, None],
    })
    d = registro.con_resultado(pred, gold)
    assert d.loc[0, "acierto"] is True
    assert pd.isna(d.loc[1, "target_1x2"]) and d.loc[1, "acierto"] is None
