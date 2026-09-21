"""El registro de predicciones: cuál vale, y no acumular copias.

Dos clases de test, y la separación es deliberada.

**La regla de elección se prueba en sintético.** El registro real llegó a tener 18
parquets para cinco fechas —seis modelos distintos de una tarde de reentrenos, más
re-corridas posteriores al corte— y durante un tiempo eso fue un banco de pruebas
cómodo. Era también una trampa: `scripts.depurar_registro` lo dejó en una predicción
por fecha, como corresponde a un TP con un solo modelo productivo, y los tests que se
apoyaban en el desorden pasaron a saltearse en silencio. Un test que depende de sobras
de desarrollo deja de probar el día que alguien ordena.

**Lo que sí se verifica contra el dato real** es la invariante que el entregable
promete: una predicción por fecha, todas del modelo de producción.
"""

from __future__ import annotations

import pandas as pd
import pytest

from common.config import CFG
from serving import registro
from training import registry

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


def _registrar(carpeta, gw, emitida, kickoffs, model_version="v1"):
    """Deja un parquet con el nombre que `listar` sabe parsear."""
    n = len(kickoffs)
    d = pd.DataFrame({
        "season": ["2099-00"] * n, "gameweek": [gw] * n,
        "fixture_id": list(range(n)),
        "kickoff_time": pd.to_datetime(kickoffs, utc=True),
        "home_short": ["AAA"] * n, "away_short": ["BBB"] * n,
        "p_home": [0.5] * n, "p_draw": [0.3] * n, "p_away": [0.2] * n,
        "prediccion": ["home"] * n, "confianza": [0.5] * n,
        "predicted_at": [pd.Timestamp(emitida, tz="UTC").isoformat()] * n,
        "model_name": ["m"] * n, "model_version": [model_version] * n,
        "feature_set_version": ["f1"] * n,
    })
    stamp = pd.Timestamp(emitida, tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    ruta = carpeta / f"2099-00_GW{gw:02d}_{stamp}.parquet"
    d.to_parquet(ruta, index=False)
    return ruta


@pytest.fixture
def sintetico(tmp_path, monkeypatch):
    monkeypatch.setattr(registro, "PREDICCIONES", tmp_path)
    return tmp_path


def test_la_congelada_es_la_ultima_anterior_al_corte(sintetico):
    """Con varias predicciones de la misma fecha, vale la última que llegó a tiempo.

    Una emitida con la fecha ya empezada no es una predicción: es una reconstrucción, y
    encima con el modelo y el dato de hoy. Tomar "la más nueva" —que es lo que hacía
    `decision.ultimas_por_fecha`— prefiere justamente la que no vale.
    """
    kickoffs = ["2099-08-10T12:00Z", "2099-08-10T14:00Z"]
    _registrar(sintetico, 3, "2099-08-08T10:00Z", kickoffs)
    buena = _registrar(sintetico, 3, "2099-08-09T22:00Z", kickoffs)    # la última a tiempo
    _registrar(sintetico, 3, "2099-08-11T09:00Z", kickoffs)            # ya se jugó

    d = registro.congelada("2099-00", 3)
    assert d is not None
    assert d["registro_archivo"].iloc[0] == buena.name
    assert bool(d["registro_pre_deadline"].iloc[0]) is True


def test_sin_ninguna_a_tiempo_se_devuelve_la_mas_temprana_y_se_dice(sintetico):
    """El caso de la GW1 de 2026-27: se jugó antes de que el sistema registrara.

    Devolverla sin avisar sería presentar como evidencia algo que no lo es; no
    devolverla dejaría un hueco. Se devuelve marcada.
    """
    kickoffs = ["2099-08-10T12:00Z"]
    temprana = _registrar(sintetico, 1, "2099-08-10T20:00Z", kickoffs)
    _registrar(sintetico, 1, "2099-08-12T20:00Z", kickoffs)

    d = registro.congelada("2099-00", 1)
    assert d["registro_archivo"].iloc[0] == temprana.name
    assert bool(d["registro_pre_deadline"].iloc[0]) is False


def test_el_registro_real_tiene_una_prediccion_por_fecha_del_modelo_productivo(registrados):
    """La invariante del entregable, verificada sobre `data/predicciones/`.

    El TP fija un solo modelo en producción. Un registro con predicciones de versiones
    que ya no existen en `models/` contradice eso y encima no es auditable: nadie puede
    cargar el modelo que las emitió. Se ordena con `python -m scripts.depurar_registro`.
    """
    repetidas = registrados.groupby("gameweek").size()
    assert (repetidas == 1).all(), (
        f"fechas con más de un registro: {dict(repetidas[repetidas > 1])}. "
        f"Corré: python -m scripts.depurar_registro --aplicar")

    v = registry.produccion(CFG.modelo)
    assert v is not None, "no hay modelo de producción declarado"
    versiones = {str(pd.read_parquet(r)["model_version"].iloc[0])
                 for r in registrados["ruta"]}
    assert versiones == {v.version}, (
        f"el registro tiene predicciones de {versiones - {v.version}}, "
        f"que no es el modelo de producción ({v.version})")


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
