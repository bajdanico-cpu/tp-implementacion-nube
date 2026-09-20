"""El contrato HTTP, de punta a punta.

Corre contra los artefactos reales del repo (Gold + el registro + el modelo versionado),
no contra un backend mockeado: si faltan, se saltea con instrucciones de qué correr antes,
igual que el resto de `tests/` (ver `conftest.py`).

Dos tests de acá valen más que el resto y conviene no aflojarlos:

* `test_predict_no_lee_silver` — es el que hace cumplir la decisión de diseño. El
  servicio sirve un artefacto pre-calculado; si alguien vuelve a construir features
  adentro del request, la imagen necesita Silver de nuevo y volvemos a los 25 segundos.
* `test_la_version_de_produccion_incluye_el_holdout` — fija el incidente del 20/09/2026,
  cuando el servicio quedó sirviendo el modelo de evaluación.
"""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from common.config import CFG
from serving import main, predict, registro
from serving.main import app
from training import registry

SEASON = CFG.current_season


@pytest.fixture
def cliente():
    """Un cliente con el estado del servicio recién cargado.

    `main.ESTADO` es global y se cachea entre requests —que es justamente su razón de
    ser—, así que cada test lo resetea para no heredar el Gold de otro.
    """
    main.ESTADO = main.Estado()
    with TestClient(app) as c:
        yield c
    main.ESTADO = main.Estado()


def _gold_o_skip():
    from common.storage import table_exists

    if not table_exists(predict.GOLD, layer="gold"):
        pytest.skip("No hay tabla Gold. Corré: python -m features.gold_tp")
    return predict.cargar_gold()


def _proxima_o_skip():
    gw = predict.proxima_en_gold(_gold_o_skip())
    if gw is None:
        pytest.skip("No hay fecha predecible en Gold.")
    return gw


def _jugada_o_skip():
    gold = _gold_o_skip()
    d = gold[(gold["season"] == SEASON) & (gold["split"] != "inferencia")]
    if d.empty:
        pytest.skip(f"No hay fechas jugadas de {SEASON} en Gold.")
    return int(d["gameweek"].max())


# ---------------------------------------------------------------------------
# Salud
# ---------------------------------------------------------------------------

def test_health_reporta_modelo_y_gold(cliente):
    b = cliente.get("/health").json()
    assert b["status"] == "ok"
    assert b["model_version"] and b["gold_filas"] > 0
    assert b["backend"] == CFG.backend


def test_health_degrada_si_falta_gold(cliente, monkeypatch):
    """Un /health que miente es peor que no tenerlo."""
    def sin_gold():
        raise FileNotFoundError("no existe gold.gold_tp_match")

    monkeypatch.setattr(predict, "cargar_gold", sin_gold)
    main.ESTADO.gold, main.ESTADO.gold_at = None, 0.0

    b = cliente.get("/health").json()
    assert b["status"] == "degraded"
    assert "gold" in b["detail"].lower()


# ---------------------------------------------------------------------------
# Los tres estados de una fecha
# ---------------------------------------------------------------------------

def test_la_proxima_fecha_se_predice_en_vivo(cliente):
    gw = _proxima_o_skip()
    r = cliente.get(f"/predict/{SEASON}/{gw}")
    assert r.status_code == 200

    b = r.json()
    assert b["estado"] == "proxima" and b["origen"] == "en_vivo"
    assert b["count"] == len(b["predictions"]) > 0
    assert b["n_con_resultado"] == 0 and b["accuracy"] is None
    for m in b["predictions"]:
        assert m["prediccion"] in ("home", "draw", "away")
        assert abs(m["p_home"] + m["p_draw"] + m["p_away"] - 1.0) < 1e-6
        assert m["target_1x2"] is None


def test_una_fecha_jugada_devuelve_la_congelada_y_el_resultado(cliente):
    """No se re-predice: se devuelve lo que el sistema anunció, con lo que pasó.

    Re-predecir daría otro número —cambió el modelo, el feature set, o el dato de
    Silver— y presentarlo como "la predicción de esa fecha" sería falso.
    """
    gw = _jugada_o_skip()
    r = cliente.get(f"/predict/{SEASON}/{gw}")
    assert r.status_code == 200

    b = r.json()
    assert b["estado"] in ("jugada", "en_curso") and b["origen"] == "registro"
    assert b["n_con_resultado"] > 0
    assert 0.0 <= b["accuracy"] <= 1.0

    congelada = registro.congelada(SEASON, gw)
    assert b["predicted_at"] == str(congelada["predicted_at"].iloc[0])
    assert b["model_version"] == congelada["model_version"].iloc[0]

    con_resultado = [m for m in b["predictions"] if m["target_1x2"] is not None]
    assert con_resultado
    for m in con_resultado:
        assert m["acierto"] == (m["prediccion"] == m["target_1x2"])


def test_una_fecha_lejana_responde_409_con_la_proxima_predecible(cliente):
    """El caso que motivó todo el cambio.

    Pedir la 7 estando en la 5 devolvía 200 con features de la 4 y `dias_descanso` de 35,
    sin ninguna señal. Ahora se contesta que no está lista y cuál sí lo está.
    """
    gw = _proxima_o_skip()
    r = cliente.get(f"/predict/{SEASON}/{gw + 5}")
    assert r.status_code == 409

    b = r.json()
    assert b["proxima_predecible"] == gw
    assert str(gw) in b["detail"]


def test_una_temporada_que_no_existe_da_404(cliente):
    assert cliente.get("/predict/1800-01/1").status_code == 409
    assert cliente.get("/calendario/1800-01").status_code == 404


# ---------------------------------------------------------------------------
# Las dos propiedades que sostienen el diseño
# ---------------------------------------------------------------------------

def test_predict_no_lee_silver(cliente, monkeypatch):
    """El servicio sirve un artefacto; no construye features.

    Se vigila el backend de storage y no un módulo puntual, así que da igual quién
    intente la lectura ni cómo haya importado `read_table`.
    """
    from common import storage

    real = storage.BACKEND.read_dataframe
    leidas = []

    def vigilado(path):
        ruta = str(path).replace("\\", "/").lower()
        leidas.append(ruta)
        if "/silver/" in ruta:
            raise AssertionError(f"El servicio leyó Silver: {path}")
        return real(path)

    monkeypatch.setattr(storage.BACKEND, "read_dataframe", vigilado)
    main.ESTADO.gold, main.ESTADO.gold_at = None, 0.0   # fuerza la relectura

    gw = _proxima_o_skip()
    assert cliente.get(f"/predict/{SEASON}/{gw}").status_code == 200
    assert any("/gold/" in r for r in leidas), "se esperaba que leyera Gold"


def test_predict_responde_en_menos_de_un_segundo(cliente):
    """La razón de ser del lookup: antes eran ~25 segundos por request."""
    gw = _proxima_o_skip()
    cliente.get(f"/predict/{SEASON}/{gw}")          # calienta modelo y Gold

    inicio = time.perf_counter()
    r = cliente.get(f"/predict/{SEASON}/{gw}")
    assert r.status_code == 200
    assert (time.perf_counter() - inicio) < 1.0


# ---------------------------------------------------------------------------
# Calendario
# ---------------------------------------------------------------------------

def test_el_calendario_lista_las_fechas_con_su_estado(cliente):
    _gold_o_skip()
    b = cliente.get(f"/calendario/{SEASON}").json()
    assert b["fechas"]
    assert b["proxima_predecible"] == predict.proxima_en_gold(predict.cargar_gold())
    estados = {f["estado"] for f in b["fechas"]}
    assert estados <= {"jugada", "en_curso", "proxima", "no_preparada"}
    assert sum(f["estado"] == "proxima" for f in b["fechas"]) <= 1


# ---------------------------------------------------------------------------
# Qué modelo se sirve
#
# El 20/09/2026 `/predict` devolvía 503 en todas las fechas. La causa: no existía
# `PRODUCTION.json`, y el fallback ("la última carpeta por nombre") eligió una versión
# que había llegado por `git checkout` con su metadata y sin un solo `.ubj`, porque los
# binarios están en `.gitignore`. Estos tests fijan las dos mitades del incidente: que
# hay una versión elegida, y que el fallback no puede volver a elegir una vacía.
# ---------------------------------------------------------------------------

def test_hay_una_version_de_produccion_declarada():
    """Sin `PRODUCTION.json`, qué modelo sirve depende de cómo ordena `glob`."""
    v = registry.produccion(CFG.modelo)
    assert v is not None, (
        f"No hay models/{CFG.modelo}/PRODUCTION.json. "
        f"Corré: python -m training.registry --listar")


def test_la_version_de_produccion_tiene_boosters():
    v = registry.produccion(CFG.modelo)
    if v is None:
        pytest.skip("No hay PRODUCTION.json")
    assert registry.tiene_boosters(v), (
        f"La versión de producción {v.version} no tiene .ubj: se puede trazar, "
        f"no servir.")


def test_la_version_de_produccion_incluye_el_holdout():
    """El que sirve es el de PRODUCCIÓN, no el de EVALUACIÓN.

    Son dos modelos distintos y el repo los distingue con `incluye_holdout`: el de
    evaluación no ve 2025-26 para que sus métricas signifiquen algo, y justamente por eso
    no es el que hay que servir. Servirlo es tirar la temporada más reciente a la basura.
    """
    v = registry.produccion(CFG.modelo)
    if v is None:
        pytest.skip("No hay PRODUCTION.json")
    meta = json.loads(v.metadata.read_text(encoding="utf-8"))
    assert meta.get("incluye_holdout") is True, (
        f"{v.version} tiene incluye_holdout={meta.get('incluye_holdout')}: "
        f"es el modelo de evaluación y no debería estar sirviendo.")


def _version_falsa(raiz, nombre: str, version: str, con_boosters: bool):
    d = raiz / nombre / version
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"n_features": 279}), encoding="utf-8")
    if con_boosters:
        (d / "model_seed0.ubj").write_bytes(b"no es un booster de verdad")
    return d


def test_el_fallback_nunca_elige_una_carpeta_sin_boosters(tmp_path, monkeypatch):
    """La reproducción exacta del incidente: la más nueva es la que no se puede servir."""
    monkeypatch.setattr(registry, "RAIZ", tmp_path)
    _version_falsa(tmp_path, "falso", "20260101T000000Z", con_boosters=True)
    _version_falsa(tmp_path, "falso", "20260202T000000Z", con_boosters=False)

    assert len(registry.versiones_disponibles("falso")) == 2
    assert [v.version for v in registry.servibles("falso")] == ["20260101T000000Z"]


def test_si_produccion_no_tiene_boosters_falla_diciendo_por_que(tmp_path, monkeypatch):
    """No se cae a otra versión en silencio: servir un modelo que nadie eligió es peor."""
    monkeypatch.setattr(registry, "RAIZ", tmp_path)
    _version_falsa(tmp_path, "falso", "20260101T000000Z", con_boosters=True)
    _version_falsa(tmp_path, "falso", "20260202T000000Z", con_boosters=False)
    (tmp_path / "falso" / registry.PRODUCCION).write_text(
        json.dumps({"version": "20260202T000000Z"}), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="no tiene archivos"):
        predict.cargar_modelo("falso")


# ---------------------------------------------------------------------------
# La página
# ---------------------------------------------------------------------------

def test_la_raiz_sirve_la_pagina(cliente):
    r = cliente.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "Premier ML" in r.text


def test_montar_la_pagina_no_tapa_la_api(cliente):
    """La página se monta en `/`, así que el orden de declaración importa.

    Si `StaticFiles` se montara antes de las rutas, se comería `/health` y `/predict` y
    la API entera devolvería 404 sin que ningún otro test lo notara.
    """
    assert cliente.get("/health").status_code == 200
    assert cliente.get(f"/calendario/{SEASON}").status_code == 200
    assert cliente.get("/docs").status_code == 200


def test_la_pagina_usa_rutas_relativas(cliente):
    """Mismo origen que la API.

    Con la URL del servicio hardcodeada habría que editarla en cada despliegue y
    además configurar CORS. Con rutas relativas, la página funciona igual en local,
    en Cloud Run y en cualquier revisión.
    """
    import re

    html = cliente.get("/").text
    assert '"/health"' in html and '`/predict/' in html and '`/calendario/' in html

    externas = re.findall(r"""(?:fetch|json)\(\s*['"`](https?://[^'"`]+)""", html)
    assert not externas, f"la página le pega a un host externo: {externas}"


def test_la_tabla_de_escudos_se_sirve_y_tiene_la_forma_esperada(cliente):
    """La página la necesita para los escudos; se regenera con `python -m scripts.escudos`."""
    r = cliente.get("/escudos.json")
    assert r.status_code == 200

    equipos = r.json()
    assert len(equipos) >= 20
    for short, eq in equipos.items():
        assert len(short) == 3 and short.isupper()
        assert isinstance(eq["code"], int) and eq["nombre"]


# ---------------------------------------------------------------------------
# El calendario completo y el disparo del pipeline
# ---------------------------------------------------------------------------

def test_el_calendario_lista_todas_las_fechas_de_la_temporada(cliente):
    """Antes sólo salían las que tenían features, y el límite era invisible.

    Con Gold hasta la GW5, la 18 simplemente no aparecía en la pantalla: no se veía que
    no se puede predecir, se veía que no existe. Ahora sale marcada `no_preparada`.
    """
    _gold_o_skip()
    b = cliente.get(f"/calendario/{SEASON}").json()

    assert len(b["fechas"]) == CFG.gameweeks
    assert [f["gameweek"] for f in b["fechas"]] == list(range(1, CFG.gameweeks + 1))
    assert b["hasta_gameweek"] is not None

    lejana = next(f for f in b["fechas"] if f["gameweek"] > b["hasta_gameweek"])
    assert lejana["estado"] == "no_preparada" and lejana["n_partidos"] == 0


def test_el_diagnostico_dice_hasta_donde_llega_gold(cliente):
    b = cliente.get("/actualizar").json()
    assert isinstance(b["hace_falta"], bool)
    assert b["motivo"]
    assert b["puede_disparar"] is True


def test_disparar_devuelve_202_y_no_espera(cliente, monkeypatch):
    """El POST no corre el pipeline: lo pide. Tiene que volver enseguida."""
    from serving import tareas

    monkeypatch.setattr(tareas, "_lanzar_subproceso",
                        lambda t: setattr(t, "estado", "corriendo"))
    tareas._TAREAS.clear()

    r = cliente.post("/actualizar")
    assert r.status_code == 202
    b = r.json()
    assert b["estado"] == "corriendo" and b["id"]

    # Y mientras corre, no se puede lanzar otra: dos pipelines se pisan escribiendo Gold.
    assert cliente.post("/actualizar").status_code == 409
    assert cliente.get(f"/actualizar/{b['id']}").json()["id"] == b["id"]
    tareas._TAREAS.clear()


def test_en_la_nube_sin_token_el_disparo_da_403(cliente, monkeypatch):
    from serving import tareas

    monkeypatch.setenv("K_SERVICE", "premier-ml-api")
    monkeypatch.delenv("TP_ADMIN_TOKEN", raising=False)
    tareas._TAREAS.clear()

    r = cliente.post("/actualizar")
    assert r.status_code == 403
    assert "apagado" in r.json()["detail"]


def test_una_tarea_que_no_existe_da_404(cliente):
    assert cliente.get("/actualizar/no-existe").status_code == 404
