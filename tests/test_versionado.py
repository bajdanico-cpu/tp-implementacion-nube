"""Tests del versionado de Silver y Gold: **nunca se pisa nada**.

La garantía que estos tests defienden es una sola y es fuerte: *después de cualquier
secuencia de escrituras, todo contenido que alguna vez estuvo vivo se puede recuperar.*

Importa más que la mayoría de los tests del repo porque `data/` está en `.gitignore`: si
esto falla, no hay git del que rescatar el Gold con el que se entrenó el modelo que está
sirviendo. Es la única capa donde un bug es **destructivo e irreversible**.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from common import storage


@pytest.fixture
def capas(tmp_path, monkeypatch):
    """Un `data/` de mentira, con Silver, Gold y su carpeta de versiones."""
    silver, gold = tmp_path / "silver", tmp_path / "gold"
    silver.mkdir(), gold.mkdir()
    monkeypatch.setattr(type(storage.CFG), "data_root",
                        property(lambda self: tmp_path))
    monkeypatch.setattr(type(storage.CFG), "silver_root",
                        property(lambda self: silver))
    monkeypatch.setattr(type(storage.CFG), "gold_root",
                        property(lambda self: gold))
    monkeypatch.delenv(storage.ETIQUETA_ENV, raising=False)
    return tmp_path


def _df(n: int, extra: str | None = None) -> pd.DataFrame:
    d = pd.DataFrame({"a": range(n), "b": [f"x{i}" for i in range(n)]})
    if extra:
        d[extra] = 1.0
    return d


# ---------------------------------------------------------------------------
# La garantía central
# ---------------------------------------------------------------------------

def test_prueba_de_fuego_reescribir_no_destruye_la_version_anterior(capas):
    """El caso real: correr `features.gold_tp` encima de un Gold ya existente."""
    storage.write_table(_df(10), "gold_tp_match", layer="gold")
    original = (capas / "gold" / "gold_tp_match.parquet").read_bytes()

    storage.write_table(_df(20, extra="nueva_feature"), "gold_tp_match", layer="gold")

    vs = storage.versiones("gold", "gold_tp_match")
    assert len(vs) == 1, "la version anterior tiene que estar archivada"
    archivada = storage.versiones_root("gold", "gold_tp_match") / vs[0]["archivo"]
    assert archivada.read_bytes() == original, "el contenido archivado cambio"

    # Y lo vigente es lo nuevo.
    assert len(storage.read_table("gold_tp_match", layer="gold")) == 20


def test_toda_version_que_estuvo_viva_se_puede_recuperar(capas):
    """Tres escrituras seguidas: las dos primeras quedan recuperables enteras."""
    contenidos = []
    for n in (5, 15, 25):
        storage.write_table(_df(n), "fact_match", layer="silver")
        contenidos.append((capas / "silver" / "fact_match.parquet").read_bytes())

    archivados = {storage.sha256(
        (storage.versiones_root("silver", "fact_match") / v["archivo"]).read_bytes())
        for v in storage.versiones("silver", "fact_match")}

    for c in contenidos[:-1]:
        assert storage.sha256(c) in archivados
    # La ultima esta viva, no archivada.
    assert (capas / "silver" / "fact_match.parquet").read_bytes() == contenidos[-1]


# ---------------------------------------------------------------------------
# Deduplicación: proteger no puede significar acumular basura
# ---------------------------------------------------------------------------

def test_reescribir_el_mismo_contenido_no_archiva_dos_veces(capas):
    """Reconstruir Gold sin cambiar nada no debe ensuciar el historico.

    Se archiva como mucho una version redundante (la de la primera reescritura), y de ahi
    en adelante el hash ya esta guardado y no se repite.
    """
    for _ in range(5):
        storage.write_table(_df(10), "gold_tp_match", layer="gold")
    assert len(storage.versiones("gold", "gold_tp_match")) <= 1


def test_archivar_un_archivo_que_no_existe_no_falla(capas):
    assert storage.archivar(capas / "gold" / "no_existe.parquet", "gold") is None


# ---------------------------------------------------------------------------
# El manifiesto: un parquet sin contexto es un parquet más
# ---------------------------------------------------------------------------

def test_el_manifiesto_guarda_forma_hash_y_etiqueta(capas):
    storage.write_table(_df(7), "gold_tp_match", layer="gold")
    vivo = (capas / "gold" / "gold_tp_match.parquet").read_bytes()
    storage.write_table(_df(9), "gold_tp_match", layer="gold")

    v = storage.versiones("gold", "gold_tp_match")[0]
    assert v["filas"] == 7 and v["columnas"] == 2
    assert v["sha256"] == storage.sha256(vivo)
    assert v["tabla"] == "gold_tp_match" and v["layer"] == "gold"
    assert v["bytes"] == len(vivo)


def test_la_etiqueta_sale_de_la_variable_de_entorno(capas, monkeypatch):
    """`TP_VERSION_LABEL=... python -m features.gold_tp` es el uso previsto."""
    storage.write_table(_df(3), "gold_tp_match", layer="gold")
    monkeypatch.setenv(storage.ETIQUETA_ENV, "antes de fase 1")
    storage.write_table(_df(4), "gold_tp_match", layer="gold")
    assert storage.versiones("gold", "gold_tp_match")[0]["etiqueta"] == "antes de fase 1"


def test_dos_versiones_en_el_mismo_segundo_no_se_pisan(capas):
    """El stamp sale del mtime, y dos escrituras rapidas pueden compartirlo.

    Si el desempate fallara, la segunda version sobrescribiria a la primera **dentro del
    histórico** — el peor bug posible en este módulo, porque el mecanismo de proteccion
    seria el que destruye.
    """
    carpeta = storage.versiones_root("gold", "t")
    storage.write_table(_df(2), "t", layer="gold")
    a = (capas / "gold" / "t.parquet").read_bytes()
    storage.write_table(_df(3), "t", layer="gold")
    b = (capas / "gold" / "t.parquet").read_bytes()
    # Se fuerza el mismo mtime para las dos versiones archivadas.
    import os
    for p in carpeta.glob("*.parquet"):
        os.utime(p, (0, 0))
    storage.write_table(_df(4), "t", layer="gold")

    guardados = {p.read_bytes() for p in carpeta.glob("*.parquet")}
    assert a in guardados and b in guardados


# ---------------------------------------------------------------------------
# Restaurar, que también tiene que ser no destructivo
# ---------------------------------------------------------------------------

def test_restaurar_archiva_lo_que_estaba_vivo(capas):
    from common import versiones as mod

    storage.write_table(_df(10), "gold_tp_match", layer="gold")
    viejo = (capas / "gold" / "gold_tp_match.parquet").read_bytes()
    storage.write_table(_df(99), "gold_tp_match", layer="gold")
    nuevo = (capas / "gold" / "gold_tp_match.parquet").read_bytes()

    stamp = storage.versiones("gold", "gold_tp_match")[0]["stamp"]
    mod.restaurar("gold", "gold_tp_match", stamp)

    assert (capas / "gold" / "gold_tp_match.parquet").read_bytes() == viejo
    guardados = {(storage.versiones_root("gold", "gold_tp_match") / v["archivo"]).read_bytes()
                 for v in storage.versiones("gold", "gold_tp_match")}
    assert nuevo in guardados, "lo que estaba vivo al restaurar tiene que quedar guardado"


def test_restaurar_una_version_inexistente_falla_claro(capas):
    from common import versiones as mod

    storage.write_table(_df(3), "gold_tp_match", layer="gold")
    with pytest.raises(ValueError, match="No hay version"):
        mod.restaurar("gold", "gold_tp_match", "19990101T000000Z")


def test_el_snapshot_etiqueta_todo_lo_vigente(capas):
    from common import versiones as mod

    storage.write_table(_df(5), "fact_match", layer="silver")
    storage.write_table(_df(6), "gold_tp_match", layer="gold")
    (capas / "gold" / "prior_ascendidos.json").write_text(json.dumps({"x": 1}),
                                                          encoding="utf-8")

    res = mod.snapshot("antes de fase 1")
    assert len(res) == 3
    for layer, tabla in (("silver", "fact_match"), ("gold", "gold_tp_match"),
                         ("gold", "prior_ascendidos")):
        vs = storage.versiones(layer, tabla)
        assert vs and vs[-1]["etiqueta"] == "antes de fase 1"


# ---------------------------------------------------------------------------
# El histórico no puede contaminar las capas vivas
# ---------------------------------------------------------------------------

def test_las_versiones_viven_fuera_de_silver_y_gold(capas):
    """El lab de GCP sube `data/gold` con un rglob: si el historico viviera adentro,
    cada corrida subiria todas las versiones al bucket."""
    storage.write_table(_df(4), "gold_tp_match", layer="gold")
    storage.write_table(_df(5), "gold_tp_match", layer="gold")

    en_gold = sorted(p.name for p in (capas / "gold").iterdir())
    assert en_gold == ["gold_tp_match.parquet"]
    assert (capas / "_versiones" / "gold" / "gold_tp_match").exists()
