"""El backend de storage: mapeo de rutas, configuración y sustituibilidad.

No se habla con Google acá. Lo que se prueba es lo que se puede romper sin darse cuenta:
que una ruta absoluta de Windows se traduzca a la clave de objeto correcta, que las
variables de entorno pisen a `config.yaml`, y que todo el pipeline funcione contra un
backend que no es el disco. Eso último es la prueba de que la abstracción existe de
verdad y no es sólo un ABC decorativo.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from common import storage
from common.config import CFG, load
from common.storage import GCSBackend, StorageBackend


# ---------------------------------------------------------------------------
# Mapeo ruta -> clave de objeto
# ---------------------------------------------------------------------------

@pytest.fixture
def gcs():
    return GCSBackend(bucket="un-bucket", prefix="", project=None)


def test_una_ruta_de_gold_se_mapea_a_la_clave_del_bucket(gcs):
    assert gcs.key(CFG.gold_root / "gold_tp_match.parquet") == "gold/gold_tp_match.parquet"


def test_una_ruta_de_silver_y_una_de_predicciones(gcs):
    assert gcs.key(CFG.silver_root / "fact_match.parquet") == "silver/fact_match.parquet"
    assert gcs.key(CFG.data_root / "predicciones" / "x.parquet") == "predicciones/x.parquet"


def test_una_ruta_de_modelos_cuelga_de_models(gcs):
    """`models/` no vive bajo `data_root`, así que necesita su propia raíz."""
    ruta = CFG.models_root / "xgb_gbt" / "20260825T024144Z" / "model_seed0.ubj"
    assert gcs.key(ruta) == "models/xgb_gbt/20260825T024144Z/model_seed0.ubj"


def test_un_snapshot_de_bronze_conserva_el_ingested_at(gcs):
    ruta = CFG.bronze_dir("fpl", "2026-27", "fixtures", stamp="20260917T224455Z")
    assert gcs.key(ruta / "fixtures.json") == (
        "bronze/fpl/2026-27/fixtures/ingested_at=20260917T224455Z/fixtures.json")


def test_el_prefijo_se_antepone_a_todo():
    g = GCSBackend(bucket="b", prefix="tp/", project=None)
    assert g.key(CFG.gold_root / "x.parquet") == "tp/gold/x.parquet"


def test_una_ruta_fuera_de_las_raices_falla(gcs):
    """Mejor un error que subir un archivo a una clave inventada."""
    with pytest.raises(ValueError, match="no cuelga de ninguna raiz"):
        gcs.key(Path("C:/otra/cosa.parquet") if Path("C:/").exists()
                else Path("/otra/cosa.parquet"))


def test_las_rutas_usan_barras_normales_aunque_windows_use_contrabarras(gcs):
    clave = gcs.key(CFG.gold_root / "sub" / "x.parquet")
    assert "\\" not in clave and clave == "gold/sub/x.parquet"


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

def test_las_env_pisan_a_config_yaml(monkeypatch, tmp_path):
    monkeypatch.setenv("TP_STORAGE_BACKEND", "gcs")
    monkeypatch.setenv("TP_GCS_BUCKET", "otro-bucket")
    monkeypatch.setenv("TP_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("TP_MODELS_ROOT", str(tmp_path / "m"))

    c = load()
    assert c.backend == "gcs"
    assert c.gcs_bucket == "otro-bucket"
    assert c.data_root == tmp_path
    assert c.models_root == tmp_path / "m"


def test_sin_env_manda_config_yaml():
    assert CFG.backend == load().raw["storage"]["backend"]


def test_backend_gcs_sin_bucket_falla_al_construirse(monkeypatch):
    """Ruidoso y temprano: un bucket vacío se convierte en 404 raros mucho después."""
    monkeypatch.delenv("TP_GCS_BUCKET", raising=False)
    if CFG.gcs_bucket:
        pytest.skip("Hay un bucket configurado en config.yaml.")
    with pytest.raises(ValueError, match="necesita un bucket"):
        GCSBackend()


def test_el_backend_se_memoiza_y_se_puede_resetear():
    a = storage.backend()
    assert storage.backend() is a
    storage.reset_backend()
    assert storage.backend() is not a


def test_backend_sigue_accesible_como_atributo_del_modulo():
    """`common/versiones.py` y los tests lo usan así; PEP 562 lo mantiene vivo."""
    assert storage.BACKEND is storage.backend()


# ---------------------------------------------------------------------------
# El pipeline contra un backend que no es el disco
# ---------------------------------------------------------------------------

class BackendFalso(StorageBackend):
    """Un backend en memoria. Si el pipeline funciona con esto, la abstracción sirve."""

    def __init__(self):
        self.objetos: dict[str, bytes] = {}

    @staticmethod
    def _k(path: Path) -> str:
        return Path(path).as_posix()

    def write_bytes(self, path, data):
        self.objetos[self._k(path)] = data

    def read_bytes(self, path):
        try:
            return self.objetos[self._k(path)]
        except KeyError:
            raise FileNotFoundError(path) from None

    def exists(self, path):
        return self._k(path) in self.objetos

    def write_dataframe(self, df, path):
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        self.objetos[self._k(path)] = buf.getvalue()

    def read_dataframe(self, path):
        return pd.read_parquet(io.BytesIO(self.read_bytes(path)))

    def list_dirs(self, path):
        raiz = self._k(path).rstrip("/") + "/"
        sub = {k[len(raiz):].split("/")[0] for k in self.objetos
               if k.startswith(raiz) and "/" in k[len(raiz):]}
        return [Path(path) / s for s in sorted(sub)]

    def list_files(self, path, patron="*"):
        from fnmatch import fnmatch

        raiz = self._k(path).rstrip("/") + "/"
        hijos = [k[len(raiz):] for k in self.objetos if k.startswith(raiz)]
        return [Path(path) / h for h in sorted(hijos)
                if "/" not in h and fnmatch(h, patron)]


@pytest.fixture
def falso():
    b = BackendFalso()
    storage.set_backend(b)
    yield b
    storage.reset_backend()


def test_el_pipeline_lee_y_escribe_contra_un_backend_falso(falso):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    storage.write_table(df, "tabla_de_prueba", layer="silver")

    assert falso.objetos, "no escribió nada"
    assert storage.table_exists("tabla_de_prueba")
    pd.testing.assert_frame_equal(storage.read_table("tabla_de_prueba"), df)


def test_una_tabla_que_no_existe_da_un_error_util(falso):
    with pytest.raises(FileNotFoundError, match="transform.silver"):
        storage.read_table("no_existe")


def test_bronze_sigue_siendo_append_only_contra_otro_backend(falso):
    a = storage.write_raw("fpl", "2099-00", "fixtures", "f.json", b"uno",
                          stamp="20990101T000000Z")
    b = storage.write_raw("fpl", "2099-00", "fixtures", "f.json", b"dos",
                          stamp="20990102T000000Z")
    assert a != b
    assert falso.read_bytes(a) == b"uno"          # el primero no se pisó
    assert storage.latest_snapshot("fpl", "2099-00", "fixtures").name.endswith(
        "ingested_at=20990102T000000Z")


# ---------------------------------------------------------------------------
# Los modelos, ahora que también pasan por el backend
# ---------------------------------------------------------------------------

def test_un_booster_guardado_por_bytes_predice_igual_que_uno_por_archivo(tmp_path):
    """Protege el cambio de `save_model(ruta)` a `save_raw()` + `write_bytes`.

    Es el mismo espíritu que `test_modelo_entrenado_en_gpu_predice_igual_en_cpu`: el
    formato tiene que sobrevivir al cambio de medio sin mover un decimal.
    """
    import xgboost as xgb

    from training import registry

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = (X[:, 0] + rng.normal(scale=0.3, size=60) > 0).astype(int)
    modelo = xgb.XGBClassifier(n_estimators=8, max_depth=2, tree_method="hist")
    modelo.fit(X, y)
    booster = modelo.get_booster()

    por_archivo = tmp_path / "a.ubj"
    booster.save_model(str(por_archivo))

    por_bytes = tmp_path / "b.ubj"
    por_bytes.write_bytes(bytes(booster.save_raw(raw_format="ubj")))

    dm = xgb.DMatrix(X)
    a = registry.cargar_booster(por_archivo).predict(dm)
    b = registry.cargar_booster(por_bytes).predict(dm)
    np.testing.assert_array_equal(a, b)


def test_promover_y_leer_la_produccion_contra_un_backend_falso(falso):
    from training import registry

    v = registry.Version("falso", "20990101T000000Z",
                         CFG.models_root / "falso" / "20990101T000000Z")
    registry.promover(v, "prueba")
    leida = registry.produccion("falso")
    assert leida is not None and leida.version == v.version
