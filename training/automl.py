"""Vertex AI AutoML Tabular sobre el mismo Gold, para comparar contra nuestro modelo.

    python -m training.automl --export                    # el CSV, local
    python -m training.automl --export --subir --bucket X # y lo sube a GCS
    python -m training.automl --entrenar --dataset-id ID  # lanza el AutoML (async)
    python -m training.automl --metricas --model-id ID    # lo compara contra el nuestro

**Qué es AutoML.** Le das una tabla, le decís qué columna predecir y qué métrica
optimizar, y Google prueba modelos y ensambles por su cuenta. Es el contrafáctico honesto
de todo el trabajo de `training/`: si una herramienta automática saca lo mismo en dos horas
sin que nadie piense, eso hay que decirlo; y si no lo saca, también.

**Se corre desde la TERMINAL de Cloud Shell, no desde el editor.** El SDK autentica por ADC
con el proyecto de la sesión, y el kernel del editor no hereda ese entorno.

---

## Las dos decisiones que hacen que la comparación signifique algo

**1. El split se le impone, no se lo deja elegir.**

AutoML parte el dataset **al azar** por defecto. Para este problema eso es fatal y no
avisa: pone partidos de mayo en train y de agosto en test, el modelo ve el futuro, y sale
un número altísimo que no vale nada. Es exactamente contra lo que está construido el resto
del pipeline.

Se le pasa `predefined_split_column_name="ml_use"`, con la misma partición temporal que usa
nuestro modelo:

    2022-23, 2023-24  -> TRAIN
    2024-25           -> VALIDATE     (la de early stopping, igual que nosotros)
    2025-26           -> TEST         (el holdout, los mismos 380 partidos)

La temporada en curso se excluye entera: no es objetivo de entrenamiento en ningún modelo
de este proyecto.

**2. Se le dan EXACTAMENTE nuestras 279 features.**

Ni una más. Nada de fechas, ids, marcadores ni cuotas: si AutoML recibiera `home_goals`
ganaría con trampa, y si recibiera las cuotas la comparación dejaría de ser contra un
modelo que no las usa. Mismas columnas, mismo split, mismo holdout — la única diferencia
es quién eligió el modelo.

## La métrica

`minimize-log-loss`. Es un problema de **tres clases** y lo que importa es la calidad de la
probabilidad, no sólo el acierto: el bloque de apuestas necesita `p` calibrada para calcular
valor esperado. `maximize-au-roc` —lo que usa el caso guía de churn— es para binario.

## El costo, dicho antes de correrlo

El presupuesto mínimo es **1 node-hour**, pero el reloj tarda alrededor de **dos horas**:
el entrenamiento corre server-side y consume crédito. No se lanza en vivo en una defensa.

Lo que **no** cuesta por hora: el modelo queda en el Model Registry sin desplegar. Un
endpoint desplegado sí factura la máquina esté o no en uso — este módulo no crea ninguno.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common.config import CFG, PROJECT_ROOT
from common.logging_setup import get_logger, setup
from features import spec
from training import dataset

log = get_logger(__name__)

SALIDA = PROJECT_ROOT / "data" / "automl"
CSV = SALIDA / "gold_automl.csv"

TARGET = "target_1x2"
COL_SPLIT = "ml_use"

# Vertex espera estos tres valores, en mayúsculas.
TRAIN, VALIDATE, TEST = "TRAIN", "VALIDATE", "TEST"


# --------------------------------------------------------------------------- #
#  Export
# --------------------------------------------------------------------------- #

def construir_csv(gold: pd.DataFrame | None = None) -> pd.DataFrame:
    """Las 279 features + el target + la columna de split. Nada más."""
    gold = dataset.cargar() if gold is None else gold

    train_seasons = set(CFG.seasons_for_training()) - {CFG.valid_season}
    def _split(season: str) -> str | None:
        if season == CFG.holdout_season:
            return TEST
        if season == CFG.valid_season:
            return VALIDATE
        if season in train_seasons:
            return TRAIN
        return None            # la temporada en curso queda afuera

    d = gold.copy()
    d[COL_SPLIT] = d["season"].map(_split)

    antes = len(d)
    d = d[d[COL_SPLIT].notna() & d[TARGET].notna()]
    if antes != len(d):
        log.info("excluidas %d filas sin split o sin target (la temporada en curso)",
                 antes - len(d))

    faltan = [c for c in spec.FEATURES if c not in d.columns]
    if faltan:
        raise KeyError(f"Gold no tiene {len(faltan)} features del spec: {faltan[:5]}")

    out = d[[*spec.FEATURES, TARGET, COL_SPLIT]]

    # AutoML Tabular pide un mínimo de 1.000 filas; con menos ni arranca.
    if len(out) < 1000:
        raise ValueError(f"AutoML Tabular exige >= 1.000 filas y hay {len(out)}. "
                         "¿Falta ingesta?")
    return out


def exportar(subir_a: str | None = None) -> Path:
    out = construir_csv()
    SALIDA.mkdir(parents=True, exist_ok=True)
    out.to_csv(CSV, index=False)

    reparto = out[COL_SPLIT].value_counts().reindex([TRAIN, VALIDATE, TEST])
    log.info("CSV escrito: %s — %d filas x %d columnas (%d features + target + split)",
             CSV, len(out), out.shape[1], len(spec.FEATURES))
    print("\nreparto del split que se le IMPONE a AutoML:")
    print(reparto.to_string())
    print("\ndistribucion del target:")
    print(out.groupby(COL_SPLIT)[TARGET].value_counts().unstack().to_string())

    if subir_a:
        from google.cloud import storage

        cliente = storage.Client()
        destino = f"automl/{CSV.name}"
        cliente.bucket(subir_a).blob(destino).upload_from_filename(CSV)
        uri = f"gs://{subir_a}/{destino}"
        log.info("subido a %s", uri)
        print(f"\nURI para crear el dataset tabular:\n  {uri}")
    return CSV


# --------------------------------------------------------------------------- #
#  Entrenamiento
# --------------------------------------------------------------------------- #

def entrenar(dataset_id: str, project: str, region: str, bucket: str,
             budget_milli_node_hours: int = 1000):
    """Lanza el AutoML y vuelve enseguida: corre server-side ~2 h."""
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=region,
                    staging_bucket=f"gs://{bucket}")
    log.info("proyecto %s | region %s | dataset %s", project, region, dataset_id)

    ds = aiplatform.TabularDataset(dataset_id)

    job = aiplatform.AutoMLTabularTrainingJob(
        display_name="premier-1x2-automl",
        optimization_prediction_type="classification",
        # Tres clases y probabilidades que tienen que estar calibradas: log-loss.
        optimization_objective="minimize-log-loss",
    )

    modelo = job.run(
        dataset=ds,
        target_column=TARGET,
        # LA LINEA QUE HACE QUE ESTO SIRVA. Sin ella el split es aleatorio y el
        # modelo ve el futuro. Ver el encabezado del módulo.
        predefined_split_column_name=COL_SPLIT,
        budget_milli_node_hours=budget_milli_node_hours,
        model_display_name="premier-1x2-automl",
        disable_early_stopping=False,
        sync=False,
    )

    try:
        job.wait_for_resource_creation()
    except Exception as exc:  # noqa: BLE001
        log.warning("no se pudo esperar la creacion del recurso: %s", exc)

    print("\ntraining pipeline:", getattr(job, "resource_name", "(ver en la consola)"))
    print("monitorear en:")
    print(f"  https://console.cloud.google.com/vertex-ai/training/training-pipelines"
          f"?project={project}")
    print("\nCuando termine, el modelo aparece en Registro de modelos con su evaluacion.")
    print("Para compararlo:  python -m training.automl --metricas --model-id <id>")
    return modelo


# --------------------------------------------------------------------------- #
#  Comparación
# --------------------------------------------------------------------------- #

def _nuestras_metricas() -> dict | None:
    """Las del modelo de EVALUACION que ya está guardado, si lo hay."""
    from training import registry

    for v in sorted((registry.RAIZ / CFG.modelo).glob("2*"), reverse=True):
        meta = json.loads((v / "metadata.json").read_text(encoding="utf-8"))
        if not meta.get("metricas_son_de_generalizacion"):
            continue                      # ese entrenó con el holdout: no sirve comparar
        m = json.loads((v / "metrics.json").read_text(encoding="utf-8"))
        h = m.get("holdout", m)
        return {"version": v.name, "accuracy": h.get("accuracy"),
                "log_loss": h.get("log_loss"), "n": h.get("n"),
                "baselines": h.get("baselines", {}),
                "feature_set_version": meta.get("feature_set_version")}
    return None


def metricas(model_id: str, project: str, region: str) -> None:
    """La evaluación que Vertex calculó sobre NUESTRO holdout, al lado de la nuestra."""
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=region)
    modelo = aiplatform.Model(model_id)
    print(f"modelo AutoML: {modelo.display_name} ({model_id})\n")

    evals = list(modelo.list_model_evaluations())
    if not evals:
        print("Todavia no hay evaluacion: el entrenamiento no termino.")
        return

    # El slice TEST es, por construcción, nuestro holdout 2025-26.
    m = dict(evals[0].to_dict().get("metrics", {}))
    interesan = ("logLoss", "auRoc", "auPrc", "confidenceMetrics")
    automl = {k: m.get(k) for k in interesan if k in m}

    print("--- lo que reporta Vertex sobre el slice TEST (= holdout 2025-26) ---")
    for k, v in automl.items():
        if k == "confidenceMetrics":
            continue
        print(f"  {k:12s} {v}")

    nuestro = _nuestras_metricas()
    print("\n--- nuestro modelo de EVALUACION, sobre las mismas 380 filas ---")
    if nuestro is None:
        print("  No hay ningun modelo con metricas de generalizacion guardado.")
        print("  Corre: python -m training.run --sin-holdout")
        return

    print(f"  version      {nuestro['version']}")
    print(f"  accuracy     {nuestro['accuracy']:.4f}")
    print(f"  log_loss     {nuestro['log_loss']:.4f}")

    # Si el modelo guardado es de otro feature set, la comparacion no es contra el
    # mismo sistema y hay que decirlo antes de que alguien cite el numero.
    if nuestro.get("feature_set_version") != spec.FEATURE_SET_VERSION:
        print()
        print(f"  !! Ese modelo es del feature set "
              f"{nuestro.get('feature_set_version')} y el actual es "
              f"{spec.FEATURE_SET_VERSION}.")
        print("     A AutoML se le dieron las features de AHORA: no es la misma comparacion.")
        print("     Para emparejarlas: python -m training.run --sin-holdout")
    for nombre, b in nuestro["baselines"].items():
        if b.get("log_loss") == b.get("log_loss"):   # descarta NaN
            print(f"  {nombre:20s} accuracy {b['accuracy']:.4f}  "
                  f"log-loss {b.get('log_loss')}")

    print("\nLa comparacion vale porque las dos usan el MISMO split y las MISMAS")
    print("279 features. Con 380 partidos el IC de la accuracy es de +-5 puntos:")
    print("una diferencia menor a eso no distingue a los dos modelos.")


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="AutoML Tabular sobre el Gold del TP.")
    ap.add_argument("--export", action="store_true", help="escribe el CSV")
    ap.add_argument("--entrenar", action="store_true", help="lanza el AutoML")
    ap.add_argument("--metricas", action="store_true", help="compara contra el nuestro")
    ap.add_argument("--subir", action="store_true", help="sube el CSV al bucket")
    ap.add_argument("--bucket", default=None, help="nombre del bucket (sin gs://)")
    ap.add_argument("--project", default=None)
    ap.add_argument("--region", default="us-central1")
    ap.add_argument("--dataset-id", default=None)
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--budget", type=int, default=1000,
                    help="milli node hours (1000 = 1 node-hour, el minimo)")
    args = ap.parse_args()

    setup(CFG.log_level, CFG.log_format)

    if not (args.export or args.entrenar or args.metricas):
        ap.error("elegí una acción: --export, --entrenar o --metricas")

    if args.export:
        exportar(args.bucket if args.subir else None)

    if args.entrenar or args.metricas:
        import os
        project = (args.project or os.environ.get("GOOGLE_CLOUD_PROJECT")
                   or os.environ.get("DEVSHELL_PROJECT_ID"))
        if not project:
            ap.error("falta --project (o la variable GOOGLE_CLOUD_PROJECT)")

        if args.entrenar:
            if not args.dataset_id:
                ap.error("--entrenar necesita --dataset-id (se crea en la consola)")
            entrenar(args.dataset_id, project, args.region,
                     args.bucket or f"{project}-premier-ml", args.budget)

        if args.metricas:
            if not args.model_id:
                ap.error("--metricas necesita --model-id")
            metricas(args.model_id, project, args.region)


if __name__ == "__main__":
    main()
