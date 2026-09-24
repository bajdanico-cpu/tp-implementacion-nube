# Arquitectura del deploy — qué arma `scripts/preparar_demo.sh`

Resumen de los componentes que provisiona `scripts/preparar_demo.sh`, para entender el
deploy completo y poder responder preguntas en la defensa. Los números de línea
(`:90`, `:236`…) refieren a ese script.

> **Ojo: el deploy real del proyecto `tp-mlops-premier-2026` no salió de este script tal
> cual.**
>
> - El script crea el bucket `${PROJECT_ID}-premier-ml`; el real es
>   `tp-mlops-premier-2026-bucket`.
> - El script corre el servicio como `premier-api@`; esa service account **no existe** en
>   el proyecto real, así que todo corre como la service account por defecto de Compute.
>
> Lo que sigue es la arquitectura **como la diseña el script**; donde el deploy real
> difiere, está marcado con *cursiva*.

---

## La arquitectura en un dibujo

```
                 ┌──────────── Cloud Build ────────────┐
  repo (código) ─┤  Dockerfile → imagen (sin datos)    ├─► Artifact Registry
                 └─────────────────────────────────────┘     mlops-2026/premier-ml-api:latest
                                                                   │  la MISMA imagen
                                      ┌────────────────────────────┴───────────────┐
                                      ▼                                            ▼
                     Cloud Run SERVICE  premier-ml-api            Cloud Run JOB  premier-ml-pipeline
                     uvicorn serving.main:app                     python -m pipeline.pre_deadline
                     API + página web, pública                    ingesta → Silver → Gold → predice
                     identidad: premier-api@ (sólo LEE)           identidad: premier-job@ (ESCRIBE)
                          │   │  POST /actualizar ──(run.invoker)──►   │
                          │   │                                        │
                    lee   ▼   │                                        ▼ escribe
                  ┌───────────────── Bucket GCS ─────────────────────────┐
                  │ gold/  predicciones/  models/  silver/  bronze/     │
                  └──────────────────────────────────────────────────────┘
                          │ stdout JSON (los dos)
                          ▼
                     Cloud Logging  (jsonPayload.evento = prediccion, pipeline_disparo, …)
```

---

## Los componentes, uno por uno

| # | Componente | Qué es | Por qué así |
|---|---|---|---|
| 1 | **APIs** (`:90`) | storage, artifactregistry, cloudbuild, run, logging | Se habilitan por proyecto; habilitar es gratis, se paga el uso |
| 2 | **Bucket** (`:100`) | `gs://${PROJECT_ID}-premier-ml`, en us-central1 | Es el **único lugar con estado**: dato y modelos. *Real: `tp-mlops-premier-2026-bucket`* |
| 3 | **Dos service accounts** (`:109`) | `premier-api@` → `objectViewer`; `premier-job@` → `objectAdmin` | Mínimo privilegio: la API **no puede** corromper Gold aunque tenga un bug. *Real: no se crearon; corre la SA por defecto de Compute, que es Editor del proyecto* |
| 4 | **Dato congelado** (`:135`) | `rsync` de la máquina local al bucket: gold, predicciones, models, silver, bronze | **No corre el pipeline** a propósito: si lo hiciera, ingestaría la GW5 ya jugada y se perdería la transición de la demo |
| 5 | **Imagen** (`:197`) | Cloud Build → Artifact Registry `mlops-2026` | **Sólo código.** Dato y modelo se leen del bucket (ver comentario en el `Dockerfile`): cambiar el dato no obliga a rebuildear |
| 6 | **Cloud Run Job** (`:219`) | misma imagen, comando `python -m pipeline.pre_deadline`, 2 GiB / 2 CPU, 30 min, 1 reintento | Es el batch: baja fuentes, reconstruye Silver y Gold, predice la próxima fecha y la registra. **Se despliega pero no se ejecuta** |
| 7 | **Cloud Run Service** (`:236`) | misma imagen con uvicorn, 1 GiB / 1 CPU, público | `/health`, `/predict/{season}/{gw}`, `/calendario` y la página web. `TP_ADMIN_TOKEN` protege el `POST /actualizar` |
| 8 | **Permiso API → Job** (`:248`) | `run.invoker` sobre el Job | La API puede **pedir** que corra el pipeline, pero sigue sin poder escribir el bucket |
| 9 | **Verificación** (`:256`) | `/health` ok, próxima = GW5, GW5 → 200, GW6 → 409, `/actualizar` dice `hace_falta` | El checklist previo a la demo, al estilo del `predemo_check.py` del profesor |

---

## Las variables de entorno: cómo sabe la imagen dónde leer

La imagen **no trae** el switch local/GCS: el `Dockerfile` no tiene ningún
`ENV TP_STORAGE_BACKEND`. Lo pone el **deploy** con `--set-env-vars`.

| Variable | Service | Job | Para qué |
|---|---|---|---|
| `TP_STORAGE_BACKEND=gcs` | ✓ | ✓ | El switch local → bucket |
| `TP_GCS_BUCKET` | ✓ | ✓ | Qué bucket |
| `TP_GCP_PROJECT`, `TP_REGION`, `TP_JOB_NAME` | ✓ | | Para que la API sepa qué Job disparar |
| `TP_ADMIN_TOKEN` | ✓ | | Llave del `POST /actualizar` |
| `K_SERVICE` / `CLOUD_RUN_JOB` | automáticas | automáticas | Las pone Cloud Run; con ellas el logging pasa a JSON (`common/logging_setup.py`) |

El camino en el código:

1. `config.yaml:84` — `storage.backend: "local"` es el valor por defecto.
2. `common/config.py:144` — `os.getenv("TP_STORAGE_BACKEND") or config.yaml`: la variable
   gana si existe.
3. `common/storage.py:220` — `backend()` crea `GCSBackend` o `LocalBackend`. Todo el I/O
   del proyecto pasa por ahí.

Con GCS, las rutas se traducen así (`common/storage.py:108`):

```
data/gold/gold_tp_match.parquet     ->  gs://BUCKET/gold/gold_tp_match.parquet
data/predicciones/x.parquet         ->  gs://BUCKET/predicciones/x.parquet
models/xgb_gbt/<v>/model_seed0.ubj  ->  gs://BUCKET/models/xgb_gbt/<v>/model_seed0.ubj
```

Para levantar la API en Cloud Shell leyendo lo mismo que producción, sin copiar nada:

```bash
pip install -r requirements-serving.txt     # trae google-cloud-storage; requirements.txt lo tiene comentado
export TP_STORAGE_BACKEND=gcs TP_GCS_BUCKET=tp-mlops-premier-2026-bucket
uvicorn serving.main:app --host 0.0.0.0 --port 8080
```

---

## Los tres ciclos de vida (la idea de fondo del diseño)

| Qué | Cuándo cambia | Dónde vive | Cómo se actualiza |
|---|---|---|---|
| **Código** | cuando cambia la lógica | imagen en Artifact Registry | build + deploy → revisión nueva de Cloud Run |
| **Dato** | cada fecha | `gold/`, `predicciones/` en el bucket | el Job, disparado desde la página |
| **Modelo** | cuando se reentrena | `models/xgb_gbt/<versión>/` + `PRODUCTION.json` | `python -m training.registry --promover` |

Es la razón por la que la imagen no lleva datos. Y es también el punto a tener en cuenta
para el rollback: modelo y dato son punteros **compartidos** por todas las revisiones, así
que volver el tráfico a una revisión vieja de Cloud Run revierte **sólo el código**.

---

## El flujo de la demo

1. `preparar_demo.sh` deja el estado inicial: GW1-4 con resultado, GW5 predicha, GW6 en gris.
2. En la página, **"Actualizar datos"** → `POST /actualizar` → dispara el Job.
3. El Job ingesta la GW5, reconstruye Gold, predice y registra la GW6, y lo loguea todo en JSON.
4. El servicio relee Gold (TTL de 5 min) → la GW6 aparece predicha.
5. `bash scripts/preparar_demo.sh --reset` vuelve a subir el estado congelado y reinicia el
   servicio, para ensayar sin gastar la demo.

---

## Preguntas probables en la defensa, con respuesta corta

- **¿Por qué Service y Job con la misma imagen?** Comparten todo el código; sólo cambia el
  comando. No hay dos imágenes que puedan desincronizarse.
- **¿Por qué el modelo no va dentro de la imagen, como en el repo del profesor?** Porque el
  dato cambia cada fecha. La clase 6 (p. 15) lo plantea: "horneado es más simple; bajarlo
  permite cambiarlo sin reconstruir". Elegimos lo segundo.
- **¿Por qué dos identidades?** Mínimo privilegio: la API es pública y sólo lee; el Job
  escribe. *En el deploy real no se aplicó — conviene decirlo, o crearlas.*
- **¿Qué pasa si el Job falla a mitad de camino?** `common.storage.archivar` guarda la
  versión anterior de Gold antes de reescribirla, y el servicio sigue sirviendo el último
  Gold bueno.
- **¿Cómo se ve lo que pasa en producción?** Logs JSON en Cloud Logging, filtrables por
  `jsonPayload.evento` (`prediccion`, `prediccion_error`, `pipeline_disparo`…).
