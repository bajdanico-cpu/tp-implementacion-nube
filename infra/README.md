# Infraestructura

Pendiente. **No hay proyecto GCP todavía** — por eso todo corre local y la capa de
storage está abstraída.

## Recursos previstos

| Recurso | Para qué |
|---|---|
| Bucket GCS | Bronze (JSON/CSV crudos, append-only) y artefactos de modelo |
| BigQuery dataset | Silver y Gold |
| Cloud Run Job — ingesta | disparado por Scheduler, pre-deadline y post-fecha |
| Cloud Run Service — API | serving del modelo |
| Cloud Scheduler | los dos cron semanales |
| Artifact Registry | imágenes de los contenedores |

## Qué hace falta para activarlo

1. Instalar el SDK de `gcloud` (hoy no está en la máquina).
2. Completar `storage.gcp` en `config.yaml`: `project_id`, `region`, `bucket`,
   `bq_dataset`.
3. Descomentar `google-cloud-storage` y `google-cloud-bigquery` en `requirements.txt`.
4. Implementar `GCSBackend` en `common/storage.py` — los métodos ya están declarados
   con la firma correcta.
5. Cambiar `storage.backend` a `"gcs"`.

Ningún paso toca la lógica de ingesta ni de transformación. Ése era el objetivo del
diseño local-first.

## Costo

Cloud Run Jobs cobra por ejecución; con dos disparos semanales el costo es marginal.
La alternativa (Composer / Airflow administrado) factura 24/7 y no se justifica para
dos triggers por semana — vale la pena decirlo en la defensa.
