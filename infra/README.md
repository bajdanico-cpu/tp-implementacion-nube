# Infraestructura

**El pipeline ya corre en Cloud Shell hasta la predicción de una fecha**, con el dato y los
artefactos persistidos en un bucket de GCS:

- [`notebooks/01_gcp_cloudshell.ipynb`](../notebooks/01_gcp_cloudshell.ipynb) — el lab: se
  clona el repo, se corre celda por celda, cada paso deja un recurso visible en la consola,
  y **la última celda borra todo** para no dejar nada facturando.
- [`gcp/paso-a-paso.md`](../gcp/paso-a-paso.md) — el recorrido completo desde crear el
  proyecto en GCP, para alguien que arranca de cero.
- [`gcp/runbook.md`](../gcp/runbook.md) — los mismos pasos por terminal, con los chequeos
  de limpieza.

Lo que **todavía no está** es el servicio permanente (Cloud Run), la imagen y los dos jobs
programados. Por eso el resto de esta página sigue siendo el plan.

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
