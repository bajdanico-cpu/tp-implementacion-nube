# Infraestructura

**El despliegue está construido.** La guía completa, de una cuenta vacía a un servicio
operado, es [`gcp/GUIA-DEPLOY.md`](../gcp/GUIA-DEPLOY.md).

```
        Cloud Storage (gs://$BUCKET)
        bronze/  silver/  gold/  predicciones/  models/
              │                          ▲
              │ lee gold + predicciones  │ escribe todo
              ▼                          │
     Cloud Run Service            Cloud Run Job
     premier-ml-api               premier-ml-pipeline
     (API + la página)            (python -m pipeline.pre_deadline)
```

| Recurso | Estado | Para qué |
|---|---|---|
| Bucket GCS | **hecho** | Bronze, Silver, Gold, predicciones y modelos |
| Artifact Registry | **hecho** | la imagen |
| Cloud Run Service | **hecho** | la API y la página, con escala a cero |
| Cloud Run Job | **hecho** | el pipeline pre-deadline |
| Cloud Scheduler | *pendiente* | ver abajo |
| BigQuery dataset | *descartado* | ver abajo |

## Tres decisiones que conviene poder defender

**Una sola imagen para el Service y el Job.** Comparten todo el código y sólo cambia el
comando (`--command python --args -m,pipeline.pre_deadline`). Dos imágenes serían dos
cosas que se pueden desincronizar, y la que sirve tiene que tener exactamente el mismo
código de features que la que entrena: es la defensa contra el train/serve skew.

**El dato no va en la imagen.** Antes sí: `COPY models` y `COPY data/silver`. Eso ataba
tres ciclos de vida distintos —el código cambia cuando cambia la lógica, el dato cada
fecha, el modelo cuando se reentrena— al mismo artefacto de build, y obligaba a
redesplegar para predecir una fecha nueva. Hoy se leen del bucket vía `GCSBackend`, y se
configura con `TP_STORAGE_BACKEND=gcs` sin reconstruir nada.

**Dos service accounts.** La API sólo tiene `objectViewer`; el Job tiene `objectAdmin`.
Con una sola identidad todo andaría igual, y justamente por eso vale nombrarlo: un bug en
el camino de lectura **no puede** corromper Gold, en vez de simplemente no deber hacerlo.

## Lo que queda

**Cloud Scheduler.** El Job está listo para que lo dispare, pero el disparo correcto va
atado al `deadline_time` de cada fecha y ése no cae en un día fijo: la Premier mueve
horarios por TV. Un cron semanal sería una aproximación, y preferimos decirlo antes que
presentar como automático algo que se desfasa solo.

**BigQuery para Silver y Gold: descartado.** Estaba en el plan original. Gold son 1.570
filas y 1,5 MB; un parquet en el bucket se lee entero en memoria en milisegundos y no
agrega un servicio más que explicar, configurar y apagar. BigQuery empieza a tener sentido
con volúmenes que este caso no tiene.
