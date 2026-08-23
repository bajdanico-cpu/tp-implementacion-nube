# Serving

Pendiente.

## Diseño

**Cloud Run + FastAPI.** Modelo versionado en GCS (o en Vertex AI Model Registry si
se quiere mostrar el servicio administrado).

```
GET  /health                      → estado y versión del modelo cargado
GET  /predict/{season}/{gameweek} → predicciones de esa fecha
POST /predict                     → predicción ad-hoc con features explícitas
```

Toda predicción servida se **registra** con su versión de modelo y su snapshot de
features. Sin ese registro, el monitoreo de la carpeta `monitoring/` no tiene con qué
trabajar.

## Ojo con CORS

La API de FPL **no se puede llamar desde un frontend**: tiene política de CORS. Toda
llamada sale del servidor. Si se hace una UI, tiene que pegarle a este servicio, no a
FPL directamente.

## Orquestación

**Cloud Scheduler → Cloud Run Jobs**, dos disparos por semana:

1. **Pre-deadline** — ingesta del snapshot de FPL, generación de features y predicción
   de la fecha siguiente. Tiene que correr **antes** del `deadline_time`, que es lo
   que hace válida la predicción.
2. **Post-fecha** — ingesta de resultados, actualización de Silver/Gold, cálculo de
   métricas de la predicción anterior y evaluación del retraining.

Composer es overkill para esto y factura 24/7. Cloud Run Jobs + Scheduler alcanza y es
más defendible en costo.

## Migración desde local

El código no cambia. `common/storage.py` abstrae todo el I/O: hay que implementar
`GCSBackend` (los métodos ya están declarados, levantan `NotImplementedError` con las
instrucciones) y cambiar `storage.backend` a `"gcs"` en `config.yaml`.

Las rutas de Bronze ya están armadas como claves de objeto
(`bronze/fpl/2026-27/bootstrap/ingested_at=.../`), así que mapean directo a un bucket.
