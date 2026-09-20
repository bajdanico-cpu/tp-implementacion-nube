# Serving

**Construido y desplegado.** La API vive en `serving/main.py`, la predicción en
`predict.py`, el registro en `registro.py` y la regla de decisión en `decision.py`.
La guía de despliegue es [`gcp/GUIA-DEPLOY.md`](../gcp/GUIA-DEPLOY.md).

El servicio **no construye features**: hace un lookup sobre la tabla Gold, donde el
pipeline ya las dejó calculadas con el mismo código del entrenamiento. Antes reconstruía
las 279 features en cada request —25 segundos— y por eso la imagen tenía que llevar las
ocho tablas de Silver adentro. Hay un test que impide que vuelva a pasar
(`test_predict_no_lee_silver`).

Una fecha se responde según su estado: jugada devuelve la predicción **congelada** del
registro más el resultado real, la próxima se predice en vivo, y una que todavía no tiene
fila responde 409 diciendo cuál sí.

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

## La regla de decisión es un paso aparte (`decision.py`)

El modelo devuelve **tres probabilidades**; convertirlas en una clase es otra decisión, y
está separada a propósito. La de producción es `argmax`. Las **candidatas** corren en
paralelo: dejan su propia columna en cada predicción registrada, se miden fecha a fecha, y
no cambian lo que el sistema anuncia hasta que el McNemar las respalde.

```
python -m serving.decision              # las reglas activas y dónde discrepan
python -m serving.decision --backfill   # etiqueta las predicciones ya registradas
```

Hoy hay una candidata: `umbral_empate_030` — llamar empate cuando `p_draw >= 0,30`. En el
holdout pasa de 4 empates predichos a 36, pero **la accuracy no mejora de forma
demostrable**: el delta va de −0,005 a +0,026 según la semilla (`training/decision_eval.py`),
o sea más chico que el ruido, y el AUC del empate es 0,515 — mueve **cuántos** empates se
anuncian, no **cuáles**. Corre igual porque es la palanca de negocio correcta y porque la
temporada en curso es la única muestra que todavía no se miró.

**No es un cambio de modelo.** Mismos boosters, mismas probabilidades: por eso el candidato
no tiene `.ubj` propio, no hace falta reentrenar, y las predicciones viejas se pueden
etiquetar hacia atrás desde las probabilidades ya guardadas (`--backfill`, que no toca ni
`p_*` ni `predicted_at`). Cada candidata declara `desde` para que quede claro qué fechas
son medición y cuáles son sólo etiquetado retrospectivo.

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
