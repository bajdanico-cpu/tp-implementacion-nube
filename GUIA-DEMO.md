# Guía de demo — probar la versión nueva en GCP y presentar

Lo que muestra, en el orden de la materia:

| Clase | Qué se muestra | Paso |
|---|---|---|
| 6 — Docker y Cloud Run | imagen construida en la nube, servicio con URL pública | 2 |
| 5 — APIs | `/health` y `/predict` respondiendo con su contrato | 3a |
| 7 — latencia | carga desde el **cliente** (p50/p95/max, cold start) | 3b |
| 7 — logs | la misma carga vista desde el **servidor** en Cloud Logging | 3c, 3d |
| 7 — una métrica | latencia del servidor en Metrics Explorer | 3e |
| 7 — rollback | romper un deploy a propósito y volver en segundos | 3f |

Detalle y fundamentos: `gcp/OBSERVABILIDAD.md` y `gcp/ROLLBACK.md`.

---

## 0. Punto de retorno (una vez, en tu PC, ANTES de subir nada)

```bash
git tag demo-estable 65cc851        # el main que hoy está desplegado y anda
git push origin demo-estable
```

---

## 1. Cloud Shell: traer el código y fijar variables

```bash
cd ~/tp-premier-ml && git pull
export PROJECT_ID=$(gcloud config get-value project)     # tp-mlops-premier-2026
export REGION=us-central1 SERVICE=premier-ml-api REPO=mlops-2026

# ANOTAR: la revisión que anda hoy. Es la red de seguridad (Plan B).
export BUENA=$(gcloud run services describe $SERVICE --region $REGION --format='value(status.latestReadyRevisionName)')
echo "BUENA=$BUENA"
```

---

## 2. Construir y desplegar la versión nueva

Tag propio para la imagen: `:latest`, la que anda hoy, **no se pisa**.

```bash
export IMAGE_NUEVA="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:logs-rollback"
gcloud builds submit --tag "$IMAGE_NUEVA" .

gcloud run deploy $SERVICE --image "$IMAGE_NUEVA" --region $REGION \
  --update-env-vars "TP_MODEL_VERSION=20260825T024144Z,TP_SEASON=2026-27"

export SERVICE_URL=$(gcloud run services describe $SERVICE --region $REGION --format='value(status.url)')
curl -s $SERVICE_URL/health | python3 -m json.tool
```

Tiene que decir `"status": "ok"` y `"model_version": "20260825T024144Z"`.
**Si no → Plan B (abajo), y listo: el servicio vuelve a como estaba.**

> `--update-env-vars`, nunca `--set-env-vars`: el segundo borra `TP_STORAGE_BACKEND` y
> `TP_GCS_BUCKET` y el servicio deja de leer el bucket.

Las métricas basadas en logs, una sola vez (cuentan desde que se crean; hacerlo el día
anterior para tener datos):

```bash
gcloud logging metrics create premier_prediccion_latencia --config-from-file=gcp/metricas/prediccion_latencia.yaml
gcloud logging metrics create premier_prediccion_errores  --config-from-file=gcp/metricas/prediccion_errores.yaml
```

---

## 3. La demo

**3a. El servicio vivo y su contrato**

```bash
curl -s $SERVICE_URL/health | python3 -m json.tool
curl -s $SERVICE_URL/predict/2026-27/6 | python3 -m json.tool | head -30
```

La página: abrir `$SERVICE_URL` en el navegador.

**3b. Latencia desde el CLIENTE**

```bash
python scripts/smoke_load.py --n 20
```

p50/p95/max. El `max` es el arranque en frío. Al final imprime una línea
`python scripts/logs_servidor.py --corrida smoke-XXXX`: **copiarla**.

**3c. La misma corrida desde el SERVIDOR** (esperar ~30 s a que llegue a Cloud Logging)

```bash
python scripts/logs_servidor.py --corrida smoke-XXXX
```

La tabla cliente / borde de Cloud Run / app: cuánto es red, cuánto es Cloud Run y cuánto
es nuestro modelo. Más la revisión que atendió y el modelo servido.

**3d. Los logs en la consola**

Logging → **Explorador de registros** (no Log Analytics) · proyecto `tp-mlops-premier-2026`
· rango "Últimas 3 horas":

```
resource.type="cloud_run_revision"
resource.labels.service_name="premier-ml-api"
jsonPayload.evento="prediccion"
```

Abrir una línea: campos sueltos (`gameweek`, `latencia_ms`, `model_version`,
`corrida`). Ni una probabilidad por partido, ni una feature: sólo agregados. Los errores:
`jsonPayload.evento="prediccion_error"`.

**3e. Una métrica**

Monitoring → **Explorador de métricas** → `logging.googleapis.com/user/premier_prediccion_latencia`
→ agregación "95th percentile", agrupar por `estado`. Y los signos vitales que Cloud Run da
gratis: Cloud Run → premier-ml-api → **MÉTRICAS**.

**3f. Rollback: romper a propósito y volver**

```bash
bash scripts/rollback.sh                    # revisiones: modelo, temporada, % de tráfico

# Romper: la revisión nueva pide un modelo que no existe
gcloud run services update $SERVICE --region $REGION --update-env-vars TP_MODEL_VERSION=no-existe
curl -s $SERVICE_URL/health                 # "degraded", y dice por qué
python scripts/smoke_load.py --n 10         # 503 desde el cliente
python scripts/logs_servidor.py --desde 10m --errores   # 503 desde el servidor, y qué revisión

# Volver: 100 % a la revisión anterior, sin rebuild. Verifica /health solo.
bash scripts/rollback.sh --anterior
python scripts/smoke_load.py --n 10         # 200 otra vez
```

El punto a decir: cada revisión **fija su modelo** (`TP_MODEL_VERSION`), así que el
rollback por tráfico vuelve atrás código y modelo juntos. Los otros escenarios (Gold malo,
fuente caída, cambio de temporada) están en `gcp/ROLLBACK.md`.

> Después del simulacro **no usar `--to-latest`**: la última revisión es la rota. El tráfico
> queda donde lo dejó `rollback.sh`.
>
> Si el simulacro no rompe nada (el `/health` sigue ok), el tráfico estaba fijado a una
> revisión y la rota nació sin tráfico. Se arregla con
> `gcloud run services update-traffic $SERVICE --region $REGION --to-latest` y se repite.

---

## 4. Si algo falla

**Plan B — segundos, sin tocar git.** El servicio vuelve a la revisión de hoy:

```bash
gcloud run services update-traffic $SERVICE --region $REGION --to-revisions $BUENA=100
curl -s $SERVICE_URL/health
```

(Si se perdió la variable: `bash scripts/rollback.sh` lista las revisiones; `$BUENA` es la
más NUEVA de las que dicen `(PRODUCTION.json)` en la columna del modelo: todas las
revisiones de antes de este cambio no fijan modelo.)

**Plan C — volver a lo de hoy desde cero.** Código de `demo-estable` y todo reconstruido:

```bash
cd ~/tp-premier-ml
git fetch --tags && git checkout demo-estable
export IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:latest"
gcloud builds submit --tag "$IMAGE" .
gcloud run deploy $SERVICE --image "$IMAGE" --region $REGION \
  --remove-env-vars TP_MODEL_VERSION,TP_SEASON
curl -s $(gcloud run services describe $SERVICE --region $REGION --format='value(status.url)')/health
```

Y en `main`, si hace falta deshacer el commit: `git revert HEAD && git push`.
