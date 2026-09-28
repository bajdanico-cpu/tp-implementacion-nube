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

## 4. Grabar el video: de la GW5 a la GW6, con sus logs

El estado de hoy ya tiene la GW6 predicha (el Job corrió el 22/09). Para grabar la
transición hay que volver el **Gold del bucket** a la versión anterior, donde la GW5 es la
próxima. Es el escenario C de `gcp/ROLLBACK.md` (rollback de datos) hecho en vivo: **no se
borra nada**, restaurar archiva lo que reemplaza.

**4a. Volver Gold a la GW5** (Cloud Shell, en `~/tp-premier-ml`)

```bash
pip install -q -r requirements-cloud.txt          # sólo si falta pandas / google-cloud-storage
export TP_STORAGE_BACKEND=gcs TP_GCS_BUCKET=tp-mlops-premier-2026-bucket

python -m common.versiones --diff gold_tp_match   # historia de Gold EN EL BUCKET
```

La tabla lista las versiones **archivadas**, de la más vieja (arriba) a la más nueva
(abajo); la vigente, la de 1580 filas que escribió el Job con la GW6 próxima, no aparece.
Buscar la **última fila con 1570 filas**: es el Gold con la GW5 próxima. Copiar su `stamp`.

```bash
python -m common.versiones --restaurar gold_tp_match <STAMP_CON_1570_FILAS>
```

El servicio relee Gold cada 5 minutos. Verificar:

```bash
curl -s $SERVICE_URL/health | python3 -m json.tool   # proxima_predecible: 5, gold_filas: 1570
curl -s $SERVICE_URL/actualizar | python3 -m json.tool # hace_falta: true
```

Si en 5 minutos sigue en 6: `gcloud run services update $SERVICE --region $REGION
--update-env-vars TP_RESET_AT=$(date +%s)` (crea una revisión nueva con el mismo código y
fuerza la relectura).

> Si `--diff` no muestra ninguna versión de 1570 filas, **no improvisar**: la PC de Nico
> tiene esa versión (`20260920T224839Z` en `data/_versiones/`) y se sube de ahí.

**4b. Grabar**

1. Abrir `$SERVICE_URL`. La GW5 predicha; la GW6 en adelante, en gris.
2. El panel "Estado del dato" dice que conviene actualizar.
3. Apretar **"Actualizar datos"** (pide el token la primera vez; ver abajo cómo leerlo).
4. Mientras corre (varios minutos), mostrar la ejecución: consola → **Cloud Run → Jobs →
   premier-ml-pipeline → Ejecuciones** → la que está en curso → **Registros**. Se ven los
   pasos: `bronze_fpl`, `bronze_vaastav`, `bronze_fd`, `bronze_opta`, `silver`,
   `competencias`, `opta`, `gold`, `predecir`.
5. Al terminar, la página se recarga sola: la GW5 con resultado, la GW6 predicha.

El token del botón, si lo pide:

```bash
gcloud run services describe $SERVICE --region $REGION --format=json \
  | python3 -c "import json,sys; e=json.load(sys.stdin)['spec']['template']['spec']['containers'][0].get('env',[]); print({x['name']:x.get('value') for x in e}.get('TP_ADMIN_TOKEN'))"
```

**4c. La evidencia en logs** (al terminar; quedan en archivos para el informe)

```bash
# El servicio: quién pidió actualizar y cómo terminó
gcloud logging read 'resource.type="cloud_run_revision" AND resource.labels.service_name="premier-ml-api"
  AND (jsonPayload.evento="pipeline_disparo" OR jsonPayload.evento="pipeline_fin")' --freshness=3h \
  --format='table(timestamp, jsonPayload.evento, jsonPayload.tarea, jsonPayload.estado, jsonPayload.segundos)' \
  | tee evidencia_disparo.txt

# El Job: paso por paso
gcloud logging read 'resource.type="cloud_run_job" AND resource.labels.job_name="premier-ml-pipeline"' \
  --freshness=3h --limit=200 --order=asc --format='table(timestamp, severity, jsonPayload.message)' \
  | tee evidencia_job.txt

# La ejecución, con su duración y resultado
gcloud run jobs executions list --job premier-ml-pipeline --region $REGION | tee evidencia_ejecuciones.txt

# Y la GW6 servida después
python scripts/smoke_load.py --n 10       # y el logs_servidor.py --corrida que imprime
```

Después del video, el estado queda igual que hoy: GW6 próxima. No hay que deshacer nada.

---

## 5. Si algo falla

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
