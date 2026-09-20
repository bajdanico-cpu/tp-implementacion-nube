# Guía de despliegue en GCP, de cero a operado

De una cuenta vacía a un servicio vivo con su página, su job de pipeline y la evidencia
de operación que pide la clase 7. Todo es copiar y pegar en la **terminal de Cloud
Shell**; donde haga falta la consola web, se dice.

> **Antes de empezar:** [`paso-a-paso.md`](paso-a-paso.md) es el lab de la clase 4 —el
> pipeline corriendo en Cloud Shell, sin desplegar nada. Esta guía es el siguiente paso y
> asume que el proyecto ya existe.

---

## 0 · Qué vas a tener al final

```
        Cloud Storage (gs://$BUCKET)
        bronze/  silver/  gold/  predicciones/  models/
              │                          ▲
              │ lee gold + predicciones  │ escribe todo
              ▼                          │
     Cloud Run Service            Cloud Run Job
     premier-ml-api  ──────────▶  premier-ml-pipeline
     · GET  /              (la página)      ▲
     · GET  /health                         │
     · GET  /calendario/{season}            │ POST /actualizar
     · GET  /predict/{season}/{gw}          │ lo dispara, no lo corre
     · GET  /actualizar     (¿hace falta?)  │
     · POST /actualizar     ────────────────┘
     · GET  /actualizar/{id}
```

Dos unidades de cómputo, **una sola imagen**: comparten el código y sólo cambia el
comando. El dato y el modelo no viajan adentro de la imagen; se leen del bucket.

| Recurso | Cuesta | Por qué existe |
|---|---|---|
| Bucket | centavos por mes | Bronze/Silver/Gold, predicciones y modelos |
| Artifact Registry | centavos | la imagen |
| Cloud Run **Service** | ~0 con escala a cero | la API y la página, 24/7 |
| Cloud Run **Job** | por ejecución | el pipeline pre-deadline |

---

## El atajo: un solo comando

Si lo que querés es **dejar todo andando para la defensa**, no hace falta leer el resto:

```bash
bash scripts/preparar_demo.sh
```

Hace los ocho pasos de esta guía —APIs, bucket, las dos identidades, los permisos, la
imagen, el Job, el servicio— y al final **verifica** que haya quedado como tiene que
quedar. Es idempotente: se puede correr dos veces.

> ### Antes: llevar el dato a Cloud Shell
>
> **El dato no está en git.** `data/` y los `.ubj` están en `.gitignore` a propósito, así
> que un `git pull` trae el código y nada más. Y regenerarlo allá **no sirve para la
> demo**: ingestaría la GW5, que ya se jugó, y la transición que queremos mostrar
> desaparecería.
>
> Dos caminos, según si tenés `gcloud` en tu máquina:
>
> **(a) Con gcloud local** — el más directo, y además sube Bronze, con lo cual la primera
> corrida del Job no re-descarga cinco temporadas:
>
> ```bash
> gcloud storage rsync -r data/bronze "gs://${BUCKET}/bronze"
> gcloud storage rsync -r data/silver "gs://${BUCKET}/silver"
> gcloud storage rsync -r data/gold   "gs://${BUCKET}/gold"
> gcloud storage rsync -r data/predicciones "gs://${BUCKET}/predicciones"
> gcloud storage rsync -r models "gs://${BUCKET}/models"
> ```
>
> **(b) Sin gcloud local** — se arma un paquete de 5 MB y se sube por la interfaz:
>
> ```powershell
> python -m scripts.bundle_demo          # deja demo-premier-ml.zip
> ```
>
> En Cloud Shell: menú de tres puntos → **Subir** → elegí el zip. Después:
>
> ```bash
> cd ~/tp-implementacion-nube
> unzip -o ~/demo-premier-ml.zip
> bash scripts/preparar_demo.sh
> ```
>
> El paquete lleva Gold, las predicciones registradas, Silver y el modelo de producción.
> **No lleva Bronze** (son ~300 MB), así que la primera corrida del Job baja las fuentes
> de nuevo y tarda varios minutos más. Para la demo en vivo conviene (a).

El resto de la guía explica qué hace cada paso y por qué. Conviene leerla antes de la
defensa, porque las preguntas van a ser sobre eso y no sobre el script.

### El estado que deja, y por qué ése

El script **sube el dato congelado y no corre el pipeline**. No es un detalle de
implementación: es lo que hace que haya algo para mostrar en vivo.

La GW5 se jugó el 18–20 de septiembre y sus resultados todavía no están ingestados. Ese
desfasaje —entre que algo pasa en el mundo y que el sistema se entera— es exactamente lo
que el ciclo cerrado viene a resolver, y es lo que se demuestra apretando un botón:

```
   ANTES del botón                    DESPUES del botón
   GW1-4  con resultado               GW1-5  con resultado
   GW5    predicha, sin jugar   →     GW6    predicha      <- aparece sola
   GW6+   en gris, sin features       GW7+   en gris
```

Si el script corriera el pipeline al provisionar, ingestaría la GW5 y la demo se quedaría
sin transición: mostraría un sistema que ya está al día, que es mucho menos interesante
que uno que se pone al día mientras lo mirás.

**El Job lo disparás vos**, desde la página, cuando estés grabando.

---

## 1 · Variables

Todo lo demás se deriva de acá. Si abrís una terminal nueva, volvé a correr este bloque.

```bash
export PROJECT_ID="$(gcloud config get-value project)"
export REGION="us-central1"
export BUCKET="${PROJECT_ID}-premier-ml"
export REPO="mlops-2026"
export SERVICE="premier-ml-api"
export JOB="premier-ml-pipeline"
export IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:latest"

echo "proyecto=$PROJECT_ID  bucket=$BUCKET  imagen=$IMAGE"
```

## 2 · Prender las APIs

Se habilitan **por proyecto** y un proyecto nuevo nace con casi todo apagado. Habilitar
es gratis; se paga el uso.

```bash
gcloud services enable \
  storage.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  logging.googleapis.com
```

---

## 3 · El dato y el modelo, al bucket

El repo **no trae datos ni `.ubj`**: están en `.gitignore` a propósito. Se regeneran.

```bash
git clone <url-del-repo> tp-premier-ml && cd tp-premier-ml
pip install -q -r requirements-cloud.txt

# La cadena entera: ingesta -> Silver -> Gold -> predicción registrada.
python -m pipeline.pre_deadline

# El modelo. `--sin-holdout` es el que se REPORTA; sin flag, el que SIRVE.
python -m training.run --sin-holdout
python -m training.run
python -m training.registry --listar
```

> **`requirements-cloud.txt` y no `requirements.txt`.** El de local está pinneado a
> wheels `cp314` y Cloud Shell trae otro Python: pip intentaría compilar numpy y pandas
> desde el fuente y la sesión se cae por tiempo o memoria. Ver el encabezado del archivo.

**Fijá qué modelo sirve.** Sin esto, cuál se sirve depende de cómo ordena `glob` — y ya
pasó: el 20/09/2026 el servicio quedó devolviendo 503 en todas las fechas porque eligió
una versión que había llegado por git con su metadata y sin binarios.

```bash
python -m training.registry --promover <VERSION> --motivo "modelo de produccion del TP"
```

Elegí una con **`holdout=True`**: ése es el modelo de producción, entrenado también con la
última temporada. El de `holdout=False` es el de evaluación, y sus métricas significan
algo justamente porque no la vio.

### Subir

> **¿Ya tenés un bucket con cosas adentro?** Reusalo: exportá `BUCKET` con su nombre
> antes de correr nada y todo lo demás se acomoda. `rsync` **suma y actualiza**, no borra
> lo que ya estaba, así que convivir con otros archivos no es problema. Lo único que el
> código espera son los prefijos `gold/`, `predicciones/` y `models/`; si el bucket usa
> otra estructura, `TP_GCS_PREFIX` mete todo bajo una carpeta y listo.

```bash
gcloud storage buckets create "gs://${BUCKET}" --location="${REGION}" || true

gcloud storage rsync -r data/silver        "gs://${BUCKET}/silver"
gcloud storage rsync -r data/gold          "gs://${BUCKET}/gold"
gcloud storage rsync -r data/predicciones  "gs://${BUCKET}/predicciones"
gcloud storage rsync -r models             "gs://${BUCKET}/models"
gcloud storage rsync -r data/bronze        "gs://${BUCKET}/bronze"   # sólo lo necesita el Job

gcloud storage ls "gs://${BUCKET}/"
```

El layout del bucket es **el mismo que las rutas locales**, que es lo que hace que subir a
mano y leer desde el código sean la misma cosa. El mapeo vive en `GCSBackend.key()`.

### Probar el backend **antes** de tocar Cloud Run

Este paso ahorra media tarde de logs: separa "el backend anda" de "Cloud Run anda".

```bash
export TP_STORAGE_BACKEND="gcs"
export TP_GCS_BUCKET="${BUCKET}"
pip install -q google-cloud-storage

python -c "from common.storage import read_table; print(read_table('gold_tp_match', layer='gold').shape)"
python -m serving.predict --gw 6 --no-guardar

unset TP_STORAGE_BACKEND TP_GCS_BUCKET   # para que el resto siga en local
```

---

## 4 · Identidades: dos, no una

```bash
gcloud iam service-accounts create premier-api  --display-name "Premier ML API (solo lee)"
gcloud iam service-accounts create premier-job  --display-name "Premier ML pipeline (escribe)"

export SA_API="premier-api@${PROJECT_ID}.iam.gserviceaccount.com"
export SA_JOB="premier-job@${PROJECT_ID}.iam.gserviceaccount.com"

# La API sólo LEE.
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${SA_API}" --role="roles/storage.objectViewer"

# El Job escribe.
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${SA_JOB}" --role="roles/storage.objectAdmin"
```

Con una sola identidad todo esto funcionaría igual, y por eso conviene explicar la
decisión: **un bug en el camino de lectura no puede corromper Gold**. Es gratis y es la
diferencia entre "no debería escribir" y "no puede".

### Que la API pueda pedir una actualización

El botón *Actualizar datos* de la página hace `POST /actualizar`, y eso **no corre el
pipeline adentro del request**: le pide al Job que lo corra. Para eso la identidad de la
API necesita poder ejecutarlo, y nada más:

```bash
gcloud run jobs add-iam-policy-binding "${JOB}" --region "${REGION}" \
  --member="serviceAccount:${SA_API}" --role="roles/run.invoker"
```

Sigue sin poder escribir en el bucket: escribe el Job, con su propia identidad. La API
sólo tiene permiso para **pedir** que arranque.

---

## 5 · Construir la imagen

```bash
gcloud artifacts repositories create "${REPO}" \
  --repository-format=docker --location="${REGION}" || true

gcloud builds submit --tag "${IMAGE}" .
```

El `.gcloudignore` deja afuera `data/` y `models/`: el contexto de build es **sólo
código**, menos de un mega. Antes eran ~60 MB porque la imagen los horneaba adentro.

---

## 6 · Desplegar el servicio

**El token del botón.** `POST /actualizar` gasta cómputo y el servicio es público para
que la página lo sea. La regla: en Cloud Run hace falta el header `X-Admin-Token`, y si
`TP_ADMIN_TOKEN` no está definido el endpoint queda **apagado** en vez de abierto.

```bash
export ADMIN_TOKEN="$(openssl rand -hex 16)"
echo "guardalo: $ADMIN_TOKEN"
```

```bash
gcloud run deploy "${SERVICE}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --service-account "${SA_API}" \
  --set-env-vars "TP_STORAGE_BACKEND=gcs,TP_GCS_BUCKET=${BUCKET},TP_GCP_PROJECT=${PROJECT_ID},TP_REGION=${REGION},TP_JOB_NAME=${JOB},TP_ADMIN_TOKEN=${ADMIN_TOKEN}" \
  --memory 1Gi \
  --cpu 1

export SERVICE_URL="$(gcloud run services describe "${SERVICE}" \
  --region "${REGION}" --format='value(status.url)')"
echo "$SERVICE_URL"
```

**`--set-env-vars` y no un `config.yaml` distinto**: el mismo contenedor corre contra el
disco local o contra el bucket según la variable. No hay que reconstruir nada para
cambiar de entorno, y no hay dos configuraciones que se puedan desincronizar.

`--memory 1Gi` porque el proceso mantiene la tabla Gold y cinco boosters en memoria. Con
512 Mi el arranque queda justo.

### Probar

```bash
curl -s "${SERVICE_URL}/health" | jq .

# Fecha ya jugada: la predicción CONGELADA + el resultado real.
curl -s "${SERVICE_URL}/predict/2026-27/2" | jq '{estado, origen, pre_deadline, accuracy}'

# La próxima: se predice en vivo.
curl -s "${SERVICE_URL}/predict/2026-27/6" | jq '.predictions[] |
  {home_short, away_short, p_home, p_draw, p_away, prediccion}'

# Una lejana: 409, y dice cuál sí se puede.
curl -s -i "${SERVICE_URL}/predict/2026-27/20" | head -20

curl -s "${SERVICE_URL}/calendario/2026-27" | jq '.proxima_predecible'
```

Y **la página**: abrí `$SERVICE_URL` en el browser. Sale de la misma app, así que no hay
CORS ni una URL aparte que mantener.

### El botón de actualizar

```bash
# ¿Hace falta? Se contesta mirando Gold, sin tocar Silver.
curl -s "${SERVICE_URL}/actualizar" | jq '{hace_falta, motivo, proxima_predecible}'

# Pedirlo. Responde 202 y NO espera: la corrida tarda minutos.
TAREA=$(curl -s -X POST -H "X-Admin-Token: ${ADMIN_TOKEN}" "${SERVICE_URL}/actualizar" | jq -r .id)

# Seguirla.
curl -s "${SERVICE_URL}/actualizar/${TAREA}" | jq '{estado, destino, segundos}'
```

Sin el header da **403**; con una corrida ya en curso, **409**. Las dos cosas están bien y
conviene mostrarlas en la demo: son el control de abuso y el de concurrencia.

---

## 7 · El Job del pipeline

Misma imagen, otro comando.

```bash
gcloud run jobs deploy "${JOB}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --service-account "${SA_JOB}" \
  --set-env-vars "TP_STORAGE_BACKEND=gcs,TP_GCS_BUCKET=${BUCKET}" \
  --memory 2Gi \
  --cpu 2 \
  --task-timeout 30m \
  --max-retries 1 \
  --command python \
  --args "-m,pipeline.pre_deadline"

gcloud run jobs execute "${JOB}" --region "${REGION}" --wait
```

Cada corrida deja su JSON en `gs://$BUCKET/pipeline/runs/`, con el detalle paso por paso:

```bash
gcloud storage ls "gs://${BUCKET}/pipeline/runs/"
gcloud storage cat "gs://${BUCKET}/pipeline/runs/<stamp>.json" | jq '.pasos[] | {paso, estado, segundos}'
```

> **Sobre Cloud Scheduler.** El disparo correcto va atado al `deadline_time` de cada
> fecha, que no cae en un día fijo: la Premier mueve horarios por TV. Un cron semanal
> sería una aproximación y conviene decirlo así en la defensa. El Job está listo para que
> Scheduler lo invoque; queda como trabajo siguiente.

---

## 8 · Operar: la evidencia que pide la clase 7

### Está respondiendo

```bash
curl -s "${SERVICE_URL}/health" | jq '{status, model_version, gold_built_at, proxima_predecible}'
```

### Cuánto tarda

```bash
for i in $(seq 1 20); do
  curl -s -o /dev/null -w "%{time_total}\n" "${SERVICE_URL}/predict/2026-27/6"
done | sort -n | awk '{a[NR]=$1} END {printf "p50 %.3fs   p95 %.3fs   max %.3fs (arranque en frio)\n", a[int(NR*0.5)], a[int(NR*0.95)], a[NR]}'
```

Referencia medida en local: **p50 ≈ 47 ms**. Antes de servir desde Gold eran ~25 s por
request, porque cada uno reconstruía las 279 features desde Silver.

### Los logs, por campo

En Cloud Run el servicio emite **una línea de JSON por evento** —se detecta solo, con la
variable `K_SERVICE`— y por eso Cloud Logging los deja consultar por campo en vez de por
texto:

```bash
gcloud run services logs read "${SERVICE}" --region "${REGION}" --limit 20

# Todas las predicciones servidas, como tabla
gcloud logging read 'jsonPayload.evento="prediccion"' --limit 15 \
  --format='table(jsonPayload.gameweek, jsonPayload.estado, jsonPayload.origen,
                  jsonPayload.latencia_ms, jsonPayload.model_version)'

# Sólo las lentas
gcloud logging read 'jsonPayload.evento="prediccion" AND jsonPayload.latencia_ms>500' --limit 10

# Quién pidió actualizar el dato
gcloud logging read 'jsonPayload.evento="pipeline_disparo"' --limit 10 \
  --format='table(jsonPayload.tarea, jsonPayload.estado, jsonPayload.destino)'
```

Así se ve una línea:

```json
{"severity":"INFO","message":"2026-27 GW5 [proxima/en_vivo] 10 partidos, 47.1 ms",
 "evento":"prediccion","season":"2026-27","gameweek":5,"estado":"proxima",
 "origen":"en_vivo","n_partidos":10,"latencia_ms":47.1,
 "model_version":"20260825T024144Z","feature_set_version":"v2.3189c9d4.279"}
```

Se guarda **la decisión y la métrica**: qué fecha, en qué estado, con qué modelo y cuánto
tardó. Ni un dato de quien consulta — acá es fácil, porque son equipos de fútbol y no
clientes, y justamente por eso conviene decir dónde se decide: `common/logging_setup.py`.
El día que el caso tenga PII, la regla ya está escrita en el lugar correcto.

En local no cambia nada: sin `K_SERVICE` los logs salen como texto legible. Se fuerza con
`TP_LOG_FORMAT=json`.

### Romper y volver

```bash
gcloud run revisions list --service "${SERVICE}" --region "${REGION}"

# Romper a propósito: se apunta a un bucket que no existe.
gcloud run services update "${SERVICE}" --region "${REGION}" \
  --set-env-vars "TP_STORAGE_BACKEND=gcs,TP_GCS_BUCKET=no-existe-este-bucket"

curl -s "${SERVICE_URL}/health" | jq '{status, detail}'     # degraded
curl -s -o /dev/null -w "%{http_code}\n" "${SERVICE_URL}/predict/2026-27/6"   # 503

# Rollback: el tráfico vuelve a la última revisión buena, en segundos y sin rebuildear.
export BUENA="$(gcloud run revisions list --service "${SERVICE}" --region "${REGION}" \
  --format='value(name)' --sort-by=~metadata.creationTimestamp | sed -n 2p)"
gcloud run services update-traffic "${SERVICE}" --region "${REGION}" --to-revisions "${BUENA}=100"

curl -s "${SERVICE_URL}/health" | jq '.status'              # ok
```

Que `/health` diga `degraded` en vez de tirar un 500 es lo que hace que esto se pueda
demostrar: el servicio está arriba y **avisa** que no puede servir.

### El modelo se degrada aunque el servicio esté sano

```bash
python -m monitoring.temporada_actual
```

Accuracy, log-loss y RPS por fecha contra los baselines —siempre local, prior de clase y
cuotas de cierre— sobre las mismas filas. Comparar contra un baseline de otro período
confunde *"el modelo empeoró"* con *"la liga se puso impredecible"*.

---

## 9 · Revisar todo desde la consola

Todo lo de arriba se puede ver y tocar desde [console.cloud.google.com](https://console.cloud.google.com).
Conviene conocer el camino de memoria: en la defensa es mucho más convincente mostrar el
recurso vivo en la consola que leer un `curl`.

**Antes de nada, arriba a la izquierda: el selector de proyecto.** Casi todo lo que
"no aparece" es que estás parado en otro proyecto.

### Los seis lugares que importan

| Recurso | Dónde | Qué tiene que decir |
|---|---|---|
| **El dato** | Cloud Storage → Buckets → `TU-PROYECTO-premier-ml` | carpetas `gold/`, `predicciones/`, `models/`, y `silver/` y `bronze/` si subiste todo |
| **El servicio** | Cloud Run → Servicios → `premier-ml-api` | verde, con URL pública y **Última implementación** reciente |
| **El Job** | Cloud Run → **Trabajos** → `premier-ml-pipeline` | existe y, después de la demo, con una ejecución **Correcta** |
| **La imagen** | Artifact Registry → `mlops-2026` | un solo tag `latest`, unos cientos de MB |
| **Las identidades** | IAM y administración → Cuentas de servicio | `premier-api` y `premier-job` |
| **Los builds** | Cloud Build → Historial | el build de la imagen, en verde |

### El servicio, pestaña por pestaña

Entrando a **Cloud Run → `premier-ml-api`**, cada solapa contesta una pregunta distinta de
la clase 7:

| Solapa | Qué mirar |
|---|---|
| **MÉTRICAS** | Latencia de solicitudes (p50/p95/p99), solicitudes por segundo, **porcentaje de errores** e instancias de contenedor. Es el tablero que la clase 7 llama *signos vitales*, y viene gratis |
| **REGISTROS** | Los logs en vivo. Al ser JSON, cada línea se **despliega** y muestra `evento`, `gameweek`, `latencia_ms`, `model_version` como campos, no como texto |
| **REVISIONES** | Una por despliegue, con el reparto de tráfico. Es acá donde se hace el **rollback**: elegís la revisión buena y le mandás el 100 % |
| **YAML** | La definición completa: imagen, variables de entorno, memoria, la cuenta de servicio. Sirve para mostrar que `TP_STORAGE_BACKEND=gcs` está puesto |
| **SEGURIDAD** | La cuenta de servicio con la que corre, y si permite invocaciones no autenticadas |

### Rollback desde la consola (sin tocar la terminal)

1. **Cloud Run → `premier-ml-api` → REVISIONES**
2. Botón **⋮** en la revisión buena → **Administrar tráfico**
3. Ponele **100 %** y guardá

Tarda segundos y no reconstruye nada. Es exactamente el mismo mecanismo del `gcloud run
services update-traffic`, y en pantalla se entiende mejor.

### Los logs por campo, en el Explorador

**Registro → Explorador de registros**, y en la caja de consulta:

```
resource.type="cloud_run_revision"
jsonPayload.evento="prediccion"
```

Cada resultado se despliega y muestra los campos sueltos. Dos consultas que valen para la
defensa:

```
jsonPayload.evento="prediccion" AND jsonPayload.latencia_ms > 500
jsonPayload.evento="pipeline_disparo"
```

La segunda muestra quién pidió actualizar el dato y cuándo. Que eso se pueda preguntar
—en vez de leer scrolleando— es el punto de haber emitido los logs en JSON.

### El Job y sus ejecuciones

**Cloud Run → Trabajos → `premier-ml-pipeline` → EJECUCIONES**. Cada corrida tiene su
estado, su duración y sus logs. Es lo que hay que mostrar mientras el botón de la página
está trabajando: la ejecución aparece ahí sola, disparada desde la web.

### Permisos, para la pregunta incómoda

**Cloud Storage → tu bucket → PERMISOS**. Se ven los dos bindings:

- `premier-api@…` → **Visualizador de objetos de Storage**
- `premier-job@…` → **Administrador de objetos de Storage**

Si preguntan por qué dos cuentas y no una, la respuesta está en esa pantalla: un bug en el
camino de lectura **no puede** corromper Gold, en vez de simplemente no deber hacerlo.

### Lo que cuesta

**Facturación → Informes**, filtrando por proyecto y agrupando por SKU. Con el escala a
cero, Cloud Run tiende a cero y lo poco que aparece es almacenamiento. El control que no
falla: si al día siguiente marca cero, quedó limpio.

### Checklist de dos minutos antes de la defensa

- [ ] Cloud Run → `premier-ml-api` en verde, y la URL abre la página
- [ ] `/health` dice `ok` y `proxima_predecible: 5`
- [ ] El calendario muestra las 38 fechas, con la 6 en adelante en gris
- [ ] Cloud Run → Trabajos → `premier-ml-pipeline` existe
- [ ] El panel *Estado del dato* dice que conviene actualizar
- [ ] Explorador de registros con la consulta de `prediccion` ya cargada
- [ ] Facturación abierta en otra pestaña, por si preguntan

---

## 10 · Costos y apagado

**Lo que prendés, cuesta.** Por primera vez queda algo corriendo 24/7.

```bash
gcloud run services list --region "${REGION}"
gcloud run jobs list --region "${REGION}"
gcloud storage ls
```

Si lo dejás vivo para la defensa, el escala a cero hace que tienda a costar nada, pero
**"tiende a cero" no es "cero"**: chequeá que sirva la revisión buena con un `/health` en
`ok`.

Para borrar todo:

```bash
gcloud run services delete "${SERVICE}" --region "${REGION}" --quiet
gcloud run jobs delete "${JOB}" --region "${REGION}" --quiet
gcloud artifacts repositories delete "${REPO}" --location "${REGION}" --quiet
gcloud storage rm -r "gs://${BUCKET}"

gcloud run services list --region "${REGION}"        # vacío
gcloud ai endpoints list --region "${REGION}"        # vacío
```

> **El error caro clásico es un endpoint de Vertex AI**: factura por hora de máquina
> desplegada, la use alguien o no. Este despliegue no crea ninguno.

El control final, el que no falla: **consola → Facturación → Informes**, filtrando por el
proyecto. Si al día siguiente marca cero, quedó limpio.

---

## 11 · Cuando algo falla

| Síntoma | Causa casi siempre | Qué hacer |
|---|---|---|
| `/health` → `degraded`, detail menciona Gold | el bucket no tiene `gold/` o la SA no lee | `gcloud storage ls gs://$BUCKET/gold/` y revisar el binding de IAM |
| `/health` → `degraded`, detail menciona `.ubj` | se subió la metadata pero no los binarios | `gcloud storage ls gs://$BUCKET/models/<modelo>/<version>/` |
| Todo `/predict` da 503 | no hay `PRODUCTION.json` o apunta a una versión sin boosters | `python -m training.registry --listar` |
| `/predict` da 409 en la fecha que querés | esa fecha todavía no tiene fila en Gold | el cuerpo trae `proxima_predecible`; para adelantar, correr el Job |
| `/predict` da 500 con "historia de hace más de N días" | Gold se construyó con un Silver viejo | correr el pipeline completo, no sólo `features.gold_tp` |
| `ValueError: no cuelga de ninguna raiz` | una ruta fuera de `data_root`/`models_root` | revisar `TP_DATA_ROOT` y `TP_MODELS_ROOT` |
| `Bucket names must be globally unique` | el nombre ya existe en otro proyecto | cambiar `BUCKET` |
| El build sube cientos de megas | falta o está mal el `.gcloudignore` | confirmar que excluye `data/` y `models/` |

---

## Resumen: el despliegue entero

```bash
gcloud services enable storage.googleapis.com artifactregistry.googleapis.com \
                       cloudbuild.googleapis.com run.googleapis.com logging.googleapis.com
gcloud storage buckets create "gs://${BUCKET}" --location="${REGION}"
python -m pipeline.pre_deadline && python -m training.run
python -m training.registry --promover <VERSION> --motivo "produccion"
gcloud storage rsync -r data/gold "gs://${BUCKET}/gold"
gcloud storage rsync -r data/predicciones "gs://${BUCKET}/predicciones"
gcloud storage rsync -r models "gs://${BUCKET}/models"
gcloud builds submit --tag "${IMAGE}" .
gcloud run deploy "${SERVICE}" --image "${IMAGE}" --region "${REGION}" \
  --allow-unauthenticated --service-account "${SA_API}" \
  --set-env-vars "TP_STORAGE_BACKEND=gcs,TP_GCS_BUCKET=${BUCKET}" --memory 1Gi
```
