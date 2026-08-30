# Runbook GCP — TP Premier ML

Comandos parametrizados para correr el pipeline en Google Cloud. Está escrito para la
**terminal** de Cloud Shell, con `gcloud` autenticado.

> Para el lab dentro del **editor** de Cloud Shell usar
> [`notebooks/01_gcp_cloudshell.ipynb`](../notebooks/01_gcp_cloudshell.ipynb), que habla con
> la nube por las librerías Python de Google (`google-cloud-storage`). Motivo: el kernel del
> editor **no hereda el entorno de la sesión** y `gcloud` desde ahí queda sin proyecto ni
> auth. Los comandos de este runbook son para la terminal.

**Alcance.** Hasta la predicción de una fecha, corriendo como batch, con su registro en el
bucket. No se despliega nada: no hay servicio, ni imagen, ni job programado. Eso es la etapa
siguiente y todavía no está.

---

## Variables base

```bash
export PROJECT_ID="$(gcloud config get-value project)"
export REGION="us-central1"
export BUCKET="${PROJECT_ID}-premier-ml"
```

## API necesaria

Las APIs se habilitan **por proyecto**: un proyecto nuevo nace con casi todo apagado.
Habilitar es gratis; se paga el uso.

```bash
gcloud services list --enabled            # ver qué ya está prendido
gcloud services enable storage.googleapis.com
```

---

## El pipeline

El repo **no trae datos**: `data/` y los `.ubj` están en `.gitignore` a propósito. Todo se
regenera desde cuatro fuentes públicas, ninguna con credenciales. Estos comandos bajan el
dato y lo transforman desde cero.

```bash
pip install -q -r requirements-cloud.txt

python -m ingestion.run                # ~27 MB de Bronze, append-only
python -m ingestion.bronze_pulselive   # copas, Europa y stats de Opta
python -m transform.silver
python -m transform.competencias
python -m transform.opta_stats
python -m features.gold_tp             # el control anti-leakage corre acá adentro
```

> **`requirements-cloud.txt` y no `requirements.txt`.** El de local está pinneado a wheels
> `cp314` (Python 3.14.3) y Cloud Shell trae otro Python: pip intentaría compilar numpy y
> pandas desde el fuente y la sesión se cae. Ver el encabezado de ese archivo.

### El dato a la nube

```bash
gcloud storage buckets create "gs://${BUCKET}" --location="${REGION}" || true
gcloud storage rsync -r data/bronze "gs://${BUCKET}/bronze"
gcloud storage rsync -r data/silver "gs://${BUCKET}/silver"
gcloud storage rsync -r data/gold   "gs://${BUCKET}/gold"
```

### Entrenar y versionar

```bash
python -m training.run --sin-holdout   # el modelo que se REPORTA (no persiste)
python -m training.run                 # el que SIRVE: incluye 2025-26 y se guarda
gcloud storage rsync -r models "gs://${BUCKET}/models"
```

Acá entrena en **CPU**: Cloud Shell no tiene GPU y `device: auto` cae solo. Es lo esperado y
está medido — ver *Decisiones ya tomadas*.

### Predecir, y verificar que no miró el futuro

```bash
python -m serving.predict --gw 1                   # una fecha ya jugada
python -m serving.predict --gw 2                   # la que se está jugando
python -m serving.predict --gw 1 --evaluar         # contra el resultado real
gcloud storage rsync -r data/predicciones "gs://${BUCKET}/predicciones"
```

⚠️ **Correr esto durante una fecha es el momento en que el leakage temporal se cuela**, porque
la información llega de a pedazos: marcadores parciales en la API de FPL, jugadores con
minutos de partidos en curso, y la fecha todavía sin cerrar.

La defensa es una regla aplicada por el código, no un cuidado manual:

```
corte(partido) = min(kickoff_time) de todos los partidos de su (temporada, gameweek)
```

Ningún partido de la fecha N entra en las features de la fecha N, **ni siquiera los que ya
terminaron**. Cada predicción guarda `hist_kickoff_local` / `hist_kickoff_visita` —el kickoff
del último partido efectivamente usado— y `serving/predict.py` levanta un `AssertionError`
antes de escribir si alguno es posterior al corte.

Verificado el 30/08/2026, con la fecha 2 en juego (8 de 10 partidos con marcador):

| Fecha | Última historia usada | Corte | Margen |
|---|---|---|---|
| GW1 | 2026-05-24 15:00 (cierre de 2025-26) | 2026-08-21 19:00 | 89 días |
| GW2 | 2026-08-24 19:00 (FUL–CHE, último de la GW1) | 2026-08-28 19:00 | 4 días |

La fecha 1 entra como historia de la fecha 2 —que es lo correcto— y la fecha 2 no entra en
absoluto. **La prueba que convence:** si la GW1 filtrara su propio resultado acertaría 10 de
10; acierta 4.

---

## Limpieza — no es opcional

**Lo que se prende, cuesta.** El lab no despliega nada, así que el único recurso con costo es
el bucket. Igual se borra: el crédito es finito, el bucket factura almacenamiento mientras
exista, y **todo esto se regenera con los comandos de arriba en minutos**.

```bash
# 1) Borrar el bucket y todo su contenido.
gcloud storage rm -r "gs://${BUCKET}"

# 2) Confirmar que no queda NINGUN recurso con costo.
gcloud storage ls                                    # ningún bucket del lab
gcloud run services list --region "${REGION}"        # vacío
gcloud ai endpoints list --region "${REGION}"        # vacío

# 3) Apagar la API que se prendió para el lab.
gcloud services disable storage.googleapis.com --force
```

> **Un endpoint de Vertex AI es el error caro clásico:** factura por **hora de máquina
> desplegada**, la use alguien o no. Este lab no crea ninguno — pero si en algún momento
> probás uno, `undeploy` y `delete` antes de cerrar la sesión.

El control final, el que no falla: **consola → Facturación → Informes**, filtrando por el
proyecto. Si al día siguiente marca cero, quedó limpio.

---

## Decisiones ya tomadas, con número

Se miden acá y no se re-discuten cuando llegue el deploy:

1. **El Job de entrenamiento va sin GPU.** Medido con `training.benchmark_gpu` en una
   GTX 1650: a 1.140 filas la GPU **pierde 1,7×**; gana 1,5× a 11.400 y 5,4× a 114.000. En
   GCP una T4 cuesta ~2,5-3× el nodo pelado, así que se paga recién arriba de ~50.000 filas.
2. **El modelo se sube entrenado, no se reentrena en el nodo.** Misma semilla, mismas rondas,
   mismas 1.384 filas: entre GPU y CPU la diferencia máxima en probabilidad es **0,079** y
   **18 de 380 predicciones cambian de resultado**. El algoritmo `hist` de GPU no es
   bit-idéntico al de CPU. Para *inferencia* el `.ubj` es portable y hay un test que lo
   demuestra; para *reentrenar* hace falta el mismo device.
3. **La temporada en curso nunca entra al entrenamiento**, ni siquiera en el artefacto de
   producción. Sus partidos sirven de historia, no de objetivo: es la única evaluación limpia
   que le queda al proyecto.

---

## Lo que sigue *(todavía no está escrito)*

El servicio permanente: `serving/app.py` (FastAPI envolviendo `serving/predict.py`, que ya
existe y está probado), su `Dockerfile`, Artifact Registry, Cloud Run, y la orquestación con
Cloud Scheduler → Cloud Run Jobs.

Una advertencia para cuando llegue ese momento, porque el fallo aparece recién en runtime:
sin un `.gcloudignore`, `gcloud builds submit` usa `.gitignore` para armar el contexto de
build — y el de este repo excluye `models/**/*.ubj` y `*.parquet`. La imagen se hornearía
**sin modelo y sin Gold**, el servicio arrancaría igual, y el health check devolvería
`model_loaded: false`.
