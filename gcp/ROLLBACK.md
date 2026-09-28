# Estrategia de rollback

> "Cada despliegue es una apuesta; el rollback es la red." — Clase 7

En este proyecto **no reentrenamos durante la temporada**, así que el rollback no es sobre
todo un tema de "el modelo nuevo salió peor". Hay cuatro cosas que pueden salir mal y un
momento del año —el **cambio de temporada**— en que cambian todas juntas.

---

## 1. Qué se puede volver atrás: tres artefactos, tres mecanismos

El diseño separa tres ciclos de vida (ver [`ARQUITECTURA-DEPLOY.md`](../ARQUITECTURA-DEPLOY.md)),
y cada uno tiene su propio mecanismo de vuelta atrás:

| Artefacto | Dónde vive | Cómo se vuelve atrás | Tiempo |
|---|---|---|---|
| **Código + configuración** | revisión de Cloud Run (imagen por digest + variables) | `update-traffic` a la revisión anterior | segundos |
| **Modelo** | `models/xgb_gbt/<versión>/` en el bucket | fijado por revisión (`TP_MODEL_VERSION`) → viaja con el `update-traffic`; o `training.registry --promover` | segundos / 1 min |
| **Dato** (Gold) | `gold/` en el bucket, con histórico en `_versiones/` | `python -m common.versiones --restaurar` | segundos + TTL de 5 min |

**Lo que no se vuelve atrás, a propósito:** las **predicciones registradas**. Cada fecha se
predice antes del deadline y queda congelada en `predicciones/`. Es un registro de lo que
el modelo dijo *antes* de que se jugara, que es lo que hace honesta la medición. Si un
modelo resultó malo, su predicción de la GW7 sigue siendo lo que se predijo para la GW7.

### El hueco que había, y cómo quedó cerrado

Una revisión de Cloud Run congela la imagen (por digest, aunque el tag sea `:latest`) y
sus variables de entorno. **Pero el modelo y la temporada no eran parte de la revisión:**

- el modelo salía de `PRODUCTION.json` en el bucket, un puntero **compartido** por todas las
  revisiones;
- la temporada salía de `config.yaml`, horneado en la imagen.

Entonces el rollback de la clase 7 (`update-traffic` a la revisión anterior) devolvía el
código viejo **con el modelo nuevo**. Ahora hay dos variables que una revisión puede fijar:

| Variable | Qué fija | Sin ella |
|---|---|---|
| `TP_MODEL_VERSION` | la versión del modelo que carga esa revisión | `PRODUCTION.json` del bucket |
| `TP_SEASON` | la temporada "actual" de esa revisión | `seasons.current` de `config.yaml` |

Con las dos fijadas, **la revisión es la unidad de despliegue y de rollback**: código,
modelo y temporada van juntos, y `update-traffic` los devuelve juntos.

> Ojo con `PRODUCTION.json` sin pin: el servicio cachea el modelo en memoria
> (`serving/main.py`, `Estado.modelo`). Promover otra versión **no cambia lo que sirve una
> instancia que ya está viva**; recién lo toma la próxima instancia que arranque. Por eso
> el camino recomendado es el pin, que hace el cambio explícito y visible en la revisión.

---

## 2. Escenarios

### A. Un deploy de código sale roto

**Síntoma:** `/health` en `degraded`, `/predict` en 503, `prediccion_error` con status 5xx en
los logs, el % de errores sube en Cloud Run → Métricas.

**Acción:** rollback por tráfico. Nada se reconstruye.

```bash
bash scripts/rollback.sh                 # ver revisiones: modelo, temporada, % de tráfico
bash scripts/rollback.sh --anterior      # 100 % a la última lista anterior, y verifica /health
```

**Verificación:** el script corre `/health` al final; y del lado del servidor,
`python scripts/logs_servidor.py --desde 10m` muestra qué revisión atendió cada request.

**Prevención (despliegue sin tráfico, "blue/green"):** la revisión nueva nace sin tráfico y
con una URL propia para probarla antes de exponerla:

```bash
gcloud run deploy premier-ml-api --image "${IMAGE}" --region us-central1 \
  --no-traffic --tag candidata \
  --update-env-vars "TP_MODEL_VERSION=20260825T024144Z,TP_SEASON=2026-27"
# URL de la candidata: https://candidata---premier-ml-api-<hash>-uc.a.run.app
curl -s "https://candidata---<...>/health"
python scripts/smoke_load.py --url "https://candidata---<...>" --n 10

gcloud run services update-traffic premier-ml-api --region us-central1 --to-tags candidata=10   # canary
gcloud run services update-traffic premier-ml-api --region us-central1 --to-latest              # todo
```

### B. El modelo nuevo resulta peor

Hoy no aplica (no reentrenamos en temporada). Aplica al **cambio de temporada** (escenario E).

**Síntoma:** no es técnico, el servicio responde perfecto. Se ve en
`python -m monitoring.temporada_actual`: accuracy por debajo de "siempre local" o de las
cuotas, varias fechas seguidas. Con n=10 por fecha, una sola fecha mala **no** es señal
(el IC de la accuracy de una fecha va de 0,10 a 0,70).

**Acción:**

```bash
# Con pin (recomendado): la revisión anterior tiene el modelo anterior.
bash scripts/rollback.sh --anterior

# Sin pin: mover el puntero y forzar que las instancias relean.
python -m training.registry --promover 20260825T024144Z --motivo "rollback: <por qué>"
gcloud run services update premier-ml-api --region us-central1 \
  --update-env-vars "TP_RESET_AT=$(date +%s)"
```

`--promover` deja el motivo escrito en `PRODUCTION.json`: el rollback queda documentado.

### C. El Job escribió un Gold malo

**Síntoma:** `/health` muestra `gold_filas` que cae o `proxima_predecible` que salta; el
control anti-leakage del pipeline falla; `/predict` da 500 con `AssertionError`.

**Acción:** volver Gold a la versión anterior. Antes de cada escritura, el pipeline archiva
la vigente en `gs://BUCKET/_versiones/gold/gold_tp_match/`.

```bash
export TP_STORAGE_BACKEND=gcs TP_GCS_BUCKET=tp-mlops-premier-2026-bucket
python -m common.versiones                                   # qué hay, con fecha y etiqueta
python -m common.versiones --diff gold_tp_match              # qué cambió
python -m common.versiones --restaurar gold_tp_match <stamp> # la anterior vuelve a ser la vigente
```

Restaurar también archiva lo que reemplaza: se puede ir y volver. El servicio relee Gold
cada 5 minutos, o de inmediato con el `TP_RESET_AT` de arriba.

### D. Una fuente externa se cae (FPL, football-data, Opta)

**Síntoma:** la ejecución del Job falla (`gcloud run jobs executions list`), el evento
`pipeline_fin` con `estado=error`, `/actualizar` sigue diciendo `hace_falta: true`.

**Acción: ninguna de rollback.** Es el caso que el diseño resuelve solo. Según la tabla de
`pipeline/pre_deadline.py`: si falla FPL, sigue con el snapshot previo; las cuotas de
football-data son opcionales; si falla Opta o Silver, el Job **corta antes de Gold**; y si
el control anti-leakage de Gold falla, no escribe nada. En todos los casos el servicio
sigue sirviendo el último Gold bueno. Se reintenta el Job cuando la fuente vuelva.

**El riesgo real** es el deadline: si la fuente no vuelve antes del primer partido, esa
fecha queda sin predicción pre-deadline. El pipeline se niega a registrar una predicción
después del corte (`pipeline/pre_deadline.py`, `_guard_pre_deadline`), así que no hay forma
de "rellenarla" después. Se asume y se documenta.

### E. Cambio de temporada (2026-27 → 2027-28)

Es el despliegue más riesgoso del año porque cambian **las tres cosas a la vez**:

- **temporada**: `current` pasa a 2027-28;
- **dato**: tres ascendidos sin historia en la Premier (cold start), los ids de FPL se
  reinician, Gold suma una temporada;
- **modelo**: es el momento natural de reentrenar, con 2026-27 incluida.

Procedimiento, con la revisión vieja como red en cada paso:

```bash
# 1. Cerrar la temporada: congelar con nombre lo que sirvió todo el año.
python -m common.versiones --snapshot "cierre 2026-27"

# 2. Preparar el dato de la temporada nueva con el Job, fijando la temporada.
gcloud run jobs update premier-ml-pipeline --region us-central1 --update-env-vars TP_SEASON=2027-28
gcloud run jobs execute premier-ml-pipeline --region us-central1 --wait

# 3. (Si se reentrena) el modelo nuevo se guarda pero NO se promueve todavía.
#    Se evalúa contra el anterior con el mismo protocolo del TP (training/README.md).

# 4. Revisión nueva SIN tráfico, con temporada y modelo fijados.
gcloud run deploy premier-ml-api --image "${IMAGE}" --region us-central1 \
  --no-traffic --tag t2027 \
  --update-env-vars "TP_SEASON=2027-28,TP_MODEL_VERSION=<versión elegida>"
curl -s "https://t2027---<...>/health"      # season_actual 2027-28, proxima_predecible 1

# 5. Pasar el tráfico.
gcloud run services update-traffic premier-ml-api --region us-central1 --to-tags t2027=100
```

**El rollback de la temporada** es un `update-traffic` a la revisión anterior, que sigue
fijada en `TP_SEASON=2026-27` con su modelo. Gold contiene todas las temporadas, así que la
revisión vieja sigue encontrando sus filas. Para el Job, que no tiene tráfico ni
revisiones, el rollback es volver la variable: `--update-env-vars TP_SEASON=2026-27`.

---

## 3. Lo que decidimos NO hacer, y por qué

| No hacemos | Por qué |
|---|---|
| Rollback automático (por error rate o latencia) | Con el tráfico del TP, una alerta dispararía por nuestros propios simulacros. El rollback manual tarda segundos y es observable |
| Canary automatizado | El tráfico es bajísimo: un 10 % de canary son uno o dos requests. `--no-traffic --tag` + prueba manual da más señal |
| Vertex AI Model Monitoring | El modelo se sirve en Cloud Run, no como endpoint de Vertex. El drift se mira fuera de línea, con `monitoring.temporada_actual` |
| Tags de imagen por commit | `:latest` alcanza para el rollback por tráfico, porque la revisión guarda el digest. **Es la mejora siguiente** si hubiera que *reconstruir* una versión vieja: `IMAGE=...:$(git rev-parse --short HEAD)` |

---

## 4. Simulacro: la evidencia que pide la clase 7

Romper a propósito **usando el pin del modelo**. Es el mismo mecanismo que protege el
rollback, así que el simulacro prueba las dos cosas a la vez:

```bash
bash scripts/rollback.sh                                  # anotar la revisión buena

# Romper: la revisión nueva pide un modelo que no existe.
gcloud run services update premier-ml-api --region us-central1 \
  --update-env-vars TP_MODEL_VERSION=no-existe
curl -s "${SERVICE_URL}/health"                           # "degraded", detail nombra "no-existe"
python scripts/smoke_load.py --n 10                       # 503
python scripts/logs_servidor.py --desde 10m --errores     # 503 del lado del servidor, qué revisión

# Volver.
bash scripts/rollback.sh --anterior                       # verifica /health = ok
python scripts/smoke_load.py --n 10                       # 200
python scripts/logs_servidor.py --desde 10m               # la revisión buena atendiendo
```

La salida de `logs_servidor.py` antes y después, con la columna "Revisión que atendió", es
la evidencia del rollback del lado del servidor. La de `smoke_load.py`, la del cliente.
