# Observabilidad — logs del servidor y del cliente

La tarea de la clase 7 pide tres evidencias del servicio operando: **líneas de log del
servicio respondiendo**, **una métrica mirada** (la latencia sirve) y **un rollback probado**.
Este documento cubre las dos primeras, medidas desde los dos lados. El rollback está en
[`ROLLBACK.md`](ROLLBACK.md).

---

## Dos lados, tres capas

Un request atraviesa tres capas, y cada una ve una latencia distinta:

```
 cliente ──red/TLS──► borde de Cloud Run ──cola / cold start──► FastAPI ──► handler /predict
 smoke_load.py         run.googleapis.com/requests                          jsonPayload.latencia_ms
 (lo mide afuera)      (lo escribe Cloud Run)                               (lo escribe la app)
```

| Capa | Quién la mide | Dónde queda | Qué incluye |
|---|---|---|---|
| **cliente** | `scripts/smoke_load.py` | la terminal + `monitoring/output/smoke/<corrida>.json` | todo: red, TLS, cold start, app |
| **borde** | Cloud Run | log `run.googleapis.com/requests`, campo `httpRequest.latency` | desde que el request entra a Google hasta que sale |
| **app** | `serving/observability.py` | log `run.googleapis.com/stdout`, campo `jsonPayload.latencia_ms` | sólo nuestro handler |

Medir sólo desde el cliente mezcla la red de quien consulta con la salud del servicio. El
lado del servidor es lo que el servicio **sabe de sí mismo**, y es lo único que existe
cuando el tráfico es de usuarios reales y no de un script propio.

---

## Qué deja cada request en Cloud Logging

Salida real del proyecto `tp-mlops-premier-2026` (23/09/2026), antes de los cambios de este
documento. Un `GET /predict/2026-27/37` que da 409:

```
22:34:36.485  run.googleapis.com/requests  (httpRequest: GET 409, latency 0.25s)   <- Cloud Run
22:34:36.492  run.googleapis.com/stdout    INFO: ... "GET /predict/2026-27/37" 409  <- uvicorn, texto
22:34:36.492  run.googleapis.com/stdout    evento=prediccion_error gameweek=37 409  <- la app, JSON
```

No era duplicación: eran **tres fuentes distintas**. La del medio (la línea de acceso de
uvicorn) sí sobraba, porque repite lo que Cloud Run ya escribe arriba con más datos. Y
nada unía las tres, salvo el timestamp.

**Lo que cambió:**

| Cambio | Dónde | Efecto |
|---|---|---|
| `--no-access-log` en el `CMD` | `Dockerfile` | desaparece la línea de texto de uvicorn: quedan **dos** entradas por request |
| `logging.googleapis.com/trace` en cada línea JSON | `common/logging_setup.py` + middleware en `serving/main.py` | Cloud Logging **anida** el evento de la app bajo la línea del request. En el Explorador, desplegás el request y adentro está su `prediccion` |
| `corrida` en cada línea JSON | ídem, desde la cabecera `X-Corrida` | se piden al servidor exactamente los requests de una corrida de `smoke_load` |
| `smoke_load.py` sin URL fija | `scripts/smoke_load.py` | antes caía en silencio en el servicio de **otro proyecto**, y por eso "no aparecían" los logs. Ahora usa `--url`, `SERVICE_URL`, o le pregunta a `gcloud`, y si no, corta con error |
| `logs_servidor.py` | `scripts/` | la mitad del servidor: percentiles del borde y de la app, errores, revisión y modelo |
| Métricas basadas en logs | `gcp/metricas/*.yaml` | `latencia_ms` y los errores como métricas de Monitoring, graficables y alertables |

Una línea más, que no es un request: `"Prediciendo 2026-27 GW6 - 10 partidos…"` sale de
`serving/predict.py` en cada predicción en vivo. Es JSON pero sin `evento`, por eso en una
tabla por `jsonPayload.evento` aparece vacía. Ahora lleva `trace` y `corrida` como el resto.

---

## El circuito, para la defensa

En Cloud Shell, con el repo actualizado y el servicio redesplegado con estos cambios:

```bash
export SERVICE_URL="$(gcloud run services describe premier-ml-api --region us-central1 --format='value(status.url)')"

# 1. Cliente: carga marcada con una corrida
python scripts/smoke_load.py --n 30
#   ... p50/p95/p99 del CLIENTE
#   python scripts/logs_servidor.py --corrida smoke-20260928T213000Z   <- lo imprime al final

# 2. Servidor: los mismos 30 requests, leídos de Cloud Logging (esperar ~30 s)
python scripts/logs_servidor.py --corrida smoke-20260928T213000Z
```

Salida esperada de la segunda:

```
Latencia (ms)            n       p50       p95       max
  cliente (smoke_load)       30     280.1     410.3    3900.2
  borde   (Cloud Run)        30     250.4     380.0    3850.7
  app     (latencia_ms)      30      45.2      60.1      80.3
    └ proxima/en_vivo        30      45.2      60.1      80.3

  En la mediana: red +29.7 ms · Cloud Run + FastAPI +205.2 ms · app 45.2 ms

Status en el borde : {200: 30}
Revisión que atendió: {'premier-ml-api-00007-xyz': 30}
Modelo servido      : {'20260825T024144Z': 30}
```

*(los números son ilustrativos: la forma es la real, los valores dependen de la corrida)*

Cómo se lee: el `max` del cliente y del borde es el **arranque en frío**; en la app no
aparece, porque la app todavía no existía cuando el request esperaba. La diferencia
borde − app es lo que Cloud Run y FastAPI agregan; la diferencia cliente − borde es la red
entre Cloud Shell y Google.

Sin `--corrida` resume la última hora de todo el servicio, que es lo que sirve cuando el
tráfico no es propio:

```bash
python scripts/logs_servidor.py --desde 6h --errores
```

---

## En la consola

**Registro → Explorador de registros.** Tres cosas dejan la pantalla vacía aunque los logs
estén: el **selector de proyecto** de arriba (tiene que ser `tp-mlops-premier-2026`), el
**rango de tiempo** (por defecto, última hora) y estar en **Log Analytics** en vez del
Explorador (Log Analytics es SQL y sale vacío si el bucket no está actualizado para eso).

```
resource.type="cloud_run_revision"
resource.labels.service_name="premier-ml-api"
jsonPayload.evento="prediccion"
```

Consultas que valen para la defensa:

```
jsonPayload.evento="prediccion_error" AND jsonPayload.status>=500     # los nuestros
jsonPayload.evento="prediccion" AND jsonPayload.latencia_ms>200         # los lentos
jsonPayload.corrida="smoke-20260928T213000Z"                            # una corrida
jsonPayload.evento="pipeline_disparo"                                   # quién pidió actualizar
```

Con el `trace`, cada línea `run.googleapis.com/requests` muestra a la derecha el ícono de
"registros relacionados": desplegándolo aparece el evento de la app de ese mismo request.

---

## La métrica: latencia del servidor en Monitoring

Una vez por proyecto (no es retroactiva: cuenta desde que se crea):

```bash
gcloud logging metrics create premier_prediccion_latencia \
  --config-from-file=gcp/metricas/prediccion_latencia.yaml
gcloud logging metrics create premier_prediccion_errores \
  --config-from-file=gcp/metricas/prediccion_errores.yaml
gcloud logging metrics list --format='table(name, description)'
```

**Monitoring → Explorador de métricas** → métrica
`logging.googleapis.com/user/premier_prediccion_latencia` → agregación **95th percentile**,
agrupado por `estado`. Es la latencia de la app a lo largo del tiempo, separada entre la
fecha que corre el modelo (`proxima`) y las que salen del registro (`jugada`).

Las de la plataforma, sin crear nada: **Cloud Run → premier-ml-api → MÉTRICAS** (latencia
p50/p95/p99 del borde, requests por segundo, % de errores, instancias). Es el "tablero de
signos vitales" de la clase 7.

Opcional, si se quiere mostrar alertas: una política sobre `premier_prediccion_errores`
con `status` 5xx > 0 en 5 minutos (Monitoring → Alertas → Crear política). No la dejamos
creada: con el tráfico del TP dispararía por nuestros propios simulacros.

---

## Qué NO va al log

Sólo agregados: cuántos partidos, cuántos de cada clase, la confianza media, la latencia,
la versión del modelo. Nunca las probabilidades por partido ni las features. La corrida es
un id que genera el cliente y se recorta a 64 caracteres. Lo vigila
`test_el_log_no_filtra_datos_por_partido` (`tests/test_serving_api.py`); lo nuevo, en
`tests/test_observabilidad.py`.
