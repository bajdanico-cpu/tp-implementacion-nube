# Machete de la demo

Todo es para **Cloud Shell** (bash), parado en `~/tp-premier-ml`. Copiar y pegar por bloque.
Lo que va en la **consola web** está marcado como 🌐.

---

## 0. Al abrir Cloud Shell (siempre, primero)

```bash
cd ~/tp-premier-ml && git pull
export PROJECT_ID=$(gcloud config get-value project) REGION=us-central1 \
       SERVICE=premier-ml-api JOB=premier-ml-pipeline REPO=mlops-2026 \
       TP_STORAGE_BACKEND=gcs TP_GCS_BUCKET=tp-mlops-premier-2026-bucket
export SERVICE_URL=$(gcloud run services describe $SERVICE --region $REGION --format='value(status.url)')
export TOKEN=$(gcloud run services describe $SERVICE --region $REGION --format=json \
  | python3 -c "import json,sys; e=json.load(sys.stdin)['spec']['template']['spec']['containers'][0].get('env',[]); print({x['name']:x.get('value') for x in e}.get('TP_ADMIN_TOKEN',''))")
echo "$PROJECT_ID  $SERVICE_URL  token=${TOKEN:0:6}…"
```

Si `TOKEN` sale vacío, el botón está apagado (ver §9).

---

## 1. Chequeo previo (5 min antes)

```bash
bash scripts/rollback.sh                          # 100 % en la revisión DE ARRIBA, modelo 20260825T024144Z
curl -s $SERVICE_URL/health | python3 -m json.tool   # status ok · proxima_predecible 5 · gold_filas 1570
curl -s $SERVICE_URL/actualizar | python3 -m json.tool  # hace_falta true
for i in 1 2 3; do curl -s -o /dev/null -w "%{http_code} %{time_total}s\n" $SERVICE_URL/health; done   # precalentar
curl -s -o /dev/null -w "%{http_code} %{time_total}s\n" $SERVICE_URL/predict/2026-27/5
```

---

## 2. Mostrar la API

🌐 **La página:** `$SERVICE_URL/` → calendario de 38 fechas, GW5 predicha, GW6+ en gris.

🌐 **El contrato:** `$SERVICE_URL/docs` → cada endpoint con su esquema, "Try it out" para
probarlo. `$SERVICE_URL/openapi.json` es el contrato en crudo.

> En `/docs`, el **POST /actualizar da 403**: el token va en un header que Swagger no muestra.
> Es el control funcionando ("sin token no se gasta cómputo"). El POST de verdad se hace
> desde la página (botón) o con `curl` (§3).

Por terminal, los mismos endpoints:

```bash
curl -s $SERVICE_URL/health | python3 -m json.tool
curl -s $SERVICE_URL/calendario/2026-27 | python3 -m json.tool | head -40
curl -s $SERVICE_URL/predict/2026-27/2 | python3 -m json.tool | head -40    # jugada: congelada + resultado real
curl -s $SERVICE_URL/predict/2026-27/5 | python3 -m json.tool | head -40    # próxima: en vivo
curl -s $SERVICE_URL/predict/2026-27/20 | python3 -m json.tool              # 409: no preparada, dice cuál sí
curl -s $SERVICE_URL/actualizar | python3 -m json.tool                      # ¿hace falta actualizar?

# Un partido, legible:
curl -s $SERVICE_URL/predict/2026-27/5 | python3 -c "
import json,sys
for p in json.load(sys.stdin)['predictions']:
    print(f\"{p['home_short']}-{p['away_short']}  L {p['p_home']:.2f}  E {p['p_draw']:.2f}  V {p['p_away']:.2f}  -> {p['prediccion']}\")"
```

---

## 3. Actualizar el dato (el POST) y seguirlo

Desde la **página**: botón "Actualizar datos" (pide el token la primera vez: `echo $TOKEN`).
Desde la **terminal**:

```bash
# Primero el control: sin token -> 403
curl -s -X POST $SERVICE_URL/actualizar | python3 -m json.tool

# El de verdad -> 202 con el id de la tarea
export TAREA=$(curl -s -X POST -H "X-Admin-Token: $TOKEN" $SERVICE_URL/actualizar \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('id') or d)")
echo $TAREA

# Seguirla cada 15 s hasta que termine (Ctrl+C para cortar)
while true; do curl -s $SERVICE_URL/actualizar/$TAREA | python3 -c "
import json,sys; t=json.load(sys.stdin); print(t.get('estado'), '-', str(t.get('detalle',''))[:100])"; sleep 15; done
```

🌐 Mientras corre: **Cloud Run → Trabajos → premier-ml-pipeline → Ejecuciones** → la corrida →
**Registros**. Se ven los pasos `--- bronze_fpl ---` … `--- predecir ---`.

**Un 409 en el POST** = ya hay una corrida en curso (control de concurrencia): se sigue esa.

### El resumen de la ejecución, en un comando

```bash
python scripts/resumen_actualizacion.py | tee evidencia_actualizacion.txt
```

Junta: el pedido (`pipeline_disparo`) y su cierre (`pipeline_fin`), la ejecución del Job,
**los pasos con sus segundos** (del diario en el bucket), los WARNING/ERROR de esa ejecución,
y cómo quedó el servicio (`/health`: próxima fecha, filas de Gold).

---

## 4. Logs

```bash
# Predicciones servidas
gcloud logging read 'resource.type="cloud_run_revision" AND resource.labels.service_name="premier-ml-api" AND jsonPayload.evento="prediccion"' \
  --freshness=3h --limit=10 --format='table(timestamp, jsonPayload.gameweek, jsonPayload.estado, jsonPayload.latencia_ms, jsonPayload.model_version)'

# Errores (4xx de quien pide, 5xx nuestros)
gcloud logging read 'resource.type="cloud_run_revision" AND resource.labels.service_name="premier-ml-api" AND jsonPayload.evento="prediccion_error"' \
  --freshness=3h --limit=10 --format='table(timestamp, jsonPayload.gameweek, jsonPayload.status, jsonPayload.error_type)'

# Quién pidió actualizar y cómo terminó
gcloud logging read 'resource.type="cloud_run_revision" AND resource.labels.service_name="premier-ml-api" AND (jsonPayload.evento="pipeline_disparo" OR jsonPayload.evento="pipeline_fin")' \
  --freshness=6h --format='table(timestamp, jsonPayload.evento, jsonPayload.tarea, jsonPayload.estado, jsonPayload.segundos)'

# El Job, paso por paso (la última hora)
gcloud logging read 'resource.type="cloud_run_job" AND resource.labels.job_name="premier-ml-pipeline"' \
  --freshness=1h --limit=100 --order=asc --format='table(timestamp, severity, jsonPayload.message)'

# Ejecuciones del Job
gcloud run jobs executions list --job $JOB --region $REGION

# Los 403/409 del POST (quedan en el log de requests de Cloud Run)
gcloud logging read 'resource.type="cloud_run_revision" AND httpRequest.requestMethod="POST"' \
  --freshness=3h --limit=10 --format='table(timestamp, httpRequest.requestUrl, httpRequest.status)'
```

🌐 **Explorador de registros** (proyecto `tp-mlops-premier-2026`, rango "Últimas 3 horas",
NO "Log Analytics"). Pegar en la caja de consulta:

```
resource.type="cloud_run_revision"
resource.labels.service_name="premier-ml-api"
jsonPayload.evento="prediccion"
```

Variantes: `jsonPayload.evento="pipeline_disparo"` · `jsonPayload.status>=500` ·
`jsonPayload.latencia_ms>100` · `jsonPayload.corrida="smoke-…"` · `resource.type="cloud_run_job"`.

---

## 5. Latencia: cliente contra servidor

```bash
python scripts/smoke_load.py --n 20                      # cliente: p50/p95/max. Copiar el --corrida del final
python scripts/logs_servidor.py --corrida smoke-XXXXXXXX # (esperar ~30 s) cliente / borde / app
python scripts/logs_servidor.py --desde 1h               # todo el servicio, sin filtrar corrida
```

Referencia del 28/09: **cliente 82 ms · borde 32 ms · app 24 ms** → más de la mitad es red.

🌐 **Monitoring → Explorador de métricas** → `logging.googleapis.com/user/premier_prediccion_latencia`
→ 95th percentile, agrupar por `estado`. · 🌐 **Cloud Run → premier-ml-api → MÉTRICAS**.

---

## 6. Simulacro de rollback

```bash
bash scripts/rollback.sh        # PRECONDICIÓN: 100 % en la revisión de ARRIBA. Si no -> §6 "limpiar" primero

# Romper
gcloud run services update $SERVICE --region $REGION --update-env-vars TP_MODEL_VERSION=no-existe
curl -s $SERVICE_URL/health | python3 -m json.tool          # degraded, detail dice por qué
curl -s -o /dev/null -w "%{http_code}\n" $SERVICE_URL/predict/2026-27/5   # 503 (la próxima)
curl -s -o /dev/null -w "%{http_code}\n" $SERVICE_URL/predict/2026-27/2   # 200 (jugada: sale del registro)
python scripts/logs_servidor.py --desde 10m --errores       # el 503 visto desde el servidor

# Volver (segundos, sin rebuild)
bash scripts/rollback.sh --anterior

# Limpiar (OBLIGATORIO después): plantilla sana + tráfico siguiendo a la última
gcloud run services update $SERVICE --region $REGION --update-env-vars TP_MODEL_VERSION=20260825T024144Z
gcloud run services update-traffic $SERVICE --region $REGION --to-latest
bash scripts/rollback.sh
```

---

## 7. Volver Gold a la GW5 (para repetir el botón)

```bash
python -m common.versiones --diff gold_tp_match                       # buscar el stamp
python -m common.versiones --restaurar gold_tp_match 20260921T184956Z # la de 1570 filas
curl -s $SERVICE_URL/health | python3 -m json.tool                    # en ≤5 min: proxima 5, 1570
```

---

## 8. Variables de entorno de la revisión

```bash
# Ver todas las de la revisión que está sirviendo
gcloud run services describe $SERVICE --region $REGION --format=json \
  | python3 -c "import json,sys; [print(f\"{x['name']}={x.get('value','')[:40]}\") for x in json.load(sys.stdin)['spec']['template']['spec']['containers'][0].get('env',[])]"

# Cambiar UNA sin tocar el resto (crea revisión nueva)
gcloud run services update $SERVICE --region $REGION --update-env-vars CLAVE=valor

# Sacar una
gcloud run services update $SERVICE --region $REGION --remove-env-vars CLAVE
```

⚠️ **Nunca `--set-env-vars`**: reemplaza TODAS y borra `TP_STORAGE_BACKEND`, `TP_GCS_BUCKET`,
`TP_ADMIN_TOKEN`… y el servicio deja de leer el bucket.

| Variable | Qué hace |
|---|---|
| `TP_STORAGE_BACKEND=gcs` · `TP_GCS_BUCKET` | lee dato y modelo del bucket |
| `TP_MODEL_VERSION` | fija el modelo de la revisión (rollback lo arrastra) |
| `TP_SEASON` | fija la temporada de la revisión |
| `TP_ADMIN_TOKEN` | llave del POST /actualizar; sin ella el endpoint está apagado |
| `TP_GCP_PROJECT` · `TP_REGION` · `TP_JOB_NAME` | qué Job dispara la API |
| `TP_RESET_AT` | cualquier valor nuevo = revisión nueva = relee Gold ya |

---

## 9. Si algo falla

| Síntoma | Comando |
|---|---|
| Primer request lento | es el cold start: nombrarlo. `for i in 1 2 3; do curl -s -o /dev/null $SERVICE_URL/health; done` |
| `/health` degraded | leer `detail` en voz alta · `bash scripts/rollback.sh --anterior` |
| Todo roto, no hay tiempo | **Plan B**: `gcloud run services update-traffic $SERVICE --region $REGION --to-revisions premier-ml-api-00005-fr9=100` |
| El simulacro no rompe nada | el tráfico está fijado: §6 "limpiar" y repetir |
| POST 403 desde la página | token mal guardado: F12 → Application → Local Storage → borrar `tp_admin_token` y reintentar con `echo $TOKEN` |
| `TOKEN` vacío (botón apagado) | `gcloud run services update $SERVICE --region $REGION --update-env-vars TP_ADMIN_TOKEN=$(openssl rand -hex 16)` y repetir §0 |
| POST 409 | ya hay una corrida: seguirla (§3) |
| El contador sigue aunque el Job terminó | el servicio no puede consultar la operación (403, visto el 29/09). Confirmar: el resumen dice `ok=1` pero sin `FIN`. El dato ya está: F5. Arreglo: `roles/run.viewer` a la cuenta (§10), o la imagen con el plan B del diario |
| El Job falla | `python scripts/resumen_actualizacion.py` dice en qué paso. El Gold anterior queda intacto |
| El Job no termina a tiempo | video de respaldo, y seguir con §5 |
| La web no cambia tras el Job | Ctrl+F5; `/health` dice la verdad |
| `gcloud logging read` vacío | proyecto (`gcloud config get-value project`), `--freshness`, y que el tráfico haya ido a ESTE servicio |
| Se perdieron las variables | repetir §0 |

---

## 10. Permisos de la cuenta del servicio

```bash
# Qué roles tiene la cuenta con la que corren el Service y el Job
gcloud projects get-iam-policy tp-mlops-premier-2026 --flatten="bindings[].members" \
  --filter="bindings.members:443531272820-compute@developer.gserviceaccount.com" --format="value(bindings.role)"

# Que pueda consultar cómo va el Job que lanza (sin esto, el contador del botón no termina)
gcloud projects add-iam-policy-binding tp-mlops-premier-2026 \
  --member="serviceAccount:443531272820-compute@developer.gserviceaccount.com" --role="roles/run.viewer"
```

**Plan C** (volver a lo de antes de todo esto): `git checkout demo-estable`, rebuild de
`:latest` y redeploy con `--remove-env-vars TP_MODEL_VERSION,TP_SEASON` (`GUIA-DEMO.md` §5).
