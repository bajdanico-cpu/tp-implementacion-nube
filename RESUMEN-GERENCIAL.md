# TP Premier ML — resumen para el grupo

Para repasar juntos antes de presentar. Qué construimos, cómo está armado, dónde está
cada concepto de la materia, y qué contestar. El detalle técnico vive en:
[`ARQUITECTURA-DEPLOY.md`](ARQUITECTURA-DEPLOY.md) · [`GUIA-DEMO.md`](GUIA-DEMO.md) ·
[`gcp/OBSERVABILIDAD.md`](gcp/OBSERVABILIDAD.md) · [`gcp/ROLLBACK.md`](gcp/ROLLBACK.md).

---

## 1. El producto en tres líneas

- **Qué hace:** predice el resultado (local / empate / visitante) de cada partido de la
  Premier League, **antes** de que se juegue, fecha por fecha.
- **Cómo se usa:** una página web pública y una API en Cloud Run. Un botón actualiza el
  dato cuando termina una fecha, y la siguiente aparece predicha sola.
- **Qué vale:** el ciclo completo de MLOps en la nube —dato, modelo, API, despliegue y
  operación— funcionando con datos reales de la temporada 2026-27.

## 2. El resultado, dicho con honestidad

| | accuracy | log-loss |
|---|---|---|
| "siempre gana el local" / prior de clase | 0,426 | 1,085 |
| **nuestro modelo** (holdout 2025-26, 380 partidos) | **0,495** | 1,030 |
| cuotas de cierre de las casas de apuestas | 0,495 | 1,012 |

- Le gana por **7 puntos** a los baselines triviales.
- **Empata con el mercado** en aciertos (188 de 380 los dos) y calibra un poco peor.
- El punto débil es el **empate**: predice 3 de 104. En la práctica es un clasificador de
  local contra visitante. Por eso corre en paralelo una regla candidata (umbral 0,30).
- La restricción que manda es el **dato**: 1.140 partidos de entrenamiento. Probamos cinco
  mejoras con criterio fijado de antemano y las cinco se rechazaron
  (`training/README.md`). Es un resultado, no una falta de trabajo.

> Cuidado en la defensa: el modelo de producción se reentrenó **incluyendo** el holdout, así
> que sus métricas propias son in-sample (llegan a dar un ROI de +90 %, que es falso). Los
> números reportables son los de arriba, los del modelo de evaluación.

---

## 3. Arquitectura

```
 FUENTES                    CLOUD RUN JOB  (batch, lo dispara el botón)
 FPL · vaastav ──────►  bronze ─► silver ─► gold (279 features) ─► predice la próxima fecha
 football-data            │         │         │                        │
 Opta (Premier)           ▼         ▼         ▼                        ▼
                     ┌──────────────── BUCKET GCS (el único estado) ────────────────┐
                     │ bronze/  silver/  gold/  predicciones/  models/  _versiones/  │
                     └───────────────────────────────▲───────────────────────────────┘
                                                     │ lee (Gold + modelo)
 USUARIO ── página / curl ──► CLOUD RUN SERVICE (API FastAPI + página web)
                               /health  /predict  /calendario  /actualizar
                                         │ stdout JSON
                                         ▼
                               CLOUD LOGGING ──► métricas basadas en logs ──► MONITORING
```

**Las tres ideas de diseño que conviene poder explicar:**

1. **Medallón (bronze → silver → gold).** Bronze guarda lo que dio la fuente, sin tocar y
   con fecha. Silver lo limpia y lo une. Gold tiene una fila por partido con sus 279
   features, calculadas solo con información anterior al partido (control anti-leakage).
2. **El servicio no calcula features: las busca.** El pipeline arma Gold y el servicio
   hace un *lookup*. Por eso responde en ~24 ms en lugar de 25 segundos, y usa exactamente
   el mismo código que el entrenamiento (sin *train/serve skew*).
3. **Tres ciclos de vida separados.** El **código** cambia cuando cambia la lógica (vive
   en la imagen). El **dato** cambia cada fecha (vive en el bucket). El **modelo** cambia
   cuando se reentrena (vive en el bucket, versionado). La imagen no lleva datos ni
   modelo: actualizar una fecha no obliga a reconstruir nada.

---

## 4. Los conceptos de la materia, en el TP

| Clase | Concepto | Dónde está en el TP |
|---|---|---|
| 1 — Pilares MLOps | ciclo dato → modelo → deploy → operación; reproducibilidad | el pipeline completo y versionado: Bronze append-only, Silver y Gold con histórico (`common/versiones.py`), modelos con `metadata.json` (git sha, librerías, hiperparámetros) |
| 2 — ML Canvas | problema, decisión, métricas, datos, riesgos | el canvas del TP (Word). La decisión es qué anunciar por partido; métrica primaria RPS/log-loss; baseline "siempre local" y cuotas |
| 3 — Del canvas al TFI | alcance, piezas entregables | modelo + API + despliegue + operación, más la demo en vivo |
| 4 — Vertex / nube | proyecto, bucket, dato en la nube, entrenamiento | bucket `tp-mlops-premier-2026-bucket`; notebook `01_gcp_cloudshell.ipynb`; AutoML de Vertex documentado como contrafáctico opcional (runbook) |
| 5 — APIs | contrato, validación, `/health`, códigos de error | FastAPI con esquemas Pydantic (`serving/schemas.py`); `/health` que informa en vez de tirar; 409 para "fecha no preparada", 503 si el modelo no cargó |
| 6 — Docker y Cloud Run | imagen, registry, build en la nube, servicio con URL | `Dockerfile` (sólo código), Artifact Registry `mlops-2026`, Cloud Build, Cloud Run Service y Job |
| 7 — Operación | latencia p95, errores 4xx/5xx, costo, logs vs métricas, sin datos personales, revisiones y rollback, drift | `smoke_load.py` + `logs_servidor.py` (cliente contra servidor), logs JSON, métricas basadas en logs, `rollback.sh`, `monitoring.temporada_actual` (degradación del modelo) |

---

## 5. Glosario: imagen, contenedor, servicio, revisión, job…

Es lo que más se confunde, con nuestros nombres reales:

| Concepto | Qué es | En el TP |
|---|---|---|
| **Dockerfile** | la receta, en git | `Dockerfile`: Python 3.14, dependencias, el código, `uvicorn` |
| **Imagen** | la receta ya cocinada: un paquete inmutable | `…/mlops-2026/premier-ml-api:latest` (la de ayer) y `:logs-rollback` (la de hoy) |
| **Tag** de imagen | un nombre que apunta a una imagen; se puede mover | `:latest` se pisa en cada build; por eso hoy construimos con `:logs-rollback` |
| **Digest** | la huella inmutable de una imagen (`sha256:…`) | lo que guarda cada revisión, aunque se haya desplegado con un tag |
| **Artifact Registry** | el "bucket de imágenes" | repositorio `mlops-2026` |
| **Contenedor** | una imagen corriendo | cada instancia que Cloud Run prende para atender requests |
| **Servicio** (Cloud Run) | la API siempre disponible, con URL | `premier-ml-api`. **Uno solo**, aunque tenga dos URLs (formato viejo `…-45hwwliisa-uc.a.run.app` y nuevo `…-443531272820.us-central1.run.app`: las dos son el mismo servicio) |
| **Revisión** | una foto inmutable de imagen + variables + configuración; cada deploy crea una | `premier-ml-api-00005-fr9` (la de hoy); `$BUENA` (la de ayer). Las viejas no se borran: son la red del rollback |
| **Tráfico** | qué porcentaje de requests va a cada revisión | hoy 100 % a la `00005`; el rollback es mover ese 100 % |
| **Variable de entorno** | configuración de la revisión, fuera de la imagen | `TP_STORAGE_BACKEND=gcs`, `TP_GCS_BUCKET`, `TP_ADMIN_TOKEN`, y ahora `TP_MODEL_VERSION` y `TP_SEASON` |
| **Job** (Cloud Run) | un proceso que corre, termina y se apaga; sin URL | `premier-ml-pipeline`: la misma imagen, con el comando `python -m pipeline.pre_deadline` |
| **Ejecución** | una corrida de un Job | cada vez que se aprieta "Actualizar datos" |
| **Instancia / cold start** | el contenedor prendido; el primero tarda más | el `max` del smoke; escala a cero si nadie consulta, así que casi no cuesta |

Una frase que lo resume: **la imagen es el qué, la revisión es el qué con su
configuración, y el tráfico decide cuál revisión atiende.**

---

## 6. Las APIs

| Endpoint | Qué hace |
|---|---|
| `GET /health` | el pulso: modelo cargado, Gold leído, qué fecha es la próxima. Nunca tira error: informa `ok` o `degraded` con el motivo |
| `GET /predict/{temporada}/{fecha}` | las probabilidades de cada partido. Si la fecha ya se jugó, devuelve la predicción **congelada** antes del partido, con el resultado real al lado. Si es la próxima, predice en vivo. Si todavía no está lista, 409 |
| `GET /calendario/{temporada}` | las 38 fechas y el estado de cada una (lo que pinta la página) |
| `GET /actualizar` | **¿hace falta actualizar?** Mira Gold y contesta si ya hay resultados nuevos para incorporar, y si se puede disparar |
| `POST /actualizar` | **dispara el pipeline.** No lo ejecuta: le pide a Cloud Run que corra el Job y responde 202 al instante. Exige el header `X-Admin-Token`; sin token configurado, el endpoint queda apagado |
| `GET /actualizar/{id}` | en qué anda esa corrida. Cuando termina bien, el servicio relee Gold en el acto |

**Por qué el servicio no corre el pipeline él mismo:** tarda minutos y chocaría con el
timeout de un request. Además obligaría a darle permiso de escritura a la parte pública.
El Job tiene su propio timeout, sus recursos y su registro de ejecuciones.

**El flujo del botón:**

```
página ─POST /actualizar─► servicio ─(Cloud Run Admin API)─► Job: bronze → silver → gold → predecir
   ▲                                                                    │
   └──── GET /actualizar/{id} cada 4 s ◄── "ok" ◄───────────────────────┘ escribe en el bucket
         al terminar: la página se recarga, la fecha jugada muestra el resultado y aparece la siguiente
```

---

## 7. Observabilidad: medido desde afuera **y** desde adentro

Medición real del 28/09 (20 requests a la GW6):

| Capa | p50 | Quién lo mide |
|---|---|---|
| cliente | 82 ms | `smoke_load.py`, desde Cloud Shell |
| borde de Cloud Run | 32 ms | Cloud Run, en su log de requests |
| **nuestra app** | **24 ms** | el evento `prediccion` de la API |

**El mensaje:** mirando solo el cliente diríamos "la API tarda 82 ms". Tarda 24; más de
la mitad es red. Lo que el servicio sabe de sí mismo es lo único que existe cuando el
tráfico es de usuarios reales.

**Qué agregamos para poder mostrarlo:**

| Pieza | Para qué |
|---|---|
| Logs JSON, una línea por evento | Cloud Logging los consulta por campo (`jsonPayload.latencia_ms > 200`) |
| `trace` en cada línea | la consola agrupa el evento de la app bajo la línea del request |
| `corrida` en cada línea | se piden al servidor exactamente los requests de una prueba |
| `--no-access-log` | una línea menos por request: la de uvicorn repetía lo que ya escribe Cloud Run |
| `smoke_load.py` sin URL fija | antes le pegaba en silencio al servicio de otro proyecto: por eso "no aparecían" los logs |
| `logs_servidor.py` | la tabla cliente / borde / app, más errores, revisión y modelo |
| 2 métricas basadas en logs | latencia y errores en Monitoring, graficables y con alertas posibles |

**Qué no va al log:** ningún dato por partido ni ninguna feature. Solo agregados:
cuántos partidos, cuántos de cada clase, la confianza media, la latencia y el modelo. Hay
un test que lo vigila.

---

## 8. Rollback: qué se vuelve atrás y cómo

### Tres artefactos, tres mecanismos

| Si falla… | Se vuelve atrás con | Tarda |
|---|---|---|
| el **código** (un deploy roto) | `bash scripts/rollback.sh --anterior` → 100 % del tráfico a la revisión anterior | segundos |
| el **modelo** | la revisión anterior tiene su modelo fijado (`TP_MODEL_VERSION`), así que viaja con el mismo `update-traffic` | segundos |
| el **dato** (Gold malo) | `python -m common.versiones --restaurar gold_tp_match <versión>` | segundos + 5 min de relectura |
| una **fuente externa** caída | nada: el Job corta antes de escribir Gold, y el servicio sigue con el último bueno | — |

Las **predicciones registradas no se revierten nunca**: son lo que el modelo dijo antes
de cada partido, y son la base de la evaluación honesta.

### El hueco que encontramos y cerramos

Antes, el modelo salía de `PRODUCTION.json` en el bucket y la temporada de `config.yaml`,
**compartidos por todas las revisiones**. El rollback de la clase 7 devolvía el código
viejo **con el modelo nuevo**. Ahora cada revisión puede fijar los dos (`TP_MODEL_VERSION`,
`TP_SEASON`), y **la revisión pasa a ser la unidad de despliegue y de rollback.**

### Qué hace cada archivo

| Archivo | Qué hace | Qué rompe / cómo se arregla |
|---|---|---|
| `scripts/preparar_demo.sh` | provisiona todo en un comando: APIs, bucket, dos identidades (la API solo lee, el Job escribe), sube el dato congelado, build, despliega Job y Service, verifica. `--reset` vuelve el dato al estado inicial de la demo | no rompe nada: es idempotente. **Diferencias con nuestro deploy real:** el bucket real es `tp-mlops-premier-2026-bucket` y no se crearon las dos identidades (corre la cuenta por defecto). Para mañana **no lo usamos**: el reset se hace restaurando la versión de Gold (`GUIA-DEMO.md` §4) |
| `scripts/rollback.sh` | lista revisiones con modelo, temporada y % de tráfico; `--anterior` o `--a <revisión>` mueven el 100 % y verifican `/health` | es el arreglo |
| `scripts/smoke_load.py` | carga desde el cliente, marcada con un id de corrida | — |
| `scripts/logs_servidor.py` | la misma corrida vista desde el servidor; después de un rollback muestra qué revisión atendió | es la evidencia |
| `common/versiones.py` | historia de Silver y Gold: listar, comparar, etiquetar, restaurar. Restaurar también archiva lo que reemplaza | el arreglo de un dato malo |
| `training/registry.py` | versiones de modelo; `--promover` escribe `PRODUCTION.json` **con el motivo** | el arreglo de un modelo malo, con registro de por qué |

### El simulacro (lo que se muestra)

1. **Romper:** `gcloud run services update … --update-env-vars TP_MODEL_VERSION=no-existe`.
   Se crea una revisión nueva que pide un modelo inexistente.
2. **Qué se rompe:** `/health` pasa a `degraded` y dice por qué. `/predict` de la próxima
   fecha da **503**. Las fechas ya jugadas **siguen respondiendo 200**, porque salen del
   registro congelado y no necesitan el modelo: la degradación es parcial, no total.
3. **Detectarlo desde los dos lados:** `smoke_load.py` ve los 503; `logs_servidor.py` ve
   los `prediccion_error` con status 503 y en qué revisión ocurrieron.
4. **Arreglarlo:** `bash scripts/rollback.sh --anterior`. En segundos, sin rebuild. El
   script verifica `/health = ok`, y `logs_servidor.py` muestra que atiende la revisión buena.

---

## 9. Cambio de temporada y reentrenamiento (teórico; no se despliega)

Al pasar de 2026-27 a 2027-28 cambian tres cosas a la vez: la **temporada**, el **dato**
(tres ascendidos sin historia, los ids de FPL se reinician) y, probablemente, el
**modelo**. Es el momento natural de reentrenar con 2026-27 incluida. El repo ya tiene la
regla escrita (`training/promotion.py`): **reentrenar es barato, promover exige evidencia.**

### Campeón y retador

```
1. Reentrenar          python -m training.run            → versión nueva en el registry, SIN promover
2. Evaluar offline     contra el campeón, en el holdout fijo (mismas filas, mismos baselines)
3. Correr en sombra    revisión nueva con --no-traffic --tag retador y TP_MODEL_VERSION=<nuevo>
                       el Job registra también sus predicciones: los dos predicen los MISMOS partidos
4. Decidir             McNemar pareado sobre 10 fechas (~100 partidos), alpha 0,05,
                       el retador gana en los pares donde discrepan, Y no empeora en el holdout
5. Promover            training.registry --promover <nuevo> --motivo "McNemar p=… en GW1-10"
                       + update-traffic --to-tags retador=100   (+ el Job con el mismo TP_MODEL_VERSION)
6. Confirmar           /health y logs_servidor.py: "Modelo servido" = el nuevo;
                       la métrica de latencia agrupada por model_version; monitoring.temporada_actual
```

**Por qué 10 fechas y no una:** con 10 partidos por fecha, el error estándar de la
accuracy es de ±15,7 puntos. Promover mirando una sola fecha es tirar una moneda:
elegirías al peor la mitad de las veces.

**Por qué McNemar:** los dos modelos predicen los mismos partidos, así que la comparación
es pareada y solo informan los partidos donde discrepan. Es mucho más potente que
comparar dos accuracies sueltas.

**Por qué el holdout como red:** un retador puede ganar en las últimas fechas por
sobreajustarse al período reciente. Si empeora en el holdout fijo, no se promueve.

### Rollback de la promoción

```bash
gcloud run services update-traffic premier-ml-api --region us-central1 --to-revisions <revisión del campeón>=100
python -m training.registry --promover <campeón> --motivo "rollback: <qué se vio>"
gcloud run jobs update premier-ml-pipeline --region us-central1 --update-env-vars TP_MODEL_VERSION=<campeón>
```

- **Primera línea:** la API vuelve en segundos, porque la revisión del campeón tiene su
  modelo fijado.
- **Segunda línea:** deja `PRODUCTION.json` coherente y con el motivo escrito.
- **Tercera línea:** que el Job registre las próximas fechas con el mismo modelo que sirve
  la API.
- Las predicciones que el retador ya registró **quedan**: son historia y sirven para
  medir cuánto costó.

**Las señales para revertir:** accuracy por debajo de "siempre local" o del mercado
durante varias fechas (no una); la distribución de lo que anuncia cambia bruscamente (el
campo `anunciadas` del log, por ejemplo "todo local"); errores 5xx en
`premier_prediccion_errores`.

---

## 10. Preguntas probables

- **¿Por qué no le gana al mercado?** Las casas agregan información que no tenemos
  (lesiones de último momento, alineaciones, dinero). Empatar en accuracy con 1.140
  partidos es razonable. La competencia internacional de 2023 la ganó el consenso de las
  casas, no un modelo.
- **¿Por qué el modelo no va dentro de la imagen, como en el ejemplo del profesor?** La
  clase 6 plantea las dos opciones. Elegimos bajarlo del bucket porque el dato cambia
  cada semana y no queríamos reconstruir la imagen por eso.
- **¿Cuánto cuesta?** Casi nada: Cloud Run escala a cero y cobra por request. El Job
  cobra solo mientras corre.
- **¿Qué pasa si el pipeline falla un sábado?** El servicio sigue con el último Gold
  bueno. El riesgo real es pasar el deadline sin predicción registrada, y eso se asume: el
  pipeline se niega a registrar después de que arrancó el partido.
- **¿Hay drift?** El EDA mide cuánto se mueven las distribuciones entre temporadas, y
  `monitoring.temporada_actual` mide la degradación en vivo, fecha a fecha, contra
  baselines calculados sobre las mismas filas. No usamos Vertex Model Monitoring porque
  el modelo no se sirve como endpoint de Vertex.
- **¿Es reproducible?** El dato está versionado (Bronze append-only, Silver y Gold
  archivados) y cada modelo guarda git sha, librerías e hiperparámetros. El modelo de
  producción se entrenó con el árbol de git sucio: lo congelan sus `.ubj` y su
  `metadata.json`, no el commit. Conviene decirlo antes de que lo pregunten.

---

## 11. Guion sugerido (≈ 15 min)

| Min | Qué | Quién | Apoyo |
|---|---|---|---|
| 0-2 | El problema y el resultado honesto | | §1-2 |
| 2-5 | Arquitectura: medallón, lookup en Gold, tres ciclos de vida | | §3 |
| 5-7 | En vivo: página, `/health`, `/predict` | | `GUIA-DEMO.md` §3a |
| 7-9 | Video: de la GW5 a la GW6 con el botón, y los logs del Job | | §4 de la guía |
| 9-11 | Cliente contra servidor: `smoke_load` → `logs_servidor`, la consola, la métrica | | §3b-3e |
| 11-13 | Rollback en vivo: romper y volver | | §3f |
| 13-15 | Cambio de temporada y promoción (teórico), cierre | | §9 |

Plan B si algo falla en vivo: `update-traffic` a `$BUENA` (`GUIA-DEMO.md` §5), y seguir
con el video.
