# TP — Implementación en la nube de modelos de ML

**Documento de contexto para retomar el proyecto desde Claude Code.**
Fecha de redacción: 17 de agosto de 2026.

---

## 1. Qué hay que entregar

Trabajo práctico grupal de la materia de implementación en la nube de modelos de ML. El requisito es **entrenar un modelo y ponerlo en producción**.

**Dato clave para priorizar:** el peso de la nota está en el **pipeline / MLOps en la nube**, no en la performance predictiva del modelo. Toda decisión de diseño se resuelve a favor de lo que simplifique el pipeline y haga más demostrable el ciclo de vida del modelo.

## 2. Caso elegido

Predecir el resultado de los partidos de la fecha de la **Premier League**.

**Target recomendado: 1X2 multiclase** (gana local / empate / gana visitante).

Razón: un Poisson bivariado (predecir goles de cada equipo y derivar probabilidades) es el enfoque más correcto en la literatura de fútbol, pero exige dos modelos, una capa de post-procesamiento y métricas que hay que explicar. Con 1X2 se usa log-loss + accuracy y se tienen dos baselines listos:

- **Baseline trivial:** "siempre gana el local" → ~45% de accuracy.
- **Baseline duro:** las cuotas de casas de apuestas → ~53-55% de accuracy.

Si el modelo queda en 50% no es un problema para la nota; lo que importa es tener el benchmark explícito y una lectura honesta del resultado.

> Estado: el target **todavía no está confirmado por el grupo**. Si se cambia a Poisson, revisar el impacto en la capa Gold y en las métricas de monitoreo.

## 3. El argumento fuerte del TP

Este dominio tiene una propiedad que la mayoría de los TPs de MLOps no tiene: **ground truth automático a las dos horas**.

Se predice el sábado a la mañana, el partido termina al mediodía y ya se sabe si se acertó — sin etiquetado manual, sin esperar semanas. Eso habilita a montar y **demostrar en vivo el ciclo completo**:

```
predicción → registro → llegada del resultado real → cálculo de métricas
   → detección de degradación → retraining automático → nueva versión del modelo
```

Este ciclo cerrado debe ser el eje de la presentación y de la arquitectura. Es lo que diferencia el trabajo de un "entrené un modelo y lo puse en un endpoint".

## 4. Viabilidad de los datos — verificada

### 4.1 API oficial de Fantasy Premier League

Pública, **sin autenticación ni API key**. Activa en la temporada 2026/27.

| Endpoint | Contenido |
|---|---|
| `https://fantasy.premierleague.com/api/bootstrap-static/` | IDs de jugadores y equipos, `strength` de cada equipo, deadlines de cada gameweek. En `elements`: puntos, precio, `status` (disponibilidad/lesión) y stats de partido por jugador. |
| `https://fantasy.premierleague.com/api/fixtures/` | Todos los fixtures de la temporada. |
| `https://fantasy.premierleague.com/api/fixtures/?event={GW}` | Fixtures de una fecha puntual. |
| `https://fantasy.premierleague.com/api/fixtures/?future=1` | Solo fixtures futuros. |
| `https://fantasy.premierleague.com/api/event/{GW}/live/` | Resultados y stats en vivo de esa fecha, para todos los jugadores. |
| `https://fantasy.premierleague.com/api/element-summary/{player_id}/` | Historial gameweek a gameweek de un jugador + temporadas anteriores. |

Notas de implementación:

- Los objetos de fixture traen `team_h_score` / `team_a_score` → **el target sale directo de ahí**.
- También traen `team_h_difficulty` / `team_a_difficulty` (el FDR que calcula FPL) → feature gratis.
- En `players_raw.csv` / `elements`, el campo `element_type` codifica la posición: **1 = GK, 2 = DEF, 3 = MID, 4 = FWD**.
- **La API tiene política de CORS**: no se puede llamar desde un frontend. Hay que consumirla desde el servidor (Cloud Run / Cloud Function).

### 4.2 Histórico

**`vaastav/Fantasy-Premier-League`** (GitHub) es el estándar de facto. Archivo histórico armado desde la API oficial y complementado con xG/xA de Understat. Cobertura **desde 2016-17**, con carpetas ya creadas hasta **2026-27**.

Estructura relevante:

```
data/{season}/gws/merged_gw.csv        ← stats fecha a fecha de todos los jugadores (EL archivo clave)
data/{season}/gws/gw{N}.csv            ← una fecha puntual
data/{season}/cleaned_players.csv      ← resumen de temporada
data/{season}/players/{name}/gws.csv   ← por jugador
data/{season}/understat/               ← xG/xA
data/cleaned_merged_seasons.csv        ← todas las temporadas concatenadas
data/master_team_list.csv              ← mapeo de IDs de equipo entre temporadas (¡necesario!)
```

Se lee directo con pandas:

```python
import pandas as pd
url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2025-26/gws/merged_gw.csv"
df = pd.read_csv(url)
```

Alternativa/complemento: **`olbauday/FPL-Core-Insights`**, que continúa la misma idea con dataset 2026/27 en `data/2026-2027/`, ratings Elo de equipos y cobertura de copas.

### 4.3 Fuentes adicionales

El grupo no está limitado a FPL. Recomendadas:

- **football-data.co.uk** — resultados históricos + cuotas de cierre desde los años 90, CSV gratis. Doble uso: **feature muy potente** y **benchmark** contra el cual medirse.
- **Understat** — xG / xA. Mucho más predictivo que los goles crudos en ventanas cortas de partidos. Ya viene mergeado en el repo de vaastav.

### 4.4 Volumen

380 partidos por temporada × ~10 temporadas ≈ **3.800 filas**.

Alcanza para un modelo razonable (regresión logística multinomial, gradient boosting), pero no para redes profundas. **Validación con split temporal, nunca aleatorio.**

## 5. Riesgo crítico: leakage temporal

Es el punto donde este proyecto se puede arruinar sin que se note.

**El problema:** los puntos FPL de una fecha se conocen *después* de que se jugó el partido. Si se arman features del tipo "puntos de los defensores en la fecha N" para predecir la fecha N, el modelo está viendo el resultado. La accuracy se dispara y el trabajo no vale nada.

**La regla de oro:**

> El snapshot de features usado para predecir la fecha N tiene que ser reproducible **usando exclusivamente datos anteriores al deadline de la fecha N**.

**Mitigaciones a implementar:**

1. Todas las features agregadas deben ser **ventanas rolling sobre las N fechas anteriores** (rolling de 3 y de 5, por ejemplo), nunca de la fecha a predecir.
2. En Bronze, **nunca sobrescribir**: el snapshot pre-deadline es distinto del post-partido, y conservar ambos es la defensa auditable contra el leakage.
3. **Escribir un test automatizado** que verifique que ninguna feature tiene timestamp posterior al deadline de la fecha objetivo. Un `assert` explícito en el pipeline. Esto solo justifica media presentación.
4. **Cuidado específico con la columna `xP`** del dataset de vaastav: se scrapea del campo `ep_this` de la API *después* de que termina la fecha. La documentación del propio repo advierte que puede reflejar información post-partido en vez de la predicción que los managers veían antes del deadline. Recomiendan aplicarle `shift(1)` agrupando por jugador, o directamente descartarla.

**Antecedente relevante:** ya apareció un problema del mismo tipo (leakage por autocorrelación espacial) en el trabajo de clasificación de litologías a partir de registros de pozo. Acá es el mismo tipo de error, pero sobre el eje temporal.

## 6. Arquitectura propuesta (GCP)

Recicla el patrón medallion ya implementado en el TP de Herramientas y Plataformas para la Gestión de Datos.

### Bronze — GCS
JSON crudo de `/bootstrap-static/`, `/fixtures/` y `/event/{GW}/live/`.
Particionado por `season/gw/fecha_ingesta`. Append-only, nunca sobrescribir.

### Silver — BigQuery
Normalizado, una tabla por entidad:
- `players`
- `teams`
- `fixtures`
- `player_gw_stats`

### Gold — BigQuery (feature table)
**Una fila por fixture.** Features construidas solo con ventanas rolling de fechas anteriores, agregadas por posición:

- Puntos / xG promedio de los delanteros del local en las últimas 3 y 5 fechas
- Goles concedidos y puntos de la defensa del visitante en la misma ventana
- Ídem para arqueros (saves, clean sheets) y mediocampistas
- `strength` de cada equipo y FDR del fixture (directo de la API)
- Forma reciente del equipo (puntos obtenidos en las últimas N fechas)
- Cantidad de jugadores clave con `status` distinto de disponible (proxy de lesiones)
- Local/visitante
- Cuotas de cierre (si se incorpora football-data.co.uk)

### Orquestación — Cloud Scheduler → Cloud Run Jobs
Dos triggers semanales:
1. **Post-fecha:** ingestar resultados, actualizar Silver/Gold, calcular métricas de la predicción anterior, evaluar retraining.
2. **Pre-deadline:** generar features y predecir la fecha siguiente.

> Composer es overkill para esto y factura 24/7. Cloud Run Jobs + Scheduler alcanza y es más defendible en costo.

### Serving — Cloud Run + FastAPI
Modelo versionado en GCS o en Vertex AI Model Registry.

### Monitoreo
- Log de cada predicción con su versión de modelo y su snapshot de features.
- Join automático contra el resultado real al cerrar la fecha.
- Métricas rodantes: log-loss y accuracy sobre las últimas K fechas.
- **Drift:** es fácil de demostrar acá — los planteles cambian entre temporadas, hay ascensos y descensos, y las distribuciones de features se mueven de forma visible.

## 7. Próximos pasos

- [ ] Confirmar el target con el grupo (1X2 vs Poisson)
- [ ] Descargar el histórico de vaastav y hacer EDA rápido para dimensionar cobertura y nulos
- [ ] Definir el esquema exacto de la tabla Gold (una fila por fixture)
- [ ] Implementar el ingestor Bronze (script Python + contenedor para Cloud Run Job)
- [ ] Implementar el test anti-leakage antes de entrenar nada
- [ ] Entrenar baseline (logística multinomial) y comparar contra los dos baselines de referencia
- [ ] Montar el ciclo de monitoreo + retraining
- [ ] Preparar la demo del ciclo cerrado para la defensa

## 8. Estructura de repo sugerida

```
tp-premier-ml/
├── ingestion/
│   ├── fpl_client.py          # cliente de la API, con retry y rate limiting
│   ├── bronze_job.py          # dump crudo a GCS
│   └── Dockerfile
├── transform/
│   ├── silver.py              # normalización a BigQuery
│   ├── gold_features.py       # feature engineering con ventanas rolling
│   └── sql/
├── training/
│   ├── train.py
│   ├── evaluate.py
│   └── baselines.py           # "siempre local" + cuotas
├── serving/
│   ├── app.py                 # FastAPI
│   └── Dockerfile
├── monitoring/
│   ├── collect_outcomes.py    # join predicción ↔ resultado real
│   └── drift.py
├── tests/
│   └── test_no_leakage.py     # ← el test que justifica media presentación
├── infra/                     # terraform o gcloud scripts
└── README.md
```
