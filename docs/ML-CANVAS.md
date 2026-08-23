# ML Canvas — Predicción 1X2 de la Premier League

Diseño del caso según el Canvas de Louis Dorard, con los diez bloques de la Clase 2.
Todos los números están **calculados sobre los datos ingestados**, no citados de memoria.

- **Fecha:** 18 de agosto de 2026
- **Estado de los datos:** Silver completo para 2022-23 → 2025-26 (1.520 partidos). La
  temporada 2026-27 arranca el **21/08/2026**; hay 380 fixtures con deadline y ningún
  partido jugado.
- **Materia:** CD.18 — Implementación de Aplicaciones de Aprendizaje Automático en la Nube (ITBA)

---

## B1 · Propuesta de valor

**Para** el manager de Fantasy Premier League, **que** tiene que decidir capitán,
transferencias y alineación antes de un deadline que vence **90 minutos antes** del
primer partido de la fecha, **nuestro servicio** es un **predictor 1X2 calibrado por
partido** **que** publica las probabilidades de los 10 partidos de la fecha *antes* del
deadline y las contrasta contra el resultado real dos horas después de que termine cada
partido.

**Objetivo de negocio.** Que la decisión del manager se apoye en una probabilidad
calibrada y auditable en vez de en la intuición o en la tabla de posiciones. La medida de
éxito no es la accuracy: es **cuántas fechas el modelo le gana a "siempre el local"**, y
qué tan cerca queda del mercado.

**Qué NO es.** No es un producto de apuestas. Las cuotas entran al proyecto como
*benchmark* y como *feature*, nunca como recomendación de apuesta.

**El argumento del proyecto.** Este dominio tiene una propiedad que casi ningún caso de
MLOps tiene: **el ground truth llega solo, dos horas después de la predicción.** Sin
etiquetado manual y sin esperar semanas. Eso permite demostrar el ciclo cerrado completo
en vivo, que es el eje de la defensa.

---

## B2 · Fuentes de datos

Tres fuentes complementarias, **no alternativas**. Cada una aporta algo que ninguna otra
tiene.

| Fuente | Grano | Cobertura | Qué aporta que nadie más aporta |
|---|---|---|---|
| **football-data.co.uk** | 1 fila = 1 partido | 2010-11 → hoy (ingestamos desde 2022-23) | **Cuotas de cierre** (el baseline duro) + tiros, córners, faltas, tarjetas |
| **vaastav/Fantasy-Premier-League** | 1 fila = jugador × fecha | 2016-17 → 2026-27 | **El histórico jugador-fecha con xG/xA.** La API oficial no lo sirve |
| **API oficial de FPL** | presente y futuro | temporada en curso | Fixtures, **deadlines**, FDR, disponibilidad y el resultado apenas termina el partido |

**Volumen y costo.** 27 MB en Bronze, 3,1 MB en Silver, para cuatro temporadas completas.
El costo de almacenamiento es despreciable; lo que crece es el histórico de snapshots
pre-deadline, ~1 MB por corrida.

**PII: ninguna.** No hay dato personal en el proyecto. Los "sujetos" son clubes y
futbolistas profesionales, y todas las estadísticas son públicas y publicadas. Esto
satisface directamente el requisito del TFI de no manejar PII real, sin necesidad de
anonimizar ni sintetizar nada.

**Licencias.** Las tres fuentes son de acceso público y gratuito. La API de FPL exige
`User-Agent` de browser y tiene CORS: se consume siempre desde el servidor.

### Frescura de cada fuente — medido el 18/08/2026

Ésta es la pregunta que decide qué fuente puede estar en el camino crítico de una
predicción en producción.

| Fuente | Cadencia real medida | ¿Sirve en producción? |
|---|---|---|
| **API de FPL** | Continua. Los fixtures y el `status` de cada jugador cambian a lo largo de la semana; el resultado aparece durante el partido | **Sí — es la fuente operativa** |
| **football-data · `fixtures.csv`** | `Last-Modified` de hoy 10:38 GMT. Trae cuotas **pre-partido** (B365, Max, Avg) | **Sí, pero sólo cubre los próximos ~2-3 días.** Hoy tiene 3 filas y ninguna de Premier |
| **football-data · archivo de temporada `E0.csv`** | Se republica durante la temporada (la página declara "Last updated: 17/08/26"). El de 2026-27 **todavía no existe**: el URL redirige (301) | Sí, para stats post-partido. **Cadencia intra-temporada a medir en la GW1** |
| **vaastav** | **Gap mediano de 10 días, máximo de 96.** En 2025-26 tocó `merged_gw.csv` 12 veces: semanal hasta la GW9, después saltó del 01/11 al 05/02 (gw29 recién el 13/03, gw38 el 17/06) | **No.** Sólo para el histórico de entrenamiento |

**Consecuencia de diseño — el pipeline se auto-alimenta el histórico.** vaastav sirve
para el arranque en frío (2022-23 → 2025-26) y nada más. Desde la GW1 de 2026-27, el
histórico jugador-fecha lo construimos nosotros: Bronze es append-only y cada corrida
post-fecha guarda el snapshot de `/event/{GW}/live/`. En producción **no dependemos de
que vaastav actualice.**

---

## B3 · Tarea de predicción

**Clasificación supervisada multiclase.** Tres clases: `home` / `draw` / `away`.

**Qué es un sample.** No es un partido: es **un partido visto desde antes del deadline de
su gameweek**. El mismo partido genera un sample distinto si se lo mira el martes o el
viernes, porque la disponibilidad de los jugadores cambia. El sample queda definido por
el par (`fixture_id`, `predicted_at`), y `predicted_at` tiene que ser anterior al
`deadline_time`.

**Balance de clases**, medido sobre las cuatro temporadas:

| | home | draw | away |
|---|---|---|---|
| Global (1.520 partidos) | **44,5 %** | **24,1 %** | **31,4 %** |
| 2022-23 | 48,4 % | 22,9 % | 28,7 % |
| 2023-24 | 46,1 % | 21,6 % | 32,4 % |
| 2024-25 | 40,8 % | 24,5 % | 34,7 % |
| 2025-26 | 42,6 % | 27,4 % | 30,0 % |

No es un desbalance severo, pero **el empate es la clase difícil**: casi nunca es la clase
más probable, ni para el modelo ni para el mercado.

**ML, no Deep Learning.** 1.140 partidos de entrenamiento. Con 30 features son ~18
observaciones por parámetro. Una red profunda no tiene con qué.

---

## B4 · Ingeniería de features

Todo sale de Silver (`fact_match`, `fact_fixture`, `fact_player_gw`, `dim_team`) y se
colapsa a **una fila por partido**.

| Grupo | Features | Fuente |
|---|---|---|
| Forma del equipo | puntos, goles a favor/en contra, tiros, tiros al arco, córners — media de los últimos **3 y 5 partidos** | `fact_match` |
| Forma local / visitante | lo mismo, restringido a partidos en esa condición | `fact_match` |
| xG | xG y xGC del equipo en los últimos 3 y 5 partidos | `fact_player_gw` agregado a equipo-fecha |
| Dificultad | `team_h_difficulty`, `team_a_difficulty` (el FDR de FPL) | `fact_fixture` |
| Disponibilidad | cantidad de jugadores con `status ≠ disponible` al momento del snapshot | API de FPL pre-deadline |
| Mercado | probabilidades implícitas de la cuota, sin el margen de la casa | `fact_match` / `fixtures.csv` |
| Descanso | días desde el partido anterior de cada equipo | `fact_match` |

### Las tres reglas que no se negocian

1. **Toda feature se calcula sobre partidos anteriores al deadline.** Nunca la fecha a
   predecir.
2. **Las ventanas rolling van sobre los últimos N partidos de cada equipo, no sobre las
   últimas N gameweeks.** En 2022-23 la GW7 no existe (se canceló entera por la muerte de
   Isabel II y sus partidos se reprogramaron), así que hay gameweeks con 7 partidos y
   otras con 16.
3. **`transform.leakage` corre antes de escribir Gold**, no sólo en los tests. El pipeline
   tiene que fallar en el momento de generar el dato malo.

Columnas prohibidas (`config.yaml` → `features.banned_columns`): `xP`, `team_h_score`,
`team_a_score`, `total_points`, `bps`, `bonus`.

### Dos variantes, siempre

Se entrena y se reporta **con cuotas** y **sin cuotas**. Las cuotas son a la vez la
feature más predictiva y el baseline: un modelo que las usa parece bueno sin haber
aprendido nada del juego. La comparación honesta contra el mercado es la variante sin
cuotas — que además es **el fallback operativo** si `fixtures.csv` no trae todavía la fila
del partido a predecir.

### Cold-start de ascendidos

Medido sobre el plantel de 2026-27: **Coventry City y Hull City no tienen ningún partido**
en la ventana ingestada; Ipswich y Sunderland tienen 38. No se dejan nulos: se define un
prior explícito para el ascendido (promedio histórico de los ascendidos en su primera
temporada) más una flag `es_ascendido`.

### Riesgo abierto: skew de escala en `strength_*`

Verificado hoy sobre `dim_team`. Los campos de fuerza del equipo **no están en la misma
escala en el histórico que en la API en vivo**:

| Temporada | `strength_overall_home` (media) | `strength_attack_home` (media) |
|---|---|---|
| 2022-23 → 2025-26 | ~1.130 – 1.148 | ~1.106 – 1.131 |
| **2026-27 (pretemporada)** | **2,85** | **0,00** — los 20 equipos en cero |

Además `strength` es nulo en los 20 equipos de 2026-27. Usar estas columnas como feature
hoy sería un train/serve skew silencioso: el modelo entrenaría con valores de cuatro
cifras y serviría con valores de un dígito. **Por eso el grupo "Fuerza" queda fuera del
feature set inicial** hasta re-verificar los valores una vez jugada la GW1.

---

## B5 · Evaluación offline

**Split temporal, nunca aleatorio.** Un split aleatorio pone partidos de mayo en el train
y de agosto en el test; el modelo ve el futuro y la métrica miente.

```
train : 2022-23, 2023-24, 2024-25   (1.140 partidos)
test  : 2025-26                     (  380 partidos)
```

**Métricas, en orden de importancia:**

1. **Log-loss** — la principal. Es probabilística y castiga la sobreconfianza.
2. **Calibración** (curva de confiabilidad) — un modelo que dice "60 %" tiene que acertar
   6 de cada 10. Para el uso real importa más que la accuracy.
3. **Accuracy** — para comparar contra los baselines, que es como se comunica.

**Las varas, calculadas sobre el holdout 2025-26:**

| Baseline | Accuracy | Log-loss |
|---|---|---|
| Siempre gana el local | 42,6 % | — |
| Prior de clase | 42,6 % | 1,085 |
| **Cuotas de cierre** | **49,5 %** | **1,012** |

La accuracy de las cuotas por temporada fue 55,5 / 60,0 / 55,5 / 49,5 %. La caída de
2025-26 no es un mercado peor informado: es la temporada con más empates (27,4 %), y el
empate casi nunca es el argmax del mercado. **El techo realista está en ~55 %, no en
49,5 %.**

**Criterio mínimo para salir a producción** (sobre la variante *sin cuotas*):

- Log-loss en el holdout **por debajo de 1,085** (el prior de clase). Si no le gana al
  prior, no aprendió nada.
- Curva de calibración sin desvío sistemático mayor a 10 puntos en los bins con masa.
- Accuracy reportada **con su intervalo**: con 380 partidos el error estándar ronda
  **±5 puntos**, así que diferencias chicas entre modelos no son distinguibles. Por eso se
  suma validación walk-forward por gameweek además del holdout.

**Costo de los errores: barato.** Es la pregunta de factibilidad de la clase. Equivocarse
en un partido le cuesta al manager unos puntos de FPL, no dinero ni salud. Eso autoriza a
salir con un modelo simple y sin human-in-the-loop obligatorio — y es la razón por la que
el proyecto es defendible como PoC.

---

## B6 · Toma de decisiones

**Qué se hace con la predicción.** Cada fecha, antes del deadline, el servicio publica las
tres probabilidades de los 10 partidos. El manager las usa para:

1. **Elegir capitán** — priorizar jugadores de equipos con `p(victoria)` alta.
2. **Elegir defensores y arquero** — priorizar equipos con probabilidad alta de no recibir
   gol, vía la probabilidad de victoria del rival.
3. **Decidir transferencias** — mirar la dificultad de las próximas fechas, no sólo la que
   viene.

**Cómo interactúa el usuario.** Endpoint HTTP (`GET /predict/{season}/{gameweek}`) y una
tabla ordenada por confianza. La predicción es una **sugerencia**, no una acción
automática: el manager decide.

**Costos ocultos.** Ninguno de etiquetado (llega solo). El costo real es el de operación:
dos disparos de Cloud Run Job por semana.

> *"Si no podés explicar cómo las predicciones serán usadas para tomar decisiones que
> provean valor al usuario final, detente aquí y no avances."* — Irina Peregud

---

## B7 · Realizando predicciones

**Batch, disparado por calendario y anclado al deadline.** No es una elección estética: la
predicción sólo es válida si `predicted_at < deadline_time`.

| Momento | Qué corre |
|---|---|
| **T − 3 h del deadline** | Ingesta del snapshot de FPL + `fixtures.csv` → features → predicción → registro |
| **T + 2 h del último partido de la fecha** | Ingesta de resultados → métricas → evaluación de degradación → decisión de retraining |

- **Servicio:** FastAPI sobre Cloud Run. Sin GPU: la inferencia son 10 filas.
- **Online, además de batch:** el endpoint queda expuesto para la demo y para recalcular
  si hay una noticia de lesión tardía.
- **Interpretabilidad:** la logística multinomial da coeficientes legibles, que son lo que
  se muestra en la defensa. El gradient boosting se acompaña de importancias.
- **Latencia:** irrelevante. Una fecha por semana, 10 partidos.

---

## B8 · Recolectando datos

**La etiqueta llega sola.** Ésta es la propiedad que define el proyecto.

```
viernes 17:30   deadline  →  se predice y se REGISTRA (snapshot de features congelado)
sábado 14:00    se juegan los partidos
sábado 16:00    /event/{GW}/live/ y /fixtures/  →  llega el resultado real
                                                →  se calculan métricas
                                                →  se evalúa degradación
                                                →  si corresponde, retraining
```

- **Entrenamiento inicial:** 1.520 partidos ya ingestados (2022-23 → 2025-26).
- **Datos nuevos:** 10 partidos por gameweek, 380 por temporada.
- **Human-in-the-loop en el etiquetado: cero.** No hay que anotar nada; el marcador es la
  etiqueta.
- **Lo que sí hay que cuidar** es que el snapshot de features quede congelado en el
  momento de predecir. Por eso Bronze es append-only y particionado por `ingested_at`: el
  snapshot pre-deadline es la prueba auditable de que no se usó información del futuro.

Cada predicción se registra con `fixture_id`, `predicted_at`, `model_version`,
`feature_set_version`, las tres probabilidades y el `features_snapshot`. Sin ese registro
no hay monitoreo, sólo un endpoint que responde.

---

## B9 · Construyendo modelos

**Modelos, en orden:**

1. **Regresión logística multinomial regularizada** — el baseline entrenado. Coeficientes
   interpretables, que sirven para la defensa.
2. **Gradient boosting chico** (LightGBM, pocas hojas, mucha regularización).

**Frecuencia de reentrenamiento: semanal**, al cierre de cada gameweek. Es lo natural en
un dominio donde el ground truth llega semanalmente.

**Disparadores explícitos, no "cuando parezca":**

| Disparador | Condición |
|---|---|
| Programado | Al cierre de cada gameweek, incorporando la fecha recién jugada |
| Por degradación | Log-loss rodante de las últimas K fechas sobre umbral **y** que la caída no se explique por el baseline de cuotas cayendo igual |
| Por drift | Cambio de temporada — es donde se mueve todo |

**Promoción del modelo.** El modelo nuevo **sólo reemplaza al viejo si le gana en el
holdout**. Si no, se registra el intento y queda el anterior.

**Stack y costo.** scikit-learn + LightGBM, entrenamiento en Cloud Run Job (o Vertex AI si
se quiere mostrar el servicio administrado). Reentrenar 1.500 filas tarda segundos: el
costo es marginal. Se descartó Composer/Airflow, que factura 24/7 para dos triggers por
semana.

---

## B10 · Monitoreo y evaluación en vivo

**Performance en producción.** Log-loss y accuracy rodantes sobre las últimas K
gameweeks, contra los mismos baselines del EDA calculados **sobre el mismo período**.

> La distinción que importa: una caída del modelo **acompañada** de una caída del baseline
> de cuotas es la liga siendo más impredecible, no el modelo degradándose. Sin comparar
> contra el mercado del mismo período, el monitoreo confunde las dos cosas.

**Drift de datos.** Ya cuantificado sobre cuatro temporadas:

| Métrica | Variación entre temporadas |
|---|---|
| Goles del visitante | 19,5 % |
| Tiros al arco del local | 19,0 % |
| Goles totales | 17,9 % |
| Córners del local | 12,9 % |
| Prob. implícita del local | 3,2 % |

A eso se suma el drift estructural: **cada temporada cambian 3 de 20 equipos** por
ascensos y descensos.

**Creación de valor.** La métrica que le habla al usuario, no al modelo:

- **% de fechas en las que el modelo le gana a "siempre el local"** — la vara de que sirve
  para algo.
- **Distancia al mercado** en log-loss, fecha a fecha, en la variante sin cuotas.
- **Calibración en vivo:** de los partidos donde dijimos 60 %, ¿cuántos salieron?

---

## Diagrama del ciclo

```
              ┌─────────────────────────────────────────────┐
              │                FUENTES (B2)                 │
              │   FPL API  ·  football-data  ·  vaastav     │
              └──────────────────────┬──────────────────────┘
                                     │  T−3h del deadline
                                     ▼
   ┌────────────┐  append-only  ┌────────────┐        ┌────────────┐
   │   BRONZE   │──────────────▶│   SILVER   │───────▶│  GOLD-TP   │
   │ ingested_at│               │ jugador-GW │  (B4)  │ 1 partido  │
   └────────────┘               └────────────┘        └─────┬──────┘
         ▲                                                  │
         │                                          ┌───────▼───────┐
         │                                          │  MODELO (B9)  │
         │                                          │  logística /  │
         │                                          │   LightGBM    │
         │                                          └───────┬───────┘
         │                                                  │  (B7)
         │                                          ┌───────▼───────┐
         │                                          │  PREDICCIÓN   │
         │                                          │  registrada   │
         │                                          │  ANTES del    │
         │                                          │   deadline    │
         │                                          └───────┬───────┘
         │                                                  │
         │                                        se juega la fecha
         │                                                  │
         │      el ground truth llega solo          ┌───────▼───────┐
         └──────────────────────────────────────────│  RESULTADO    │
                    reentrenamiento (B9)            │   +2h  (B8)   │
                              ▲                     └───────┬───────┘
                              │                             │
                              │   ┌─────────────────────────▼──────┐
                              └───│   MÉTRICAS Y DRIFT (B10)       │
                                  │   vs. baseline del mismo       │
                                  │   período                      │
                                  └────────────────────────────────┘
```

---

## Mapa a GCP

| Bloque del Canvas | Servicio |
|---|---|
| B2 — Datos | Cloud Storage (Bronze) + BigQuery (Silver/Gold) |
| B4, B8 — Ingesta y features | Cloud Run Jobs + Cloud Scheduler (2 disparos/semana) |
| B9 — Entrenamiento | Cloud Run Job o Vertex AI |
| B7 — Serving | FastAPI · Docker · Cloud Run |
| Build de la imagen | Artifact Registry + Cloud Build |
| B10 — Monitoreo | Cloud Logging + métricas propias en BigQuery |

---

## Qué queda abierto

1. **Re-verificar `strength_*` después de la GW1.** Hoy están en escala incompatible con
   el histórico y en cero. Decide si el grupo "Fuerza" entra al feature set.
2. **Medir la cadencia intra-temporada de `E0.csv`.** Registrar el `Last-Modified` en cada
   corrida de ingesta desde la GW1: define si las features de forma (tiros, córners) se
   pueden refrescar a tiempo para el deadline siguiente.
3. **Confirmar los campos de `/event/{GW}/live/`.** En pretemporada devuelve `elements`
   vacío; hay que verificar en la GW1 que trae `expected_goals` por jugador, que es lo que
   nos independiza de vaastav.
4. **Cerrar el umbral de degradación de B10** con un número, no con "cuando parezca".
