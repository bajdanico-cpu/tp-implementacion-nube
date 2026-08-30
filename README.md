# TP Premier ML — predicción 1X2 de la Premier League

Trabajo práctico de **Implementación de Aplicaciones de Aprendizaje Automático en la Nube** (ITBA).

Predecir el resultado (gana local / empate / gana visitante) de los partidos de cada fecha de la
Premier League, con el ciclo de vida del modelo completo y automatizado.

El foco del trabajo está en el **pipeline / MLOps**, no en exprimir la performance predictiva. El
dominio se eligió por una propiedad que la mayoría de los casos no tiene: **el ground truth llega
solo, dos horas después de la predicción**. Eso permite montar y demostrar en vivo el ciclo cerrado:

```
predicción → registro → llegada del resultado real → cálculo de métricas
   → detección de degradación → retraining → nueva versión del modelo
```

---

## Setup

Requiere **Python 3.14** (verificado en 3.14.3; hay wheels `cp314` para todas las dependencias).

El venv se crea **fuera de OneDrive** a propósito: son ~30.000 archivos y OneDrive intentaría
sincronizarlos todos.

```powershell
.\scripts\setup_env.ps1          # crea el venv, instala todo y verifica
```

En Linux o macOS: `bash scripts/setup_env.sh`.

El script hace lo mismo que estos tres comandos, más la verificación:

```powershell
py -3.14 -m venv $env:USERPROFILE\.venvs\tp-premier-ml
& $env:USERPROFILE\.venvs\tp-premier-ml\Scripts\Activate.ps1
pip install -r requirements.txt
```

Se usa `py -3.14` y no `python` para fijar la versión: las dependencias están pinneadas a
wheels `cp314`, y con varios Python instalados el `python` del PATH puede no ser el 3.14.

Verificación del entorno:

```powershell
python -m common.config     # temporadas, ventanas y rutas
python -m training.device   # dice si hay GPU o si el entrenamiento cae a CPU
```

**GPU (opcional).** El wheel de XGBoost de PyPI trae CUDA de fábrica, así que con una NVIDIA
y el driver al día el entrenamiento la usa sin instalar nada más. Si no la hay, `device: auto`
cae a CPU con un warning y todo lo demás sigue igual: **la GPU no está en el camino crítico
de nada**. Verificado en una GTX 1650 (4 GB, compute capability 7.5).

---

## Cómo se corre

Todo corre en local, bajo demanda. No hace falta ninguna credencial de nube.

```powershell
# 1. Ingesta Bronze — las cuatro fuentes
python -m ingestion.run                # las tres publicas de siempre
python -m ingestion.bronze_pulselive   # la API oficial: copas, Europa y stats de Opta

# Variantes útiles
python -m ingestion.run --source fpl            # sólo snapshot de la API (el ciclo en vivo)
python -m ingestion.run --source fpl --gw 1     # + resultados de la fecha 1
python -m ingestion.run --season 2025-26        # una temporada puntual
python -m ingestion.run --force                 # ignora la caché y re-baja todo

# 2. Silver — normalización
python -m transform.silver             # FPL + football-data
python -m transform.competencias       # fact_match_comp: las cinco competencias
python -m transform.opta_stats         # fact_opta_stats: ~180 stats por equipo-partido

# 3. Tests — incluye el control anti-leakage
pytest

# 4. Gold — la tabla de features (1.530 x 301, de las cuales 279 son features)
python -m features.gold_tp
python -m features.spec --docs      # regenera docs/FEATURES.md desde el contrato

# 5. Entrenamiento y evaluación
python -m training.run --sin-holdout                       # el modelo que se REPORTA
python -m training.run                                     # el que SIRVE (incluye 2025-26)
python -m training.run --model xgb_gbt --walk-forward      # 38 folds, simula el ciclo
python -m training.ablacion                                # cuanto aporta cada bloque
python -m training.benchmark_gpu                           # CPU vs GPU con barrido de escala
python -m training.compare_models                          # grilla 7 modelos x 3 variantes
python -m training.analysis                                # donde le gana a cada vara
python -m training.reproducir                              # que hace falta para rehacer uno

# 5 bis. Operación — el ciclo cerrado
python -m serving.predict --gw 3                           # predice y registra la fecha
python -m serving.predict --gw 1 --evaluar                 # una ya jugada, contra el real
python -m monitoring.temporada_actual                      # metricas en vivo de 2026-27

# 6. El notebook que recorre todo
python notebooks/00_recorrido_completo.py                  # regenera el .ipynb
jupyter lab notebooks/00_recorrido_completo.ipynb

# 7. EDA y baselines
python -m eda.run_eda
```

**Política de caché:** las temporadas cerradas no se vuelven a bajar (no cambian). La temporada
actual sí, en cada corrida. La API de FPL nunca se cachea: cada snapshot tiene valor propio.

---

## Fuentes de datos

Las cuatro son complementarias, no alternativas. **Ninguna pide credenciales.**

| Fuente | Granularidad | Cobertura | Qué aporta que nadie más aporta |
|---|---|---|---|
| **football-data.co.uk** | 1 fila = 1 partido | 2010-11 → actual (se ingesta desde 2022-23) | **Cuotas de cierre** — el baseline duro y una feature fuerte |
| **vaastav/Fantasy-Premier-League** | 1 fila = 1 jugador × fecha | 2016-17 → 2025-26 | **El histórico jugador-fecha**. Ver nota abajo |
| **API oficial de FPL** | presente y futuro | temporada en curso | Fixtures de la fecha que viene, **deadlines** y el resultado apenas termina el partido |
| **API de premierleague.com** | 1 fila = 1 partido, todas las competencias | 22-35 temporadas | **Copas y Europa** —lo único que veía el pipeline era la Premier— y **~180 estadísticas de Opta** por equipo y partido |

### Por qué vaastav y no la API para el histórico

La API oficial **no sirve el detalle fecha a fecha del pasado**. Verificado sobre
`/element-summary/{id}/`:

- `history` → sólo la temporada **actual**, fecha a fecha.
- `history_past` → temporadas anteriores, pero **una fila por temporada** (totales agregados).
  Saka tiene 8 temporadas ahí: son 8 filas, no 8 × 38.
- Además `history_past` sólo existe para jugadores que **hoy** están en la base → sesgo de
  supervivencia: quien se fue de la Premier desaparece del todo.

vaastav archiva semanalmente el `history` de cada jugador, y por eso el pasado sólo existe ahí.

### Y cómo se combinan las dos para la temporada en curso

```
vaastav donde tenga la fecha   ->  es la version ASENTADA
la API donde falte             ->  es la que llega a tiempo
```

La decisión va por **(temporada, fecha)**, nunca por temporada entera: alcanzaba con que
vaastav publicara la fecha 1 para descartar los snapshots en vivo de todo el resto del año,
que es justo el hueco que la API existe para tapar.

**No son exactamente los mismos datos.** Comparadas las 610 filas de la GW1 de 2026-27:

| | Diferencias |
|---|---|
| minutos, goles, asistencias, vallas, goles concedidos, tarjetas, atajadas, **puntos**, bonus | **0 de 610** — idénticos |
| `expected_goals`, `xGI`, `xGC` | 7 / 10 / 27, **todas del último partido del snapshot** (se estaba cerrando cuando se tomó la foto) |
| `influence`, `creativity`, `threat`, `ict_index` | ~45 % de las filas — **ninguna feature del modelo las usa** |

Por eso vaastav gana cuando llega: la API puede agarrar un partido a medio asentar.

⚠️ **El `bootstrap` que se lee es el contemporáneo de cada fecha, no el último.** El bootstrap
dice a qué club pertenece cada jugador *hoy*: leyendo el último, Baleba —que jugó la fecha 1
en el Brighton y se fue al United— aparecía con sus stats atribuidas al United. Bronze es
append-only y fechado justamente para esto. Un jugador transferido *a mitad de fecha* queda
inclasificable y se descarta, con warning.

### Notas de implementación verificadas

- **La API exige `User-Agent` de browser** (si no, 403) y **tiene CORS**: se consume siempre desde
  el servidor, nunca desde un frontend.
- **`xG`/`xA`/`xGC` en FPL existen recién desde 2022-23.** Es lo que fija la ventana de ingesta.
- **`master_team_list.csv` sólo llega hasta 2023-24.** De 2024-25 en adelante el mapeo de equipos
  sale del `teams.csv` de cada temporada.
- **`data/2026-27/gws/` no existe hasta que se juegue la primera fecha.** Ese 404 es estado normal.
- **`mmz4281/2627/E0.csv` redirige (301) a otra división mientras la temporada no arrancó** —
  devolvía partidos de la National League con status 200. Por eso hay un guard que valida
  `Div == 'E0'` antes de aceptar nada.
- Los nombres de equipo no coinciden entre fuentes: `Man Utd`↔`Man United`, `Spurs`↔`Tottenham`,
  `Sheffield Utd`↔`Sheffield United`. Y FPL es inconsistente consigo mismo: el mismo club es
  `Ipswich` en 2024-25 y `Ipswich Town` en 2026-27. La clave canónica es `short_name` (ARS, MUN,
  TOT…), que resultó **100% estable** entre temporadas — a diferencia del `id`, que FPL reasigna
  todos los años (cambia en 18 de 27 equipos).
- **En 2022-23 no existe la GW7**: se canceló entera por la muerte de Isabel II y sus partidos se
  reprogramaron a otras fechas. Por eso hay gameweeks con 7 partidos y otras con 16. Consecuencia
  para el feature engineering: las ventanas rolling van sobre los **últimos N partidos de cada
  equipo**, no sobre las últimas N gameweeks.
- **`position == 'AM'` son los directores técnicos**, no jugadores. FPL los hizo elegibles con el
  chip *Assistant Manager*, vigente sólo entre la GW23 de 2024-25 y el final de esa temporada
  (322 filas). Tienen 0 minutos y puntúan por resultado del equipo: se excluyen en Silver.

---

## Arquitectura de datos

Patrón medallion, con **un Silver y dos Golds**:

```
Bronze  (crudo, append-only, particionado por ingested_at)
   │
Silver  (normalizado — granularidad jugador-fecha conservada)
   ├──> Gold-TP    : 1 fila por partido        → modelo 1X2  (este TP)
   └──> Gold-FPL   : 1 fila por jugador-fecha  → armado del equipo de FPL (proyecto aparte)
```

Bronze **nunca sobrescribe**. Cada corrida escribe en `ingested_at=<timestamp>`. El motivo no es
prolijidad: el snapshot tomado *antes* del deadline es el único que refleja lo que se sabía al
momento de predecir, y conservarlo junto al posterior es la defensa auditable contra el leakage.

La capa `common/storage.py` abstrae el I/O. Hoy el backend es `local` (parquet); pasar a GCS o
BigQuery es implementar un backend y cambiar una línea en `config.yaml`. La lógica de negocio no
se toca.

---

## El riesgo que puede arruinar el trabajo: leakage temporal

Los puntos y stats de una fecha se conocen **después** de que se jugó. Si una feature de la fecha N
usa datos de la fecha N, el modelo está viendo el resultado: la accuracy se dispara y el trabajo no
vale nada.

**Regla de oro:** el snapshot de features usado para predecir la fecha N tiene que ser reproducible
usando **exclusivamente** datos anteriores al `deadline_time` de la fecha N.

Mitigaciones implementadas:

1. Features agregadas siempre como **ventanas rolling sobre fechas anteriores** (3 y 5).
2. Bronze append-only con snapshots fechados.
3. `tests/test_no_leakage.py` — assert automatizado de la regla de oro.
4. `xP` **descartado**: se scrapea del campo `ep_this` *después* de que termina la fecha, y el
   propio repo de vaastav advierte que puede reflejar información post-partido. Está en
   `features.banned_columns` y hay un test de regresión para que no vuelva a entrar.

---

## Datos disponibles

Estado tras correr la ingesta y `transform.silver`:

| Tabla Silver | Grano | Filas |
|---|---|---|
| `dim_team` | temporada × equipo | 100 |
| `fact_fixture` | fixture (con **deadline**) | 1.900 |
| `fact_match` | partido (resultado + cuotas) | 1.530 |
| `fact_player_gw` | jugador × fecha | 113.880 |
| `fact_match_comp` | equipo × partido, las 5 competencias | 4.700 |
| `fact_opta_stats` | equipo × partido, stats de Opta | 3.058 |

El cruce football-data ↔ FPL da **100% en las cuatro temporadas cerradas**.

---

## Baselines

Calculados por `python -m eda.run_eda`, no citados de memoria. Holdout: 2025-26.

| Baseline | Accuracy | Log-loss |
|---|---|---|
| Siempre gana el local | 42.6% | — |
| Prior de clase | 42.6% | 1.085 |
| **Cuotas de cierre** | **49.5%** | **1.012** |

Por temporada, la accuracy de las cuotas fue **55.5% / 60.0% / 55.5% / 49.5%**. La caída de
2025-26 no es que el mercado haya estado peor informado: es la temporada con más empates (27.4%),
y el empate casi nunca es el resultado más probable para una casa de apuestas. **El techo realista
está en ~55%.**

Quedar en el medio entre 42.6% y 55% es el resultado esperable y no es un problema para el
trabajo. Lo que importa es tener el benchmark explícito y una lectura honesta.

⚠️ `sklearn.metrics.log_loss` asume las etiquetas en **orden lexicográfico** y alinea las columnas
de probabilidad con ese orden. Pasarle `['home','draw','away']` devuelve un número incorrecto y
sólo avisa por warning. Usar `eda.baselines.CLASES_ORD`.

---

## Estado

- [x] **Fase 0** — andamiaje, entorno, config, capa de storage
- [x] **Fase 1** — ingesta Bronze de las cuatro fuentes (append-only)
- [x] **Fase 2** — Silver: mapeo de equipos y normalización (cruce 100%)
- [x] **Fase 3** — tests: anti-leakage, schemas, mapeo
- [x] **Fase 4** — EDA y baselines
- [x] **Diseño del caso** — ML Canvas (`ML Canvas esquema.docx`, en la carpeta de la materia)
- [x] **Fase 5** — Gold + modelo → `features/`, `training/`, `docs/FEATURES.md`
- [~] **Fase 6** — serving, monitoreo y retraining. **La lógica está escrita y corriendo en
  local** (`serving/predict.py`, `monitoring/temporada_actual.py`); falta empaquetarla:
  `serving/app.py` + `Dockerfile`, el backend GCS de `common/storage.py`, e `infra/`

### Dónde está el trabajo, en una tabla

| | |
|---|---|
| Tabla Gold | **1.530 filas × 301 columnas**, de las cuales **279 son features** |
| Diccionario | [`docs/FEATURES.md`](docs/FEATURES.md), **generado** desde `features/spec.py` |
| Versión del feature set | `v2.3189c9d4.279` — **derivada de un hash** de la lista, no escrita a mano |
| **Modelo elegido** | **XGBoost** (`xgb_gbt`), entrenado sin las fechas con xG falso |
| Accuracy | **0,500** en holdout (baseline del canvas: 0,426; mercado: 0,495). En walk-forward, 0,516 con el set de 159 |
| Feature más importante | `dif_elo` — la diferencia de rating Elo entre los dos equipos |
| Modelos comparados | 7 modelos × 3 variantes de datos, incluida una red neuronal |
| Bloques de features medidos | copas/Europa (24) y Opta (56): **no aportan**, y está publicado |
| GPU | **Se midió**: pierde 1,7× a 1.140 filas, gana **5,4×** a 114.000 |
| Ciclo cerrado | predicción registrada → resultado real → métricas, corriendo sobre 2026-27 |
| Tests | **470**, con pruebas de fuego para cada hallazgo |

> ⚠️ **Hay dos modelos y sólo uno reporta números.** El de **evaluación**
> (`training.run --sin-holdout`) entrena hasta 2024-25 y se mide contra 2025-26: es el que
> vale como evidencia. El de **producción** (`training.run`) entrena también con 2025-26, así
> que su 0,616 sobre esa temporada **no es una mejora, es el modelo acordándose**. El
> `metadata.json` lo declara con `metricas_son_de_generalizacion: false`.

### Para entender todo de una

Abrí **[`notebooks/00_recorrido_completo.ipynb`](notebooks/00_recorrido_completo.ipynb)**:
recorre el proyecto entero paso a paso, con los números a la vista.

### Para verlo correr en GCP

**[`notebooks/01_gcp_cloudshell.ipynb`](notebooks/01_gcp_cloudshell.ipynb)** — el lab en la
nube: se clona el repo en Cloud Shell, se corre celda por celda, y cada paso deja un recurso
visible en la consola. Llega hasta la **predicción de una fecha registrada en el bucket**,
con la verificación de que no hay leakage temporal, y **la última celda borra todo** para no
dejar nada facturando. Usa `requirements-cloud.txt`
(Cloud Shell no tiene Python 3.14). Los comandos por terminal están en
[`gcp/runbook.md`](gcp/runbook.md).

### Lo que el modelo NO logra, dicho sin maquillar

- **No le gana al mercado.** Cuando discrepa de las casas de apuestas acierta 0,346 contra
  0,365 de ellas: no tiene ventaja informativa.
- **Ninguna estrategia de apuestas resultó rentable.** Todos los ROI son negativos, y el
  único positivo que apareció quedó dentro del error estándar.
- Para la propuesta de valor del canvas —un emprendimiento que gane con apuestas— el
  sistema todavía no la sostiene. Lo que sí sostiene es el ciclo de MLOps completo.



---

## Cómo arrancar desde cero

Para un compañero que clona el repo por primera vez. **No hace falta ninguna credencial:**
las cuatro fuentes de datos son públicas y sin autenticación.

```powershell
git clone <url-del-repo> tp-premier-ml
cd tp-premier-ml

.\scripts\setup_env.ps1        # crea el venv fuera de OneDrive e instala todo
                               # (en Linux/macOS: bash scripts/setup_env.sh)

python -m ingestion.run              # ~27 MB de Bronze, tarda unos minutos
python -m ingestion.bronze_pulselive # copas, Europa y las stats de Opta
python -m transform.silver           # FPL + football-data
python -m transform.competencias     # fact_match_comp
python -m transform.opta_stats       # fact_opta_stats
python -m features.gold_tp           # Gold: 1.530 x 301
pytest                               # 470 tests
python -m training.run --sin-holdout # entrena y evalúa (el numero que se reporta)
```

**`data/` no está en el repo** y no debería estar: se regenera con los dos primeros
comandos, y así el repo queda liviano y sin datos que puedan quedar desactualizados. Lo
mismo con `models/*.ubj` — pero sí se versionan los `metadata.json`, `metrics.json` y
`attempts.jsonl`, que son la trazabilidad del ciclo de promoción.

Si algo falla, el primer diagnóstico es:

```powershell
python -m common.config      # rutas, temporadas y ventanas
python -m training.device    # ¿hay GPU, o se entrena en CPU?
```
