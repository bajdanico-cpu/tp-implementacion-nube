# Training

**Estado: implementado.** `python -m training.run --todos` entrena, evalúa y persiste.

```powershell
python -m training.run --todos                             # los tres modelos
python -m training.run --model xgb_gbt --device cpu        # forzar CPU
python -m training.run --model xgb_gbt --walk-forward      # 38 folds
python -m training.run --model xgb_gbt --features podado   # top-N por importancia
python -m training.benchmark_gpu                           # CPU vs GPU
python -m training.device                                  # ¿hay GPU?
```

| Módulo | Qué hace |
|---|---|
| `device.py` | Resuelve `cuda`/`cpu` con un fit de prueba real y fallback |
| `dataset.py` | Carga Gold, split temporal, codificación de etiquetas |
| `models.py` | `xgb_gbt`, `xgb_rf` y `logreg` |
| `metrics.py` | Accuracy, F1, precision, recall, log-loss, calibración, McNemar |
| `betting.py` | Capa de decisión: valor esperado y simulación de ROI |
| `evaluate.py` | Holdout con IC bootstrap + walk-forward por gameweek |
| `promotion.py` | Cuándo un candidato reemplaza al modelo en producción |
| `registry.py` | Persistencia versionada e intentos rechazados |
| `benchmark_gpu.py` | El barrido CPU vs GPU |

---

## Los modelos

El bloque 3 del canvas nombra **"XGBoost, RF"**. Los dos corren sobre el mismo motor
—XGBoost hace Random Forest con `num_parallel_tree` y `n_estimators=1`— así que van a GPU
con el mismo código y sin dependencias nuevas. (cuML de RAPIDS, el RF en GPU "de verdad",
es **sólo Linux**.) La logística no está en el canvas pero es el piso contra el cual se
justifica la complejidad de los árboles.

### Las cinco palancas contra el overfit

1.140 filas y 143 features son **~8 observaciones por feature**. El riesgo no es falta de
capacidad, es memorizar:

1. **Árboles chatos** (`max_depth=3`) y hojas gordas (`min_child_weight=10`).
2. **`colsample_bytree=0.5`** — cada árbol ve ~70 features. Con columnas muy
   correlacionadas entre sí, es la palanca que más decorrelaciona.
3. **`max_bin=64`.** Con el default de 256 y 1.140 filas, cada bin del histograma tiene
   **4 observaciones** y los cortes son ruido. A 64 son ~18 obs/bin. Es regularización
   estructural, no una optimización de velocidad.
4. **Early stopping temporal** contra 2024-25 — **nunca contra el holdout**. Usar el
   holdout para decidir cuándo parar es la forma sutil de contaminarlo. El
   `best_iteration` se guarda y el modelo final se refitea con las tres temporadas usando
   ese número fijo.
5. **Promediado de 5 semillas.** Cada fit tarda menos de un segundo y el holdout tiene ±5
   puntos de error estándar: reducir varianza sale gratis.

---

## Resultados sobre el holdout 2025-26

Train 2022-25 (1.140 partidos) → test 2025-26 (380). Split temporal, nunca aleatorio.

| Modelo | Accuracy | IC 95 % | F1 macro | Log-loss | ROI |
|---|---|---|---|---|---|
| **`xgb_rf`** | **0,505** | [0,455 – 0,555] | 0,380 | **1,029** | −0,070 |
| `xgb_gbt` | 0,492 | [0,442 – 0,542] | 0,398 | 1,046 | −0,017 |
| `logreg` | 0,458 | [0,408 – 0,508] | **0,409** | 1,138 | **−0,009** |

**Las varas, sobre las mismas 380 filas:**

| Baseline | Accuracy | Log-loss |
|---|---|---|
| Siempre local | 0,426 | — |
| **Prior de clase** (el del bloque 5) | **0,426** | **1,085** |
| Cuotas de cierre | 0,495 | 1,012 |

**El criterio del bloque 5 se cumple:** los tres modelos le ganan al promedio del dataset.
Y `xgb_rf` **iguala o supera la accuracy del mercado (0,505 vs 0,495) sin usar cuotas**,
que era la comparación honesta. En log-loss el mercado sigue adelante (1,012 vs 1,029):
está mejor calibrado, aunque acierte menos veces el argmax.

### El empate: donde la accuracy engaña

La matriz de confusión de `xgb_rf`:

```
             predicho
real      away   draw   home
away        64      0     50
draw        41      0     63
home        34      0    128
```

**La columna del empate está vacía: nunca lo predice.** Por eso su F1 macro (0,380) es el
peor de los tres pese a tener la mejor accuracy. La logística, que sí predice empates
(F1 del empate = 0,207), tiene peor accuracy pero mejor F1 macro **y mejor ROI**.

Esto es exactamente por qué el bloque 5 pide las cuatro métricas y no sólo accuracy: con
el empate en 24,1 % de los partidos y casi nunca siendo el argmax, un modelo puede subir
la accuracy simplemente ignorándolo.

---

## Walk-forward: la simulación del ciclo operativo

38 folds sobre 2025-26. Para cada fecha se reentrena con **todo lo anterior a su corte** y
se predice esa fecha. No es sólo validación: es el ciclo del bloque 9 corriendo de verdad,
y de acá sale la serie de predicciones pareadas que alimenta el McNemar.

| Métrica | Valor |
|---|---|
| Accuracy media por fecha | **0,499** |
| Log-loss media | **1,039** |
| % de fechas que le gana a "siempre local" | **52,6 %** |
| % de fechas que le gana al prior en log-loss | **65,8 %** |
| Accuracy media de "siempre local" | 0,430 |

Las dos últimas son las **métricas de creación de valor del bloque 10**: no le hablan al
modelo, le hablan al usuario.

El walk-forward (0,499) coincide con el holdout (0,492), que es la consistencia que uno
quiere ver: dos protocolos distintos midiendo lo mismo.

### El ±15,7 puntos, en vivo

La accuracy fecha a fecha del walk-forward salta así:

```
GW01 0,30   GW02 0,50   GW03 0,60   GW04 0,70   GW05 0,70   GW06 0,10
```

Con 10 partidos por fecha, eso es exactamente el ruido esperado. **Es la razón de que la
promoción no se decida sobre una fecha**, y se ve mejor acá que en cualquier argumento.

### Un bug que valió la pena encontrar

La primera corrida del walk-forward dio accuracy media **0,435** y ganaba a "siempre local"
sólo el 44,7 % de las fechas — muy por debajo del holdout. La causa no era el modelo: cada
fold entrenaba con `n_estimators=2000` **sin early stopping**, mientras que el camino del
holdout usaba el `best_iteration` (185). Los dos protocolos estaban midiendo modelos
distintos, y el del walk-forward estaba masivamente sobreajustado.

| | Con el bug | Corregido |
|---|---|---|
| Accuracy media | 0,435 | **0,499** |
| Log-loss media | 1,171 | **1,039** |
| % gana a siempre-local | 44,7 % | **52,6 %** |
| % gana al prior | 31,6 % | **65,8 %** |

Ahora el número de rondas se fija **una vez, fuera del bucle**. Además de ser correcto, es
lo que haría un reentrenamiento semanal real: se reentrena con los datos nuevos, no se
re-tunea de cero cada semana.

---

## La capa de decisión: "apostamos o no apostamos"

Es el bloque 6, y es **el único lugar donde entran las cuotas**. No son features del
modelo, y la razón es estructural:

> Si el modelo usa las cuotas como feature, aprende a copiarlas. Entonces `p ≈ 1/cuota`,
> el valor esperado da ~0 por construcción y **el sistema nunca encontraría una apuesta con
> valor**. Detectar una discrepancia con el mercado exige que las dos estimaciones sean
> independientes. Hay un test que lo demuestra.

```
EV = p · cuota − 1        # se apuesta si EV > 0,05
```

**Resultado de la simulación: ROI negativo en los tres modelos.** Era lo esperable y está
dicho de antemano en el código: el overround medido es **1,057**, o sea una comisión
implícita del 5,7 %. Para que el ROI dé positivo, el modelo tendría que estar mejor
calibrado que el mercado por más que esa comisión.

Lo que sí se ve, y es el hallazgo interesante:

| | ROI | Tasa de acierto | Cuota media |
|---|---|---|---|
| `logreg` | **−0,009** | 0,317 | 3,68 |
| `xgb_gbt` | −0,017 | — | — |
| `xgb_rf` | −0,070 | 0,261 | 4,75 |
| Apostar siempre al local | −0,086 | — | — |

**Para apostar, la calibración importa más que la accuracy.** La logística es el peor
modelo por accuracy y el mejor por ROI, porque sus probabilidades son más honestas. El
`xgb_rf`, que ignora los empates, se sobreconfía y termina apostando a cuotas altas que no
salen.

---

## Reentreno y promoción: dos cadencias, porque una sola no se puede medir

El canvas dice *"se compara en la siguiente fecha, si le gana al de producción se pasa a
producción"*. Tomado literalmente eso no se puede medir:

| | |
|---|---|
| partidos por gameweek | **10** |
| error estándar de la accuracy con n=10 | **±15,7 puntos** |
| filas que agrega una semana sobre 1.140 | **+0,9 %** |

Comparar dos modelos sobre una sola fecha es tirar una moneda: promoverías al peor
aproximadamente la mitad de las veces. Y con +0,9 % de datos nuevos, el candidato es casi
idéntico al que está en producción.

| | Cadencia | Qué pasa |
|---|---|---|
| **Reentrenamiento** | **Semanal** | Se entrena con el histórico + la fecha jugada. Tarda segundos. Se registra, **no se promueve** |
| **Evaluación** | Semanal, shadow mode | Candidato y producción predicen los mismos partidos |
| **Promoción** | Cuando el test lo respalda | McNemar pareado sobre 10 fechas (100 partidos) |

**McNemar pareado**: como los dos modelos predicen *los mismos* partidos, sólo aportan
información aquellos donde **discrepan**. Es mucho más potente que comparar dos accuracies
independientes. Se promueve si p < 0,05, el candidato gana en los discordantes, y no
empeora en el holdout fijo (la red contra el sobreajuste al período reciente).

Los candidatos rechazados quedan en `models/<modelo>/attempts.jsonl`: **un pipeline que
sólo registra lo que promovió no puede demostrar que sabe decir que no.**

> El disparador honesto no es el calendario, es el **cambio de temporada**: cada año
> cambian 3 de 20 equipos y rota ~40 % de los minutos.

---

## GPU: la hipótesis se cayó, y por eso vale

El benchmark se pre-registró con una hipótesis escrita **antes** de medir:

> *"Con 1.140 filas × 143 features la GPU pierde por un factor de 3 a 8, y el punto de
> cruce está entre 10⁵ y 3×10⁵ filas."*

**Las dos mitades resultaron falsas.** Medido en una GTX 1650 Max-Q (4 GB, CC 7.5) contra
12 hilos de CPU, con 200 árboles fijos por fit, warmup descartado, mediana de 5 corridas:

| Escala | Filas | CPU | GPU | Speedup |
|---|---|---|---|---|
| ×1 | 1.140 | 0,70 s | 1,20 s | **0,59×** — la GPU pierde |
| ×10 | 11.400 | 2,08 s | 1,36 s | **1,53×** — ya gana |
| ×100 | 114.000 | 23,45 s | 4,37 s | **5,36×** |
| ×1.000 | 1.140.000 | 177,52 s | 33,69 s | **5,27×** |

1. A escala ×1 la GPU pierde, pero por **1,7×**, no por 3-8×. El overhead de lanzamiento
   de kernels es real pero mucho menor de lo estimado.
2. **El punto de cruce está entre 1.140 y 11.400 filas** — un orden de magnitud antes de lo
   predicho. Extrapolando, ronda las **4.000-5.000 filas**.
3. El speedup satura en ~5,3×: de ×100 a ×1.000 ya no mejora, señal de que a partir de ahí
   la GPU está saturada y escala linealmente igual que la CPU.

**Qué se concluye, con números:**

- **Para este TP la GPU no se justifica.** 1.140 filas están por debajo del cruce, y media
  décima de segundo por fit no le cambia la vida a nadie. El Job de entrenamiento en la
  nube se aprovisiona **sin GPU**, y eso ahora es una decisión tomada con evidencia.
- **En GCP una T4 sobre un `n1-standard-4` cuesta ~2,5-3× el nodo pelado.** Para pagarse
  necesita ser ≥3× más rápida: eso ocurre recién arriba de ~50.000 filas.
- **Para el otro proyecto sobre el mismo Silver, sí se justifica.** `fact_player_gw` tiene
  **113.270 filas**, prácticamente la escala ×100: ahí la GPU va **5,4× más rápido** y pasa
  el umbral de costo con holgura.
- **El valor MLOps del código GPU no es la velocidad, es la portabilidad.** El bloque 7
  sirve sin GPU. `test_modelo_entrenado_en_gpu_predice_igual_en_cpu` entrena en CUDA,
  guarda `.ubj`, carga en CPU y compara: coinciden dentro de 1e-5. La tolerancia no es
  exacta porque el algoritmo `hist` de GPU no es bit-idéntico (distinto orden de reducción
  en punto flotante) — eso se documenta, no se esconde.

Salidas en `training/output/benchmark_gpu.csv` y `.png`, con el ambiente completo
registrado para que un compañero con otra máquina pueda reproducirlo.

---

## Completo vs podado: 143 features contra 40

El set `podado` no se elige a mano: sale de las importancias de ganancia del set
`completo`. Sobre el mismo holdout:

| | Accuracy | IC 95 % | F1 macro | Log-loss | ROI |
|---|---|---|---|---|---|
| `completo` (143) | **0,492** | [0,442 – 0,542] | 0,398 | **1,046** | −0,017 |
| `podado` (40) | 0,482 | [0,429 – 0,532] | **0,413** | 1,064 | **+0,014** |

**Con 40 de las 143 features se pierde 1 punto de accuracy** — muy dentro del intervalo de
confianza, o sea indistinguible. A cambio sube el F1 macro: con menos features el modelo se
sobreconfía menos y reparte algo más de probabilidad al empate.

Es el argumento para promover `podado` a la `v2` del feature set: mismo rendimiento con un
cuarto de las columnas, menos superficie de fallo en serving y menos que calcular antes de
cada deadline. Se deja para la siguiente iteración, con las 143 documentadas.

### El único ROI positivo, y por qué no alcanza para festejar

`podado` es la única configuración con ROI positivo: **+0,014** sobre 391 apuestas. Pero
hay que decir el error:

```
ROI      +0,014
error    ±0,091      (n=391, acierto 0,320, cuota media 3,84)
IC 95%   [-0,164, +0,192]
```

**El cero está cómodamente adentro.** Con cuotas medias de 3,84 la varianza por apuesta es
enorme, y 391 apuestas no alcanzan ni de cerca para distinguir +1,4 % de 0. Reportarlo como
"el modelo es rentable" sería exactamente el error que el resto del proyecto se ocupa de
evitar. Lo honesto: **ninguna configuración demostró ser rentable, y el overround del 5,7 %
sigue siendo la explicación más simple de todos los ROI observados.**

---

## Qué features pesan

De `training/output/importancias_xgb_gbt.csv`. **142 de 143 features tienen ganancia > 0**,
así que casi nada es peso muerto.

**Top 10 por ganancia:** `dif_pos_tabla_camp` · `dif_pts_camp` · `fdr_dif` · `fdr_visita` ·
`dif_ppp_camp` · `local_es_ascendido` · `fdr_local` · `dif_n_hist` · `local_xg_cond_u5` ·
`local_pos_tabla_camp`

Lo que manda es **la diferencia de posición en la tabla y de puntos de campeonato** —
exactamente el *"puntaje campeonato"* que pide el bloque 4.

| Grupo | Peso |
|---|---|
| Forma reciente | 33,4 % |
| Puntaje de campeonato | 16,1 % |
| **Puntaje por línea** (def/med/del/arq) | **13,8 %** |
| Forma según condición local/visitante | 7,8 % |
| Forma intra-temporada | 7,7 % |
| Head-to-head | 5,7 % |
| Dificultad (FDR) | 5,7 % |
| Diferenciales | 5,3 % |
| Continuidad y rotación de plantel | 4,5 % |

### El "período de tiempo a definir" del canvas, respondido

El bloque 4 dejaba abierta la ventana temporal. Medido por peso en las importancias:

| Ventana | Peso |
|---|---|
| **`u5`** (últimos 5 partidos) | **26,4 %** |
| `u3` (últimos 3) | 20,2 % |
| `camp` (acumulado de temporada) | 16,1 % |
| `u5_temp` (últimos 5 de esta temporada) | 10,1 % |
| `cond_u5` (últimos 5 en la misma condición) | 7,8 % |

**La ventana de 5 pesa más que la de 3, pero las dos aportan**, así que se conservan las
dos. Es una decisión tomada con un número, no por gusto.

---

## Persistencia

```
models/xgb_gbt/
  PRODUCTION.json          -> qué versión está en producción
  attempts.jsonl           -> los candidatos rechazados, con su motivo
  <stamp>/
    model.ubj              formato nativo: portable CPU<->GPU, sobrevive upgrades
    metadata.json          el contrato con serving
    metrics.json
    importancias.csv
```

`.ubj` en vez de pickle: no arrastra estado de device, así que un modelo entrenado en GPU
se carga y predice en CPU sin tocar nada — el escenario real del bloque 7.

El `metadata.json` incluye `feature_names` **ordenado** (si el serving arma las columnas en
otro orden, XGBoost no se queja y devuelve basura), `classes_`, hiperparámetros,
`best_iteration`, `device_requested` vs `device_used`, el **prior de ascendidos congelado**,
versiones de librerías, `git_sha` y el hash de Gold.

---

## Detalles que muerden

- ⚠️ **`training/metrics.py` es el único módulo que llama a `log_loss`.** sklearn asume las
  columnas de probabilidad en orden lexicográfico. En scikit-learn 1.9 pasar
  `labels=['home','draw','away']` ya reordena y avisa, pero si **las columnas de tu matriz**
  están en ese orden el número sale mal **sin ningún warning**. Por eso las etiquetas se
  codifican con `CLASES_ORD` (`['away','draw','home']`) desde `dataset.py`: la alineación
  queda garantizada por construcción, no por una convención que alguien tenga que recordar.
- ⚠️ **En scikit-learn 1.9 `LogisticRegression` ya no acepta `multi_class`**, y `penalty`
  quedó deprecado en 1.8. Con `lbfgs` y 3 clases el ajuste es multinomial por defecto.
- ⚠️ **XGBoost exige etiquetas numéricas 0..n-1**; no acepta strings.
