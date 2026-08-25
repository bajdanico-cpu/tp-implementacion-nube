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

## El empate: no es una falla del modelo

La objecion natural al ver la matriz de confusion es "si nunca predice empate, no sirve".
Vale la pena mirarlo con datos, porque la conclusion es la contraria.

**El mercado de apuestas —casas reales, con plata de verdad— tampoco lo predice nunca:**

| Sobre las 380 fechas del holdout | |
|---|---|
| Veces que el empate es el argmax **del mercado** | **0 de 380** |
| Probabilidad de empate que asigna el mercado | media 0,248, **maximo 0,312** |
| Partidos donde el mercado le da al empate mas de 1/3 | **0** |
| Empates que efectivamente ocurrieron | **27,4 %** |

El empate ocurre en uno de cada cuatro partidos pero **nunca es el resultado mas probable**:
con local ~43 %, visitante ~30 % y empate ~25 %, siempre queda tercero. Un modelo
perfectamente calibrado tampoco lo pondria de argmax.

O sea: **el argmax es la salida equivocada para este problema**, y el canvas ya lo sabia —
el bloque 6 decide con probabilidades ("apostamos o no apostamos"), no con la clase ganadora.

Lo que si importa es si la probabilidad del empate esta bien estimada. Medido sobre
`xgb_gbt`: dice 0,236 en promedio y ocurren 0,274 — la subestima en 3,8 puntos. Y de hecho
llega a asignarle hasta 0,443 en algunos partidos, **mas que el maximo del mercado (0,312)**.

### Se puede forzar a que prediga empates?

Si, pesando la clase. Y muestra exactamente el canje:

| Peso del empate | Empates predichos | F1 del empate | F1 macro | Accuracy | Log-loss |
|---|---|---|---|---|---|
| x1 (base) | 6 | 0,073 | 0,391 | 0,490 | 1,040 |
| **x1,5** | **42** | **0,192** | **0,422** | 0,482 | **1,038** |
| x2,5 | 172 | 0,304 | 0,385 | **0,395** | 1,075 |

Con x1,5 se gana F1 macro y hasta un poquito de log-loss, a cambio de menos de un punto de
accuracy: es un canje razonable si lo que importa es no ignorar una clase entera. Con x2,5
el modelo se vuelve un predictor de empates y la accuracy se desploma nueve puntos.

---

## Experimentos: que mejora y que no

`python -m training.experiments` corre todas las variantes contra el mismo holdout, con el
mismo protocolo. Cada una cambia **una** cosa.

### Entrenar mas rondas mejora?

**No.** El early stopping corta en 184-201 rondas de 2.000 y ahi se queda lo que hay:

| Variante | Rondas | Accuracy | Log-loss |
|---|---|---|---|
| base (lr 0,03) | 184 | 0,490 | 1,040 |
| lr 0,01 con hasta 6.000 rondas | 550 | **0,490** | 1,039 |
| lr 0,10 | 52 | 0,497 | 1,042 |
| `max_depth=6` en vez de 3 | 123 | 0,479 | 1,048 |

Entrenar tres veces mas lento y tres veces mas rondas llega **exactamente al mismo lugar**.
Y con arboles mas profundos empeora, que es la confirmacion de que el limite no es
capacidad del modelo: es cuanta senial hay en 1.140 partidos.

### Conviene sacar del entrenamiento una temporada con datos incompletos?

**Si, y bastante.** Esta fue la mejor idea de toda la ronda. La clave es la separacion:
esos partidos salen como **objetivo de entrenamiento** pero se conservan como **historia**
para las ventanas de los partidos posteriores. Las features de Gold se calculan sobre la
historia completa, asi que filtrar filas del train no les quita nada a las demas.

| Variante | Filas de train | Accuracy | F1 macro | Log-loss |
|---|---|---|---|---|
| base | 1.140 | 0,490 | 0,391 | 1,040 |
| sin las fechas de 2022-23 sin xG | 1.004 | 0,497 | **0,406** | 1,039 |
| **sin 2022-23 entera** | **760** | **0,505** | 0,399 | **1,034** |

Con **un tercio menos de datos de entrenamiento el modelo es mejor en todo**. La temporada
2022-23 no solo aporta poco: contamina. Tiene sentido — el 37,9 % de sus partidos no tiene
xG real, y el futbol de hace tres anios se parece menos al de hoy que el del anio pasado.

### La calibracion por temperatura ayuda?

Marginalmente: T = 1,008 y el log-loss baja de 1,0396 a 1,0350. El modelo ya estaba bien
calibrado, asi que no hay mucho que arreglar.

> **Un error propio que vale documentar.** La primera corrida daba log-loss 1,205, mucho
> peor. La causa: la temperatura se ajustaba con el modelo **refiteado**, que ya habia visto
> la temporada de validacion. Contra predicciones artificialmente buenas, el ajuste elegia
> un T que *agudizaba* en vez de aplanar. Es un leakage sutil y silencioso: no falla nada,
> solo empeora el resultado. Se corrigio ajustando T con el modelo previo al refit.

---

## Features nuevas: Elo y estado del equipo

Las ventanas moviles tienen un limite: **tratan igual a todos los rivales**. Ganarle al
ultimo pesa lo mismo que ganarle al primero. El Elo resuelve exactamente eso.

Se agregaron 14 columnas (`features/elo.py`), 7 por lado:

| Feature | Que aporta que las ventanas no aportan |
|---|---|
| `elo` | cada resultado vale segun **contra quien** fue. K=20, ventaja de localia 65, margen de victoria atenuado por logaritmo, regresion del 25 % a la media entre temporadas |
| `xg_diff_u5` | goles menos xG: **suerte de definicion**, fuertemente reversible a la media. Ni `gf_u5` ni `xg_u5` lo capturan por separado |
| `xgc_diff_u5` | lo mismo del lado defensivo (incluye rendimiento del arquero) |
| `tiros_conc_u5` | lo que el equipo **regala**; las ventanas de `tiros` miden lo que genera |
| `partidos_14d` | congestion de calendario |
| `racha` | puntos de los ultimos 3 contra el promedio de la temporada: si esta por encima o por debajo de su nivel |

**Validacion del Elo**, sin ajustar nada: al cierre de 2024-25 pone arriba a Liverpool
(campeon), Arsenal y City; y abajo a Southampton, Ipswich y Leicester — **los tres
descendidos**.

| Variante | Accuracy | F1 macro | Log-loss |
|---|---|---|---|
| base (143 features) | 0,490 | 0,391 | 1,040 |
| **+ Elo y estado (159)** | **0,497** | 0,396 | **1,033** |
| + Elo, sin 2022-23 | 0,497 | 0,382 | **1,028** |

La mejora es chica y **cae dentro del intervalo de confianza**, asi que por si sola no es
concluyente con n=380. Pero es consistente en accuracy y log-loss a la vez, y sobre todo:
**`dif_elo` paso a ser la feature mas importante del modelo** (ganancia 14,9, por encima de
`dif_pos_tabla_camp` con 12,2). La combinacion con sacar 2022-23 da log-loss **1,028**, el
mejor de todo lo probado y ya cerca del mercado (1,012).

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

## Otros modelos probados

`training/models_alt.py`. Todos sobre el mismo holdout, mismas features.

| Modelo | Accuracy | F1 macro | F1 empate | Log-loss | Empates predichos |
|---|---|---|---|---|---|
| **`hgb`** (HistGradientBoosting de sklearn) | **0,500** | **0,395** | 0,055 | **1,031** | 6 |
| `xgb_gbt` | 0,492 | 0,385 | 0,037 | 1,044 | 5 |
| Poisson bivariado | 0,484 | 0,367 | **0,000** | 1,042 | **0** |
| Logit ordinal | 0,432 | 0,385 | **0,184** | 1,131 | 70 |
| **Red neuronal (MLP)** | **0,426** | 0,378 | 0,129 | 1,100 | 67 |
| Ensamble de los cinco | 0,492 | 0,384 | 0,036 | 1,033 | 6 |

Referencias: mercado 0,495 / 1,012 — prior de clase 0,426 / 1,085.

**La red neuronal es de las peores**, apenas por encima del baseline trivial. Con 1.140
filas y 159 features no tiene con qué: es el caso de manual donde una red sobreajusta. Ya
no es una intuición, está medido. `hgb` quedó incorporado a la CLI por ser el mejor.

### El Poisson bivariado: la prueba estructural sobre el empate

Este modelo no predice la clase. Predice **cuántos goles hace cada equipo** y de ahí deriva
las tres probabilidades: `P(empate) = Σₖ P(local=k)·P(visita=k)`. El empate deja de ser una
etiqueta arbitraria y pasa a ser la diagonal de la distribución conjunta.

Y aun así **predice cero empates**, con un máximo de probabilidad de 0,302:

| | Media de P(empate) | Máximo | Veces que es argmax |
|---|---|---|---|
| Poisson bivariado | 0,237 | **0,302** | **0** |
| `xgb_gbt` | 0,238 | 0,421 | 5 |
| **Mercado real** | 0,248 | **0,312** | **0** |

Un modelo que *deduce* el empate de la estructura del marcador —y que no puede "elegir" no
predecirlo— llega al mismo lugar que las casas de apuestas. **No es un artefacto del
clasificador: es una propiedad del fútbol.** Con local ~43 %, visitante ~30 % y empate
~25 %, el empate nunca es el resultado más probable.

### Logit ordinal: el empate como franja, no como clase

Aprovecha que las clases tienen orden natural (derrota < empate < victoria) sobre un eje
latente de superioridad. Entrena dos umbrales acumulados y define `P(empate)` como la
franja entre ambos. Es el único que predice empates en cantidad (70) y tiene el mejor F1 de
esa clase (0,184) — pero paga con la peor accuracy y el peor log-loss. Sirve como
recordatorio de que forzar la clase del medio tiene un costo.

---

## La grilla completa: modelo x datos de entrenamiento

`python -m training.compare_models`. Las rondas anteriores dejaron un hueco: las ablaciones
de datos se midieron con el feature set viejo (143 columnas, **sin Elo**) y los modelos
alternativos con el nuevo (159) **pero entrenando con todo**. Nunca se cruzaron, y al
cruzarlos una conclusión cambia.

### Accuracy — mercado 0,4947, prior de clase 0,4263

| Modelo | todo (1.140) | sin_xg_falso (1.004) | sin_2022_23 (760) |
|---|---|---|---|
| `xgb_gbt` | 0,4868 | **0,5132** | 0,5000 |
| `xgb_rf` | 0,5000 | 0,4921 | 0,4947 |
| `hgb` | 0,5053 | 0,4895 | 0,4974 |
| `logreg` | 0,4316 | 0,4342 | 0,4553 |
| `poisson` | 0,4947 | 0,4974 | 0,4737 |
| `ordinal` | 0,4316 | 0,4658 | 0,4553 |
| `mlp` | 0,4579 | 0,4868 | 0,4579 |

### Log-loss — mercado 1,0118, prior 1,0845

| Modelo | todo | sin_xg_falso | sin_2022_23 |
|---|---|---|---|
| `xgb_gbt` | 1,0379 | 1,0364 | 1,0351 |
| `xgb_rf` | 1,0293 | 1,0315 | 1,0310 |
| `hgb` | 1,0304 | 1,0365 | **1,0277** |
| `logreg` | 1,1446 | 1,1437 | 1,1865 |
| `poisson` | 1,0379 | 1,0390 | 1,0427 |
| `ordinal` | 1,1312 | 1,1308 | 1,2234 |
| `mlp` | 1,0680 | 1,0679 | 1,0690 |

**Mejores combinaciones:** `xgb_gbt` + `sin_xg_falso` para accuracy (0,5132) y `hgb` +
`sin_2022_23` para log-loss (1,0277).

### Corrección: sacar 2022-23 entera ya no es la mejor idea

| Modelos que mejoran respecto de entrenar con todo | Accuracy | Log-loss |
|---|---|---|
| `sin_xg_falso` | **5 de 7** | **4 de 7** |
| `sin_2022_23` | 3 de 7 | 2 de 7 |

Medido con **143 features**, sacar la temporada entera ganaba claro: 0,505 contra 0,490.
Medido con **159** —tras agregar Elo— esa ventaja se diluye, y encima destroza a los
modelos lineales (`logreg` pasa de 1,1446 a 1,1865 de log-loss; `ordinal` de 1,1312 a
1,2234).

La explicación es que **el Elo ya codifica la fuerza de largo plazo** que aportaba la
temporada extra, así que su valor marginal cae; pero el ruido de sacar un tercio de los
datos sigue igual. Es una interacción entre features y datos que sólo se ve corriendo la
grilla, no probando una cosa a la vez.

**Lo que sí aguanta es la versión quirúrgica de la idea:** sacar únicamente las fechas con
xG falso. Mejora 5 de 7 modelos, da el mejor resultado absoluto y no rompe a ninguno. La
diferencia con la versión agresiva es que ahí se saca lo que está *mal medido*; acá se
sacaba también lo que estaba bien.

Nótese además que **la limpieza de datos ayuda más a los modelos débiles**: `mlp` sube de
0,4579 a 0,4868 y `ordinal` de 0,4316 a 0,4658, mientras que los boosting apenas se mueven.
Un modelo con menos capacidad de ignorar ruido depende más de que el dato esté limpio.

---

## Dónde le gana el modelo a cada vara

`python -m training.analysis`. Un promedio global no dice si el modelo sirve; lo que
importa es **en qué situaciones** gana.

### Fecha a fecha

| Le gana a | % de las 38 fechas |
|---|---|
| "Siempre local" en accuracy | **55,3 %** |
| El mercado en accuracy | **23,7 %** |
| El mercado en log-loss | **39,5 %** |

### Por favoritismo del mercado

| Tramo | n | Modelo | Mercado | Diferencia |
|---|---|---|---|---|
| muy parejo (<40 %) | 60 | 0,333 | 0,400 | **−0,067** |
| parejo (40-50 %) | 129 | 0,442 | 0,426 | +0,016 |
| favorito claro (50-60 %) | 98 | 0,500 | 0,480 | +0,020 |
| cantado (>60 %) | 93 | 0,656 | 0,667 | −0,011 |

El modelo aporta algo en la franja intermedia y **pierde claramente en los partidos muy
parejos**, que son justo los que más margen dejarían.

### La prueba decisiva: cuando el modelo discrepa del mercado

| Situación | n | Acierta el modelo | Acierta el mercado |
|---|---|---|---|
| Coinciden | 328 | 0,515 | 0,515 |
| **DISCREPAN** | **52** | **0,346** | **0,365** |

**Cuando el modelo tiene opinión propia, se equivoca más que el mercado.** Ésta es la
medición que explica el ROI negativo mejor que ninguna otra: el modelo no tiene una ventaja
informativa sobre las casas de apuestas. Reproduce bien lo que el mercado ya sabe y agrega
ruido cuando se aparta.

Para la propuesta de valor del bloque 1 —ganar plata con apuestas— **la conclusión honesta
es que el sistema todavía no la sostiene.**

### Dónde se pierde el log-loss

| Resultado real | n | Log-loss modelo | Log-loss mercado | Diferencia |
|---|---|---|---|---|
| away | 114 | 1,025 | 1,039 | **−0,014** |
| draw | 104 | **1,472** | 1,388 | **+0,085** |
| home | 162 | 0,770 | 0,751 | +0,019 |

El empate es donde más se pierde: es la clase peor estimada, y por lejos.

### ¿Hace falta predecir el empate?

Sí, aunque nunca sea el argmax, y por una razón aritmética: **las tres probabilidades suman
1**. Subestimar el empate en 3,6 puntos —que es lo que hace el modelo— reparte esos puntos
entre local y visitante, e infla el valor esperado de *todas* las apuestas. Con cuotas de 2
a 4, un punto de probabilidad de más son 2 a 4 puntos de EV inflado, y el umbral de apuesta
son 5 puntos: alcanza para disparar apuestas que no tenían valor.

Corregir ese sesgo mejora el log-loss (1,0387 → 1,0331), aunque no arregla el ROI.

**Sobre apostar sólo al empate.** En una corrida dio ROI +0,092 y parecía la única
estrategia rentable. Medido con cuidado, no lo es:

```
xgb_gbt, apostando sólo al empate:
  umbral EV 0,00   n=125   ROI −0,001 ± 0,160
  umbral EV 0,05   n=101   ROI +0,029 ± 0,184
  umbral EV 0,10   n= 86   ROI +0,018 ± 0,199
```

**El error estándar se come el resultado.** Con cuotas medias de 4,4 la varianza por apuesta
es enorme y cien apuestas no alcanzan para distinguir +3 % de 0. Es exactamente el tipo de
conclusión apresurada que el resto del proyecto se ocupa de evitar.

---

## Cuanto tarda entrenar, y por que tan poco

| Modelo | Un fit | Nota |
|---|---|---|
| `xgb_gbt` | **8,4 s** con 2.000 rondas | en el pipeline el early stopping corta en ~150, o sea **~0,8 s** |
| `xgb_rf` | 5,6 s | 300 arboles en paralelo, una sola ronda |
| `rf_sklearn` | 1,0 s | |
| `hgb` | 0,5 s | |
| `logreg` | 0,2 s | |

Que sea rapido no es sospechoso, es aritmetica: **el dataset son 624 KB** (1.004 filas x
159 features en float32), los arboles tienen profundidad 3, y el early stopping corta en
~150 rondas de 2.000. Un reentrenamiento completo del modelo de produccion —la pasada de
early stopping mas el refit de 5 semillas— tarda unos 10 segundos.

Esto es lo que hace **viable el reentreno semanal** del bloque 9: el costo de computo es
irrelevante. Lo caro no es entrenar, es *decidir si el modelo nuevo es mejor* — y para eso
no alcanza con una fecha (ver la regla de promocion).

---

## Combinar modelos: probado, y empeora

La pregunta natural es si apilar modelos ayuda. Se implemento stacking como corresponde
—predicciones **out-of-fold con folds temporales** de los modelos base, y un meta-modelo
que aprende a combinarlas— no un simple promedio.

| | Accuracy | Log-loss |
|---|---|---|
| base: `xgb_gbt` | **0,500** | 1,0428 |
| base: `xgb_rf` | 0,4895 | 1,0323 |
| base: `poisson` | 0,4974 | 1,0400 |
| base: `ordinal` | 0,4632 | 1,1713 |
| promedio simple | 0,4921 | **1,0319** |
| stacking, meta lineal | 0,4947 | 1,0332 |
| **stacking, meta XGBoost** | **0,4737** | **1,0557** |
| stacking, meta + features originales | 0,4789 | 1,0477 |

**El stacking no mejora, y cuanto mas flexible el meta-modelo, peor.** La razon es el
tamano: el meta-modelo solo dispone de **793 filas out-of-fold** para aprender a combinar
12 probabilidades. Con eso, un XGBoost como meta sobreajusta la relacion entre modelos base
y pierde tres puntos de accuracy.

El promedio simple empata al mejor modelo individual en log-loss (1,0319 contra 1,0323) sin
aportar nada. Conclusion: **un solo modelo bien regularizado le gana a cualquier
combinacion**, a esta escala.

---

## La confianza: mal calibrada, pero util para seleccionar

Accuracy por tramo de confianza en el holdout:

| Confianza | n | Accuracy | Confianza media | Desvio |
|---|---|---|---|---|
| < 0,40 | 34 | 0,559 | 0,376 | **+0,18** |
| 0,40 - 0,45 | 70 | 0,471 | 0,426 | +0,05 |
| 0,45 - 0,50 | 66 | **0,394** | 0,477 | **−0,08** |
| 0,50 - 0,60 | 123 | 0,496 | 0,541 | −0,05 |
| > 0,60 | 87 | **0,632** | 0,671 | −0,04 |

La correlacion entre confianza y acierto es apenas **0,098**, y en la franja media el modelo
se **sobreconfia**: dice 0,477 y acierta 0,394.

Pero como criterio de **seleccion** si funciona:

| Si solo predijeramos... | Accuracy |
|---|---|
| el 25 % mas confiado | **0,642** |
| el 50 % mas confiado | 0,568 |
| el 75 % mas confiado | 0,519 |
| todos | 0,511 |

Es directamente aplicable al bloque 6: **actuar solo donde el modelo esta seguro**. Y explica
el caso de FUL-CHE, un partido de ida y vuelta que termino 2-3: el modelo dio
`0,377 / 0,260 / 0,363` — practicamente un empate a tres bandas. **Detecto que era
impredecible**, aunque el argmax fallara.

### `sorpresa_u5` y `sorpresa_u10`

De ahi sale una feature nueva: cuanto se apartaron los ultimos N resultados de un equipo de
lo que el Elo esperaba, `|real − esperado|` promediado. Mide **que tan impredecible viene
siendo**, no en que direccion.

Es la version legitima de "que tan bien viene acertando el modelo": usar las predicciones
propias como feature seria un bucle de realimentacion, y ademas imposible de calcular en
entrenamiento sin leakage. La expectativa del Elo sale solo de resultados pasados.

Validacion, sin ajustar nada: los mas impredecibles de 2025-26 fueron **CHE, NEW y AVL**;
los mas predecibles **BUR y BRE** — ser consistentemente malo tambien es predecible.

Aporte medido: log-loss 1,0318 -> 1,0303. Marginal, dentro del ruido. Se conserva porque el
costo es cero y porque es la unica feature que habla de *confiabilidad* en vez de direccion.

---

## Dos modelos: el que se reporta y el que sirve

Con 1.004 filas de entrenamiento, mantener 380 partidos afuera para siempre es caro — y
son ademas **los mas recientes**, los mas parecidos a lo que viene. Pero incorporarlos
destruye la evidencia. La salida es tener los dos, con la separacion explicita:

| | Entrena hasta | Filas | Para que |
|---|---|---|---|
| **evaluacion** | 2024-25 | 1.004 | **Es el que se REPORTA.** Justifica cada decision de disenio. `python -m training.run --sin-holdout` |
| **produccion** | 2025-26 | **1.384** | **Es el que SIRVE.** Su unico test honesto es la temporada en curso |

El holdout se reservo para **elegir**: modelo, features, hiperparametros. Esa funcion ya la
cumplio. Lo que no se puede hacer es seguir usandolo como prueba de generalizacion despues
de haberlo entrenado — por eso el `metadata.json` guarda `incluye_holdout` y
`metricas_son_de_generalizacion`, y hay un test que verifica la coherencia.

**La temporada en curso NUNCA entra**, ni siquiera en produccion. Sus partidos jugados
entran a Gold para servir de historia a las fechas siguientes, pero usarlos como objetivo
dejaria al proyecto sin ninguna evaluacion limpia. Tambien hay un test.

### El test disponible: la GW1 de 2026-27

Diez partidos que ninguna de las dos versiones vio.

| | Filas de train | Aciertos | Accuracy | Log-loss |
|---|---|---|---|---|
| evaluacion (hasta 24-25) | 1.004 | **5/10** | 0,500 | 1,1435 |
| **produccion (hasta 25-26)** | **1.384** | 4/10 | 0,400 | **1,0108** |

Las dos lecturas apuntan en direcciones opuestas, y vale entender por que:

- **La diferencia en aciertos es UN partido**: FUL-CHE, donde evaluacion dio 0,369 al
  visitante y produccion 0,337 al local. Los dos, practicamente un empate a tres bandas.
  Con n=10 un partido son diez puntos de accuracy: es ruido puro.
- **El log-loss mejora 12 %** (1,144 -> 1,011). Esa metrica usa la probabilidad completa
  en vez del argmax, asi que con muestras chicas es mucho menos ruidosa. Los 380 partidos
  extra mejoraron la **calibracion**.

La conclusion razonable es que sumar el holdout ayuda, pero con diez partidos no se puede
afirmar. Por eso existe `monitoring/temporada_actual.py`: la respuesta se acumula sola,
diez partidos por semana.

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
