# Plan de mejoras: cuatro fases, un protocolo, y el criterio fijado de antemano

Este documento gobierna las próximas fases del modelo. Existe por una razón concreta: en el
proyecto ya pasó **dos veces** que un número prometedor no sobrevivió a mirarlo bien.

- La búsqueda de hiperparámetros con CV temporal encontró una config con log-loss 0,970
  contra 1,013 de la actual, y en el holdout dio **peor** (1,041 vs 1,032).
- El umbral de empate 0,30 daba +0,0079 de accuracy, y resultó ser **ruido de semilla**: el
  delta va de −0,005 a +0,026 según qué semilla se use, y la corrida por defecto (`seed=42`)
  era la mejor de cinco.

Las dos veces el problema fue el mismo: **se decidió después de ver el número.** Este plan
fija el protocolo y el criterio *antes*.

---

## Regla cero: no se pisa nada, nunca

**Ninguna fase puede destruir el estado anterior.** Ni una versión de Gold, ni una de
Silver, ni un modelo. Si una fase sale mal, tiene que poder deshacerse entera.

Hasta el 05/09/2026 esto **no estaba garantizado**. El inventario era:

| | ¿se pisaba? |
|---|---|
| Bronze | no — append-only por `ingested_at=<stamp>` desde el día uno |
| Modelos | no — cada corrida es `models/<modelo>/<stamp>/` |
| Predicciones | no — cada una lleva su stamp en el nombre |
| **Silver y Gold** | **sí — `write_table` escribía encima** |

Y como `data/` está en `.gitignore` —tiene que estarlo, son 180 MB regenerables— para esas
dos capas **no había red de git**. Una corrida distraída de `python -m features.gold_tp`
destruía sin rastro el Gold con el que se entrenó el modelo que está sirviendo.

Ahora `common.storage.archivar` aparta la versión vigente antes de cada escritura:

    data/gold/gold_tp_match.parquet                      <- la vigente
    data/_versiones/gold/gold_tp_match/<stamp>.parquet   <- todas las anteriores
    data/_versiones/gold/gold_tp_match/<stamp>.json      <- qué era cada una

El histórico vive **fuera** de `data/silver` y `data/gold` para que las carpetas de capa
sigan teniendo exactamente una versión de cada tabla, y el lab de GCP suba sólo lo vigente.

    python -m common.versiones                        # que hay guardado
    python -m common.versiones --diff gold_tp_match   # que cambio entre versiones
    python -m common.versiones --restaurar gold_tp_match <stamp>

**Antes de cada fase**, la primera acción es etiquetar el estado del que se parte:

    python -m common.versiones --snapshot "antes de fase N: <que se va a cambiar>"

o, si la fase reconstruye Gold, dejar que lo haga el propio pipeline:

    TP_VERSION_LABEL="fase N: pi-ratings" python -m features.gold_tp

La etiqueta es lo que después permite contestar *"¿con qué Gold se entrenó el modelo
`20260825T024144Z`?"* sin adivinar por fecha. Un parquet archivado sin etiqueta es un
parquet más.

> El estado previo a todas estas fases ya quedó congelado con la etiqueta
> `"estado previo a las fases de mejora (commit 5398fd3)"`: las 6 tablas de Silver, el Gold
> de 1.540 × 301 y el `prior_ascendidos.json` del modelo en producción.

Y `--restaurar` tampoco destruye: archiva la versión que está viva antes de reemplazarla,
así que ir y volver entre versiones es seguro.

---

## El protocolo, por fase

Cada fase se mide en tres bancos. Los tres, siempre, aunque el cambio "obviamente" funcione.

### Banco A — holdout fijo (el que se reporta)

Train 2022-23 → 2024-25 (1.140 partidos), test 2025-26 (380).

    python -m training.run --sin-holdout

- **Métrica primaria: RPS.** Es ordinal —sabe que el empate está en el medio— y permite
  compararse con las Soccer Prediction Challenges. Ver `training/metrics.rps`.
- Secundarias: accuracy con **McNemar pareado** contra el modelo de la fase anterior, y
  log-loss (que es el que le importa a la capa de apuestas).
- **Con control de semillas, siempre.** Se corre con *k* semillas y se reporta **media ±
  desvío**, nunca un número solo. Es la lección del umbral, y sin esto el banco miente.

### Banco B — walk-forward sobre 2025-26

Reentrena en cada fecha y predice la siguiente: es la simulación del ciclo operativo.

    python -m training.decision_eval --walk-forward     # o evaluate.walk_forward

No es evidencia independiente del Banco A —son los mismos 380 partidos— pero es el
protocolo más parecido a producción. Sirve para detectar un cambio que ayuda entrenando una
vez y no ayuda reentrenando semanalmente.

### Banco C — temporada en curso (2026-27)

Entrena hasta 2025-26 inclusive y predice fecha a fecha.

    python -m monitoring.temporada_actual

**⚠️ Las fechas 1 y 2 ya no son ciegas.** Se miraron muchas veces: 4/10 y 6/10, el
retrospectivo del umbral, la comparación de modelos de 192 contra 279 features. Usarlas como
evidencia de que una fase funcionó sería exactamente el error que este documento existe para
evitar. **La muestra virgen arranca en la fecha 3.**

Lo que el Banco C sí aporta y ningún otro puede dar es el **cold-start de temporada**: es el
único lugar donde se ven ascendidos reales (Coventry y Hull no tienen ningún partido en la
ventana ingestada) y ratings recién regresados a la media. Para eso las fechas 1-2 siguen
sirviendo **como diagnóstico**, porque el fallo es estructural y está identificado: los dos
errores más caros de la GW1 fueron ascendidos ganando de local (HUL-MUN, IPS-SUN) con el
modelo respaldando al equipo establecido.

Dos unidades de medida por fecha, porque dicen cosas distintas:

- **aciertos sobre 10** — legible, pero se satura y con n=10 el error estándar es ±15,7 puntos;
- **log-verosimilitud conjunta de la fecha** — la suma de los `log p` de los diez resultados.
  Un solo número por fecha, con mucha más resolución: distingue "acertó 6 con dudas" de
  "acertó 6 con convicción", y no necesita que el argmax cambie para moverse.

---

## El criterio de adopción, fijado antes de correr

Un cambio se adopta si cumple **las tres**:

1. **Banco A**: el delta de RPS supera el desvío entre semillas. El signo no alcanza.
2. **Banco B**: no empeora.
3. **Si el cambio apunta a un subgrupo** (ascendidos, arranque de temporada), mejora **en ese
   subgrupo**. Ahí es donde vive la hipótesis y donde hay más señal por fila; pedirle a un
   cambio dirigido que mueva la métrica global es pedirle que mueva 1.140 partidos con un
   arreglo que afecta a 40.

Si no las cumple: **se registra el rechazo** en `models/xgb_gbt/attempts.jsonl` y se sigue.
Ya hay precedente (`decision:umbral_empate_030`), y esa entrada vale tanto como una mejora:
es la evidencia de que el pipeline sabe decir que no.

### Una regla contra el multiple testing

**Cada fase es UNA decisión sobre el Banco A.** No se prueban seis variantes y se elige la
mejor: los hiperparámetros de cada fase (λ y γ de pi-ratings, la profundidad del burn-in, el
offset de división) se fijan por **validación temporal dentro del train**, y al Banco A llega
**una sola** configuración. El holdout decide *si* la fase entra, no *cuál* de sus variantes.

Es la misma disciplina que ya salvó al proyecto: el holdout 2025-26 se reservó para elegir, y
el modelo de producción que lo incluye declara `metricas_son_de_generalizacion: false`.

---

## ¿Anidadas o cada una por separado?

**Las dos cosas, y no es indecisión: depende de si los cambios son independientes.**

### Las fases 1-3 van ANIDADAS, porque tocan el mismo bloque

Historia profunda, pi-ratings y ratings de ataque/defensa **no son features que se suman**:
son tres versiones del mismo bloque, el de rating.

- pi-ratings no se agrega *al lado* del Elo actual: lo **reemplaza** en buena parte. Meter los
  dos daría dos columnas con correlación altísima, y el proyecto ya midió qué pasa con eso
  (24 pares con r > 0,95: la accuracy no cambia, pero **se diluyen las importancias**).
- La historia profunda no agrega columnas: cambia el **insumo** de cualquier rating que se
  use después. Medirla "por separado" de pi-ratings no tiene sentido, porque pi-ratings sobre
  historia corta y sobre historia larga son dos features distintas.

Así que van en orden, cada una sobre el mejor stack vigente. Y si una fase no pasa el
criterio, la siguiente se construye sobre el stack **anterior**, no sobre la rechazada.

### La fase 4 va SUELTA, porque es un bloque nuevo

Los estilos y sus interacciones no tocan el rating: son features nuevas sobre las 56 de Opta.
Se miden contra el mejor stack vigente al momento de llegar.

### Y al final, una ablación HACIA ATRÁS

El marginal de un cambio anidado depende del orden en que entró. Para la atribución final —lo
que va al informe— se corre `python -m training.ablacion` sobre el set completo: sacar cada
bloque del modelo terminado y ver qué se pierde. Eso sí es independiente del orden, y es la
tabla que contesta "¿cuánto aportó cada cosa?".

---

## Las cuatro fases

### Fase 1 — Historia profunda para los ratings · ❌ RECHAZADA (05/09/2026)

**No entra.** Se construyó entera —37.840 partidos de E0/E1/E2 desde 2000, la separación
entre divisiones emergiendo sola de los ascensos— y no pasó ninguno de los tres criterios:

- Banco A: delta RPS **+0,0010** ± 0,0011 (peor), cruza el cero entre semillas; McNemar
  p ∈ [0,146 – 1,000].
- Banco B: delta RPS **+0,0006** (peor).
- Subgrupo objetivo: ascendidos 0,4796 → **0,4778**; arranque 0,5320 → **0,5240**.

Y el diagnóstico decisivo: el Elo sembrado es **peor como predictor por sí solo**, con AUC
local en partidos de ascendidos cayendo de 0,6964 a 0,6620 — o sea que la hipótesis es falsa,
no la implementación. La explicación probable es que el 1500 de un ascendido no era
ignorancia sino *shrinkage*, y `features/cold_start.py` ya cubría esa información con un
prior ajustado.

La infraestructura queda en el repo detrás de `features.elo_sembrado_con_historia: false`,
porque la Fase 2 la usa. Detalle completo en `training/README.md` y en `attempts.jsonl`.

> **Lo que la fase sí dejó, y vale:** el protocolo funcionó. Se etiquetó el estado previo, se
> midió con control de semillas, se miró el subgrupo, se restauró el Gold anterior y se
> registró el rechazo. La versión sembrada quedó archivada, así que la Fase 2 puede volver a
> ella sin recalcular nada.

<details>
<summary>El planteo original de la fase</summary>

#### Historia profunda para los ratings (no como filas de entrenamiento)

**Qué NO es:** entrenar el clasificador con partidos de 2006. Cambió el fútbol, cambiaron los
planteles, y las features ricas (xG, FPL) no existen ahí. Descartado.

**Qué es:** usar los resultados crudos de `football-data.co.uk` —que ya se ingestan, gratis,
~20 temporadas y varias divisiones— **sólo para alimentar el rating**, con las filas de
entrenamiento intactas en 2022-23 → 2025-26.

Dos cosas que arregla:

1. **El burn-in.** Hoy el Elo arranca a todos en 1500 en 2022-23 y regresa 25 % por temporada.
   En la GW1 de 2022-23 el rating **no contiene información** y tarda media temporada en
   converger. Con historia previa llega ya convergido.
2. **Los ascendidos**, que es el fallo medido. `football-data.co.uk` sirve también E1/E2. Un
   ascendido entraría con el rating construido con su temporada real en el Championship,
   corregido por un **offset de división** estimado con lo que históricamente hicieron los
   ascendidos — en vez de 1500 y el parche de `features/cold_start.py`.

**Subgrupo objetivo:** partidos de equipos ascendidos, y partidos de GW ≤ 5.

</details>

### Fase 2 — pi-ratings · ❌ RECHAZADA (05/09/2026)

**No entra.** Implementada completa, con λ=0,20 y γ=0,70 ajustados dentro del train (MAE
1,3839 goles contra 1,4833 de la vara trivial). Ninguno de los tres criterios:

- Banco A: delta RPS **+0,0002** ± 0,0013, cruza el cero; McNemar p ≥ 0,30 en las 5 semillas.
- Banco B: delta RPS **+0,0006**.
- Subgrupo objetivo: ascendidos 0,4796 → **0,4704**; arranque 0,5320 → **0,5200**.

Y el diagnóstico que la cierra: **la ventaja de localía por equipo no persiste**, r = 0,093
entre las dos mitades de los datos, contra r = 0,746 de la misma prueba sobre el nivel del
equipo. La dispersión observada es ruido de estimación, no un rasgo del club — o sea que la
constante `VENTAJA_LOCAL = 65` del Elo, que parecía una simplificación grosera, es
defendible con los datos que hay.

Detalle en `training/README.md`, incluida la tensión de que el ajuste sí rechaza γ=1,0.
Queda detrás de `features.pi_ratings_activo: false`.

<details>
<summary>El planteo original de la fase</summary>

#### pi-ratings

Cada equipo con **dos** ratings: local y visitante. Se actualizan con el error en diferencia
de goles, amortiguado. Dos parámetros: `λ` (cuánto se corrige el rating que jugó) y `γ`
(cuánto se transfiere al otro rating del mismo equipo).

Lo que compra: hoy la ventaja de localía es `VENTAJA_LOCAL = 65.0`, **una constante idéntica
para los 20 equipos**. Con pi-ratings pasa a ser una propiedad medida de cada club.

Referencia: Constantinou & Fenton (2013). El mejor modelo ML de la Soccer Prediction
Challenge 2023 fue CatBoost sobre pi-ratings.

</details>

### Fase 3 — Ratings de ataque y defensa (Berrar)

Ratings separados de ataque y defensa, actualizados con goles marcados y concedidos contra lo
esperado. Ganó la Challenge 2017 (k-NN sobre estas features).

**Precondición, antes de escribir una línea:** medir la correlación de `pts_arq/def/med/del_u5`
contra `gc_u5`, `xgc_u5`, `tasa_atajadas_u5` y `gf_u5`. El sistema de puntaje de FPL es una
función de goles, asistencias y vallas invictas con pesos por posición —un defensor cobra 6
por gol y 4 por valla— así que **`pts_def` puede ser casi el récord defensivo, que ya está**.
Si `pts_med` sobrevive con correlación baja, ahí sí hay una dimensión propia (creación) y el
rating de mediocampo vale. Si no, es `dif_elo` disfrazado.

> Ojo con la asimetría: ataque y defensa tienen base observable (goles marcados y concedidos,
> atribuibles por separado). **Mediocampo no**: no hay "goles de mediocampo". Ese rating se
> apoya en el puntaje de FPL, que es un constructo, no una medición.

### Fase 5 — Inversión del plantel: el dato que sí existe para un ascendido

> Idea de Nico (05/09/2026). Agregada después de medir el problema de los ascendidos y ver
> que lo único que sobrevive al escrutinio es que **el mercado tiene información de plantel
> que nosotros no**: altas, bajas, lesiones, precios de transferencia.

**La intuición.** Un ascendido es el equipo con mayor rotación de plantel de la liga, y casi
todas nuestras features salen del historial de sus jugadores *en Premier*, que no existe.
Hace falta una medida de **cuánto vale el plantel** que esté disponible el día cero.

**La fuente principal: valores y transferencias reales de Transfermarkt.** Scrapear
Transfermarkt directamente va contra sus términos y no tiene API pública, pero existe
**[`dcaribou/transfermarkt-datasets`](https://github.com/dcaribou/transfermarkt-datasets)**
— un dataset derivado, mantenido y **actualizado semanalmente**, publicado como CSV en
GitHub y Kaggle. Doce tablas, y las tres que importan acá:

| tabla | qué trae |
|---|---|
| `player_valuations` | **520.000+ valuaciones históricas fechadas** |
| `transfers` | 99.000+ transferencias con monto |
| `clubs` / `players` | plantel por club y temporada |

**Y cubre el Championship.** Eso es lo decisivo: un ascendido llega con valores de mercado
reales de *antes* de pisar la Premier, que es exactamente la información que el mercado de
apuestas tiene y nosotros no. Ninguna otra fuente del proyecto puede dar eso.

Que las valuaciones estén **fechadas** es lo que la hace usable sin leakage: se toma la
vigente al `corte` de cada partido, por el mismo `merge_asof` que todo lo demás. Hay que
verificarlo igual, como con cualquier fuente nueva.

**El precio FPL, como análogo barato y como control.** `fact_player_gw.value` ya está
ingestado —precio FPL por jugador y por fecha, todas las temporadas— y FPL le pone precio a
todos los jugadores de Premier antes de la primera fecha. No es valor de mercado sino valor
*fantasy*, así que no reemplaza a Transfermarkt; sirve para (a) tener la feature funcionando
en un día, sin fuente nueva, y (b) **contrastar**: si las dos versiones dicen lo mismo, la
señal es robusta; si difieren, la diferencia misma es informativa.

**Medido antes de planificarlo**, con el análogo FPL (suma del precio de los 15 más caros
por equipo en la GW1,
z-score dentro de la temporada, contra los puntos finales):

| temporada | r |
|---|---|
| 2022-23 | 0,542 |
| 2023-24 | 0,882 |
| 2024-25 | 0,665 |
| 2025-26 | 0,659 |
| **global** | **0,690** (n=80) |

Y separa exactamente donde hace falta:

| | inversión GW1 (z) | puntos finales |
|---|---|---|
| ascendidos (n=9) | **−1,06** | 27,6 |
| resto (n=71) | +0,13 | 55,6 |

**Qué construir.** Valor del plantel total y **por línea** (arq / def / med / del), en dos
momentos, con las dos fuentes (Transfermarkt como principal, FPL como análogo):

- **`inv_*_inicial`** — la de la GW1 de cada temporada. Es la información genuinamente
  nueva: un juicio externo sobre el plantel, disponible antes de cualquier partido.
- **`inv_*_actual`** y **`inv_*_delta`** — la de la fecha en curso y cuánto se movió desde
  el arranque. Cubre el refuerzo de mercado de invierno, que es la otra mitad del pedido.

Más los derivados que dan la forma del plantel y no sólo su tamaño: proporción por línea
(¿el equipo invirtió en defensa o en ataque?), concentración (una estrella o once parejos),
y la diferencia entre los dos equipos, que es lo que el modelo realmente usa.

Y con Transfermarkt se suma lo que FPL no puede dar: **el gasto neto del mercado de pases**
(altas menos bajas, del verano y del invierno), que es la medida directa de "cuánto invirtió
este equipo" que Nico pidió, y que para un ascendido es justamente donde se juega la
temporada.

**Las advertencias, al frente:**

1. **El precio FPL no es valor de mercado.** Es el valor *fantasy* estimado por FPL: un
   defensor barato de un equipo ordenado cobra vallas invictas y sube de precio sin que su
   valor de transferencia se mueva. Correlaciona con calidad, pero es otro constructo — por
   eso es el análogo, no la fuente.
1b. **Transfermarkt trae una identidad de club nueva que hay que mapear.** Es exactamente
   donde el proyecto ya se comió un bug (Coventry / Hull / Ipswich sin historia en la Fase 1
   por hacer match exacto en vez de usar el resolvedor). El control de "ningún equipo de la
   ventana se queda sin datos" va desde el primer día, no al final.
2. **El precio de mitad de temporada se mueve por popularidad**, no sólo por calidad: sube
   cuando lo compran muchos managers, o sea *después* de que rinda. Riesgo de que
   `inv_actual` sea un eco tardío de features que ya tenemos. El de la GW1 no tiene ese
   problema, y es el que carga la hipótesis.
3. **Deriva de escala entre temporadas** (la media pasa de 48,2 a 51,6): hay que normalizar
   dentro de cada temporada, nunca usar el valor crudo.
4. **No es leakage** si se respeta el corte: el precio de la fecha N refleja hasta la N−1, y
   entra por el mismo `merge_asof` que todo lo demás. Hay que verificarlo igual.

**Subgrupo objetivo:** ascendidos, y arranque de temporada.

> Nota de prioridad: por lo medido, esta fase pinta más fuerte que la Fase 4. Va acá porque
> Nico pidió seguir el orden planificado, pero si la 2 y la 3 no dan, conviene saltear a
> ésta antes que a los estilos.

### Fase 4 — Estilos e interacciones de matchup

La hipótesis: el estilo A le gana al B más de lo que predicen sus ratings.

Dos advertencias que van al frente:

1. **XGBoost ya modela interacciones**: los árboles las capturan nativamente. La pregunta no
   es si el modelo puede verlas, es si alcanzan los datos para estimarlas. Con ~5 clusters de
   estilo son 25 celdas de enfrentamiento sobre 1.140 partidos: ~45 por celda. Es el régimen
   donde aparece un patrón espurio.
2. **El historial baja la expectativa**: las 24 features de competencias y las 56 de Opta se
   usan (9,5 % y 19,8 % de la ganancia) y **no** se tradujeron en mejora.

Por eso la versión barata primero: en vez de clusterizar estilos, **media docena de términos
de matchup explícitos y teóricos** sobre las features de Opta que ya están (centros del local
× duelos aéreos del visitante, altura de presión del local × precisión de pase del visitante).
Pocos parámetros, interpretables, y se testean de a uno con `training/ablacion.py`.

---

## Registro

Cada fase:

1. **Abre** con `--snapshot "antes de fase N: ..."` o con `TP_VERSION_LABEL` puesto. Regla
   cero: el estado del que se parte queda etiquetado antes de tocar nada.
2. Corre **los tres bancos**, con control de semillas.
3. **Cierra** con el resultado anotado —en `training/README.md` si entró, en
   `attempts.jsonl` si no— y **un commit**.

La fase siguiente arranca del stack que quedó, no del que se esperaba que quedara. Y si una
fase hay que deshacerla, `python -m common.versiones --restaurar` la deshace sin perder lo
que se probó en el intento.
