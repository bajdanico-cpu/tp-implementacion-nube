# Gold — capa de features

**Estado: implementado.** `python -m features.gold_tp` produce
`data/gold/gold_tp_match.parquet` — **1.520 filas × 165 columnas, 143 de ellas features**.

El diccionario completo, feature por feature con su fórmula, está en
[`docs/FEATURES.md`](../docs/FEATURES.md). **Ese archivo se genera** con
`python -m features.spec --docs` y hay un test que falla si queda desfasado del código.

```
silver.fact_player_gw  (jugador × fecha) ─┐
silver.fact_match      (partido)          ├─> features/gold_tp.py ─> 1 fila por partido
silver.fact_fixture    (fixture)          │
silver.dim_team                           ┘
```

---

## Módulos

| Archivo | Qué hace |
|---|---|
| `spec.py` | El contrato: qué columnas existen, cuáles son features, cómo se calcula cada una. Única fuente de verdad; genera `docs/FEATURES.md` |
| `player_agg.py` | `fact_player_gw` → estadísticas de equipo × fixture (xG, puntajes por línea, rotación) |
| `team_form.py` | Tabla larga equipo-partido, ventanas rodantes, acumulado de campeonato, el `merge_asof` anti-leakage |
| `h2h.py` | Historial entre los dos equipos, en dos variantes |
| `cold_start.py` | `es_ascendido` derivado y el prior de los recién ascendidos |
| `gold_tp.py` | Orquesta todo y corre los controles antes de escribir |

---

## Las cuatro reglas que no se negocian

**1 · El corte es el inicio de la fecha.**

```
corte(partido) = min(kickoff_time) de todos los partidos de su (temporada, gameweek)
```

Toda feature usa únicamente partidos **terminados antes** de ese momento. Es el criterio
más conservador: cuando se publica la tanda de predicciones de una fecha, ninguna usó
información de esa fecha.

**2 · El mecanismo es `merge_asof`, no `shift(1)`.**

> **shift cuenta partidos. merge_asof cuenta tiempo.**

Hay **85 pares (temporada, gameweek, equipo) donde el equipo juega dos veces en la misma
fecha** (42 en 2022-23, 23 / 10 / 10 después). Con `shift(1)`, el segundo partido usaría
el resultado del primero — que se jugó *después* del corte. Es leakage silencioso en ~5,6 %
de las filas. Con `merge_asof` los dos comparten el mismo vector de features, que es lo
correcto. De paso resuelve los partidos reprogramados de la GW7 de 2022-23, que no existe.

Verificado sobre datos reales: los dos partidos del Arsenal en la GW23 de 2022-23 (vs
Brentford y vs Man City) salen con `pts_u5`, `xg_u5`, `n_hist` y `hist_kickoff` idénticos.

**3 · Los controles corren ANTES de escribir**, no en los tests. `gold_tp.run()` llama a
`transform.leakage.assert_no_banned_columns`, verifica que `hist_kickoff_* < corte` en las
1.520 filas, y audita fecha por fecha contra el **deadline de FPL** — un criterio todavía
más estricto que el corte propio, porque cae 90 minutos antes. Margen mínimo medido:
**22,5 horas**. El detalle queda en `features/output/gold_audit.csv`.

**4 · Ventanas configuradas en `config.yaml`** → `features.rolling_windows` (hoy 3 y 5).

---

## Los cinco hallazgos que condicionaron el diseño

Medidos sobre Silver, no asumidos.

| # | Hallazgo | Qué se hizo |
|---|---|---|
| 1 | **El xG de 2022-23 viene hardcodeado en CERO hasta la GW15** — 0,0 para los 20 equipos, el 37,9 % de esa temporada | Se enmascara a NaN + flag `xg_available`. Como cero, el modelo aprendería que "xG bajo" y "arranque de temporada" van juntos: un artefacto del calendario de publicación de FPL |
| 2 | **`expected_goals_conceded` no se puede sumar**: se cuenta una vez por cada jugador del plantel. Medido en 2024-25, la suma da media **15,75** contra 1,47 goles concedidos reales | El xGC es **el xG del rival** en el mismo fixture (media 1,44, que sí calza) |
| 3 | **`dim_team.promoted` no significa "ascendido"**: significa "primera temporada en la ventana ingestada", y marca 1 equipo en 2024-25 cuando ascendieron 3 | `es_ascendido` se deriva como "no estaba en la temporada anterior" |
| 4 | **No hay histórico de disponibilidad.** `fact_player_gw` dice lo que cada jugador *hizo* (post-partido); el `status` de FPL dice quién estaba disponible *antes*, y la API sólo sirve el estado de hoy | Se reemplaza por proxies de rotación: `n_jugadores`, `mins_hhi` y `continuidad_plantel_u5` |
| 5 | **`fpl_player_id` no es estable entre temporadas**: el 90 % de los ids apunta a otro futbolista según el año (el id 1 es Cédric Soares en 2022-23 y David Raya en 2025-26) | La clave entre temporadas es `player_name`. Con el id, la continuidad de plantel daba 9,4 % en vez de 61,3 % |

---

## Pases de jugadores y cambios de temporada

**La pregunta:** si un jugador pasa del equipo A al B, ¿a quién le suman sus puntos?

**La respuesta cae del orden de las operaciones:**

```
paso 1: fact_player_gw  →  se agrega a (temporada, fixture, equipo)
paso 2: sobre ese resultado se calculan las ventanas
```

Como el paso 1 agrega **por partido**, cada jugador ya quedó atribuido al equipo con el que
efectivamente jugó ese día. Sus goles de agosto quedan para siempre en el historial de A y
los de febrero suman al de B: **ni doble conteo ni puntos huérfanos**. Y un jugador nuevo
suma a su equipo desde su primer partido, sin necesitar pasado propio.

**El límite real**, medido: entre temporadas sólo el **61,3 % / 66,0 % / 57,3 %** de los
minutos los juega gente que ya estaba en ese equipo. O sea que ~40 % rota cada año. Se
aborda con tres señales, no ignorándolo:

1. `continuidad_plantel_u5` — qué proporción de los minutos de la ventana la jugaron
   futbolistas que también jugaron el último partido. Leak-free, y se desploma justo
   cuando el plantel se renovó.
2. Ventanas intra-temporada (`_u5_temp`) — sólo la temporada actual, donde el plantel sí es
   el de hoy. NaN al arranque.
3. `pj_camp` — cuántos partidos lleva jugados, para saber cuánto confiar en la señal 2.

---

## Arranque de temporada y ascendidos

Los partidos donde el modelo está más ciego: **40 en la GW1 (2,6 %), 199 en GW≤5 (13,1 %)**.

- Las ventanas `_u3` / `_u5` **cruzan el borde de temporada**: siempre hay dato.
- Las `_u5_temp` **no lo cruzan**: quedan NaN en la fecha 1 para los veinte equipos por
  igual. Es un estado compartido y bien definido, no un faltante.
- El **prior de ascendidos** rellena sólo las ventanas cruzadas —las que a un recién
  llegado le quedarían vacías para siempre— y sólo donde `n_hist == 0`. Se ajusta
  **únicamente con las temporadas de train** y se congela en `data/gold/prior_ascendidos.json`
  y en el `metadata.json` del modelo: recalcularlo en serving sería train/serve skew.
- Coventry y Hull en 2026-27 son el caso extremo: cero partidos en la ventana.

---

## Lo que quedó afuera, y por qué

| Grupo | Motivo |
|---|---|
| **Cuotas** | No son features. Si el modelo las copia, el valor esperado da ~0 por construcción y el sistema nunca encontraría una apuesta con valor. Viven en Gold sólo para el baseline y la simulación de ROI, y hay un test que verifica que no entraron al feature set |
| **`strength_*`** | Skew de escala: promedia ~1.130 en 2022-26 y **2,85** en 2026-27, con `strength_attack_*` en cero para los 20 equipos. Entrenar con cuatro cifras y servir con un dígito sería train/serve skew silencioso. Hay un test que impide re-agregarlas sin discutirlo |
| **Disponibilidad** | No hay histórico (hallazgo 4) |

---

## Gold-FPL — pendiente

`gold_fpl.py` (una fila por jugador × fecha, para el armado del equipo de FPL) todavía no
está. Consume la misma `fact_player_gw`, que por eso conserva la granularidad fina.

⚠️ **Bloqueante para ese proyecto:** el hallazgo 5. Cualquier feature que siga a un jugador
entre temporadas tiene que usar `player_name`, nunca `fpl_player_id`.
