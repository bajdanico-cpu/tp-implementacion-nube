# Diccionario de features — Gold-TP

> **Archivo generado.** No lo edites a mano: se produce con `python -m features.spec --docs`
> a partir de `features/spec.py`, y hay un test (`test_docs_features_esta_sincronizado_con_spec`)
> que falla si quedan desfasados. Para cambiar una feature, tocá el spec.

## Cómo leer los nombres

```
{lado}_{estadística}_{ventana}

lado     local | visita        el equipo local y el visitante de ESE partido
ventana  u3 | u5               media de los últimos N partidos del equipo,
                               CRUZANDO el borde de temporada
         u5_temp               igual, pero sólo partidos de la temporada actual
         cond_u5               igual, pero sólo partidos en la misma condición
         camp                  acumulado de la temporada al momento del corte
```

**Todo es del equipo, mirando hacia atrás.** Ningún número describe el partido a predecir.
La construcción pasa por una tabla intermedia con grano equipo × partido (3.040 filas),
donde cada equipo tiene su propia fila; las ventanas se calculan ahí y recién al final se
pivotea a ancho, pegando la historia del local con prefijo `local_` y la del visitante con
`visita_`.

## La regla anti-leakage

```
corte(partido) = min(kickoff_time) de todos los partidos de su (temporada, gameweek)
```

Toda feature usa **únicamente partidos terminados antes del corte**. El corte es el inicio
de la fecha, así que todos los partidos de una misma gameweek se predicen con la misma
información — incluidas las 85 dobles fechas, donde un equipo juega dos veces en la misma
gameweek.

El mecanismo es `merge_asof`, no `shift(1)`: **shift cuenta partidos, merge_asof cuenta
tiempo.** Con shift, el segundo partido de una doble fecha vería el resultado del primero,
que se jugó después del corte.

Las columnas `hist_kickoff_local` / `hist_kickoff_visita` guardan el kickoff del último
partido efectivamente usado: son la **prueba auditable** de que no se miró el futuro, y hay
un assert que las verifica antes de escribir la tabla.

## Pases de jugadores y cambios de temporada

`fact_player_gw` se agrega primero a `(temporada, fixture, equipo)` y recién después se
calculan las ventanas. Como el primer paso agrega **por partido**, cada jugador ya quedó
atribuido al equipo con el que efectivamente jugó. Si un delantero pasa de A a B en enero,
sus goles de agosto quedan para siempre en el historial de A y los de febrero suman al de B:
**no hay doble conteo ni puntos huérfanos**, y un jugador nuevo suma a su equipo desde su
primer partido sin necesitar pasado.

El límite real: la forma pasada incluye gente que ya se fue. Medido, entre temporadas sólo
el **61,3 % / 66,0 % / 57,3 %** de los minutos los juega gente que ya estaba en el mismo
equipo. Por eso existen `continuidad_plantel_u5`, las ventanas `u5_temp` y `pj_camp`.

> ⚠️ **`fpl_player_id` NO es estable entre temporadas.** El 90 % de los ids (725 de 808)
> apunta a un futbolista distinto según el año: el id 1 es Cédric Soares en 2022-23 y David
> Raya en 2025-26. La clave entre temporadas es `player_name`.


## Resumen

- **Versión del feature set:** `v2.906eed82.220`
- **Features:** 220
- **Columnas totales:** 242 (220 features + 22 no-features)
- **Grano:** un partido = una fila. 1.520 filas (4 temporadas × 380).

| Grupo | Columnas |
|---|---|
| Forma reciente | 72 |
| Forma intra-temporada | 16 |
| Forma según condición | 12 |
| Puntaje de campeonato | 14 |
| Head-to-head | 10 |
| Continuidad de plantel | 2 |
| Elo y estado | 36 |
| Otras competencias | 24 |
| Contexto | 10 |
| Dificultad | 3 |
| Diferenciales | 21 |
| **TOTAL DE FEATURES** | **220** |

## Las 16 estadísticas base

Se calculan una vez por equipo y por partido, sobre la tabla larga de 3.040 filas. **Todas las features de forma salen de estas 16.**

| # | Nombre | Fórmula | Fuente |
|---|---|---|---|
| 1 | `pts` | 3 si ganó, 1 si empató, 0 si perdió | `fact_match` |
| 2 | `gf` | goles a favor | `fact_match` |
| 3 | `gc` | goles en contra | `fact_match` |
| 4 | `tiros` | tiros totales | `fact_match` |
| 5 | `tiros_arco` | tiros al arco | `fact_match` |
| 6 | `corners` | córners | `fact_match` |
| 7 | `faltas` | faltas cometidas | `fact_match` |
| 8 | `tarjetas` | amarillas + 2 x rojas | `fact_match` |
| 9 | `xg` | suma de expected_goals de los jugadores del equipo | `fact_player_gw` |
| 10 | `xgc` | el xg DEL RIVAL en ese mismo fixture. NO la suma de expected_goals_conceded, que se cuenta una vez por cada jugador del plantel e infla el valor x11 (medido: media 15,75 contra 1,47 goles concedidos reales) | `fact_player_gw` |
| 11 | `xa` | suma de expected_assists | `fact_player_gw` |
| 12 | `pts_arq` | suma de total_points de los jugadores con position == GK | `fact_player_gw` |
| 13 | `pts_def` | suma de total_points de position == DEF | `fact_player_gw` |
| 14 | `pts_med` | suma de total_points de position == MID | `fact_player_gw` |
| 15 | `pts_del` | suma de total_points de position == FWD | `fact_player_gw` |
| 16 | `n_jugadores` | cantidad de jugadores con minutes > 0 | `fact_player_gw` |
| 17 | `atajadas` | atajadas del arquero en el partido | `fact_player_gw` |
| 18 | `tasa_atajadas` | atajadas / (atajadas + goles recibidos): que proporcion de los remates al arco termina detenida. Es el equivalente defensivo de xg_por_tiro -- separa 'concede poco' de 'concede mucho pero lo atajan'. Envejece distinto que las otras defensivas: conceder pocos remates es estructural y persiste, mientras que una tasa de atajadas alta es en buena parte varianza del arquero y revierte a la media | `fact_player_gw` |

Más `dg`, derivada en la tabla larga: diferencia de gol del partido (gf - gc).

> **`total_points` está en `banned_columns` y esto no la viola.** Está prohibida como columna cruda del partido a predecir, donde es el resultado. Una media de partidos **anteriores** es una feature de forma, que es literalmente lo que pide el bloque 4 del canvas. Lo garantiza el `merge_asof`, y los nombres finales (`local_pts_def_u5`) no colisionan con la lista de prohibidas.

## Features, una por una

### Forma reciente — 72 columnas

| Columna | Lado | Ventana | Fuente | Cálculo |
|---|---|---|---|---|
| `local_pts_u3` | local | u3 | `fact_match` | 3 si ganó, 1 si empató, 0 si perdió. media de los últimos 3 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `local_gf_u3` | local | u3 | `fact_match` | goles a favor. media de los últimos 3 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `local_gc_u3` | local | u3 | `fact_match` | goles en contra. media de los últimos 3 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `local_tiros_u3` | local | u3 | `fact_match` | tiros totales. media de los últimos 3 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `local_tiros_arco_u3` | local | u3 | `fact_match` | tiros al arco. media de los últimos 3 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `local_corners_u3` | local | u3 | `fact_match` | córners. media de los últimos 3 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `local_faltas_u3` | local | u3 | `fact_match` | faltas cometidas. media de los últimos 3 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `local_tarjetas_u3` | local | u3 | `fact_match` | amarillas + 2 x rojas. media de los últimos 3 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `local_xg_u3` | local | u3 | `fact_player_gw` | suma de expected_goals de los jugadores del equipo. media de los últimos 3 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `local_xgc_u3` | local | u3 | `fact_player_gw` | el xg DEL RIVAL en ese mismo fixture. NO la suma de expected_goals_conceded, que se cuenta una vez por cada jugador del plantel e infla el valor x11 (medido: media 15,75 contra 1,47 goles concedidos reales). media de los últimos 3 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `local_xa_u3` | local | u3 | `fact_player_gw` | suma de expected_assists. media de los últimos 3 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `local_pts_arq_u3` | local | u3 | `fact_player_gw` | suma de total_points de los jugadores con position == GK. media de los últimos 3 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `local_pts_def_u3` | local | u3 | `fact_player_gw` | suma de total_points de position == DEF. media de los últimos 3 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `local_pts_med_u3` | local | u3 | `fact_player_gw` | suma de total_points de position == MID. media de los últimos 3 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `local_pts_del_u3` | local | u3 | `fact_player_gw` | suma de total_points de position == FWD. media de los últimos 3 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `local_n_jugadores_u3` | local | u3 | `fact_player_gw` | cantidad de jugadores con minutes > 0. media de los últimos 3 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `local_atajadas_u3` | local | u3 | `fact_player_gw` | atajadas del arquero en el partido. media de los últimos 3 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `local_tasa_atajadas_u3` | local | u3 | `fact_player_gw` | atajadas / (atajadas + goles recibidos): que proporcion de los remates al arco termina detenida. Es el equivalente defensivo de xg_por_tiro -- separa 'concede poco' de 'concede mucho pero lo atajan'. Envejece distinto que las otras defensivas: conceder pocos remates es estructural y persiste, mientras que una tasa de atajadas alta es en buena parte varianza del arquero y revierte a la media. media de los últimos 3 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `local_pts_u5` | local | u5 | `fact_match` | 3 si ganó, 1 si empató, 0 si perdió. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `local_gf_u5` | local | u5 | `fact_match` | goles a favor. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `local_gc_u5` | local | u5 | `fact_match` | goles en contra. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `local_tiros_u5` | local | u5 | `fact_match` | tiros totales. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `local_tiros_arco_u5` | local | u5 | `fact_match` | tiros al arco. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `local_corners_u5` | local | u5 | `fact_match` | córners. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `local_faltas_u5` | local | u5 | `fact_match` | faltas cometidas. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `local_tarjetas_u5` | local | u5 | `fact_match` | amarillas + 2 x rojas. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `local_xg_u5` | local | u5 | `fact_player_gw` | suma de expected_goals de los jugadores del equipo. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `local_xgc_u5` | local | u5 | `fact_player_gw` | el xg DEL RIVAL en ese mismo fixture. NO la suma de expected_goals_conceded, que se cuenta una vez por cada jugador del plantel e infla el valor x11 (medido: media 15,75 contra 1,47 goles concedidos reales). media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `local_xa_u5` | local | u5 | `fact_player_gw` | suma de expected_assists. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `local_pts_arq_u5` | local | u5 | `fact_player_gw` | suma de total_points de los jugadores con position == GK. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `local_pts_def_u5` | local | u5 | `fact_player_gw` | suma de total_points de position == DEF. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `local_pts_med_u5` | local | u5 | `fact_player_gw` | suma de total_points de position == MID. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `local_pts_del_u5` | local | u5 | `fact_player_gw` | suma de total_points de position == FWD. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `local_n_jugadores_u5` | local | u5 | `fact_player_gw` | cantidad de jugadores con minutes > 0. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `local_atajadas_u5` | local | u5 | `fact_player_gw` | atajadas del arquero en el partido. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `local_tasa_atajadas_u5` | local | u5 | `fact_player_gw` | atajadas / (atajadas + goles recibidos): que proporcion de los remates al arco termina detenida. Es el equivalente defensivo de xg_por_tiro -- separa 'concede poco' de 'concede mucho pero lo atajan'. Envejece distinto que las otras defensivas: conceder pocos remates es estructural y persiste, mientras que una tasa de atajadas alta es en buena parte varianza del arquero y revierte a la media. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `visita_pts_u3` | visita | u3 | `fact_match` | 3 si ganó, 1 si empató, 0 si perdió. media de los últimos 3 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `visita_gf_u3` | visita | u3 | `fact_match` | goles a favor. media de los últimos 3 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `visita_gc_u3` | visita | u3 | `fact_match` | goles en contra. media de los últimos 3 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `visita_tiros_u3` | visita | u3 | `fact_match` | tiros totales. media de los últimos 3 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `visita_tiros_arco_u3` | visita | u3 | `fact_match` | tiros al arco. media de los últimos 3 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `visita_corners_u3` | visita | u3 | `fact_match` | córners. media de los últimos 3 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `visita_faltas_u3` | visita | u3 | `fact_match` | faltas cometidas. media de los últimos 3 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `visita_tarjetas_u3` | visita | u3 | `fact_match` | amarillas + 2 x rojas. media de los últimos 3 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `visita_xg_u3` | visita | u3 | `fact_player_gw` | suma de expected_goals de los jugadores del equipo. media de los últimos 3 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `visita_xgc_u3` | visita | u3 | `fact_player_gw` | el xg DEL RIVAL en ese mismo fixture. NO la suma de expected_goals_conceded, que se cuenta una vez por cada jugador del plantel e infla el valor x11 (medido: media 15,75 contra 1,47 goles concedidos reales). media de los últimos 3 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `visita_xa_u3` | visita | u3 | `fact_player_gw` | suma de expected_assists. media de los últimos 3 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `visita_pts_arq_u3` | visita | u3 | `fact_player_gw` | suma de total_points de los jugadores con position == GK. media de los últimos 3 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `visita_pts_def_u3` | visita | u3 | `fact_player_gw` | suma de total_points de position == DEF. media de los últimos 3 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `visita_pts_med_u3` | visita | u3 | `fact_player_gw` | suma de total_points de position == MID. media de los últimos 3 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `visita_pts_del_u3` | visita | u3 | `fact_player_gw` | suma de total_points de position == FWD. media de los últimos 3 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `visita_n_jugadores_u3` | visita | u3 | `fact_player_gw` | cantidad de jugadores con minutes > 0. media de los últimos 3 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `visita_atajadas_u3` | visita | u3 | `fact_player_gw` | atajadas del arquero en el partido. media de los últimos 3 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `visita_tasa_atajadas_u3` | visita | u3 | `fact_player_gw` | atajadas / (atajadas + goles recibidos): que proporcion de los remates al arco termina detenida. Es el equivalente defensivo de xg_por_tiro -- separa 'concede poco' de 'concede mucho pero lo atajan'. Envejece distinto que las otras defensivas: conceder pocos remates es estructural y persiste, mientras que una tasa de atajadas alta es en buena parte varianza del arquero y revierte a la media. media de los últimos 3 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `visita_pts_u5` | visita | u5 | `fact_match` | 3 si ganó, 1 si empató, 0 si perdió. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `visita_gf_u5` | visita | u5 | `fact_match` | goles a favor. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `visita_gc_u5` | visita | u5 | `fact_match` | goles en contra. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `visita_tiros_u5` | visita | u5 | `fact_match` | tiros totales. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `visita_tiros_arco_u5` | visita | u5 | `fact_match` | tiros al arco. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `visita_corners_u5` | visita | u5 | `fact_match` | córners. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `visita_faltas_u5` | visita | u5 | `fact_match` | faltas cometidas. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `visita_tarjetas_u5` | visita | u5 | `fact_match` | amarillas + 2 x rojas. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `visita_xg_u5` | visita | u5 | `fact_player_gw` | suma de expected_goals de los jugadores del equipo. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `visita_xgc_u5` | visita | u5 | `fact_player_gw` | el xg DEL RIVAL en ese mismo fixture. NO la suma de expected_goals_conceded, que se cuenta una vez por cada jugador del plantel e infla el valor x11 (medido: media 15,75 contra 1,47 goles concedidos reales). media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `visita_xa_u5` | visita | u5 | `fact_player_gw` | suma de expected_assists. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `visita_pts_arq_u5` | visita | u5 | `fact_player_gw` | suma de total_points de los jugadores con position == GK. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `visita_pts_def_u5` | visita | u5 | `fact_player_gw` | suma de total_points de position == DEF. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `visita_pts_med_u5` | visita | u5 | `fact_player_gw` | suma de total_points de position == MID. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `visita_pts_del_u5` | visita | u5 | `fact_player_gw` | suma de total_points de position == FWD. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `visita_n_jugadores_u5` | visita | u5 | `fact_player_gw` | cantidad de jugadores con minutes > 0. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `visita_atajadas_u5` | visita | u5 | `fact_player_gw` | atajadas del arquero en el partido. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |
| `visita_tasa_atajadas_u5` | visita | u5 | `fact_player_gw` | atajadas / (atajadas + goles recibidos): que proporcion de los remates al arco termina detenida. Es el equivalente defensivo de xg_por_tiro -- separa 'concede poco' de 'concede mucho pero lo atajan'. Envejece distinto que las otras defensivas: conceder pocos remates es estructural y persiste, mientras que una tasa de atajadas alta es en buena parte varianza del arquero y revierte a la media. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo). Cruza el borde de temporada, así que siempre hay dato, incluso en la GW1 |

### Forma intra-temporada — 16 columnas

| Columna | Lado | Ventana | Fuente | Cálculo |
|---|---|---|---|---|
| `local_pts_u5_temp` | local | u5_temp | `fact_match` | 3 si ganó, 1 si empató, 0 si perdió. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo), pero SÓLO con partidos de la temporada actual. NaN en la fecha 1: es el contrapunto a la ventana cruzada, porque acá el plantel sí es el de hoy |
| `local_gf_u5_temp` | local | u5_temp | `fact_match` | goles a favor. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo), pero SÓLO con partidos de la temporada actual. NaN en la fecha 1: es el contrapunto a la ventana cruzada, porque acá el plantel sí es el de hoy |
| `local_gc_u5_temp` | local | u5_temp | `fact_match` | goles en contra. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo), pero SÓLO con partidos de la temporada actual. NaN en la fecha 1: es el contrapunto a la ventana cruzada, porque acá el plantel sí es el de hoy |
| `local_xg_u5_temp` | local | u5_temp | `fact_player_gw` | suma de expected_goals de los jugadores del equipo. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo), pero SÓLO con partidos de la temporada actual. NaN en la fecha 1: es el contrapunto a la ventana cruzada, porque acá el plantel sí es el de hoy |
| `local_xgc_u5_temp` | local | u5_temp | `fact_player_gw` | el xg DEL RIVAL en ese mismo fixture. NO la suma de expected_goals_conceded, que se cuenta una vez por cada jugador del plantel e infla el valor x11 (medido: media 15,75 contra 1,47 goles concedidos reales). media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo), pero SÓLO con partidos de la temporada actual. NaN en la fecha 1: es el contrapunto a la ventana cruzada, porque acá el plantel sí es el de hoy |
| `local_pts_def_u5_temp` | local | u5_temp | `fact_player_gw` | suma de total_points de position == DEF. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo), pero SÓLO con partidos de la temporada actual. NaN en la fecha 1: es el contrapunto a la ventana cruzada, porque acá el plantel sí es el de hoy |
| `local_pts_med_u5_temp` | local | u5_temp | `fact_player_gw` | suma de total_points de position == MID. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo), pero SÓLO con partidos de la temporada actual. NaN en la fecha 1: es el contrapunto a la ventana cruzada, porque acá el plantel sí es el de hoy |
| `local_tiros_arco_u5_temp` | local | u5_temp | `fact_match` | tiros al arco. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo), pero SÓLO con partidos de la temporada actual. NaN en la fecha 1: es el contrapunto a la ventana cruzada, porque acá el plantel sí es el de hoy |
| `visita_pts_u5_temp` | visita | u5_temp | `fact_match` | 3 si ganó, 1 si empató, 0 si perdió. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo), pero SÓLO con partidos de la temporada actual. NaN en la fecha 1: es el contrapunto a la ventana cruzada, porque acá el plantel sí es el de hoy |
| `visita_gf_u5_temp` | visita | u5_temp | `fact_match` | goles a favor. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo), pero SÓLO con partidos de la temporada actual. NaN en la fecha 1: es el contrapunto a la ventana cruzada, porque acá el plantel sí es el de hoy |
| `visita_gc_u5_temp` | visita | u5_temp | `fact_match` | goles en contra. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo), pero SÓLO con partidos de la temporada actual. NaN en la fecha 1: es el contrapunto a la ventana cruzada, porque acá el plantel sí es el de hoy |
| `visita_xg_u5_temp` | visita | u5_temp | `fact_player_gw` | suma de expected_goals de los jugadores del equipo. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo), pero SÓLO con partidos de la temporada actual. NaN en la fecha 1: es el contrapunto a la ventana cruzada, porque acá el plantel sí es el de hoy |
| `visita_xgc_u5_temp` | visita | u5_temp | `fact_player_gw` | el xg DEL RIVAL en ese mismo fixture. NO la suma de expected_goals_conceded, que se cuenta una vez por cada jugador del plantel e infla el valor x11 (medido: media 15,75 contra 1,47 goles concedidos reales). media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo), pero SÓLO con partidos de la temporada actual. NaN en la fecha 1: es el contrapunto a la ventana cruzada, porque acá el plantel sí es el de hoy |
| `visita_pts_def_u5_temp` | visita | u5_temp | `fact_player_gw` | suma de total_points de position == DEF. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo), pero SÓLO con partidos de la temporada actual. NaN en la fecha 1: es el contrapunto a la ventana cruzada, porque acá el plantel sí es el de hoy |
| `visita_pts_med_u5_temp` | visita | u5_temp | `fact_player_gw` | suma de total_points de position == MID. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo), pero SÓLO con partidos de la temporada actual. NaN en la fecha 1: es el contrapunto a la ventana cruzada, porque acá el plantel sí es el de hoy |
| `visita_tiros_arco_u5_temp` | visita | u5_temp | `fact_match` | tiros al arco. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo), pero SÓLO con partidos de la temporada actual. NaN en la fecha 1: es el contrapunto a la ventana cruzada, porque acá el plantel sí es el de hoy |

### Forma según condición — 12 columnas

| Columna | Lado | Ventana | Fuente | Cálculo |
|---|---|---|---|---|
| `local_pts_cond_u5` | local | cond_u5 | `fact_match` | 3 si ganó, 1 si empató, 0 si perdió. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo), restringida a los partidos que el equipo jugó de local. Captura la ventaja de localía propia de cada equipo |
| `local_gf_cond_u5` | local | cond_u5 | `fact_match` | goles a favor. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo), restringida a los partidos que el equipo jugó de local. Captura la ventaja de localía propia de cada equipo |
| `local_gc_cond_u5` | local | cond_u5 | `fact_match` | goles en contra. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo), restringida a los partidos que el equipo jugó de local. Captura la ventaja de localía propia de cada equipo |
| `local_dg_cond_u5` | local | cond_u5 | `fact_match` | diferencia de gol del partido (gf - gc). media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo), restringida a los partidos que el equipo jugó de local. Captura la ventaja de localía propia de cada equipo |
| `local_xg_cond_u5` | local | cond_u5 | `fact_player_gw` | suma de expected_goals de los jugadores del equipo. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo), restringida a los partidos que el equipo jugó de local. Captura la ventaja de localía propia de cada equipo |
| `local_xgc_cond_u5` | local | cond_u5 | `fact_player_gw` | el xg DEL RIVAL en ese mismo fixture. NO la suma de expected_goals_conceded, que se cuenta una vez por cada jugador del plantel e infla el valor x11 (medido: media 15,75 contra 1,47 goles concedidos reales). media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo), restringida a los partidos que el equipo jugó de local. Captura la ventaja de localía propia de cada equipo |
| `visita_pts_cond_u5` | visita | cond_u5 | `fact_match` | 3 si ganó, 1 si empató, 0 si perdió. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo), restringida a los partidos que el equipo jugó de visitante. Captura la ventaja de localía propia de cada equipo |
| `visita_gf_cond_u5` | visita | cond_u5 | `fact_match` | goles a favor. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo), restringida a los partidos que el equipo jugó de visitante. Captura la ventaja de localía propia de cada equipo |
| `visita_gc_cond_u5` | visita | cond_u5 | `fact_match` | goles en contra. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo), restringida a los partidos que el equipo jugó de visitante. Captura la ventaja de localía propia de cada equipo |
| `visita_dg_cond_u5` | visita | cond_u5 | `fact_match` | diferencia de gol del partido (gf - gc). media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo), restringida a los partidos que el equipo jugó de visitante. Captura la ventaja de localía propia de cada equipo |
| `visita_xg_cond_u5` | visita | cond_u5 | `fact_player_gw` | suma de expected_goals de los jugadores del equipo. media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo), restringida a los partidos que el equipo jugó de visitante. Captura la ventaja de localía propia de cada equipo |
| `visita_xgc_cond_u5` | visita | cond_u5 | `fact_player_gw` | el xg DEL RIVAL en ese mismo fixture. NO la suma de expected_goals_conceded, que se cuenta una vez por cada jugador del plantel e infla el valor x11 (medido: media 15,75 contra 1,47 goles concedidos reales). media de los últimos 5 partidos del equipo, min_periods=1, anclada al corte con merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo), restringida a los partidos que el equipo jugó de visitante. Captura la ventaja de localía propia de cada equipo |

### Puntaje de campeonato — 14 columnas

| Columna | Lado | Ventana | Fuente | Cálculo |
|---|---|---|---|---|
| `local_pts_camp` | local | camp | `fact_match` | puntos acumulados en la temporada al momento del corte. expanding() en lugar de rolling(), mismo anclaje al corte por merge_asof |
| `local_pj_camp` | local | camp | `fact_match` | partidos jugados en la temporada. expanding() en lugar de rolling(), mismo anclaje al corte por merge_asof |
| `local_ppp_camp` | local | camp | `fact_match` | puntos por partido (pts / pj): el acumulado normalizado, comparable entre la fecha 3 y la 30. expanding() en lugar de rolling(), mismo anclaje al corte por merge_asof |
| `local_gf_camp` | local | camp | `fact_match` | goles a favor acumulados. expanding() en lugar de rolling(), mismo anclaje al corte por merge_asof |
| `local_gc_camp` | local | camp | `fact_match` | goles en contra acumulados. expanding() en lugar de rolling(), mismo anclaje al corte por merge_asof |
| `local_dg_camp` | local | camp | `fact_match` | diferencia de gol acumulada. expanding() en lugar de rolling(), mismo anclaje al corte por merge_asof |
| `local_pos_tabla_camp` | local | camp | `fact_match` | posición en la tabla, rankeando por (pts, dg, gf) dentro de (temporada, corte): el criterio de desempate real de la Premier. expanding() en lugar de rolling(), mismo anclaje al corte por merge_asof |
| `visita_pts_camp` | visita | camp | `fact_match` | puntos acumulados en la temporada al momento del corte. expanding() en lugar de rolling(), mismo anclaje al corte por merge_asof |
| `visita_pj_camp` | visita | camp | `fact_match` | partidos jugados en la temporada. expanding() en lugar de rolling(), mismo anclaje al corte por merge_asof |
| `visita_ppp_camp` | visita | camp | `fact_match` | puntos por partido (pts / pj): el acumulado normalizado, comparable entre la fecha 3 y la 30. expanding() en lugar de rolling(), mismo anclaje al corte por merge_asof |
| `visita_gf_camp` | visita | camp | `fact_match` | goles a favor acumulados. expanding() en lugar de rolling(), mismo anclaje al corte por merge_asof |
| `visita_gc_camp` | visita | camp | `fact_match` | goles en contra acumulados. expanding() en lugar de rolling(), mismo anclaje al corte por merge_asof |
| `visita_dg_camp` | visita | camp | `fact_match` | diferencia de gol acumulada. expanding() en lugar de rolling(), mismo anclaje al corte por merge_asof |
| `visita_pos_tabla_camp` | visita | camp | `fact_match` | posición en la tabla, rankeando por (pts, dg, gf) dentro de (temporada, corte): el criterio de desempate real de la Premier. expanding() en lugar de rolling(), mismo anclaje al corte por merge_asof |

### Head-to-head — 10 columnas

| Columna | Lado | Ventana | Fuente | Cálculo |
|---|---|---|---|---|
| `h2h_pts_local` | — | h2h | `fact_match` | puntos que sacó el equipo que HOY es local, en esos partidos. Media sobre TODOS los enfrentamientos previos entre estos dos equipos anteriores al corte, en cualquier condición. Medido: media 2,83 antecedentes en la ventana, 5,15 en el holdout |
| `h2h_pts_visita` | — | h2h | `fact_match` | puntos que sacó el equipo que HOY es visitante, en esos partidos. Media sobre TODOS los enfrentamientos previos entre estos dos equipos anteriores al corte, en cualquier condición. Medido: media 2,83 antecedentes en la ventana, 5,15 en el holdout |
| `h2h_gf_local` | — | h2h | `fact_match` | goles que hizo el equipo que hoy es local. Media sobre TODOS los enfrentamientos previos entre estos dos equipos anteriores al corte, en cualquier condición. Medido: media 2,83 antecedentes en la ventana, 5,15 en el holdout |
| `h2h_gc_local` | — | h2h | `fact_match` | goles que recibió el equipo que hoy es local. Media sobre TODOS los enfrentamientos previos entre estos dos equipos anteriores al corte, en cualquier condición. Medido: media 2,83 antecedentes en la ventana, 5,15 en el holdout |
| `h2h_n` | — | h2h | `fact_match` | cuántos enfrentamientos previos hay. Media sobre TODOS los enfrentamientos previos entre estos dos equipos anteriores al corte, en cualquier condición. Medido: media 2,83 antecedentes en la ventana, 5,15 en el holdout |
| `h2h_cond_pts_local` | — | h2h_cond | `fact_match` | puntos que sacó el equipo que HOY es local, en esos partidos. Igual que h2h_pts_local pero SÓLO los partidos en que este mismo equipo fue local contra este mismo rival. Flaco por construcción: con 4 temporadas el máximo posible es 3 (media medida 1,16; 2,33 en el holdout) |
| `h2h_cond_pts_visita` | — | h2h_cond | `fact_match` | puntos que sacó el equipo que HOY es visitante, en esos partidos. Igual que h2h_pts_visita pero SÓLO los partidos en que este mismo equipo fue local contra este mismo rival. Flaco por construcción: con 4 temporadas el máximo posible es 3 (media medida 1,16; 2,33 en el holdout) |
| `h2h_cond_gf_local` | — | h2h_cond | `fact_match` | goles que hizo el equipo que hoy es local. Igual que h2h_gf_local pero SÓLO los partidos en que este mismo equipo fue local contra este mismo rival. Flaco por construcción: con 4 temporadas el máximo posible es 3 (media medida 1,16; 2,33 en el holdout) |
| `h2h_cond_gc_local` | — | h2h_cond | `fact_match` | goles que recibió el equipo que hoy es local. Igual que h2h_gc_local pero SÓLO los partidos en que este mismo equipo fue local contra este mismo rival. Flaco por construcción: con 4 temporadas el máximo posible es 3 (media medida 1,16; 2,33 en el holdout) |
| `h2h_cond_n` | — | h2h_cond | `fact_match` | cuántos enfrentamientos previos hay. Igual que h2h_n pero SÓLO los partidos en que este mismo equipo fue local contra este mismo rival. Flaco por construcción: con 4 temporadas el máximo posible es 3 (media medida 1,16; 2,33 en el holdout) |

### Continuidad de plantel — 2 columnas

| Columna | Lado | Ventana | Fuente | Cálculo |
|---|---|---|---|---|
| `local_continuidad_plantel_u5` | local | u5 | `fact_player_gw` | proporción de los minutos de los últimos 5 partidos que jugaron futbolistas que también jugaron el partido MÁS RECIENTE del equipo. 100% leak-free: sólo mira partidos pasados. Se desploma cuando el plantel se renovó, que es justo cuando la forma pasada deja de representar al equipo actual. Medido: entre temporadas rota ~40% de los minutos |
| `visita_continuidad_plantel_u5` | visita | u5 | `fact_player_gw` | proporción de los minutos de los últimos 5 partidos que jugaron futbolistas que también jugaron el partido MÁS RECIENTE del equipo. 100% leak-free: sólo mira partidos pasados. Se desploma cuando el plantel se renovó, que es justo cuando la forma pasada deja de representar al equipo actual. Medido: entre temporadas rota ~40% de los minutos |

### Elo y estado — 36 columnas

| Columna | Lado | Ventana | Fuente | Cálculo |
|---|---|---|---|---|
| `local_elo` | local | — | `derivada` | rating Elo del equipo despues de su ultimo partido antes del corte. K=20, ventaja de localia 65 puntos, margen de victoria atenuado por logaritmo, y regresion del 25% a la media al cambiar de temporada. Resuelve lo que las medias moviles no pueden: ponderar cada resultado segun contra quien fue. Validado -- al cierre de 2024-25 da Liverpool/Arsenal/City arriba y Southampton/Ipswich/Leicester abajo, que son los tres descendidos |
| `local_elo_delta_u3` | local | — | `derivada` | cuanto GANO o PERDIO de rating Elo en sus ultimos 3 partidos. Es informacion distinta del nivel y distinta de `racha`: `elo` dice donde esta el equipo, `racha` compara puntos contra su propio promedio tratando igual a todos los rivales, y esto dice hacia donde va PONDERADO POR CONTRA QUIEN. Sumar 6 puntos contra dos rivales de arriba sube mucho mas que sumarlos contra dos de abajo |
| `local_elo_delta_u5` | local | — | `derivada` | lo mismo sobre 5 partidos |
| `local_elo_delta_u10` | local | — | `derivada` | lo mismo sobre 10: menos ruidoso. Junto con `elo` le permite al modelo distinguir cuatro situaciones que hoy se le mezclan: grande en alza, grande en caida, chico en alza y chico en caida |
| `local_tiros_conc_u5` | local | — | `derivada` | tiros que le concedieron al equipo en sus ultimos 5 partidos. Las ventanas de `tiros` miden lo que el equipo genera; esta mide lo que regala, que es informacion distinta |
| `local_tiros_arco_conc_u5` | local | — | `derivada` | tiros al arco concedidos en los ultimos 5 |
| `local_xg_diff_u5` | local | — | `derivada` | goles menos xG en los ultimos 5: suerte de definicion. Es fuertemente reversible a la media, asi que un valor alto anticipa una caida. Ni `gf_u5` ni `xg_u5` capturan esto por separado |
| `local_xgc_diff_u5` | local | — | `derivada` | goles recibidos menos xG concedido en los ultimos 5: lo mismo del lado defensivo, incluye el rendimiento del arquero |
| `local_xg_por_tiro_u5` | local | — | `derivada` | xG dividido por tiros, promediado sobre los ultimos 5: la CALIDAD de las situaciones, no la cantidad. 2,0 de xG en 3 ocasiones claras y 2,0 en 20 remates de afuera son cosas distintas y predicen distinto; el xG agregado no las separa. Es la aproximacion gratis al xG a nivel tiro que daria Understat |
| `local_xgc_por_tiro_u5` | local | — | `derivada` | lo mismo del lado defensivo: que tan claras son las situaciones que concede. Un equipo puede conceder muchos remates lejanos (bajo riesgo) o pocas ocasiones claras (alto riesgo) |
| `local_prop_tiros_arco_u5` | local | — | `derivada` | proporcion de tiros que van al arco: punteria y seleccion de remate |
| `local_prop_tiros_arco_conc_u5` | local | — | `derivada` | lo mismo entre los tiros que concede |
| `local_partidos_7d` | local | — | `derivada` | partidos de Premier jugados en los 7 dias previos. Detecta el 'jugo entre semana', que es lo mas cerca que se puede estar de identificar un compromiso de copa o de Europa sin el calendario de esas competencias |
| `local_partidos_14d` | local | — | `derivada` | partidos jugados en los 14 dias previos: la carga de dos semanas |
| `local_partidos_21d` | local | — | `derivada` | partidos en 21 dias: la carga acumulada. No es lo mismo un pico aislado que tres semanas seguidas de partido cada tres dias |
| `local_racha` | local | — | `derivada` | puntos de los ultimos 3 partidos menos el promedio de lo que va de la temporada. Captura si el equipo esta por encima o por debajo de su nivel |
| `local_sorpresa_u5` | local | — | `derivada` | cuanto se apartaron los ultimos 5 resultados de lo que el Elo esperaba: \|real - esperado\| promediado. Mide que tan IMPREDECIBLE viene siendo el equipo, no en que direccion. Es informacion sobre la confiabilidad de la prediccion. No usa las predicciones del modelo -- eso seria un bucle de realimentacion -- sino la expectativa del Elo, que sale solo de resultados pasados |
| `local_sorpresa_u10` | local | — | `derivada` | lo mismo sobre 10 partidos: menos ruidoso, mas estructural. Medido en 2025-26, los mas impredecibles fueron CHE, NEW y AVL; los mas predecibles BUR y BRE (ser consistentemente malo tambien es predecible) |
| `visita_elo` | visita | — | `derivada` | rating Elo del equipo despues de su ultimo partido antes del corte. K=20, ventaja de localia 65 puntos, margen de victoria atenuado por logaritmo, y regresion del 25% a la media al cambiar de temporada. Resuelve lo que las medias moviles no pueden: ponderar cada resultado segun contra quien fue. Validado -- al cierre de 2024-25 da Liverpool/Arsenal/City arriba y Southampton/Ipswich/Leicester abajo, que son los tres descendidos |
| `visita_elo_delta_u3` | visita | — | `derivada` | cuanto GANO o PERDIO de rating Elo en sus ultimos 3 partidos. Es informacion distinta del nivel y distinta de `racha`: `elo` dice donde esta el equipo, `racha` compara puntos contra su propio promedio tratando igual a todos los rivales, y esto dice hacia donde va PONDERADO POR CONTRA QUIEN. Sumar 6 puntos contra dos rivales de arriba sube mucho mas que sumarlos contra dos de abajo |
| `visita_elo_delta_u5` | visita | — | `derivada` | lo mismo sobre 5 partidos |
| `visita_elo_delta_u10` | visita | — | `derivada` | lo mismo sobre 10: menos ruidoso. Junto con `elo` le permite al modelo distinguir cuatro situaciones que hoy se le mezclan: grande en alza, grande en caida, chico en alza y chico en caida |
| `visita_tiros_conc_u5` | visita | — | `derivada` | tiros que le concedieron al equipo en sus ultimos 5 partidos. Las ventanas de `tiros` miden lo que el equipo genera; esta mide lo que regala, que es informacion distinta |
| `visita_tiros_arco_conc_u5` | visita | — | `derivada` | tiros al arco concedidos en los ultimos 5 |
| `visita_xg_diff_u5` | visita | — | `derivada` | goles menos xG en los ultimos 5: suerte de definicion. Es fuertemente reversible a la media, asi que un valor alto anticipa una caida. Ni `gf_u5` ni `xg_u5` capturan esto por separado |
| `visita_xgc_diff_u5` | visita | — | `derivada` | goles recibidos menos xG concedido en los ultimos 5: lo mismo del lado defensivo, incluye el rendimiento del arquero |
| `visita_xg_por_tiro_u5` | visita | — | `derivada` | xG dividido por tiros, promediado sobre los ultimos 5: la CALIDAD de las situaciones, no la cantidad. 2,0 de xG en 3 ocasiones claras y 2,0 en 20 remates de afuera son cosas distintas y predicen distinto; el xG agregado no las separa. Es la aproximacion gratis al xG a nivel tiro que daria Understat |
| `visita_xgc_por_tiro_u5` | visita | — | `derivada` | lo mismo del lado defensivo: que tan claras son las situaciones que concede. Un equipo puede conceder muchos remates lejanos (bajo riesgo) o pocas ocasiones claras (alto riesgo) |
| `visita_prop_tiros_arco_u5` | visita | — | `derivada` | proporcion de tiros que van al arco: punteria y seleccion de remate |
| `visita_prop_tiros_arco_conc_u5` | visita | — | `derivada` | lo mismo entre los tiros que concede |
| `visita_partidos_7d` | visita | — | `derivada` | partidos de Premier jugados en los 7 dias previos. Detecta el 'jugo entre semana', que es lo mas cerca que se puede estar de identificar un compromiso de copa o de Europa sin el calendario de esas competencias |
| `visita_partidos_14d` | visita | — | `derivada` | partidos jugados en los 14 dias previos: la carga de dos semanas |
| `visita_partidos_21d` | visita | — | `derivada` | partidos en 21 dias: la carga acumulada. No es lo mismo un pico aislado que tres semanas seguidas de partido cada tres dias |
| `visita_racha` | visita | — | `derivada` | puntos de los ultimos 3 partidos menos el promedio de lo que va de la temporada. Captura si el equipo esta por encima o por debajo de su nivel |
| `visita_sorpresa_u5` | visita | — | `derivada` | cuanto se apartaron los ultimos 5 resultados de lo que el Elo esperaba: \|real - esperado\| promediado. Mide que tan IMPREDECIBLE viene siendo el equipo, no en que direccion. Es informacion sobre la confiabilidad de la prediccion. No usa las predicciones del modelo -- eso seria un bucle de realimentacion -- sino la expectativa del Elo, que sale solo de resultados pasados |
| `visita_sorpresa_u10` | visita | — | `derivada` | lo mismo sobre 10 partidos: menos ruidoso, mas estructural. Medido en 2025-26, los mas impredecibles fueron CHE, NEW y AVL; los mas predecibles BUR y BRE (ser consistentemente malo tambien es predecible) |

### Otras competencias — 24 columnas

| Columna | Lado | Ventana | Fuente | Cálculo |
|---|---|---|---|---|
| `local_partidos_todo_7d` | local | — | `fact_match_comp` | partidos jugados en los 7 dias previos contando TODAS las competencias. La version que solo miraba la Premier dio un resultado nulo, justamente porque no veia los partidos de copa y de Europa: medido sobre 2025-26 son 953 partidos que faltaban |
| `local_partidos_todo_14d` | local | — | `fact_match_comp` | lo mismo sobre 14 dias |
| `local_partidos_todo_21d` | local | — | `fact_match_comp` | lo mismo sobre 21: la carga acumulada de tres semanas |
| `local_partidos_copa_7d` | local | — | `fact_match_comp` | de esos, cuantos NO fueron de liga en los ultimos 7 dias |
| `local_partidos_copa_14d` | local | — | `fact_match_comp` | idem sobre 14 dias |
| `local_copas_acumuladas` | local | — | `fact_match_comp` | partidos de FA Cup y EFL Cup jugados en lo que va de la temporada. Solo crece si el equipo avanza, asi que es 'seguir en carrera' convertido en numero |
| `local_europa_acumuladas` | local | — | `fact_match_comp` | partidos de Champions y Europa League en la temporada |
| `local_importancia_max` | local | — | `fact_match_comp` | la instancia mas avanzada alcanzada en copa: 1 = primera ronda, 6 = octavos, 7 = cuartos, 8 = semis, 9 = final. Es retrospectivo -- dice hasta donde LLEGO, no hasta donde va a llegar, que no se puede saber sin ver el sorteo |
| `local_dias_desde_ultimo_todo` | local | — | `fact_match_comp` | dias desde su ultimo partido de cualquier competencia. El `dias_descanso` de liga se equivocaba en los equipos que jugaban entre semana |
| `local_pts_todo_u5` | local | — | `fact_match_comp` | puntos por partido en los ultimos 5 de CUALQUIER competencia. Comparado con la version de liga es informativo por si mismo: un equipo que rinde distinto en copa esta rotando |
| `local_gf_todo_u5` | local | — | `fact_match_comp` | goles a favor en los ultimos 5 de cualquier competencia |
| `local_gc_todo_u5` | local | — | `fact_match_comp` | goles en contra en los ultimos 5 de cualquier competencia |
| `visita_partidos_todo_7d` | visita | — | `fact_match_comp` | partidos jugados en los 7 dias previos contando TODAS las competencias. La version que solo miraba la Premier dio un resultado nulo, justamente porque no veia los partidos de copa y de Europa: medido sobre 2025-26 son 953 partidos que faltaban |
| `visita_partidos_todo_14d` | visita | — | `fact_match_comp` | lo mismo sobre 14 dias |
| `visita_partidos_todo_21d` | visita | — | `fact_match_comp` | lo mismo sobre 21: la carga acumulada de tres semanas |
| `visita_partidos_copa_7d` | visita | — | `fact_match_comp` | de esos, cuantos NO fueron de liga en los ultimos 7 dias |
| `visita_partidos_copa_14d` | visita | — | `fact_match_comp` | idem sobre 14 dias |
| `visita_copas_acumuladas` | visita | — | `fact_match_comp` | partidos de FA Cup y EFL Cup jugados en lo que va de la temporada. Solo crece si el equipo avanza, asi que es 'seguir en carrera' convertido en numero |
| `visita_europa_acumuladas` | visita | — | `fact_match_comp` | partidos de Champions y Europa League en la temporada |
| `visita_importancia_max` | visita | — | `fact_match_comp` | la instancia mas avanzada alcanzada en copa: 1 = primera ronda, 6 = octavos, 7 = cuartos, 8 = semis, 9 = final. Es retrospectivo -- dice hasta donde LLEGO, no hasta donde va a llegar, que no se puede saber sin ver el sorteo |
| `visita_dias_desde_ultimo_todo` | visita | — | `fact_match_comp` | dias desde su ultimo partido de cualquier competencia. El `dias_descanso` de liga se equivocaba en los equipos que jugaban entre semana |
| `visita_pts_todo_u5` | visita | — | `fact_match_comp` | puntos por partido en los ultimos 5 de CUALQUIER competencia. Comparado con la version de liga es informativo por si mismo: un equipo que rinde distinto en copa esta rotando |
| `visita_gf_todo_u5` | visita | — | `fact_match_comp` | goles a favor en los ultimos 5 de cualquier competencia |
| `visita_gc_todo_u5` | visita | — | `fact_match_comp` | goles en contra en los ultimos 5 de cualquier competencia |

### Contexto — 10 columnas

| Columna | Lado | Ventana | Fuente | Cálculo |
|---|---|---|---|---|
| `local_n_hist` | local | — | `derivada` | cantidad de partidos previos disponibles para el equipo. Le dice al modelo cuán parcial es la media móvil |
| `local_dias_descanso` | local | — | `derivada` | días desde el partido anterior del equipo |
| `local_es_ascendido` | local | — | `derivada` | el equipo no estaba en la Premier la temporada anterior. Derivado así, NO desde dim_team.promoted, que en realidad significa 'primera temporada dentro de la ventana ingestada' y marca 1 equipo en 2024-25 cuando ascendieron 3 |
| `local_mins_hhi` | local | — | `derivada` | índice de Herfindahl sobre el reparto de minutos en los últimos 5 partidos: mide rotación de plantel. Reemplaza al grupo 'disponibilidad' del canvas, que no se puede construir porque la API de FPL no sirve el `status` histórico |
| `visita_n_hist` | visita | — | `derivada` | cantidad de partidos previos disponibles para el equipo. Le dice al modelo cuán parcial es la media móvil |
| `visita_dias_descanso` | visita | — | `derivada` | días desde el partido anterior del equipo |
| `visita_es_ascendido` | visita | — | `derivada` | el equipo no estaba en la Premier la temporada anterior. Derivado así, NO desde dim_team.promoted, que en realidad significa 'primera temporada dentro de la ventana ingestada' y marca 1 equipo en 2024-25 cuando ascendieron 3 |
| `visita_mins_hhi` | visita | — | `derivada` | índice de Herfindahl sobre el reparto de minutos en los últimos 5 partidos: mide rotación de plantel. Reemplaza al grupo 'disponibilidad' del canvas, que no se puede construir porque la API de FPL no sirve el `status` histórico |
| `gameweek` | — | — | `fact_fixture` | número de fecha; captura el efecto del momento de la temporada |
| `xg_available` | — | — | `derivada` | falso donde el xG viene hardcodeado en cero (2022-23 hasta la GW15, los 20 equipos). Explica el NaN en vez de dejar que el modelo lo interprete solo |

### Dificultad — 3 columnas

| Columna | Lado | Ventana | Fuente | Cálculo |
|---|---|---|---|---|
| `fdr_local` | local | — | `fact_fixture` | FDR de FPL para el local (team_h_difficulty), escala 1-5 |
| `fdr_visita` | visita | — | `fact_fixture` | FDR de FPL para el visitante (team_a_difficulty), escala 1-5 |
| `fdr_dif` | — | — | `derivada` | fdr_local - fdr_visita |

### Diferenciales — 21 columnas

| Columna | Lado | Ventana | Fuente | Cálculo |
|---|---|---|---|---|
| `dif_pts_u5` | — | — | `derivada` | local_pts_u5 - visita_pts_u5 |
| `dif_gf_u5` | — | — | `derivada` | local_gf_u5 - visita_gf_u5 |
| `dif_gc_u5` | — | — | `derivada` | local_gc_u5 - visita_gc_u5 |
| `dif_xg_u5` | — | — | `derivada` | local_xg_u5 - visita_xg_u5 |
| `dif_xgc_u5` | — | — | `derivada` | local_xgc_u5 - visita_xgc_u5 |
| `dif_pts_def_u5` | — | — | `derivada` | local_pts_def_u5 - visita_pts_def_u5 |
| `dif_pts_med_u5` | — | — | `derivada` | local_pts_med_u5 - visita_pts_med_u5 |
| `dif_pts_camp` | — | — | `derivada` | local_pts_camp - visita_pts_camp |
| `dif_pos_tabla_camp` | — | — | `derivada` | local_pos_tabla_camp - visita_pos_tabla_camp |
| `dif_ppp_camp` | — | — | `derivada` | local_ppp_camp - visita_ppp_camp |
| `dif_dias_descanso` | — | — | `derivada` | local_dias_descanso - visita_dias_descanso |
| `dif_n_hist` | — | — | `derivada` | local_n_hist - visita_n_hist |
| `dif_elo` | — | — | `derivada` | local_elo - visita_elo |
| `dif_elo_delta_u5` | — | — | `derivada` | local_elo_delta_u5 - visita_elo_delta_u5 |
| `dif_racha` | — | — | `derivada` | local_racha - visita_racha |
| `dif_sorpresa_u10` | — | — | `derivada` | local_sorpresa_u10 - visita_sorpresa_u10 |
| `dif_xg_por_tiro_u5` | — | — | `derivada` | local_xg_por_tiro_u5 - visita_xg_por_tiro_u5 |
| `dif_partidos_todo_14d` | — | — | `derivada` | local_partidos_todo_14d - visita_partidos_todo_14d |
| `dif_copas_acumuladas` | — | — | `derivada` | local_copas_acumuladas - visita_copas_acumuladas |
| `dif_importancia_max` | — | — | `derivada` | local_importancia_max - visita_importancia_max |
| `dif_pts_todo_u5` | — | — | `derivada` | local_pts_todo_u5 - visita_pts_todo_u5 |

## Columnas que NO son features

| Grupo | Columnas | Por qué están en Gold |
|---|---|---|
| Claves | `season`, `fixture_id`, `match_date`, `kickoff_time`, `home_short`, `away_short` | identificación del partido |
| Target | `target_1x2`, `home_goals`, `away_goals`, `goal_diff` | `target_1x2` es la etiqueta; los goles se guardan para no cerrarle la puerta a un modelo Poisson |
| Auditoría | `corte`, `hist_kickoff_local`, `hist_kickoff_visita`, `split`, `feature_set_version`, `gold_built_at` | `hist_kickoff_*` es la prueba de que no se miró el futuro |
| Mercado | `odds_avg_close_home`, `odds_avg_close_draw`, `odds_avg_close_away`, `p_mercado_home`, `p_mercado_draw`, `p_mercado_away` | las necesita la simulación de ROI. **Nunca son feature**: si el modelo copia al mercado, el valor esperado da ~0 por construcción y jamás encontraría una apuesta con valor |

