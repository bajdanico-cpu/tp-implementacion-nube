# Monitoring y retraining

Pendiente. Es el eje de la defensa del trabajo, así que el diseño va acá desde ahora.

## El ciclo cerrado

Lo que diferencia este TP de "entrené un modelo y lo puse en un endpoint" es que el
**ground truth llega solo, dos horas después de la predicción**.

```
viernes 17:30   deadline de la fecha  →  se predice y se REGISTRA
sábado 14:00    se juegan los partidos
sábado 16:00    llega el resultado    →  se calculan métricas
                                      →  se evalúa degradación
                                      →  si corresponde, retraining
```

Sin etiquetado manual, sin esperar semanas. El ciclo completo se puede demostrar en
vivo, y ése es el argumento central de la presentación.

## Qué se registra en cada predicción

Sin esto no hay monitoreo posible, sólo un endpoint que responde.

| Campo | Por qué |
|---|---|
| `fixture_id`, `season`, `gameweek` | identificación |
| `predicted_at` | tiene que ser **anterior al deadline**; es la prueba auditable |
| `model_version`, `feature_set_version` | atribuir la degradación al modelo correcto |
| `prob_home`, `prob_draw`, `prob_away` | la predicción completa, no sólo el argmax |
| `prediccion` + una columna por regla candidata | qué anunció producción y qué habría anunciado cada candidata |
| `features_snapshot` | reproducir exactamente la predicción meses después |

## Reglas de decisión en paralelo

`python -m monitoring.temporada_actual` mide, además del modelo, cada **regla de decisión
candidata** de `serving/decision.py` sobre las mismas probabilidades y los mismos partidos.

Es el caso más pareado que existe: no cambia el modelo, cambia sólo la función que va de
las tres probabilidades a una clase. Por eso la comparación usa **McNemar**
(`training/promotion.py`) y no dos accuracies sueltas — sólo informan los partidos donde
las dos reglas discrepan, que son uno o dos por fecha.

Cada candidata declara `desde` en `config.yaml`. Las fechas anteriores se etiquetan igual
(la regla es función pura de las probabilidades guardadas) pero se reportan aparte y **no
cuentan**: una fecha ya jugada, etiquetada con una regla elegida después, es reproducción,
no predicción.

## Recolección del resultado

`collect_outcomes.py` — después de la fecha, pega `/event/{GW}/live/` y `/fixtures/`
contra las predicciones registradas y arma la tabla de evaluación.

## Métricas rodantes

Log-loss y accuracy sobre las últimas K gameweeks, contra los mismos baselines del
EDA. Una caída del modelo **acompañada** de una caída del baseline de cuotas es la
liga siendo más impredecible, no el modelo degradándose: hay que distinguirlas.

## Drift

Fácil de demostrar acá, y el EDA ya lo cuantificó sobre 4 temporadas:

| Métrica | Variación entre temporadas |
|---|---|
| Goles del visitante | 19.5% |
| Tiros al arco del local | 19.0% |
| Goles totales | 17.9% |
| Prob. implícita del local | 3.2% |

A eso se suman los ascensos y descensos: cada temporada cambian 3 de 20 equipos, y
en 2026-27 dos de ellos (Coventry, Hull) no tienen ningún partido en la ventana
ingestada.

## Disparadores del retraining

Definir con umbrales explícitos, no "cuando parezca":

1. **Programado** — al cierre de cada gameweek, reentrenar incorporando la fecha que
   acaba de jugarse. Es lo natural en un dominio con ground truth semanal.
2. **Por degradación** — log-loss rodante de las últimas K fechas por encima de un
   umbral, y que la caída no se explique por el baseline de cuotas cayendo igual.
3. **Por drift** — cambio de temporada, que es donde se mueve todo.

El modelo nuevo sólo reemplaza al viejo si le gana en el holdout. Si no, se registra
el intento y queda el anterior.
