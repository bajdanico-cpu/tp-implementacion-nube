"""Genera `notebooks/00_recorrido_completo.ipynb`.

El notebook se produce desde acá, no se edita a mano: así queda bajo control de versiones
como texto legible, los diffs se entienden, y no se versionan salidas de ejecución que
inflan el repo y ensucian los merges.

    python notebooks/00_recorrido_completo.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = Path(__file__).with_suffix(".ipynb")


# nbformat 4.5 en adelante exige un `id` por celda. Se derivan del contenido para que
# regenerar el notebook sin cambios produzca exactamente el mismo archivo: si fueran
# aleatorios, cada corrida ensuciaría el diff con ids nuevos.
def _id(prefijo: str, texto: str) -> str:
    return f"{prefijo}-{hashlib.sha1(texto.encode('utf-8')).hexdigest()[:8]}"


def md(texto: str) -> dict:
    return {"cell_type": "markdown", "id": _id("md", texto), "metadata": {},
            "source": texto.strip("\n").splitlines(keepends=True)}


def code(texto: str) -> dict:
    return {"cell_type": "code", "id": _id("code", texto), "execution_count": None,
            "metadata": {}, "outputs": [],
            "source": texto.strip("\n").splitlines(keepends=True)}


CELDAS = [
md("""
# TP Premier ML — recorrido completo

Predicción 1X2 de la Premier League, con el ciclo de vida del modelo entero.

Este notebook reproduce **todo lo que hicimos, en orden**, con los números a la vista. No
sustituye al código: cada paso llama a los módulos del repo, así que lo que corre acá es
exactamente lo que corre en producción.

**Antes de empezar** hay que tener el entorno y los datos:

```powershell
.\\scripts\\setup_env.ps1        # crea el venv e instala todo
python -m ingestion.run        # ~27 MB de Bronze, sin credenciales
python -m transform.silver     # las 4 tablas Silver
python -m features.gold_tp     # la tabla Gold
```

**El resumen, para el apurado:**

| | |
|---|---|
| Modelo elegido | **XGBoost** (`xgb_gbt`), entrenado sin las fechas con xG falso |
| Accuracy en el holdout | **0,503** contra 0,426 del baseline |
| Accuracy en walk-forward | **0,516**, le gana a "siempre local" en el 60,5 % de las fechas |
| Feature más importante | `dif_elo` — la diferencia de rating Elo entre los dos equipos |
| ¿Le gana al mercado? | **No.** Cuando discrepa de las casas, acierta menos que ellas |
"""),

md("""
---
## 0 · Preparación

Todo sale de los módulos del repo. Si algo falla acá, faltan los pasos de arriba.
"""),
code("""
import sys, warnings
from pathlib import Path

RAIZ = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(RAIZ))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)

from common.config import CFG
from common.storage import read_table

print("temporadas de entrenamiento:", CFG.seasons_for_training())
print("temporada de validacion    :", CFG.valid_season)
print("holdout (nunca se entrena) :", CFG.holdout_season)
print("modelo de produccion       :", CFG.modelo, "|", CFG.datos_entrenamiento)
"""),

md("""
---
## 1 · Los datos: arquitectura medallion

```
Bronze  crudo, append-only, particionado por ingested_at
   |
Silver  normalizado, grano jugador-fecha conservado
   |
Gold    una fila por partido, lista para entrenar
```

Bronze **nunca sobrescribe**. No es prolijidad: el snapshot tomado *antes* del deadline es
el único que refleja lo que se sabía al momento de predecir. Conservarlo es la defensa
auditable contra el leakage.
"""),
code("""
for nombre in ("dim_team", "fact_match", "fact_fixture", "fact_player_gw"):
    d = read_table(nombre)
    print(f"{nombre:18s} {len(d):>7,} filas x {d.shape[1]:>3} columnas")
"""),

md("""
### Las tres fuentes, y por qué las tres

| Fuente | Grano | Qué aporta que nadie más aporta |
|---|---|---|
| **football-data.co.uk** | partido | cuotas de cierre (el baseline duro), tiros, córners, tarjetas |
| **vaastav/Fantasy-Premier-League** | jugador × fecha | el histórico jugador-fecha con xG. La API oficial no lo sirve |
| **API oficial de FPL** | presente y futuro | fixtures, deadlines, y el resultado apenas termina el partido |

La API de FPL **no sirve el detalle fecha a fecha del pasado**: `history` da sólo la
temporada actual y `history_past` una fila por temporada, y sólo de jugadores que hoy están
en la base (sesgo de supervivencia). Por eso vaastav no es reemplazable.
"""),

md("""
---
## 2 · La regla que sostiene todo: el corte temporal

```
corte(partido) = min(kickoff_time) de todos los partidos de su (temporada, gameweek)
```

Toda feature usa **únicamente partidos terminados antes de ese momento**. El ancla es el
inicio de la fecha: cuando se publica la tanda de predicciones, ninguna usó información de
la fecha misma.

**El mecanismo es `merge_asof`, no `shift(1)`**, y la diferencia no es de estilo:

> shift cuenta **partidos**. merge_asof cuenta **tiempo**.

Hay 85 pares (temporada, gameweek, equipo) donde el equipo juega **dos veces en la misma
fecha**. Con `shift(1)` el segundo partido usaría el resultado del primero, que se jugó
*después* del corte. Es leakage silencioso en ~5,6 % de las filas.
"""),
code("""
gold = read_table("gold_tp_match", layer="gold")
print(f"Gold: {len(gold):,} filas x {gold.shape[1]} columnas")

# La prueba de fuego, sobre datos reales: los dos partidos del Arsenal en la GW23 de
# 2022-23 tienen que compartir features, porque comparten el corte.
sub = gold[(gold.season == "2022-23") & (gold.gameweek == 23) &
           ((gold.home_short == "ARS") | (gold.away_short == "ARS"))]
filas = []
for _, r in sub.iterrows():
    lado = "local" if r.home_short == "ARS" else "visita"
    filas.append({"rival": r.away_short if lado == "local" else r.home_short,
                  "lado": lado,
                  "pts_u5": r[f"{lado}_pts_u5"], "xg_u5": round(r[f"{lado}_xg_u5"], 5),
                  "n_hist": r[f"{lado}_n_hist"], "elo": round(r[f"{lado}_elo"]),
                  "hist_kickoff": r[f"hist_kickoff_{lado}"], "corte": r["corte"]})
print("\\nDoble fecha: los dos partidos del Arsenal en la GW23 de 2022-23")
display(pd.DataFrame(filas))
print("Identicos. Con shift(1), el segundo habria visto el resultado del primero.")
"""),

md("""
### El control corre antes de escribir, no en los tests

`features/gold_tp.py` verifica, **en el momento de generar la tabla**, que toda la historia
usada sea anterior al corte. Un test que corre después ya llegó tarde para producción.

Además audita contra el **deadline de FPL**, que cae 90 minutos antes del corte: un
criterio todavía más estricto.
"""),
code("""
from transform import leakage

leakage.assert_no_banned_columns(gold, context="gold_tp_match")   # falla si hay prohibidas

for lado in ("local", "visita"):
    hk = gold[f"hist_kickoff_{lado}"]
    con_dato = hk.notna()
    assert (hk[con_dato] < gold.loc[con_dato, "corte"]).all()
print("OK: toda la historia usada es anterior al corte, en las 1.520 filas.")

audit = pd.read_csv(RAIZ / "features" / "output" / "gold_audit.csv")
print(f"\\nMargen minimo contra el deadline de FPL: {audit.margen_horas_min.min():.1f} horas")
"""),

md("""
---
## 3 · Las features

**159 columnas**, todas del equipo y mirando hacia atrás. El diccionario completo, campo
por campo con su fórmula, está en [`docs/FEATURES.md`](../docs/FEATURES.md) — y se
**genera** desde `features/spec.py`, con un test que falla si queda desfasado.
"""),
code("""
from features import spec

print(f"feature set {spec.FEATURE_SET_VERSION}: {len(spec.FEATURES)} features\\n")
for g, feats in spec.grupos().items():
    print(f"  {g:26s} {len(feats):3d}   ej: {feats[0].nombre}")
"""),

md("""
### Todo es por equipo, y aparece dos veces

El grano de Gold es un partido = una fila, y **cada estadística aparece dos veces**: una por
cada lado.

```
fila = Arsenal (local) vs Chelsea (visitante)

  local_pts_def_u5   = puntos FPL de LA DEFENSA DE ARSENAL en sus ultimos 5 partidos
  visita_pts_def_u5  = puntos FPL de LA DEFENSA DE CHELSEA en sus ultimos 5
```

La construcción pasa por una tabla intermedia con grano **equipo × partido** (3.040 filas),
donde cada equipo tiene su propia fila. Las ventanas se calculan ahí y recién al final se
pivotea a ancho.
"""),
code("""
ejemplo = gold[(gold.season == "2024-25") & (gold.gameweek == 30)].iloc[0]
print(f"{ejemplo.home_short} vs {ejemplo.away_short}  ({ejemplo.match_date.date()})\\n")
comp = pd.DataFrame({
    "local": [ejemplo[f"local_{c}"] for c in
              ("elo", "pos_tabla_camp", "pts_camp", "pts_def_u5", "pts_med_u5", "xg_u5")],
    "visita": [ejemplo[f"visita_{c}"] for c in
               ("elo", "pos_tabla_camp", "pts_camp", "pts_def_u5", "pts_med_u5", "xg_u5")],
}, index=["Elo", "posicion en la tabla", "puntos de campeonato",
          "puntaje defensa (u5)", "puntaje mediocampo (u5)", "xG (u5)"])
display(comp.round(2))
print("resultado real:", ejemplo.target_1x2)
"""),

md("""
### El Elo: lo que las medias móviles no pueden

Las ventanas rodantes **tratan igual a todos los rivales**: ganarle al último pesa lo mismo
que ganarle al primero. El Elo pondera cada resultado según contra quién fue, y por eso es
la feature clásica del fútbol.

Se valida solo: al cierre de 2024-25 tiene que poner arriba a los campeones y abajo a los
descendidos, **sin que le hayamos dicho quiénes son**.
"""),
code("""
ult = (gold[gold.season == "2024-25"]
       .melt(id_vars=["corte"], value_vars=["home_short", "away_short"],
             value_name="equipo")
       .drop(columns="variable"))
elos = pd.concat([
    gold[gold.season == "2024-25"][["corte", "home_short", "local_elo"]]
        .rename(columns={"home_short": "equipo", "local_elo": "elo"}),
    gold[gold.season == "2024-25"][["corte", "away_short", "visita_elo"]]
        .rename(columns={"away_short": "equipo", "visita_elo": "elo"}),
]).sort_values("corte").groupby("equipo").tail(1).sort_values("elo", ascending=False)

print("Elo al cierre de 2024-25\\n")
print("  TOP 5:"); display(elos.head(5)[["equipo", "elo"]].round(0))
print("  FONDO 3 (los tres descendidos reales fueron SOU, IPS, LEI):")
display(elos.tail(3)[["equipo", "elo"]].round(0))
"""),

md("""
---
## 4 · Los hallazgos de datos que cambiaron el diseño

Cinco cosas que aparecieron midiendo, no suponiendo. Cada una tiene su test.
"""),
code("""
# 1) El xG de 2022-23 viene HARDCODEADO EN CERO hasta la GW15.
p = read_table("fact_player_gw")
t = p.groupby(["season", "gameweek", "team_short"], as_index=False).agg(xg=("expected_goals", "sum"))
share = t.assign(cero=t.xg == 0).groupby("season")["cero"].mean()
print("share de equipo-fecha con xG == 0, por temporada:")
print(share.round(3).to_string())
print("\\n-> en 2022-23 es el 37,9%. No falta: viene 0,0 para los 20 equipos hasta la GW15.")
print("   Como cero, el modelo aprende que 'xG bajo' y 'arranque de temporada' van juntos:")
print("   un artefacto del calendario de publicacion de FPL. Se enmascara a NaN.")
"""),
code("""
# 2) expected_goals_conceded NO se puede sumar: se cuenta una vez por jugador.
q = p[p.season == "2024-25"].groupby(["fixture_id", "team_short"], as_index=False).agg(
    xg_sum=("expected_goals", "sum"),
    xgc_sum=("expected_goals_conceded", "sum"),
    gc=("goals_conceded", "max"), gf=("goals_scored", "sum"))
print("2024-25, medias por equipo-partido:")
print(f"  sum(expected_goals)            {q.xg_sum.mean():6.2f}   contra {q.gf.mean():.2f} goles convertidos  OK")
print(f"  sum(expected_goals_conceded)   {q.xgc_sum.mean():6.2f}   contra {q.gc.mean():.2f} goles concedidos   INFLADO x11")
print("\\n-> el xGC correcto es el xG DEL RIVAL en el mismo fixture.")
"""),
code("""
# 3) fpl_player_id NO es estable entre temporadas. Bloqueante para el Gold-FPL.
pj = p[p.minutes > 0]
chk = pj.groupby(["fpl_player_id", "season"])["player_name"].agg(lambda s: s.mode().iloc[0]).unstack()
inconsistentes = chk.apply(lambda r: r.dropna().nunique() > 1, axis=1)
print(f"ids que apuntan a MAS DE UN jugador segun la temporada: "
      f"{inconsistentes.sum()} de {len(chk)} ({inconsistentes.mean():.0%})\\n")
display(chk[inconsistentes].head(3))
print("-> el id 1 es Cedric Soares en 2022-23 y David Raya en 2025-26.")
print("   La clave entre temporadas es player_name. Con el id, la continuidad de")
print("   plantel daba 9,4% en vez del 61,3% real.")
"""),

md("""
---
## 5 · El modelo

Split **temporal, nunca aleatorio**: un split aleatorio pone partidos de mayo en el train y
de agosto en el test, el modelo ve el futuro y la métrica miente.

```
train    2022-23 + 2023-24   (sin las fechas con xG falso)
valid    2024-25             para el early stopping -- NUNCA el holdout
holdout  2025-26             380 partidos que el modelo no ve jamas
```

Usar el holdout para decidir cuándo parar es la forma sutil de contaminarlo: el número de
rondas pasaría a estar elegido con información del test.
"""),
code("""
from training import dataset, evaluate
from training.device import resolve

info = resolve("auto")
print(f"device: {info.used} ({info.reason})")
if info.gpu_name:
    print(f"GPU   : {info.gpu_name}")

sp = dataset.preparar(gold)
print(f"\\ntrain {len(sp.y_train)} | valid {len(sp.y_valid)} | holdout {len(sp.y_test)}")
"""),
code("""
res = evaluate.evaluar_holdout(CFG.modelo, info, gold=gold)
rep = res["reporte"]
ic = rep["accuracy_ic95"]

print(f"{CFG.modelo}  ({rep['n_train']} filas de train -> {rep['n']} de holdout)")
print(f"  rondas de boosting (las eligio el early stopping): {rep['best_iteration']}\\n")
print(f"  accuracy   {rep['accuracy']:.4f}   IC95 [{ic[0]:.3f}, {ic[1]:.3f}]")
print(f"  f1 macro   {rep['f1_macro']:.4f}")
print(f"  log-loss   {rep['log_loss']:.4f}")
print("\\n  las varas, sobre las MISMAS 380 filas:")
for k, v in rep["baselines"].items():
    if "accuracy" in v:
        ll = f"  log-loss {v['log_loss']:.4f}" if v.get("log_loss") else ""
        print(f"    {k:22s} accuracy {v['accuracy']:.4f}{ll}")
"""),

md("""
### El intervalo de confianza no es decorativo

Con 380 partidos el error estándar de la accuracy ronda **±5 puntos**. Diferencias chicas
entre modelos **no son distinguibles**, y reportar sólo el punto invita a concluir de más.
Por eso toda métrica va con su intervalo.
"""),

md("""
---
## 6 · El empate: la pregunta que más discutimos

La objeción natural al ver la matriz de confusión es *"si casi no predice empates, no
sirve"*. Los datos dicen lo contrario.
"""),
code("""
print("matriz de confusion (filas = real, columnas = predicho)\\n")
print("           " + "  ".join(f"{c:>6s}" for c in rep["clases"]))
for c, fila in zip(rep["clases"], rep["matriz_confusion"]):
    print(f"    {c:>6s} " + "  ".join(f"{v:6d}" for v in fila))
"""),
code("""
# El MERCADO REAL -- casas de apuestas con plata de verdad -- tampoco lo predice nunca.
h = gold[gold.season == CFG.holdout_season]
P_mkt = h[["p_mercado_away", "p_mercado_draw", "p_mercado_home"]].to_numpy()

print("El mercado sobre las mismas 380 fechas:")
print(f"  veces que el empate es su resultado mas probable : {int((P_mkt.argmax(1) == 1).sum())} de 380")
print(f"  probabilidad de empate: media {P_mkt[:,1].mean():.3f}, MAXIMO {P_mkt[:,1].max():.3f}")
print(f"  empates que realmente ocurrieron                 : {(h.target_1x2 == 'draw').mean():.1%}")
print()
print("-> el empate ocurre en 1 de cada 4 partidos pero NUNCA es el mas probable.")
print("   Con local ~43%, visitante ~30% y empate ~25%, siempre queda tercero.")
print("   Un modelo perfectamente calibrado tampoco lo pondria de argmax.")
"""),
code("""
# La prueba estructural: el Poisson bivariado NO predice la clase. Predice cuantos goles
# hace cada equipo y DEDUCE P(empate) como la diagonal de la distribucion conjunta.
# No puede "elegir" no predecirlo. Y aun asi llega al mismo lugar.
from training import models_alt as ma

full = gold[gold.season.isin(CFG.seasons_for_training())]
X = dataset.matriz(full, spec.FEATURES)
Xte = dataset.matriz(h, spec.FEATURES)
pb = ma.PoissonBivariado(device=info.used).fit(
    X, full.home_goals.to_numpy(), full.away_goals.to_numpy())
P_poi = pb.predict_proba(Xte)

print("Probabilidad de EMPATE que alcanza cada uno:\\n")
for n, P in [("Poisson bivariado", P_poi), (f"{CFG.modelo}", res["proba"]),
             ("mercado real", P_mkt)]:
    print(f"  {n:20s} media {P[:,1].mean():.3f}  maximo {P[:,1].max():.3f}  "
          f"argmax {int((P.argmax(1)==1).sum()):3d} veces")
print("\\n-> no es artefacto del clasificador: es la estructura del futbol.")
"""),

md("""
**¿Entonces hace falta predecir el empate?** Sí, aunque nunca sea el argmax, y por una
razón aritmética: **las tres probabilidades suman 1**. Subestimar el empate reparte esos
puntos entre local y visitante, e infla el valor esperado de *todas* las apuestas. Con
cuotas de 2 a 4, un punto de probabilidad de más son 2 a 4 puntos de EV inflado — y el
umbral de apuesta son 5 puntos.
"""),

md("""
---
## 7 · Qué features pesan

142 de 159 tienen ganancia mayor que cero: casi nada es peso muerto.
"""),
code("""
from training.run import _importancias

imp = _importancias(res["modelos"], res["features"])
display(imp.head(12).round(2))

imp["ventana"] = imp.feature.str.extract(r"_(u3|u5_temp|cond_u5|u5|camp)$")[0]
peso = (imp.groupby("ventana").ganancia.sum() / imp.ganancia.sum()).sort_values(ascending=False)
print("\\nEl 'periodo de tiempo a definir' del canvas, respondido con numeros:")
print(peso.round(3).to_string())
"""),

md("""
---
## 8 · Dónde le gana el modelo a cada vara

Un promedio global no dice si el modelo sirve. Lo que importa es **en qué situaciones**
gana, porque si se pueden identificar de antemano, se apuesta sólo ahí.
"""),
code("""
from training import analysis

an = analysis.correr(CFG.modelo)

print("Sobre las 38 fechas del holdout:")
for k, v in an["resumen_fechas"].items():
    print(f"  {k:32s} {v:.1%}" if isinstance(v, float) else f"  {k:32s} {v}")

print("\\n--- por favoritismo del mercado ---")
display(an["por_favoritismo"].round(3))
"""),
code("""
print("--- LA MEDICION QUE ORDENA TODO: cuando el modelo discrepa del mercado ---\\n")
display(an["por_acuerdo"].round(3))
print("Cuando el modelo tiene opinion propia, se equivoca MAS que el mercado.")
print("El modelo no tiene ventaja informativa sobre las casas: reproduce bien lo que")
print("el mercado ya sabe y agrega ruido cuando se aparta.")
print("\\nPara la propuesta de valor del bloque 1 -- ganar plata con apuestas --")
print("la conclusion honesta es que el sistema todavia NO la sostiene.")
"""),

md("""
---
## 9 · La simulación de apuestas

Es el bloque 6 del canvas, y **el único lugar donde entran las cuotas**. No son features del
modelo, y la razón es estructural:

> Si el modelo usa las cuotas como feature, aprende a copiarlas. Entonces `p ≈ 1/cuota`, el
> valor esperado da ~0 por construcción y **el sistema nunca encontraría una apuesta con
> valor**. Detectar una discrepancia exige que las dos estimaciones sean independientes.

```
EV = p x cuota - 1        # se apuesta si EV > 0,05
```
"""),
code("""
from training import betting

roi = betting.reporte(sp.filas_test, res["proba"])
m = roi["modelo"]
print(f"overround medio del mercado: {roi['overround_medio']:.3f} "
      f"(o sea ~{(roi['overround_medio']-1)*100:.1f}% de comision implicita)\\n")
print(f"  {m['n_apuestas']} apuestas | ROI {m['roi']:+.3f} | acierto {m['tasa_acierto']:.3f}")
print(f"  referencia 'siempre al local': ROI {roi['siempre_local']['roi']:+.3f}")
print("\\n--- por clase ---")
display(an["apuestas"].round(3))
"""),

md("""
**Cuidado con leer de más esta tabla.** En una corrida el empate dio ROI +0,092 y parecía la
única estrategia rentable. Medido con el error estándar:

```
apostando solo al empate:  n=101   ROI +0,029 +- 0,184
```

Con cuotas medias de 4,4 la varianza por apuesta es enorme y cien apuestas no alcanzan para
distinguir +3 % de cero. **No hay ninguna estrategia rentable demostrada.** El overround del
5,7 % sigue siendo la explicación más simple de todos los ROI observados.
"""),

md("""
---
## 10 · Walk-forward: la simulación del ciclo operativo

Para cada una de las 38 fechas se reentrena con **todo lo anterior al corte** y se predice
esa fecha. No es sólo validación: es el ciclo del bloque 9 corriendo de verdad.

*(Tarda unos minutos: son 38 reentrenamientos.)*
"""),
code("""
wf = evaluate.walk_forward(CFG.modelo, info, gold=gold)
resumen = evaluate.resumen_walk_forward(wf)
for k, v in resumen.items():
    print(f"  {k:34s} {v:.4f}" if isinstance(v, float) else f"  {k:34s} {v}")

print("\\nAccuracy fecha a fecha (las primeras 10):")
print("  " + "   ".join(f"GW{int(r.gameweek):02d} {r.accuracy:.2f}"
                        for r in wf.head(10).itertuples()))
print("\\n-> con 10 partidos por fecha, el error estandar de la accuracy es +-15,7 puntos.")
print("   ESA es la razon de que la promocion NO se decida sobre una sola fecha.")
"""),

md("""
### La regla de promoción

El canvas dice *"si le gana al de producción se pasa a producción"*. Tomado literalmente no
se puede medir:

| | |
|---|---|
| partidos por gameweek | 10 |
| error estándar de la accuracy con n=10 | **±15,7 puntos** |
| filas que agrega una semana sobre 1.140 | **+0,9 %** |

Por eso se separan las cadencias: **reentrenar semanal** (barato, y demuestra el ciclo), pero
**promover sólo con test estadístico** — McNemar pareado sobre 10 fechas. Como ambos modelos
predicen los mismos partidos, sólo cuentan aquellos donde **discrepan**, lo que lo hace mucho
más potente que comparar dos accuracies sueltas.
"""),
code("""
from training import promotion

# Un candidato que gana 6-4 en UNA fecha no se promueve.
d = promotion.decidir(np.array([True]*6 + [False]*4), np.array([False]*6 + [True]*4))
print(f"gana 6-4 en una fecha  -> promover: {d.promover}")
print(f"  motivo: {d.motivo}\\n")

# Una ventaja sostenida sobre 100 partidos si.
rng = np.random.default_rng(0)
prod = rng.random(100) < 0.42
cand = prod.copy(); cand[np.flatnonzero(~prod)[:20]] = True
d2 = promotion.decidir(cand, prod)
print(f"ventaja sostenida en 10 fechas -> promover: {d2.promover}")
print(f"  motivo: {d2.motivo}")
"""),

md("""
---
## 11 · GPU: la hipótesis se cayó

Antes de medir dejamos escrita esta predicción:

> *"Con 1.140 filas × 159 features la GPU pierde por un factor de 3 a 8, y el punto de cruce
> está entre 10⁵ y 3×10⁵ filas."*

**Las dos mitades resultaron falsas.** Medido en una GTX 1650, 200 árboles fijos, warmup
descartado, mediana de 5 corridas:

| Escala | Filas | CPU | GPU | Speedup |
|---|---|---|---|---|
| ×1 | 1.140 | 0,70 s | 1,20 s | **0,59×** — pierde, pero sólo 1,7× |
| ×10 | 11.400 | 2,08 s | 1,36 s | **1,53×** — ya gana |
| ×100 | 114.000 | 23,45 s | 4,37 s | **5,36×** |
| ×1.000 | 1.140.000 | 177,52 s | 33,69 s | **5,27×** |

El cruce está entre 1.140 y 11.400 filas: **un orden de magnitud antes** de lo predicho.

**Qué se concluye, con números:**

- Para este TP la GPU **no se justifica**: el Job de entrenamiento en la nube se aprovisiona
  sin GPU, y ahora es una decisión con evidencia.
- En GCP una T4 cuesta ~2,5-3× el nodo pelado: para pagarse necesita ser ≥3× más rápida,
  cosa que ocurre recién arriba de ~50.000 filas.
- **Para el otro proyecto sobre el mismo Silver sí se justifica**: `fact_player_gw` tiene
  113.270 filas, prácticamente la escala ×100, donde va 5,4× más rápido.
- El valor MLOps del código GPU es la **portabilidad**, no la velocidad: se entrena en un
  device y se sirve en otro. Hay un test que lo demuestra.

Para reproducirlo: `python -m training.benchmark_gpu`
"""),

md("""
---
## 12 · Por qué éste es el modelo elegido

Se corrió la grilla entera —**7 modelos × 3 variantes de datos**— con
`python -m training.compare_models`, y después el walk-forward de los finalistas.
"""),
code("""
comp = pd.read_csv(RAIZ / "training" / "output" / "comparacion_completa.csv")
print("=== ACCURACY por modelo y variante de datos (mercado 0,4947) ===")
display(comp.pivot(index="modelo", columns="datos", values="accuracy")
            .loc[["xgb_gbt","xgb_rf","hgb","logreg","poisson","ordinal","mlp"],
                 ["todo","sin_xg_falso","sin_2022_23"]].round(4))
"""),
code("""
wfc = pd.read_csv(RAIZ / "training" / "output" / "walkforward_candidatos.csv")
print("=== WALK-FORWARD de los finalistas, 38 fechas ===")
display(wfc.round(4))
"""),

md("""
### La decisión

**`xgb_gbt` (XGBoost), entrenado sin las fechas con xG falso.**

| Motivo | Evidencia |
|---|---|
| Mejor en la métrica principal del canvas | accuracy **0,516** en walk-forward y **0,503** en holdout — los dos protocolos coinciden |
| El que menos ignora el empate | F1 macro **0,4205**, contra 0,35-0,38 de los demás |
| Mejor en la métrica de valor del bloque 10 | le gana a "siempre local" en el **60,5 %** de las fechas, contra 52,6 % del resto |
| Listo para servir en la nube | el formato `.ubj` es portable CPU↔GPU y sobrevive upgrades de librería, a diferencia de un pickle de sklearn |

`xgb_rf` tiene mejor log-loss (1,0328 contra 1,0390), pero la diferencia es de 0,006 —
ruido a esta escala— y pierde 2,3 puntos de accuracy y 7 de F1 macro.

**La red neuronal quedó de las peores** (0,426, apenas por encima del baseline trivial). Con
1.140 filas y 159 features es el caso de manual donde una red sobreajusta. Ya no es
intuición: está medido.
"""),

md("""
---
## 13 · Qué falta: la fase de nube

Lo que está listo para el deploy:

- **`data/gold/gold_tp_match.parquet`** — la tabla de features, reproducible con un comando.
- **`models/xgb_gbt/<version>/model.ubj`** — el modelo serializado en formato portable.
- **`metadata.json`** — el contrato con el serving: `feature_names` **ordenado** (si el
  serving arma las columnas en otro orden, XGBoost no se queja y devuelve basura),
  `classes_`, hiperparámetros, el prior de ascendidos congelado, versiones de librerías,
  `git_sha` y el hash de Gold.
- **`common/storage.py`** — la capa de I/O con backend intercambiable. Migrar a GCS es
  implementar seis métodos y cambiar una línea de `config.yaml`; la lógica de negocio no se
  toca.
- **`training/promotion.py`** — la regla de promoción, con el registro de intentos
  rechazados.

Lo que falta escribir:

| Carpeta | Qué va |
|---|---|
| `serving/` | `app.py` (FastAPI) y `Dockerfile`. El endpoint carga el `.ubj`, valida el orden de features contra el metadata y devuelve las tres probabilidades |
| `monitoring/` | recolección del resultado real y métricas rodantes contra los baselines del mismo período |
| `infra/` | Cloud Run + Cloud Scheduler (dos disparos por semana), Artifact Registry, y el bucket de GCS |

**Decisión de arquitectura ya tomada con números:** el Job de entrenamiento **sin GPU**. A
1.140 filas no se paga, y está medido dónde empezaría a pagarse.
"""),
code("""
import json
from training import registry

v = registry.produccion(CFG.modelo)
if v is None:
    from pathlib import Path as _P
    dirs = sorted((RAIZ / "models" / CFG.modelo).glob("2*"))
    v = registry.Version(CFG.modelo, dirs[-1].name, dirs[-1]) if dirs else None

if v:
    meta = json.loads(v.metadata.read_text(encoding="utf-8"))
    print(f"modelo persistido: {v.ruta.name}\\n")
    for k in ("feature_set_version", "n_features", "classes_", "best_iteration",
              "device_used", "n_train", "n_test", "git_sha"):
        print(f"  {k:22s} {meta.get(k)}")
    print(f"\\n  las primeras 5 features, EN ORDEN: {meta['feature_names'][:5]}")
    print("  (el serving valida este orden exacto: XGBoost no avisa si se lo cambian)")
else:
    print("No hay modelo persistido. Corre: python -m training.run --model xgb_gbt")
"""),

md("""
---
## Resumen para la defensa

**Lo que funciona:**

- Pipeline completo Bronze → Silver → Gold → modelo, reproducible con cinco comandos y sin
  ninguna credencial.
- Control anti-leakage que **corre antes de escribir**, con prueba de fuego para cada
  hallazgo. Margen medido contra el deadline de FPL: 22,5 horas.
- El modelo le gana al baseline del canvas (0,503 contra 0,426) y **empata la accuracy del
  mercado sin usar cuotas**.
- Todas las decisiones tomadas con medición: el feature set, la variante de datos, el
  modelo, y la decisión de no usar GPU en la nube.

**Lo que no funciona, dicho sin maquillar:**

- **El modelo no le gana al mercado.** Cuando discrepa de las casas, acierta 0,346 contra
  0,365. No tiene ventaja informativa.
- **Ninguna estrategia de apuestas resultó rentable.** Todos los ROI son negativos, y el
  único positivo que apareció quedó dentro del error estándar.
- Para la propuesta de valor del bloque 1 —un emprendimiento que gane con apuestas— **el
  sistema todavía no la sostiene**. Lo que sí sostiene es el ciclo de MLOps completo, que
  es donde está el peso de la nota.

**Tres cosas que aprendimos midiendo, y que no se ven en el resultado final:**

1. El empate no es una falla del modelo: ni el mercado ni un Poisson bivariado lo ponen
   nunca como resultado más probable. Es una propiedad del fútbol.
2. Entrenar más no mejora: con lr 0,01 y 6.000 rondas se llega exactamente al mismo lugar
   que con 184. El límite es cuánta señal hay en 1.140 partidos.
3. Menos datos pueden ser mejores: sacar las fechas con xG falso mejora 5 de 7 modelos,
   porque esas filas enseñan un artefacto del calendario de publicación de FPL.
"""),
]


def main() -> None:
    nb = {
        "cells": CELDAS,
        "metadata": {
            "kernelspec": {"display_name": "tp-premier-ml", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.14"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    DESTINO.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    n_code = sum(1 for c in CELDAS if c["cell_type"] == "code")
    print(f"Escrito {DESTINO}")
    print(f"{len(CELDAS)} celdas ({n_code} de codigo, {len(CELDAS) - n_code} de texto)")


if __name__ == "__main__":
    main()
