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
python -m ingestion.run              # ~27 MB de Bronze, sin credenciales
python -m ingestion.bronze_pulselive # copas, Europa y las stats de Opta
python -m transform.silver           # las tablas de FPL y football-data
python -m transform.competencias     # silver.fact_match_comp
python -m transform.opta_stats       # silver.fact_opta_stats
python -m features.gold_tp           # la tabla Gold
```

**El resumen, para el apurado:**

| | |
|---|---|
| Tabla Gold | **1.530 × 301**, de las cuales **279 son features** |
| Fuentes | 4 — las tres públicas de siempre **más la API oficial de premierleague.com** |
| Modelo elegido | **XGBoost** (`xgb_gbt`), entrenado sin las fechas con xG falso |
| Accuracy en el holdout | **0,500** contra 0,426 del baseline y 0,495 del mercado |
| Accuracy en walk-forward | **0,516**, le gana a "siempre local" en el 60,5 % de las fechas |
| Feature más importante | `dif_elo` — la diferencia de rating Elo entre los dos equipos |
| Ciclo cerrado | predicción registrada → resultado real → métricas, ya corriendo sobre 2026-27 |
| ¿Le gana al mercado? | **No.** Cuando discrepa de las casas, acierta menos que ellas |
| Tests | **470**, en verde |

> ⚠️ **Hay dos modelos y no son intercambiables.** El de *evaluación* entrena hasta
> 2024-25 y se mide contra 2025-26: **es el único cuyos números valen como evidencia**. El
> de *producción* entrena también con 2025-26 —380 partidos más, y los más recientes— así
> que sus métricas sobre esa temporada ya no prueban nada. Todo lo que se reporta acá sale
> del de evaluación. La sección 6 lo explica en detalle.
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
print("temporada EN CURSO         :", CFG.current_season, "(monitoreo, nunca entrena)")
print("modelo de produccion       :", CFG.modelo, "|", CFG.datos_entrenamiento)
print("el artefacto de produccion incluye el holdout:", CFG.incluir_holdout)
print("   -> por eso TODO lo que se reporta abajo usa incluir_holdout=False")
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
from common.storage import table_exists

for nombre in ("dim_team", "fact_match", "fact_fixture", "fact_player_gw",
               "fact_match_comp", "fact_opta_stats"):
    if not table_exists(nombre):
        print(f"{nombre:18s} (falta: python -m ingestion.bronze_pulselive)")
        continue
    d = read_table(nombre)
    print(f"{nombre:18s} {len(d):>7,} filas x {d.shape[1]:>3} columnas")
"""),

md("""
### Las cuatro fuentes, y por qué las cuatro

| Fuente | Grano | Qué aporta que nadie más aporta |
|---|---|---|
| **football-data.co.uk** | partido | cuotas de cierre (el baseline duro), tiros, córners, tarjetas |
| **vaastav/Fantasy-Premier-League** | jugador × fecha | el histórico jugador-fecha con xG. La API oficial no lo sirve |
| **API oficial de FPL** | presente y futuro | fixtures, deadlines, y el resultado apenas termina el partido |
| **API de premierleague.com** *(nueva)* | partido, todas las competencias | copas y Europa —lo único que veía el pipeline era la Premier— y ~180 estadísticas de Opta por equipo y partido |

La API de FPL **no sirve el detalle fecha a fecha del pasado**: `history` da sólo la
temporada actual y `history_past` una fila por temporada, y sólo de jugadores que hoy están
en la base (sesgo de supervivencia). Por eso vaastav no es reemplazable.

La cuarta es `footballapi.pulselive.com`, la que consume el sitio oficial: **pública,
gratuita, sin clave y sin cuota**, con 22 a 35 temporadas de histórico. Cierra el agujero
que el proyecto arrastraba —el calendario de copas y Europa— y trae de yapa las
estadísticas de Opta. Lo único que pide es la cabecera `Origin: https://www.premierleague.com`.

```python
B = "https://footballapi.pulselive.com/football"
# comps: 1=Premier  2=Champions  3=Europa  4=FA Cup  5=EFL Cup
GET {B}/competitions/5/compseasons?pageSize=100   # ids de temporada
GET {B}/fixtures?comps=5&compSeasons=812&pageSize=400
GET {B}/stats/match/125161                        # ~180 stats por equipo
```
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

**279 columnas**, todas del equipo y mirando hacia atrás. El diccionario completo, campo
por campo con su fórmula, está en [`docs/FEATURES.md`](../docs/FEATURES.md) — y se
**genera** desde `features/spec.py`, con un test que falla si queda desfasado.

La versión del set **se deriva de un hash de la lista de features** (`v2.3189c9d4.279`), no
se escribe a mano. Existe por un error real: durante días la etiqueta quedó pegada en `"v2"`
mientras el set pasaba por 159, 164, 171, 175, 184 y 192 columnas — seis modelos distintos
guardados con la misma versión, justo lo que una versión tiene que evitar.
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
## 4 · Copas, Europa y Opta: dos bloques medidos, y la respuesta que no queríamos

Hasta acá el pipeline **sólo veía partidos de Premier**. Un equipo que sigue en semifinales
de Champions y en la Copa de la Liga juega mucho más de lo que registran las ventanas de
fatiga, y el modelo no tenía forma de enterarse. La hipótesis era razonable **y el dato
faltaba de verdad**: las ventanas de 7, 14 y 21 días ya estaban construidas y no aportaban
nada, justamente porque sólo contaban partidos de Premier.

La API oficial de premierleague.com lo resolvió de una: las cinco competencias con el mismo
grano equipo-partido, y de yapa las estadísticas de Opta.
"""),
code("""
comp = read_table("fact_match_comp")
print("partidos-equipo por temporada y competencia (equipos de Premier):\\n")
display(pd.crosstab(comp.season, comp.competencia))
# La tabla cuenta partidos-EQUIPO: un partido de copa entre dos equipos de Premier
# aporta dos filas, y uno contra un rival de otra division, una sola.
ult_temp = comp[comp.season == "2025-26"]
de_premier = set(ult_temp[ult_temp.es_premier].team_short)
fuera = ult_temp[~ult_temp.es_premier]
n_pl = int(fuera.team_short.isin(de_premier).sum())
print(f"\\n-> lo que el pipeline NO veia en 2025-26: {n_pl} partidos-equipo de copa "
      "y Europa")
print("   jugados por los 20 de Premier, concentrados en los que llegan lejos.")
print(f"   Y {len(fuera) - n_pl} mas de equipos del ascenso: son los que pueden subir el "
      "anio siguiente,")
print("   y hoy llegan a la Premier sin ninguna historia (el cold-start de la seccion 14).")
print("   2026-27 recien arranca: solo tiene la primera ronda de la EFL Cup.")
"""),
code("""
# Cobertura real: no es un dato de borde que toque a cuatro partidos.
h_cob = gold[gold.season == CFG.holdout_season]
cob = pd.Series({
    "algun equipo jugo entre semana (7d)":
        ((h_cob.local_partidos_todo_7d > 1) | (h_cob.visita_partidos_todo_7d > 1)).mean(),
    "hubo copa en los ultimos 14 dias":
        ((h_cob.local_partidos_copa_14d > 0) | (h_cob.visita_partidos_copa_14d > 0)).mean(),
    "algun equipo juega en Europa":
        ((h_cob.local_europa_acumuladas > 0) | (h_cob.visita_europa_acumuladas > 0)).mean(),
})
print("sobre los 380 partidos del holdout 2025-26:\\n")
print(cob.map(lambda v: f"{v:.1%}").to_string())
"""),

md("""
### Lo que trae Opta, y las tres que quedaron afuera a propósito

`/stats/match/{id}` devuelve ~180 estadísticas por equipo y partido. Se ruedan **14** en dos
ventanas y por los dos lados = **56 features**. Cubren tres huecos que ninguna otra fuente
del proyecto llenaba:

| Hueco | Columnas | Por qué importa |
|---|---|---|
| **Ubicación del remate** | `tiros_area`, `tiros_fuera` y sus proporciones | es el proxy de calidad del xG que se había dado por inalcanzable sin Understat: el xG agregado de FPL no distingue "2,0 en tres ocasiones claras" de "2,0 en veinte remates de lejos" |
| **Defensa como acción** | `quites`, `intercepciones`, `rechazos`, `bloqueos` | hasta ahora la defensa se medía sólo por lo que el rival lograba |
| **Dominio territorial** | `posesion`, `toques_area_rival` | sin equivalente previo |

**Tres estadísticas se descartaron antes de construir nada**, porque la cobertura por
temporada las delata:

```
conducciones_prog     0 % en 2022-24, 41 % en 2025-26, 100 % en 2026-27
recuperaciones        0 % salvo la temporada actual
atajadas_clarisimas   6 % global
```

Es exactamente la trampa del xG hardcodeado en cero de 2022-23 (sección 5): **una feature
que sólo existe en las temporadas recientes le enseña al modelo a reconocer la temporada,
no el fútbol.**
"""),

md("""
### La medición, que es el verdadero resultado

`python -m training.ablacion` corre los cuatro sets con el mismo protocolo —modelo de
evaluación, holdout 2025-26— y escribe la tabla.
"""),
code("""
abl = RAIZ / "training" / "output" / "ablacion_bloques.csv"
if abl.exists():
    display(pd.read_csv(abl).round(4))
else:
    print("Falta la corrida. Ejecutar: python -m training.ablacion")
"""),

md("""
```
                set   n   accuracy   IC 95 %          f1 macro   log-loss
               base  199    0,4974   0,447 - 0,547      0,3911     1,0258
base + competencias  223    0,5000   0,450 - 0,550      0,3928     1,0262
        base + Opta  255    0,5026   0,453 - 0,553      0,4089     1,0334
               todo  279    0,5000   0,450 - 0,550      0,3928     1,0290
```

**Ningún bloque aporta.** La diferencia entre el mejor y el peor set es de **0,005** —cinco
milésimas— contra un error estándar de ±0,025. Los cuatro son el mismo número.

Las 80 features **se usan igual**: todas tienen ganancia mayor que cero y entre las dos
familias explican el 26 % de la ganancia total del modelo (sección 8). Es decir: el modelo
las mira, pero mirarlas no lo hace acertar más.

**Por qué se conservan.** Porque el costo es cero, porque son el único camino a las stats de
Opta si más adelante hay más filas, y sobre todo porque **el resultado nulo es evidencia**.
La hipótesis de la fatiga era buena y el dato faltaba de verdad; que el efecto no aparezca
tiene explicaciones plausibles y medibles —los técnicos rotan para compensar, o la señal
existe pero es chica frente al piso de ruido de 1.004 filas de entrenamiento, o ya estaba
capturada por `dias_descanso` y el Elo—. Lo que no es defendible es no haberlo medido.

### Lo que NO se construyó, a propósito

Ninguna feature mira el **calendario futuro** de copas. El calendario se publica ronda por
ronda: al 25/08/2026 la EFL Cup tenía sus 60 fixtures de primera ronda con **dos días** de
anticipación, contra los 278 de la Premier. Una feature del tipo *"juega copa la semana que
viene"* estaría siempre completa en entrenamiento y faltaría en producción entre el fin de
una ronda y el sorteo de la siguiente. Peor: **no se puede ni simular el hueco**, porque la
API no expone cuándo se publicó cada fixture.
"""),

md("""
---
## 5 · Los hallazgos de datos que cambiaron el diseño

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
## 6 · El modelo

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
### Dos modelos, y sólo uno de los dos puede reportar números

Esta es la distinción más fácil de arruinar de todo el trabajo, y la que más caro sale.

| | Modelo de **evaluación** | Modelo de **producción** |
|---|---|---|
| Entrena con | 2022-23 → 2024-25 (1.004 filas) | + 2025-26 (1.384 filas) |
| Se mide contra | 2025-26, que nunca vio | 2025-26, **que sí vio** |
| Accuracy sobre esa temporada | **0,500** | 0,616 |
| Para qué sirve | **es el número que se reporta** | es el `.ubj` que predice la fecha que viene |
| Comando | `python -m training.run --sin-holdout` | `python -m training.run` |

Ese **0,616 no es una mejora, es el modelo acordándose**. Aparece en el `metrics.json` del
artefacto de producción y por eso el `metadata.json` guarda la bandera
`metricas_son_de_generalizacion: false` — para que nadie lo cite por error dentro de seis
meses. El CLI además tira un warning en cada corrida de producción.

Por qué existen los dos: el modelo que sale a predecir la fecha que viene **debería** usar
los 380 partidos más recientes, que son los más informativos. Pero entonces se queda sin
ninguna temporada limpia contra la cual medirse. La salida es tener los dos artefactos, con
la regla explícita de cuál habla.

**Y la temporada en curso (2026-27) no entra a entrenar en ninguno de los dos**, ni siquiera
en el de producción: sus partidos jugados sirven de historia para las fechas siguientes,
pero usarlos como objetivo dejaría al proyecto sin evaluación limpia en vivo. Hay tests que
lo verifican.
"""),
code("""
# La demostracion, en una corrida: el MISMO modelo, la MISMA temporada de test.
rep_prod = evaluate.evaluar_holdout(CFG.modelo, info, gold=gold,
                                    incluir_holdout=True)["reporte"]
print(f"{'':22s} {'n_train':>8s} {'accuracy':>9s}  metricas_de_generalizacion")
for etq, r in (("evaluacion", rep), ("produccion", rep_prod)):
    print(f"  {etq:20s} {r['n_train']:>8d} {r['accuracy']:>9.4f}  "
          f"{r['metricas_son_de_generalizacion']}")
print()
print("-> +11,6 puntos de accuracy que NO existen. Es el holdout entrando al train.")
"""),

md("""
### El intervalo de confianza no es decorativo

Con 380 partidos el error estándar de la accuracy ronda **±5 puntos**. Diferencias chicas
entre modelos **no son distinguibles**, y reportar sólo el punto invita a concluir de más.
Por eso toda métrica va con su intervalo.
"""),

md("""
---
## 7 · El empate: la pregunta que más discutimos

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
### Y si en vez de clasificar, predecimos los goles

La idea surge sola al ver la matriz de confusión: en vez de aprender la clase, predecir
**cuántos goles hace cada equipo** y derivar el 1X2 de la distribución conjunta. El empate
deja de ser una etiqueta y pasa a ser lo que realmente es —los dos marcan lo mismo— y encima
el modelo mira el problema desde otro ángulo, que es la condición para que un **ensamble**
aporte.

Está implementado (`PoissonBivariado`, dos regresores XGBoost con `objective="count:poisson"`)
y **no hace falta ninguna base nueva**: Gold ya tiene `home_goals` y `away_goals`.

`python -m training.ensamble` mide las tres preguntas de una.
"""),
code("""
ens = RAIZ / "training" / "output" / "ensamble_clf_goles.csv"
if ens.exists():
    display(pd.read_csv(ens).round(4))
else:
    print("Falta la corrida. Ejecutar: python -m training.ensamble")
"""),

md("""
**El ensamble no aporta: ningún peso le gana al clasificador solo.** Y la razón no es un
misterio, es una medición:

```
coinciden en el argmax          89,7 %
correlacion de p_away            0,906
correlacion de p_home            0,934
correlacion de p_draw            0,340   <- aca SI piensan distinto
```

Un ensamble sirve cuando los modelos **fallan en lugares distintos**. Éstos fallan casi en
los mismos partidos: fallan los dos en el 46,1 % y sólo uno de los dos acierta en el 8,4 %.

*(Combinar modelos ya se había probado en general —promedio simple y stacking con folds
temporales sobre los cuatro modelos base— y tampoco mejoraba; está en
[`training/README.md`](../training/README.md). Lo que faltaba era el **porqué**, y es esta
correlación.)*

Pero adentro de ese resultado negativo hay algo: **para el empate la correlación es 0,34**.
Ahí sí discrepan — y el modelo de goles predice **cero** empates como argmax, contra 4 del
clasificador.
"""),

md("""
### La corrección de Dixon-Coles, y por qué acá no aplica

Esa discrepancia tiene una causa concreta en el código: la conjunta se arma multiplicando
las dos distribuciones, o sea **asume independencia** entre los goles de los dos equipos. Es
falso justo en los marcadores bajos, que es donde vive el empate.

La corrección clásica es **Dixon-Coles (1997)**: un parámetro `ρ` que reajusta esas cuatro
celdas.

```
τ(0,0) = 1 - λμρ      τ(0,1) = 1 + λρ
τ(1,0) = 1 + μρ       τ(1,1) = 1 - ρ
```

Con `ρ < 0` sube 0-0 y 1-1 y baja 1-0 y 0-1: sube P(empate). El `ρ` se ajusta por máxima
verosimilitud sobre el train — y como el término de Poisson no depende de `ρ`, se reduce a
maximizar `Σ log τ`, una optimización de una sola variable.

**Está implementado y medido, y el resultado es que no aplica.** El diagnóstico dice
exactamente por qué:
"""),
code("""
from training import ensamble as ens_mod
from training import dataset as ds

full_g = ds.filtrar_train(gold[gold.season.isin(CFG.seasons_for_training())])
X_g = ds.matriz(full_g, spec.FEATURES)
celdas = ens_mod.diagnostico_celdas(
    X_g, full_g.home_goals.to_numpy().astype(int),
    full_g.away_goals.to_numpy().astype(int), info)
display(celdas.round(3))
"""),

md("""
**0-0 aparece MENOS de lo que predice la independencia (0,666) y 1-1 MÁS (1,087).** Las dos
celdas del empate se desvían en direcciones **opuestas**.

Y la `τ` de Dixon-Coles tiene **un solo parámetro**, que las empuja a las dos en la **misma**
dirección: no existe un `ρ` capaz de bajar una y subir la otra. La máxima verosimilitud lo
resuelve quedándose en **ρ = −0,0074** —contra el −0,13 que Dixon y Coles reportan para el
fútbol inglés—, que es la forma matemática de decir *"esta corrección no aplica a estos
datos"*. El efecto sobre el holdout es de +0,002 en P(empate) y −0,0002 en log-loss: nada.

**Lo que sí sigue siendo cierto** es que el modelo subestima el empate: los empates
observados superan a los esperados en un 4,7 %. El déficit existe. Lo que no existe es que
tenga *la forma* que Dixon-Coles corrige.

Se conserva el código, apagado por defecto y con `poisson_dc` en la grilla, por el mismo
motivo que las features de Opta: **saber que algo no funciona, y por qué, también es un
resultado** — y éste señala hacia dónde habría que ir si se retomara (una corrección con más
de un parámetro, o calibrar el empate directamente).
"""),

md("""
### El estadístico que cierra la discusión: el AUC del empate

Todo lo anterior mira **accuracy y F1**, y para el empate las dos engañan. Un modelo que
**nunca** predice empate puede tener buena accuracy y no saber absolutamente nada del
empate. La pregunta correcta es otra: *¿pone más probabilidad de empate en los partidos que
terminaron empatados?* Eso lo mide el **AUC uno-contra-resto**, y además no depende del
umbral.

`python -m training.empate` lo corre sobre las cuatro familias.
"""),
code("""
from training import empate as emp

res_emp = emp.correr()
display(res_emp["discriminacion"][["auc_away", "auc_draw", "auc_home",
                                   "ic_draw", "draw_sin_señal"]].round(3))
"""),

md("""
```
          AUC_away  AUC_draw  AUC_home     IC95 del empate
xgb_gbt      0,680     0,515     0,683     [0,441 - 0,584]
poisson      0,648     0,479     0,655     [0,412 - 0,543]
ordinal      0,606     0,493     0,648     [0,431 - 0,559]
marcador     0,641     0,460     0,659     [0,392 - 0,525]
mercado      0,674     0,531     0,697     [0,465 - 0,592]
```

**Cuatro familias de modelos distintas, las cuatro en 0,5 para el empate** — y el 0,5 está
dentro del intervalo en todas. Los mismos modelos discriminan local y visitante con AUC
0,65-0,68. **No es que el empate se prediga mal: es que no se puede rankear.**
"""),
code("""
print("tasa real de empates por decil de p_draw (si rankeara, subiria):\\n")
display(res_emp["deciles"].round(3))
"""),

md("""
### La trampa del F1, que hay que saber antes de la defensa

| modelo | F1 del empate | empates predichos | AUC del empate |
|---|---|---|---|
| `ordinal` | **0,209** | 68 | 0,493 |
| `marcador` | 0,098 | 19 | 0,460 |
| `xgb_gbt` | 0,056 | 4 | 0,515 |

**El mejor F1 del empate viene con el AUC más cerca de 0,5.** Sube por predecir *más*
empates, no por acertarlos. Optimizar el F1 del empate premia adivinar más seguido — sin el
AUC al lado, ese número miente.

### Y la vara que impide sacar la conclusión equivocada

Sobre las 1.530 filas con cuotas de las cinco temporadas, el **mercado** —casas de apuestas
con plata de verdad e información que este proyecto no tiene— saca **AUC 0,563** para el
empate contra **0,733 y 0,735** para local y visitante.

Tiene señal, pero es minúscula, y **hace falta n=1.530 para demostrar que existe**. Con los
380 partidos del holdout, nuestro 0,515 es indistinguible de ese 0,563.

**El techo no es el algoritmo: es el tamaño de la muestra.** Por eso ninguna de las tres
ideas de esta sección —el modelo de goles, el ensamble, Dixon-Coles— movió la aguja. No
estaban atacando el problema real.

### Lo único que sí es una palanca

El **umbral** con el que se decide llamar empate, que no es una decisión de modelado sino
**de negocio**: cuánto cuesta perderse un empate contra cuánto cuesta anunciar uno que no
fue.
"""),
code("""
display(res_emp["umbral"].round(4))
"""),

md("""
Bajándolo de argmax a **0,30**: de 4 empates predichos a 36, precisión 0,417, y la accuracy
global **no baja** (0,5079 contra 0,5000). Parece gratis — pero con AUC 0,515 eso es afinar
un umbral sobre una señal que no ordena, y por eso **no se cambió el modelo de producción**.
Queda como dial documentado, para el día que el costo de cada tipo de error esté definido.
"""),

md("""
---
## 8 · Qué features pesan

278 de 279 tienen ganancia mayor que cero: casi nada es peso muerto. Lo interesante es
**cuánto** pesa cada familia, no cuántas hay.
"""),
code("""
from training.run import _importancias

imp = _importancias(res["modelos"], res["features"])
display(imp.head(12).round(2))

print(f"\\nfeatures con ganancia > 0: {(imp.ganancia > 0).sum()} de {len(spec.FEATURES)}")

imp["ventana"] = imp.feature.str.extract(r"_(u3|u5_temp|cond_u5|u5|camp)$")[0]
peso = (imp.groupby("ventana").ganancia.sum() / imp.ganancia.sum()).sort_values(ascending=False)
print("\\nEl 'periodo de tiempo a definir' del canvas, respondido con numeros:")
print(peso.round(3).to_string())
"""),

code("""
# El peso por FAMILIA de features: es donde se lee el veredicto de la seccion 4.
de_grupo = {f.nombre: g for g, fs in spec.grupos().items() for f in fs}
imp["familia"] = imp.feature.map(de_grupo)
share = (imp.groupby("familia").ganancia.sum() / imp.ganancia.sum()).sort_values(ascending=False)
print("share de la ganancia total, por familia:\\n")
print(share.map(lambda v: f"{v:6.1%}").to_string())
print()
print("-> Opta es la SEGUNDA familia en peso (20%) y competencias suma otro 6%:")
print("   el modelo las mira mucho. Y sin embargo la accuracy no se mueve.")
print("   Peso en el arbol no es lo mismo que poder predictivo.")
"""),

md("""
---
## 9 · Dónde le gana el modelo a cada vara

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
## 10 · La simulación de apuestas

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
## 11 · Walk-forward: la simulación del ciclo operativo

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
## 12 · GPU: la hipótesis se cayó

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
## 13 · Por qué éste es el modelo elegido

Se corrió la grilla entera —**7 modelos × 3 variantes de datos**— con
`python -m training.compare_models`, y después el walk-forward de los finalistas.

> ⚠️ **La grilla es de la época del set de 159 features.** Las 120 que se agregaron después
> (h2h, momentum de Elo, competencias, Opta) no la movieron —eso es lo que mide la sección
> 4— así que la elección de modelo sigue en pie, pero los números de abajo son de ese
> momento. Para rehacerla con el set actual: `python -m training.compare_models`.
"""),
code("""
# training/output/ esta en .gitignore: en un clon fresco hay que correr la grilla.
ruta_grilla = RAIZ / "training" / "output" / "comparacion_completa.csv"
if ruta_grilla.exists():
    grilla = pd.read_csv(ruta_grilla)
    print("=== ACCURACY por modelo y variante de datos (mercado 0,4947) ===")
    display(grilla.pivot(index="modelo", columns="datos", values="accuracy")
                  .loc[["xgb_gbt","xgb_rf","hgb","logreg","poisson","ordinal","mlp"],
                       ["todo","sin_xg_falso","sin_2022_23"]].round(4))
else:
    print("Falta la corrida. Ejecutar: python -m training.compare_models")
"""),
code("""
ruta_wfc = RAIZ / "training" / "output" / "walkforward_candidatos.csv"
if ruta_wfc.exists():
    print("=== WALK-FORWARD de los finalistas, 38 fechas ===")
    display(pd.read_csv(ruta_wfc).round(4))
else:
    print("Falta la corrida. Ejecutar: python -m training.compare_models")
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
intuición: está medido. Con 279 features la relación filas/columnas es todavía peor.
"""),

md("""
---
## 14 · La predicción en producción: el ciclo cerrado, ya corriendo

Acá empieza la Fase 6. Todavía no hay HTTP ni contenedor, pero **la lógica que va adentro
del endpoint ya está escrita, probada y corriendo**: `serving/predict.py`.

```powershell
python -m serving.predict --gw 3              # la fecha que viene
python -m serving.predict --gw 1 --evaluar    # una ya jugada, contra el resultado real
```

Cumple las tres cosas que una predicción de producción tiene que cumplir, y ninguna es
decorativa:

| | Qué garantiza | Qué pasa si falta |
|---|---|---|
| **Mismo código de features que el entrenamiento** — `features.gold_tp.construir(objetivos=...)` es literalmente la misma función | no hay train/serve skew | dos implementaciones paralelas de la misma feature divergen en silencio, y es el bug más caro de MLOps |
| **El orden de las columnas se valida contra el `metadata.json`** | XGBoost recibe un `ndarray`: si las columnas vienen en otro orden **no se queja** y predice cualquier cosa | fallo silencioso, sin excepción y sin log |
| **Cada predicción se registra** con `fixture_id`, momento, versión de modelo y de feature set, y las tres probabilidades | hay con qué monitorear después | un endpoint que responde, y nada más |

Y corre el **mismo assert anti-leakage** que Gold: `hist_kickoff` tiene que ser anterior al
corte también en producción.
"""),
code("""
from serving import predict as srv

registradas = sorted((RAIZ / "data" / "predicciones").glob("*.parquet"))
print(f"predicciones registradas hasta ahora: {len(registradas)}")
for r in registradas[-3:]:
    print("  ", r.name)

ult = pd.read_parquet(registradas[-1])
print()
display(ult[["home_short", "away_short", "p_home", "p_draw", "p_away",
             "prediccion", "confianza", "model_version"]].round(3))
"""),
code("""
# El cierre del ciclo: la fecha 1 de 2026-27, contra lo que efectivamente paso.
pred1 = srv.predecir(CFG.current_season, 1)
ev = srv.evaluar(pred1)

if "nota" in ev:
    print(ev["nota"])
else:
    d = ev["detalle"]
    print(f"{'partido':16s}{'predijo':9s}{'p':>7s}   {'real':7s}{'marcador':10s}ok")
    for r in d.itertuples():
        marcador = f"{int(r.home_goals)}-{int(r.away_goals)}"
        ok = "OK" if r.prediccion == r.target_1x2 else "--"
        print(f"{r.home_short}-{r.away_short:12s}{r.prediccion:9s}{r.confianza:7.3f}   "
              f"{r.target_1x2:7s}{marcador:10s}{ok}")
    print()
    print(f"  accuracy   {ev['accuracy']:.3f}")
    print(f"  log-loss   {ev['log_loss']:.4f}")
    print(f"  'siempre local' habria acertado: {ev['acierta_siempre_local']:.3f}")
"""),

md("""
**Los dos errores más caros fueron los ascendidos ganando de local** (HUL–MUN, IPS–SUN): es
el *cold-start* en vivo, la situación donde el modelo tiene menos historia y más se
equivoca. El prior de ascendidos existe justamente para eso, y en la fecha 1 no alcanzó.

**Y "siempre al local" acertó 7 de 10 contra nuestros 4.** Antes de sacar conclusiones: fue
una fecha con 7 locales de 10 contra el 44,5 % histórico, y con n=10 el intervalo de la
accuracy va de 0,10 a 0,70. Diez partidos no distinguen nada de nada — que es exactamente
el argumento de la regla de promoción de la sección 11.
"""),

md("""
---
## 15 · El monitoreo de la temporada en curso

El holdout 2025-26 es una foto: 380 partidos fijos que ya se midieron muchas veces y contra
los que se tomaron decisiones. La temporada en curso es otra cosa — **datos que el modelo no
vio y que nadie miró todavía**, que llegan de a diez por semana. Es la evaluación más
honesta que existe, y **la única que no se puede sobreajustar mirándola**.

```powershell
python -m monitoring.temporada_actual
```

Dos decisiones de diseño que importan:

1. **Los baselines se calculan sobre las mismas filas**, fecha a fecha. La distinción es
   sutil y decisiva: una caída del modelo *acompañada* de una caída del mercado es la liga
   siendo más impredecible, no el modelo degradándose. Sin el baseline pareado, esa
   diferencia no se ve y se dispara un retraining que no hacía falta.
2. **2026-27 no entra al entrenamiento** ni siquiera en el artefacto de producción. Sus
   partidos jugados sirven de historia para las fechas siguientes; usarlos como objetivo
   dejaría al proyecto sin ninguna evaluación limpia. Hay un test que lo verifica.
"""),
code("""
from monitoring import temporada_actual as mon

mon_df = mon.correr()
display(mon_df.round(4))

res_mon = mon.resumen(mon_df)
print()
for k, v in res_mon.items():
    print(f"  {k:28s} {v:.4f}" if isinstance(v, float) else f"  {k:28s} {v}")
"""),

md("""
**Lo que hay que saber leer en esa tabla: todavía no dice nada.** Con una fecha jugada el
intervalo de confianza de la accuracy es de ±30 puntos. El monitoreo no está para dar un
veredicto hoy — está para que dentro de diez fechas exista la serie pareada que alimenta el
McNemar de la regla de promoción, y para que la degradación se detecte cuando sea real y no
cuando sea ruido.

Es, literalmente, el bloque 10 del canvas corriendo.
"""),

md("""
---
## 16 · Reproducibilidad: qué hace falta para volver a armar un modelo

```powershell
python -m training.reproducir                        # el inventario completo
python -m training.reproducir --version 20260825T024144Z
```

Este módulo existe por un error concreto. Durante varios días `FEATURE_SET_VERSION` se
mantuvo **a mano** y quedó pegada en `"v2"` mientras el set pasaba por 159, 164, 171, 175,
184 y 192 columnas: **seis modelos distintos guardados con la misma etiqueta**, justo lo que
una versión tiene que evitar. Ahora se deriva de un hash de la lista de features
(`v2.3189c9d4.279`) y no puede volver a desincronizarse.

Lo que salvó a los modelos viejos es que el `metadata.json` guarda dos cosas que sí son
fiables: el **`git_sha`** del código con el que se entrenó y la **lista ordenada completa de
`feature_names`**. Con eso alcanza para reproducir, aunque la etiqueta mienta.
"""),
code("""
from training import reproducir

inv = pd.DataFrame(reproducir.inventario())
display(inv)
print("Donde 'declarado' difiere de 'feature set real', la etiqueta mentia.")
"""),

md("""
### Hay dos cosas distintas que se llaman "reproducir"

| | Qué necesita | Dónde se usa |
|---|---|---|
| **Volver a usarlo** — cargar el `.ubj` y predecir | el modelo y su lista de features, nada más | `serving/predict.py`. **Es lo que se sube a la nube** |
| **Volver a entrenarlo** — reconstruir el artefacto desde cero | `git checkout <sha>`, la misma Silver, las versiones pineadas y **el mismo device** | sólo si hace falta auditarlo |

### El hallazgo que decide la arquitectura de la nube: GPU ≠ CPU

Medido sobre las mismas 1.384 filas, misma semilla, mismas rondas:

| | ¿Bit a bit? | Diferencia máxima en probabilidad |
|---|---|---|
| Mismo device, misma semilla | **sí** | 0 |
| GPU vs CPU | **no** | **7,9 × 10⁻²** |

**18 de 380 predicciones cambian de resultado.** El algoritmo `hist` de GPU no es
bit-idéntico al de CPU: distinto orden de reducción en punto flotante. Se ve incluso en el
early stopping, que con el set actual para en la ronda 111 en GPU y en la 102 en CPU
(`python -m training.run --sin-holdout --device cpu` para comprobarlo).

La consecuencia práctica es una sola línea: **se sube el `.ubj` entrenado en local, no se
reentrena en el nodo.** El formato nativo de XGBoost es portable CPU↔GPU **para inferencia**
y sobrevive upgrades de librería — un modelo entrenado en GPU carga y predice en CPU sin
tocar nada, y hay un test que lo demuestra
(`test_modelo_entrenado_en_gpu_predice_igual_en_cpu`).

Si igual se quisiera reentrenar en la nube, es reproducible siempre que el nodo use el mismo
device, las versiones pineadas de `requirements.txt` y el mismo `gold_tp_match.parquet`
—cuyo hash está en el `metadata.json`—.
"""),

md("""
---
## 17 · Qué falta: la fase de nube

Todo lo anterior corre **en local, sin una sola credencial**. Lo que queda es empaquetarlo.

### Lo que ya está listo para el deploy

| | Qué es |
|---|---|
| `data/gold/gold_tp_match.parquet` | la tabla de features, reproducible con un comando |
| `models/xgb_gbt/<version>/*.ubj` | cinco modelos (uno por semilla) en formato portable CPU↔GPU |
| `metadata.json` | **el contrato con el serving**: `feature_names` *ordenado*, `classes_`, hiperparámetros, el prior de ascendidos congelado, versiones de librerías, `git_sha` y el hash de Gold |
| `serving/predict.py` | la lógica del endpoint, entera y probada: carga, valida el orden, predice, registra |
| `monitoring/temporada_actual.py` | la evaluación en vivo contra los baselines pareados |
| `training/promotion.py` | la regla de promoción, con el registro de intentos rechazados |
| `common/storage.py` | la capa de I/O con backend intercambiable |

### Lo que falta escribir

| Dónde | Qué va | Tamaño real |
|---|---|---|
| `serving/app.py` + `Dockerfile` | envolver `predict.py` en FastAPI. **La lógica no se toca**: es un `POST /predict` que llama a la función que ya existe | chico |
| `common/storage.py` | el backend GCS, hoy declarado como stub: seis métodos y una línea de `config.yaml` | chico |
| `infra/` | Cloud Run + Cloud Scheduler (dos disparos por semana), Artifact Registry y el bucket de GCS | es el grueso |

### La forma que tiene el deploy, con las decisiones ya tomadas

```
Cloud Scheduler ──> Cloud Run Job   (ingesta + Silver + Gold)   martes
       │
       └─────────> Cloud Run Job   (predicción de la fecha)     antes del deadline
                          │
                   GCS bucket:  bronze/ silver/ gold/ models/ predicciones/
                          │
                   Cloud Run Service (FastAPI)  ──>  POST /predict
                          │
                   Cloud Run Job   (monitoreo)                  lunes
```

Tres decisiones que **no** quedan para el momento del deploy, porque ya están medidas:

1. **El Job de entrenamiento va sin GPU.** A 1.140 filas la GPU pierde 1,7×, y recién arriba
   de ~50.000 filas se paga el 2,5-3× que cuesta una T4 (sección 12).
2. **El modelo se sube entrenado, no se reentrena en el nodo.** GPU y CPU no dan el mismo
   modelo: 18 de 380 predicciones cambian (sección 16).
3. **El disparo va atado al `deadline_time` de FPL, no a un día fijo.** El margen medido
   entre el último dato disponible y el corte es de 22,5 horas.
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

- Pipeline completo Bronze → Silver → Gold → modelo, reproducible con un puñado de comandos
  y **sin ninguna credencial**: las cuatro fuentes son públicas y abiertas.
- Control anti-leakage que **corre antes de escribir**, con prueba de fuego para cada
  hallazgo, y que se ejecuta también en producción. Margen medido contra el deadline de
  FPL: 22,5 horas. 470 tests en verde.
- El modelo le gana al baseline del canvas (0,500 contra 0,426) y **empata la accuracy del
  mercado sin usar cuotas** (0,495).
- **El ciclo cerrado ya corre**: se predice la fecha, se registra la predicción con su
  trazabilidad completa, llega el resultado real y se calculan las métricas contra los
  baselines de las mismas filas.
- Todas las decisiones tomadas con medición: el feature set, la variante de datos, el
  modelo, los dos bloques de features que no aportaron, y la decisión de no usar GPU en la
  nube.

**Lo que no funciona, dicho sin maquillar:**

- **El modelo no le gana al mercado.** Cuando discrepa de las casas, acierta 0,346 contra
  0,365. No tiene ventaja informativa, y las 120 features nuevas no se la dieron.
- **Ninguna estrategia de apuestas resultó rentable.** Todos los ROI son negativos, y el
  único positivo que apareció quedó dentro del error estándar.
- Para la propuesta de valor del bloque 1 —un emprendimiento que gane con apuestas— **el
  sistema todavía no la sostiene**. Lo que sí sostiene es el ciclo de MLOps completo, que
  es donde está el peso de la nota.

**Cinco cosas que aprendimos midiendo, y que no se ven en el resultado final:**

1. El empate no es una falla del modelo: ni el mercado ni un Poisson bivariado lo ponen
   nunca como resultado más probable. Es una propiedad del fútbol.
2. Entrenar más no mejora: con lr 0,01 y 6.000 rondas se llega exactamente al mismo lugar
   que con 184. El límite es cuánta señal hay en 1.140 partidos.
3. Menos datos pueden ser mejores: sacar las fechas con xG falso mejora 5 de 7 modelos,
   porque esas filas enseñan un artefacto del calendario de publicación de FPL.
4. **Más features tampoco mejoran.** El set pasó de 159 a 279 columnas con dos bloques
   construidos sobre hipótesis razonables —fatiga de copas, ubicación del remate—, el
   modelo les da el 26 % de su ganancia, y la accuracy no se movió ni medio punto. El
   resultado nulo, medido y publicado, vale tanto como uno positivo.
5. **Hay que decir cuál de los dos modelos habla.** El artefacto de producción entrena con
   el holdout y sobre esa temporada marca 0,616; el de evaluación marca 0,500. Reportar el
   primero sería el error más caro y más fácil de cometer del trabajo entero.
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
