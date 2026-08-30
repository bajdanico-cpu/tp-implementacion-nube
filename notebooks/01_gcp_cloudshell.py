"""Genera `notebooks/01_gcp_cloudshell.ipynb`, el lab del TP en Cloud Shell.

Mismo criterio que `00_recorrido_completo.py`: el notebook se produce desde acá y no se
edita a mano, así el diff es legible y no se versionan salidas de ejecución.

    python notebooks/01_gcp_cloudshell.py

**Alcance deliberado.** El notebook llega hasta donde llegó la clase: proyecto, bucket, dato
en la nube, Gold, un modelo entrenado y registrado, y la predicción de una fecha corriendo
como batch. No levanta servicios ni construye imágenes: eso es la clase que viene y acá no
se adelanta.

El Paso 7 predice las fechas 1 y 2 **y demuestra que no hay leakage temporal**, que es el
riesgo real de correr esto mientras se juega una fecha.

**La última celda borra todo.** No es una cortesía: el lab no puede dejar nada prendido
comiendo crédito.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

DESTINO = Path(__file__).with_name("01_gcp_cloudshell.ipynb")


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
# TP Premier ML en GCP — del dato crudo al modelo entrenado

Este notebook es el lab del TP en la nube. Se corre en el **editor de Cloud Shell**, con el
repo clonado. Cada paso es **un concepto, una celda que apretás, y un lugar de la consola
donde ver el recurso que apareció**.

**Antes de arrancar:**

- Cuenta de GCP activa y un proyecto con billing.
- API de Cloud Storage habilitada (el Paso 1 lo verifica).
- Cloud Shell abierto, este repo clonado, y estar parado en `notebooks/`.

```bash
git clone <url-del-repo> tp-premier-ml
cd tp-premier-ml
pip install -q -r requirements-cloud.txt
```

> **Por qué `requirements-cloud.txt` y no `requirements.txt`.** El de local está pinneado a
> wheels `cp314` (Python 3.14.3). Cloud Shell trae otro Python, y para esas versiones
> exactas no hay wheel: pip intentaría compilar numpy y pandas desde el fuente y la sesión
> se cae. El de nube usa cotas en vez de versiones clavadas. **Es la primera lección real
> del deploy: el entorno de la nube no es el de tu máquina, y eso hay que decidirlo, no
> descubrirlo.**

---

## El clon viene vacío de datos, y es a propósito

`data/` y los `.ubj` **están en `.gitignore`**: el repo no trae ni un byte de dato ni un
modelo. No es un olvido —es la decisión de diseño— por dos razones:

- **Todo se regenera con un comando** desde cuatro fuentes públicas. Versionar 27 MB de
  parquet sólo lograría que el repo pese y que el dato del repo se desactualice respecto del
  real.
- **El dato versionado miente rápido.** La temporada en curso cambia cada fecha: un parquet
  commiteado en agosto describe un mundo que en septiembre ya no existe.

Consecuencia práctica para este lab: **las primeras celdas bajan y transforman todo**. El
Paso 2 hace la ingesta (~27 MB, unos minutos), el Paso 3 arma Silver y el Paso 4 arma Gold.
Recién ahí hay con qué entrenar. **No hay nada que subir a mano ni que copiar desde tu
máquina.**

Es, además, la prueba de que el pipeline es reproducible de verdad: si funciona sobre un
clon limpio en una máquina que nunca vio el proyecto, funciona.

---

## Qué hace este notebook, y qué no

Llega hasta **la predicción de una fecha, con su registro y su prueba de que no miró el
futuro**:

```
proyecto → bucket → Bronze → Silver → Gold → modelo → predicción → limpieza
```

**No** levanta ningún servicio, **no** construye ninguna imagen y **no** deja ningún recurso
prendido. La predicción corre como un batch, igual que el entrenamiento. Envolverla en una
API y desplegarla es la etapa siguiente y acá no se adelanta nada.

El monitoreo de la temporada en curso también está implementado y se ve en
[`00_recorrido_completo.ipynb`](00_recorrido_completo.ipynb).

**El caso.** Predecir el resultado (gana local / empate / gana visitante) de cada fecha de
la Premier League. Lo que tiene este dominio y casi ningún otro: el ground truth **llega
solo**, dos horas después de la predicción.

> ⚠️ **La última celda borra todo lo que este lab creó en la nube.** Correla siempre antes
> de cerrar la sesión. Lo que se prende, se paga.

> ⚠️ **Si corrés esto mientras se está jugando una fecha**, leé el Paso 7 antes de sacar
> conclusiones. Es el momento exacto en que el leakage temporal se cuela, y hay una celda
> dedicada a demostrar que no pasó.
"""),

md("""
---
## Paso 0 — Dónde estás parado

Lo primero en la nube es siempre lo mismo: confirmar **en qué proyecto** vas a trabajar.
Todo lo que crees hoy vive adentro de ese proyecto.

En el editor de Cloud Shell el proyecto **no siempre se autodetecta**. Si la celda imprime
el placeholder, cambiá esa línea por el ID que ves arriba a la izquierda en la consola.
"""),
code("""
import os, sys, subprocess
from pathlib import Path

# El notebook vive en notebooks/; el repo es el padre. Todos los modulos se importan
# desde ahi, igual que en local: NO hay una copia distinta del codigo para la nube.
RAIZ = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(RAIZ))
os.chdir(RAIZ)

PROJECT_ID = (
    os.environ.get("GOOGLE_CLOUD_PROJECT")
    or os.environ.get("DEVSHELL_PROJECT_ID")
    or "cambiame-por-tu-project-id"     # <-- el ID de TU proyecto
)

# us-central1 es la region de la cursada. config.yaml propone southamerica-east1, que
# esta mas cerca pero tiene menos servicios: para el lab conviene la de clase.
REGION = "us-central1"
BUCKET = f"{PROJECT_ID}-premier-ml"

print("repo    :", RAIZ)
print("python  :", sys.version.split()[0])
print("proyecto:", PROJECT_ID)
print("region  :", REGION)
print("bucket  :", BUCKET)
"""),

md("""
---
## Paso 1 — La nube te responde

Un proyecto de GCP nace con casi todo **apagado**: cada servicio hay que habilitar su API.
Habilitar es gratis; se paga el uso.

Si la celda lista buckets (aunque sea una lista vacía), Cloud Storage responde y estás
autenticado. El cliente de Python usa las credenciales de Cloud Shell (ADC) solo.

> Si faltara la API: `gcloud services enable storage.googleapis.com` en la **terminal**
> (no acá: el kernel del editor no hereda el proyecto ni la auth de `gcloud`).
"""),
code("""
from google.cloud import storage

gcs = storage.Client(project=PROJECT_ID)

print("Buckets del proyecto:")
for b in gcs.list_buckets():
    print(" -", b.name)
"""),

md("""
---
## Paso 2 — El dato a la nube (Bronze)

El TP toma datos de **cuatro fuentes públicas, ninguna con credenciales**. La ingesta baja
~27 MB y los escribe en Bronze **append-only**: cada corrida va a su propia partición
`ingested_at=<timestamp>` y **nunca sobrescribe**.

Eso no es prolijidad. El snapshot tomado **antes** del deadline es el único que refleja lo
que se sabía al momento de predecir; conservarlo junto al posterior es la defensa auditable
contra el leakage. En un bucket la propiedad se mantiene igual: son prefijos distintos.

*(Tarda unos minutos. Las temporadas cerradas salen de caché si ya se bajaron; la actual se
re-baja siempre, porque cambia.)*

**Andá a ver:** al final del paso, *Cloud Storage → tu bucket → `bronze/`*.
"""),
code("""
def correr(modulo, *args):
    \"\"\"Corre un modulo del repo como subproceso y muestra el final del log.\"\"\"
    r = subprocess.run([sys.executable, "-m", modulo, *args],
                       capture_output=True, text=True, cwd=RAIZ)
    salida = (r.stdout + r.stderr).strip().splitlines()
    print("\\n".join(salida[-18:]))
    if r.returncode != 0:
        raise RuntimeError(f"{modulo} fallo con codigo {r.returncode}")

correr("ingestion.run")
"""),
code("""
# La cuarta fuente: la API oficial de premierleague.com. Copas, Europa y las
# estadisticas de Opta. Publica, sin clave y sin cuota.
correr("ingestion.bronze_pulselive")
"""),
code("""
# Creamos el bucket si no existe y subimos Bronze entero.
bucket = gcs.bucket(BUCKET)
if bucket.exists():
    print("El bucket ya existe:", BUCKET)
else:
    bucket = gcs.create_bucket(BUCKET, location=REGION)
    print("Bucket creado:", BUCKET)


def subir(carpeta_local, prefijo):
    \"\"\"Sube un arbol de archivos al bucket, conservando la estructura de rutas.\"\"\"
    base = RAIZ / carpeta_local
    n, mb = 0, 0.0
    for f in sorted(base.rglob("*")):
        if not f.is_file():
            continue
        bucket.blob(f"{prefijo}/{f.relative_to(base).as_posix()}").upload_from_filename(f)
        n += 1
        mb += f.stat().st_size / 1e6
    print(f"{n} archivos, {mb:.1f} MB -> gs://{BUCKET}/{prefijo}/")
    return n


subir("data/bronze", "bronze")
"""),

md("""
---
## Paso 3 — Normalizar (Silver)

Bronze es crudo; Silver es la misma información **normalizada y cruzada**. Acá se resuelve
el problema aburrido y decisivo: los nombres de equipo no coinciden entre fuentes
(`Man Utd` ↔ `Man United`, `Spurs` ↔ `Tottenham`), y FPL es inconsistente consigo mismo.

La clave canónica es `short_name` (ARS, MUN, TOT…), que resultó **100 % estable** entre
temporadas — a diferencia del `id`, que FPL reasigna todos los años.

**Mirá el log:** tiene que decir **100 % de cruce** en las cuatro temporadas cerradas. Si no
cruza, todo lo que viene después está mal y no se nota.
"""),
code("""
correr("transform.silver")
correr("transform.competencias")
correr("transform.opta_stats")
"""),
code("""
subir("data/silver", "silver")
"""),

md("""
---
## Paso 4 — Las features, con el control que corre antes de escribir (Gold)

Gold es **una fila por partido y 279 features**, todas del equipo y **mirando hacia atrás**.

El riesgo que puede arruinar el trabajo entero es el **leakage temporal**: los datos de una
fecha se conocen *después* de que se jugó. La regla es una sola:

```
corte(partido) = min(kickoff_time) de todos los partidos de su (temporada, gameweek)
```

y toda feature usa **únicamente partidos terminados antes de ese corte**.

Lo importante para la nube: **ese control corre acá adentro, antes de escribir la tabla**, no
en una suite de tests aparte. Un pipeline que sólo valida en CI puede escribir datos
contaminados en producción y enterarse el lunes.

**Mirá el log:** la línea `Controles anti-leakage OK` con el margen contra el deadline.
"""),
code("""
correr("features.gold_tp")
"""),
code("""
import pandas as pd

gold = pd.read_parquet(RAIZ / "data" / "gold" / "gold_tp_match.parquet")
print(f"Gold: {len(gold):,} filas x {gold.shape[1]} columnas")
print()
print(gold.groupby(["season", "split"], observed=True).size().to_string())
"""),
code("""
subir("data/gold", "gold")
"""),

md("""
---
## Paso 5 — Entrenar, y la distinción que más caro sale

Acá se pasa de datos a modelo. **Hay dos modelos y no son intercambiables:**

| | Modelo de **evaluación** | Modelo de **producción** |
|---|---|---|
| Entrena con | 2022-23 → 2024-25 | + 2025-26 |
| Se mide contra | 2025-26, que nunca vio | 2025-26, **que sí vio** |
| Para qué sirve | **es el número que se reporta** | es el que predeciría la fecha que viene |
| Comando | `training.run --sin-holdout` | `training.run` |

El de producción marca 0,616 sobre esa temporada, pero **eso no es una mejora: es el modelo
acordándose**. Por eso el `metadata.json` guarda `metricas_son_de_generalizacion: false`, y
el CLI tira un warning en cada corrida de producción.

**Y acá entrena en CPU.** En local hay una GTX 1650 y el pipeline la usa; Cloud Shell no
tiene GPU y `device: auto` cae a CPU con un warning. Todo lo demás sigue igual: **la GPU no
está en el camino crítico de nada**. Está medido (`training.benchmark_gpu`): a 1.140 filas la
GPU *pierde* 1,7×, y recién arriba de ~50.000 filas se paga el 2,5-3× que cuesta una T4. Por
eso, cuando se despliegue, el Job de entrenamiento va **sin GPU** — y es una decisión con
número, no una intuición.
"""),
code("""
correr("training.device")
"""),
code("""
# El que se REPORTA: entrena hasta 2024-25 y se mide contra 2025-26, que nunca vio.
correr("training.run", "--sin-holdout", "--no-guardar")
"""),
code("""
# El que SIRVE: incluye 2025-26 (380 partidos mas, y los mas recientes). Este SI se
# persiste, porque es el artefacto que iria a predecir.
correr("training.run")
"""),

md("""
---
## Paso 6 — Registrar y versionar el modelo

Un modelo sin registrar es un archivo perdido. Cada corrida escribe
`models/xgb_gbt/<timestamp>/` con:

- **cinco `.ubj`**, uno por semilla (las probabilidades se promedian);
- **`metadata.json`** — el contrato con el serving: `feature_names` **ordenado**, `classes_`,
  hiperparámetros, el prior de ascendidos congelado, versiones de librerías, `git_sha` y el
  hash de Gold;
- **`metrics.json`** e **`importancias.csv`** — la evidencia.

El orden de las features no es un detalle: XGBoost recibe un `ndarray` y **si las columnas
vienen en otro orden no se queja**, predice cualquier cosa. Es un fallo silencioso, y por eso
el orden se persiste y se valida.

> **Por qué `.ubj` y no un pickle.** El formato nativo de XGBoost sobrevive upgrades de
> librería y **no arrastra el device**: un modelo entrenado en GPU carga y predice en CPU sin
> tocar nada. Hay un test que lo demuestra.

**Andá a ver:** *Cloud Storage → tu bucket → `models/`*.
"""),
code("""
import json
from training import registry
from common.config import CFG

v = registry.produccion(CFG.modelo)
if v is None:
    dirs = sorted((RAIZ / "models" / CFG.modelo).glob("2*"))
    v = registry.Version(CFG.modelo, dirs[-1].name, dirs[-1])

meta = json.loads(v.metadata.read_text(encoding="utf-8"))
print("version:", v.ruta.name)
for k in ("feature_set_version", "n_features", "classes_", "best_iteration",
          "device_used", "n_train", "seasons_entrenadas",
          "metricas_son_de_generalizacion", "git_sha"):
    print(f"  {k:32s} {meta.get(k)}")
print()
print("  primeras 5 features, EN ORDEN:", meta["feature_names"][:5])
"""),
code("""
subir("models", "models")
"""),

md("""
---
## Paso 7 — Predecir una fecha, y probar que no miró el futuro

Esta es la inferencia. `serving/predict.py` es **la lógica que iría adentro de un endpoint**:
ya está escrita y probada, y hace tres cosas que ninguna es decorativa.

| | Qué garantiza | Qué pasa si falta |
|---|---|---|
| Usa **el mismo código de features** que el entrenamiento — `features.gold_tp.construir(objetivos=...)` es literalmente la misma función | no hay train/serve skew | dos implementaciones paralelas de la misma feature divergen en silencio, y es el bug más caro de MLOps |
| **Valida el orden de las columnas** contra el `metadata.json` | el modelo recibe lo que espera | XGBoost recibe un `ndarray`: con las columnas en otro orden **no se queja**, predice cualquier cosa. Fallo silencioso |
| **Registra cada predicción** con fixture, momento, versión de modelo y de feature set | hay con qué medir después | un endpoint que responde, y nada más |

Predecimos **dos** fechas, y son casos distintos a propósito: la **1** ya se jugó entera, así
que se puede comparar contra el resultado real; la **2** se está jugando ahora mismo, que es
el caso peligroso.
"""),
code("""
correr("serving.predict", "--gw", "1")
"""),
code("""
correr("serving.predict", "--gw", "2")
"""),

md("""
### El riesgo real: predecir una fecha que se está jugando

Los datos de un partido se conocen **después** de que se jugó. Si una feature de la fecha N
usa datos de la fecha N, el modelo está viendo el resultado: la accuracy se dispara y el
trabajo entero no vale nada.

Y correr esto **durante** una fecha es el momento en que eso se cuela sin que nadie lo note,
porque la información llega de a pedazos: al momento de escribir esto la fecha 2 tiene 8 de
10 partidos con marcador en la API de FPL, dos todavía sin jugar, y varios jugadores con
minutos cargados de partidos en curso.

La defensa no es acordarse de no mirar. Es una regla, aplicada por el código:

```
corte(partido) = min(kickoff_time) de todos los partidos de su (temporada, gameweek)
```

Toda feature usa **únicamente partidos terminados antes de ese corte**. El corte es el inicio
de la fecha, así que **ningún partido de la fecha 2 puede entrar en las features de la fecha
2**, ni siquiera los que ya terminaron.

El mecanismo es `merge_asof`, **no** `shift(1)`: *shift cuenta partidos, merge_asof cuenta
tiempo*. Y la prueba queda guardada en la propia predicción: `hist_kickoff_local` y
`hist_kickoff_visita` son el kickoff del último partido efectivamente usado.

La celda de abajo compara ese máximo contra el corte. **Si alguna vez diera negativo, hay
leakage** — y de hecho `serving/predict.py` levanta un `AssertionError` antes de llegar acá.
"""),
code("""
import pandas as pd

def evidencia(gw):
    # El ultimo partido usado como historia vs. el corte de la fecha.
    arch = sorted((RAIZ / "data" / "predicciones").glob(f"*_GW{gw:02d}_*.parquet"))[-1]
    p = pd.read_parquet(arch)
    hist = pd.concat([p["hist_kickoff_local"], p["hist_kickoff_visita"]]).max()
    corte = p["kickoff_time"].min()
    return {"fecha": f"GW{gw}", "partidos": len(p),
            "ultimo partido usado como historia": hist,
            "corte de la fecha": corte,
            "margen": corte - hist,
            "sin leakage": bool(hist < corte)}

ev = pd.DataFrame([evidencia(1), evidencia(2)]).set_index("fecha")
for f, fila in ev.iterrows():
    print(f)
    for k, v in fila.items():
        print(f"    {k:36s} {v}")
    print()
"""),

md("""
Leído con calendario en la mano:

- **Fecha 1** — la última historia es del **24/05/2026**: el cierre de la temporada pasada.
  Todavía no se había jugado nada de 2026-27, así que el margen es de casi 90 días.
- **Fecha 2** — la última historia es del **24/08/2026 19:00**, que es FUL–CHE: el último
  partido de la fecha 1. El corte de la fecha 2 es el 28/08 19:00. O sea: **la fecha 1 entra
  como historia —que es lo correcto y lo que queremos— y la fecha 2 no entra en absoluto**,
  aunque 8 de sus partidos ya se hayan jugado.

**La prueba que convence de verdad es la de abajo.** Si la fecha 1 estuviera filtrando su
propio resultado, el modelo acertaría 10 de 10. Acierta 4.
"""),
code("""
correr("serving.predict", "--gw", "1", "--evaluar", "--no-guardar")
"""),

md("""
Cuatro de diez, contra siete de "siempre al local". Antes de sacar conclusiones: fue una
fecha con **7 locales de 10** contra el 44,5 % histórico, y con n=10 el intervalo de la
accuracy va de 0,10 a 0,70. **Diez partidos no distinguen nada de nada** — que es
exactamente el argumento por el cual la regla de promoción del modelo no se decide sobre una
sola fecha.

Los dos errores más caros fueron los ascendidos ganando de local (HUL–MUN, IPS–SUN): es el
*cold-start* en vivo, la situación donde el modelo tiene menos historia y más se equivoca.

### ¿Y la fecha 2?

Todavía no se puede evaluar, y el pipeline lo dice en vez de inventar un número. La
evaluación cruza contra `fact_match`, que sale de football-data —la fuente con las cuotas—,
y esa fuente publica la fecha **cuando termina entera**. Los marcadores parciales que hoy
tiene la API de FPL no alcanzan.

Es un detalle que vale la pena mirar de frente: **el ground truth llega solo, pero llega
cuando quiere la fuente, no cuando lo necesitamos**. Un sistema honesto espera; uno apurado
completa con lo que tiene y se miente.
"""),
code("""
subir("data/predicciones", "predicciones")
"""),

md("""
---
## Paso 8 — Qué quedó en la nube

El pipeline entero corrió en Cloud Shell, **sin una sola credencial de datos**, y dejó el
bucket con la estructura medallion completa más el modelo y su trazabilidad.
"""),
code("""
from collections import defaultdict

resumen = defaultdict(lambda: [0, 0.0])
for b in gcs.list_blobs(BUCKET):
    resumen[b.name.split("/")[0]][0] += 1
    resumen[b.name.split("/")[0]][1] += (b.size or 0) / 1e6

print(f"gs://{BUCKET}/\\n")
total = 0.0
for p in sorted(resumen):
    n, mb = resumen[p]
    total += mb
    print(f"  {p + '/':16s} {n:>5} objetos   {mb:>8.1f} MB")
print(f"\\n  {'TOTAL':16s} {'':>5}           {total:>8.1f} MB")
"""),

md("""
---
## Paso 9 — Limpieza: apagar todo

**Regla de oro de la nube: lo que prendés, cuesta.** Y lo que cuesta sin que lo mires es lo
que se come el crédito.

Este lab tuvo la ventaja de no desplegar nada: **no hay endpoint, ni máquina, ni job
programado**. El único recurso con costo es el bucket. Igual la celda de abajo **borra todo**,
por dos motivos:

1. **El crédito de la cuenta es finito** y el bucket sigue facturando almacenamiento mientras
   exista, aunque nadie lo toque.
2. **Todo esto se regenera con este mismo notebook.** El dato es público, el pipeline es
   reproducible y el modelo se vuelve a entrenar en minutos. No hay nada acá que valga la
   pena conservar pagando.

La celda borra los objetos y después el bucket, y **verifica** que no haya quedado nada.
"""),
code("""
# --- BORRA EL BUCKET Y TODO SU CONTENIDO ---
bucket = gcs.bucket(BUCKET)

if not bucket.exists():
    print("El bucket ya no existe:", BUCKET)
else:
    blobs = list(gcs.list_blobs(BUCKET))
    print(f"Borrando {len(blobs)} objetos de gs://{BUCKET}/ ...")
    # force=True borra los objetos y despues el bucket. Con muchos objetos la API
    # pide hacerlo por lotes, asi que los borramos a mano primero.
    for lote in (blobs[i:i + 100] for i in range(0, len(blobs), 100)):
        bucket.delete_blobs(lote)
    bucket.delete()
    print("Bucket borrado:", BUCKET)
"""),
code("""
# Verificacion: que NO quede nada en el proyecto.
restantes = [b.name for b in gcs.list_buckets()]
print("Buckets que quedan en el proyecto:", restantes or "ninguno")

if BUCKET in restantes:
    print("\\n  !! El bucket sigue ahi. Borralo desde la consola o desde la terminal:")
    print(f"     gcloud storage rm -r gs://{BUCKET}")
else:
    print("\\n  OK: no queda nada de este lab facturando.")
"""),

md("""
### Lo último, desde la **terminal** (no desde acá)

El kernel del editor no tiene `gcloud` autenticado, así que estos dos chequeos van en la
terminal de Cloud Shell. Son los que cierran de verdad:

```bash
# 1) Confirmar que no hay NINGUN recurso con costo corriendo.
gcloud storage ls                                    # ningun bucket del lab
gcloud run services list --region us-central1        # vacio
gcloud ai endpoints list --region us-central1        # vacio (un endpoint factura por hora)

# 2) Apagar las APIs que se prendieron para el lab.
#    Habilitarlas es gratis, pero apagarlas evita que algo quede corriendo por error.
gcloud services disable storage.googleapis.com --force
```

> **Un endpoint de Vertex AI es el error caro clásico:** factura por **hora de máquina
> desplegada**, la haya usado alguien o no. Este lab no crea ninguno — pero si en algún
> momento probás uno, `undeploy` y `delete` antes de cerrar.

Y el control final, el que no falla: **consola → Facturación → Informes**, filtrando por el
proyecto. Si mañana marca cero, quedó limpio.
"""),

md("""
---
## Hasta acá llega el lab

Lo que se demostró en la nube:

| | |
|---|---|
| Proyecto y bucket | creados desde el notebook |
| Dato | cuatro fuentes públicas, ~27 MB, Bronze append-only |
| Silver | normalizado, 100 % de cruce entre fuentes |
| Gold | 1.530 × 301, con el control anti-leakage corriendo antes de escribir |
| Modelo | entrenado en CPU, versionado, con su contrato de features |
| Predicción | fechas 1 y 2 predichas y registradas, con la prueba de que no miraron el futuro |
| Costo al cerrar | **cero** — todo borrado y verificado |

**Lo que falta** es envolver la predicción en una API y desplegarla: hoy corre como un batch,
igual que el entrenamiento. El monitoreo de la temporada en curso —métricas fecha a fecha
contra los baselines calculados sobre las mismas filas— también está escrito y corre en
local; se ve en [`00_recorrido_completo.ipynb`](00_recorrido_completo.ipynb).

El ciclo cerrado está entero: se predice, se registra, llega el resultado y se mide. Lo que
falta es empaquetarlo, y eso es la clase que viene.

Los comandos equivalentes por terminal están en [`gcp/runbook.md`](../gcp/runbook.md).
"""),
]


def main() -> None:
    nb = {
        "cells": CELDAS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    DESTINO.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    n_code = sum(1 for c in CELDAS if c["cell_type"] == "code")
    print(f"Escrito {DESTINO}")
    print(f"{len(CELDAS)} celdas ({n_code} de codigo, {len(CELDAS) - n_code} de texto)")


if __name__ == "__main__":
    main()
