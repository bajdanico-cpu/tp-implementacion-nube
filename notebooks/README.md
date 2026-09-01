# Notebooks

## `00_recorrido_completo.ipynb`

Recorre todo el proyecto paso a paso, con los números a la vista. Es lo que hay que abrir
para entender qué hicimos y por qué.

```powershell
jupyter lab notebooks/00_recorrido_completo.ipynb
```

**Antes hay que tener el entorno y los datos:**

```powershell
.\scripts\setup_env.ps1
python -m ingestion.run              # las tres fuentes publicas de siempre
python -m ingestion.bronze_pulselive # la API oficial: copas, Europa y stats de Opta
python -m transform.silver
python -m transform.competencias
python -m transform.opta_stats
python -m features.gold_tp
```

El notebook no reimplementa nada: cada paso llama a los módulos del repo, así que lo que
corre ahí es exactamente lo que corre en producción. Si un número del notebook cambia, es
porque cambió el pipeline.

Tarda unos minutos: la sección 11 hace 38 reentrenamientos (el walk-forward) y la 6
entrena dos veces, para mostrar la diferencia entre el modelo de evaluación y el de
producción.

Las 17 secciones van del dato crudo al deploy: los datos y la regla anti-leakage (1-2),
las features y las dos fuentes nuevas (3-5), el modelo y sus métricas honestas (6-13),
y el ciclo operativo ya corriendo — predicción, monitoreo y reproducibilidad (14-17).

---

## `01_gcp_cloudshell.ipynb`

El **lab en la nube**: se clona el repo en Cloud Shell, se corre celda por celda, y cada
paso deja un recurso visible en la consola de GCP. Es el equivalente, para este TP, del lab
de la clase 4 del profesor.

```bash
git clone <url-del-repo> tp-premier-ml
cd tp-premier-ml
pip install -q -r requirements-cloud.txt
# abrir notebooks/01_gcp_cloudshell.ipynb en el editor de Cloud Shell
```

**El clon no trae datos** (`data/` está en `.gitignore`): las primeras celdas bajan las
cuatro fuentes y arman Silver y Gold desde cero. Eso mismo es la prueba de que el pipeline
es reproducible sobre una máquina que nunca vio el proyecto.

Llega hasta **la predicción de una fecha, registrada en el bucket** — proyecto, Bronze,
Silver, Gold, modelo, predicción — y ahí para. No levanta ningún servicio ni construye
imágenes: eso es la etapa siguiente.

El Paso 7 predice las fechas 1 y 2 y **demuestra que no hay leakage temporal**, que es el
riesgo real de correrlo mientras se está jugando una fecha.

**La última celda borra el bucket y verifica que no quedó nada facturando**, y el runbook
tiene los dos chequeos que faltan (que no haya servicios ni endpoints, y apagar la API).
Lo que se prende, se paga.

El paso a paso completo —desde crear el proyecto en GCP— está en
[`gcp/paso-a-paso.md`](../gcp/paso-a-paso.md); los comandos por terminal, en
[`gcp/runbook.md`](../gcp/runbook.md).

> **`requirements-cloud.txt` y no `requirements.txt`.** El de local está pinneado a wheels
> `cp314` (Python 3.14.3) y Cloud Shell trae otro Python: pip intentaría compilar numpy y
> pandas desde el fuente y la sesión se cae.

---

## Por qué el `.ipynb` se genera desde un `.py`

El notebook **se produce con `python notebooks/00_recorrido_completo.py`**, no se edita a
mano. Un `.ipynb` es JSON con el código embebido línea por línea: editado directamente,
los diffs de git son ilegibles y los merges entre dos personas son un infierno.

Generándolo desde un `.py`:

- el contenido se versiona como texto legible y los diffs se entienden;
- no se commitean salidas de ejecución, que inflan el repo y cambian en cada corrida;
- los `id` de celda se derivan del contenido con un hash, así que **regenerar sin cambios
  produce un archivo byte a byte idéntico** — si fueran aleatorios, cada corrida ensuciaría
  el diff.

Para cambiar cualquiera de los dos notebooks: se edita el `.py` y se regenera.
