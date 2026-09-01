# Paso a paso: el TP en GCP, desde cero

Para alguien que abre esto sin haber tocado el proyecto antes. Va desde crear la cuenta
hasta apagar todo, y no asume nada instalado en la máquina: **todo corre en Cloud Shell**,
que ya trae Python, `git` y `gcloud`.

**Tiempo total: ~30 minutos**, de los cuales la mitad es esperar a que corra el pipeline.

```
Parte 0   crear el proyecto            consola web        ~5 min
Parte 1   preparar el repo             terminal           ~5 min
Parte 2   correr el lab                editor + notebook  ~20 min
Parte 3   apagar todo                  notebook + terminal ~2 min   ← no es opcional
```

> ⚠️ **La Parte 3 no se saltea.** Lo que se prende, se paga. El lab está diseñado para no
> dejar nada corriendo, pero el bucket factura almacenamiento mientras exista.

---

## Parte 0 — Crear el proyecto (consola web)

### 1. Cuenta con facturación

Entrá a [console.cloud.google.com](https://console.cloud.google.com) con tu cuenta de
Google. Si es la primera vez, GCP ofrece una **prueba gratuita con crédito** (US$300 por 90
días al momento de escribir esto). Hay que **vincular una tarjeta**, pero durante la prueba
no se cobra sin que lo autorices explícitamente.

Si la materia les dio créditos, usá esa cuenta de facturación en vez de la prueba.

### 2. Crear el proyecto

Arriba a la izquierda, al lado del logo de Google Cloud, hay un **selector de proyecto**.
Clic ahí → **Proyecto nuevo**.

- **Nombre**: `tp-premier-ml` (o el que quieras: es la etiqueta que ves).
- **ID del proyecto**: se genera solo a partir del nombre. **Es único en todo Google y no se
  puede cambiar nunca más.** Si `tp-premier-ml` está tomado, le agrega un sufijo numérico.
  **Anotá el ID exacto**: lo vas a necesitar en la Parte 2, y no es lo mismo que el nombre.

Crear, y esperar unos segundos a que aparezca. Después **seleccionalo** en el selector: todo
lo que sigue pasa adentro de ese proyecto.

### 3. Verificar que tenga facturación vinculada

Menú (☰) → **Facturación**. Tiene que decir que el proyecto está vinculado a una cuenta.
Sin esto, el paso siguiente falla y el error no es obvio.

### 4. Habilitar la API de Cloud Storage

**Un proyecto nuevo nace con casi todo apagado.** Cada servicio hay que prenderlo. Prenderlo
es gratis: se paga el uso, no tenerlo habilitado.

Menú → **APIs y servicios** → **Habilitar API y servicios** → buscar
**Cloud Storage API** → *Habilitar*.

*(También se puede desde la terminal, en la Parte 1. Es la misma cosa.)*

### 5. Abrir Cloud Shell

Arriba a la derecha, el ícono de **terminal** (`>_`). Se abre una consola abajo.

Cloud Shell es una máquina Linux gratis, ya autenticada con tu cuenta, con **5 GB de disco
persistente** en el home. Lo que quede en `~` sobrevive entre sesiones; lo que instales
fuera del home, no.

---

## Parte 1 — Preparar el repo (terminal de Cloud Shell)

### 6. Confirmar en qué proyecto estás

```bash
gcloud config get-value project
```

Si no es el que creaste:

```bash
gcloud config set project TU-PROJECT-ID
```

Y si no habilitaste la API desde la consola, ahora:

```bash
gcloud services enable storage.googleapis.com
```

### 7. Clonar el repo

```bash
git clone https://github.com/bajdanico-cpu/tp-implementacion-nube.git tp-premier-ml
cd tp-premier-ml
```

**El repo no trae datos.** `data/` y los modelos están en `.gitignore` a propósito: todo se
regenera desde cuatro fuentes públicas, ninguna con credenciales. Las primeras celdas del
notebook los bajan.

### 8. Instalar las dependencias

```bash
pip install --user -r requirements-cloud.txt
```

**`--user` importa**: instala en `~/.local`, que está en el home persistente y que el kernel
del editor ve sin configurar nada. Sin `--user`, en algunas imágenes de Cloud Shell pip se
niega a tocar el Python del sistema.

> **`requirements-cloud.txt` y no `requirements.txt`.** El de local está pinneado a wheels
> `cp314` (Python 3.14.3) y Cloud Shell trae otro Python: pip intentaría compilar numpy y
> pandas desde el fuente y la sesión se cae por tiempo o memoria. El de nube usa cotas.

Si pip se queja con *"externally-managed-environment"*, el camino alternativo es un venv,
pero hay que registrarlo como kernel para que el editor lo vea:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-cloud.txt ipykernel
python -m ipykernel install --user --name tp-premier --display-name "TP Premier"
```

Después, en el notebook, elegí el kernel **TP Premier** arriba a la derecha.

### 9. Verificar

```bash
python3 -c "import pandas, xgboost, google.cloud.storage; print('ok')"
```

---

## Parte 2 — Correr el lab (editor de Cloud Shell)

### 10. Abrir el editor

En la barra de Cloud Shell, botón **Abrir editor**. Se abre un VS Code en el navegador con
el repo ya clonado.

Abrí **`notebooks/01_gcp_cloudshell.ipynb`**.

### 11. Poner tu PROJECT_ID

En la **primera celda de código** (Paso 0) hay esta línea:

```python
or "cambiame-por-tu-project-id"     # <-- el ID de TU proyecto
```

Cambiala por el ID que anotaste en el punto 2.

> **Por qué a mano.** El notebook lo intenta leer de `GOOGLE_CLOUD_PROJECT` y
> `DEVSHELL_PROJECT_ID`, pero **el kernel del editor no hereda el entorno de la sesión**:
> muchas veces esas variables no están. Es la misma razón por la que el notebook habla con
> la nube por las librerías de Python (`google-cloud-storage`) y no por `gcloud`.

### 12. Correr celda por celda

**No uses "Run All".** La gracia del lab es ver aparecer cada recurso en la consola. Cada
paso dice qué mirar.

| Paso | Qué hace | Dónde verlo | Tarda |
|---|---|---|---|
| **0** | dónde estás parado | — | instantáneo |
| **1** | lista los buckets | si responde, estás autenticado | segundos |
| **2** | baja ~27 MB de las 4 fuentes → Bronze, y crea el bucket | *Cloud Storage → tu bucket → `bronze/`* | **varios minutos** |
| **3** | Silver: normaliza y cruza las fuentes | el log tiene que decir **100 % de cruce** | ~1 min |
| **4** | Gold: 279 features, con el control anti-leakage | el log dice `Controles anti-leakage OK` | **el más lento** |
| **5** | entrena los dos modelos | acá cae a **CPU**: Cloud Shell no tiene GPU | ~2-3 min |
| **6** | registra y versiona el modelo | *Cloud Storage → `models/`* | segundos |
| **7** | predice las fechas 1 y 2, y prueba que no miró el futuro | *Cloud Storage → `predicciones/`* | ~2 min |
| **8** | inventario de lo que quedó en el bucket | — | segundos |
| **9** | **borra todo** | los buckets vuelven a estar vacíos | segundos |

Los tiempos son de referencia: Cloud Shell tiene 1-2 vCPU, así que va más lento que una
notebook local. El Paso 2 y el Paso 4 son los que hay que esperar.

**Los tres momentos donde vale la pena frenar a mirar:**

- **Paso 4** — el control anti-leakage corre *antes de escribir la tabla*, no en una suite
  de tests aparte. Un pipeline que sólo valida en CI puede escribir datos contaminados en
  producción y enterarse el lunes.
- **Paso 5** — hay **dos modelos** y sólo uno puede reportar números. El de producción marca
  0,616 sobre 2025-26 porque entrenó con esa temporada: no es una mejora, es el modelo
  acordándose.
- **Paso 7** — la prueba de que no hay leakage. Si la fecha 1 filtrara su propio resultado,
  el modelo acertaría 10 de 10. Acierta 4.

---

## Parte 3 — Apagar todo

### 13. La última celda del notebook

El **Paso 9** borra los objetos, borra el bucket y **verifica** que no haya quedado nada.
Correla siempre antes de cerrar.

### 14. Los tres chequeos, en la **terminal**

El kernel del editor no tiene `gcloud` autenticado, así que esto va en la terminal:

```bash
gcloud storage ls                                    # ningún bucket del lab
gcloud run services list --region us-central1        # vacío
gcloud ai endpoints list --region us-central1        # vacío

gcloud services disable storage.googleapis.com --force
```

> **Un endpoint de Vertex AI es el error caro clásico:** factura por **hora de máquina
> desplegada**, la use alguien o no. Este lab no crea ninguno — pero si probás uno por tu
> cuenta, `undeploy` y `delete` antes de cerrar.

### 15. El control que no falla

Al día siguiente: **consola → Facturación → Informes**, filtrando por el proyecto. Si marca
cero, quedó limpio.

Si no vas a volver a usar el proyecto, lo más seguro es **eliminarlo entero**:
*Menú → IAM y administración → Configuración → Cerrar proyecto*. Se borra todo lo que
tenga adentro, con 30 días de gracia para arrepentirse.

---

## Cuando algo falla

| Síntoma | Qué pasó |
|---|---|
| `403` o *"API not enabled"* al listar buckets | falta habilitar Cloud Storage (punto 4) o falta facturación (punto 3) |
| `Bucket names must be globally unique` | el nombre del bucket sale de tu `PROJECT_ID`; si el ID quedó con sufijo numérico, usá el ID exacto |
| El notebook crea el bucket en otro proyecto | quedó el placeholder en la celda del Paso 0 (punto 11) |
| `ModuleNotFoundError` en la primera celda | el kernel no ve lo que instalaste: revisá el punto 8, y si usaste venv, que el kernel elegido sea **TP Premier** |
| pip falla con *externally-managed-environment* | usá el camino del venv (punto 8) |
| La sesión se desconecta a mitad del Paso 2 | Cloud Shell corta por inactividad, pero **el estado del notebook se pierde**: hay que volver a correr desde el Paso 0. Bronze ya bajado no se re-descarga |
| *"Cloud Shell quota exceeded"* | hay un límite semanal de uso; espera a que se renueve |
| El Paso 7 predice una fecha ya jugada | es el calendario: si la próxima gameweek ya cerró, sigue funcionando pero deja de ser una predicción |

---

## Qué NO hace este lab, y por qué

Llega hasta la predicción corriendo como **batch**. No levanta ningún servicio, no construye
ninguna imagen y no programa ningún job. Eso es la etapa siguiente.

Es deliberado: cada recurso que se despliega es un recurso que hay que acordarse de apagar,
y el objetivo de este lab es que el pipeline entero corra en la nube **sin dejar nada
facturando**.

Los comandos por terminal, y las decisiones de arquitectura ya medidas para cuando llegue el
deploy, están en [`runbook.md`](runbook.md).
