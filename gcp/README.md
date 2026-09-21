# Empezá acá

Hay cuatro documentos en esta carpeta y tres scripts en `scripts/`. Éste dice cuál usar.

## Si lo que querés es dejar el TP andando en GCP

**En Cloud Shell**, y en este orden:

```bash
# 1. El código
git clone https://github.com/bajdanico-cpu/tp-implementacion-nube.git
cd tp-implementacion-nube
gcloud config set project TU_PROYECTO

# 2. ¿Qué hay ya arriba? (no toca nada)
bash scripts/verificar_bucket.sh

# 3. Si falta el dato, ver más abajo. Si está, desplegar:
bash scripts/preparar_demo.sh
```

Eso es todo. El script hace los ocho pasos —APIs, bucket, las dos cuentas de servicio,
los permisos, la imagen, el Job, el servicio— y al final verifica que haya quedado bien.
Es idempotente: se puede correr dos veces.

La explicación de **qué hace cada paso y por qué** está en
[`GUIA-DEPLOY.md`](GUIA-DEPLOY.md). Conviene leerla antes de la defensa: las preguntas
van a ser sobre eso y no sobre el script.

---

## El paso 3: el dato no viene en el `git clone`

`data/` y los `.ubj` están en `.gitignore` a propósito —pesan y se regeneran— así que el
clone trae **sólo código**. Hay tres situaciones y el verificador te dice en cuál estás:

| Si… | Hacé esto |
|---|---|
| El bucket ya tiene `gold/`, `predicciones/` y `models/` | nada: andá al paso 3 |
| Cloud Shell ya tiene `data/` de un lab anterior | `bash scripts/preparar_demo.sh` lo sube solo |
| No hay dato en ningún lado | ver abajo |

**Si no hay dato en ningún lado**, dos caminos:

**(a) Con `gcloud` en tu PC** — el mejor, porque además sube Bronze y la primera corrida
del Job no tiene que re-descargar las temporadas históricas:

```bash
gcloud storage rsync -r data/bronze gs://TU-BUCKET/bronze
gcloud storage rsync -r data/silver gs://TU-BUCKET/silver
gcloud storage rsync -r data/gold   gs://TU-BUCKET/gold
gcloud storage rsync -r data/predicciones gs://TU-BUCKET/predicciones
gcloud storage rsync -r models gs://TU-BUCKET/models
```

**(b) Sin `gcloud` local** — un paquete de 5 MB que subís por la interfaz:

```powershell
python -m scripts.bundle_demo          # en tu PC: deja demo-premier-ml.zip
```

En Cloud Shell: menú de tres puntos → **Subir** → elegí el zip. Después
`unzip -o ~/demo-premier-ml.zip` desde la raíz del repo.

> **No regeneres el dato en Cloud Shell con `pipeline.pre_deadline`.** Funciona, pero
> ingesta los resultados de la GW5 y te deja sin la transición que la demo muestra en
> vivo. Ver *El estado que deja, y por qué ése* en `GUIA-DEPLOY.md`.

---

## Qué es cada documento

| Archivo | Para qué | Cuándo |
|---|---|---|
| **[`GUIA-DEPLOY.md`](GUIA-DEPLOY.md)** | De cero a operado: los ocho pasos explicados, la evidencia de la clase 7, cómo revisar todo desde la consola web, costos y troubleshooting | **la principal** |
| [`runbook.md`](runbook.md) | Comandos sueltos de terminal: correr el pipeline, leer logs por campo, smoke test de carga | cuando ya está desplegado |
| [`paso-a-paso.md`](paso-a-paso.md) | El lab de la clase 4: el pipeline en Cloud Shell, sin desplegar nada | histórico |
| [`../infra/README.md`](../infra/README.md) | La arquitectura y las tres decisiones que hay que poder defender | antes de la defensa |

## Qué es cada script

| Script | Qué hace | Toca algo |
|---|---|---|
| `scripts/verificar_bucket.sh` | Dice qué hay en el bucket y qué falta | no |
| `scripts/preparar_demo.sh` | Despliega todo y verifica | sí |
| `scripts/preparar_demo.sh --reset` | Vuelve el dato al estado inicial de la demo | sí |
| `scripts/bundle_demo.py` | Arma el zip de 5 MB para subir a Cloud Shell | no |
| `scripts/smoke_load.py` | Carga y latencia contra la API | no |

---

## El día de la defensa

```bash
curl -s $SERVICE_URL/health | python3 -m json.tool
```

Tiene que decir `"status": "ok"` y `"proxima_predecible": 5`. El checklist completo está
al final de la sección 9 de [`GUIA-DEPLOY.md`](GUIA-DEPLOY.md).
