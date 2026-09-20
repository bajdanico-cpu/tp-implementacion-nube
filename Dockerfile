# Imagen del TP: API de predicción + el Job del pipeline. La misma para las dos cosas,
# porque comparten todo el código y lo único que cambia es el comando.
#
# ADENTRO NO VA NI UN DATO NI UN MODELO. Antes sí: `COPY models` (53 MB, catorce
# versiones) y `COPY data/silver`. Eso congelaba el dato en el momento del build, así que
# predecir una fecha nueva exigía reconstruir y redesplegar la imagen entera; y como
# `data/` y los `.ubj` están en `.gitignore`, sólo podía desplegar quien ya hubiera
# corrido el pipeline en su máquina. Ahora el dato y el modelo se leen del bucket
# (`TP_STORAGE_BACKEND=gcs`), que es lo que separa los tres ciclos de vida: el código
# cambia cuando cambia la lógica, el dato cada fecha, el modelo cuando se reentrena.

FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Las dependencias primero y solas: mientras no cambien, Docker reusa esta capa y el
# build baja de minutos a segundos.
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY common ./common
COPY features ./features
COPY transform ./transform
COPY ingestion ./ingestion
COPY training ./training
COPY serving ./serving
COPY pipeline ./pipeline
COPY web ./web
COPY config.yaml .

# El servicio. El Job del pipeline usa esta misma imagen con otro comando:
#   gcloud run jobs create premier-ml-pipeline --image ... \
#     --command python --args -m,pipeline.pre_deadline
CMD ["sh", "-c", "uvicorn serving.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
