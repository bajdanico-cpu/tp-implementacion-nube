FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY common ./common
COPY features ./features
COPY transform ./transform
COPY ingestion ./ingestion
COPY eda ./eda
COPY training ./training
COPY serving ./serving
COPY config.yaml .

COPY models ./models
COPY data/silver ./data/silver

CMD ["sh", "-c", "uvicorn serving.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
