#!/usr/bin/env bash
# Entorno del TP Premier ML — Linux / macOS.
#
# Equivalente de scripts/setup_env.ps1. Sirve para un compañero que no use Windows y,
# más adelante, como base del contenedor de Cloud Run.
#
# Nota sobre GPU: el wheel de XGBoost para Linux también trae CUDA, así que
# `python -m training.device` va a detectarla si el driver está. En un contenedor sin
# GPU cae a CPU solo, que es el comportamiento buscado para el serving.
set -euo pipefail

RUTA="${1:-$HOME/.venvs/tp-premier-ml}"
RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "== Entorno del TP Premier ML =="
echo "  proyecto : $RAIZ"
echo "  venv     : $RUTA"

if [ ! -d "$RUTA" ]; then
  echo
  echo "Creando el venv..."
  PY="$(command -v python3.14 || command -v python3)"
  "$PY" -m venv "$RUTA"
else
  echo
  echo "El venv ya existe; se reutiliza (borralo a mano para rehacerlo)."
fi

PYTHON="$RUTA/bin/python"

echo
echo "Instalando dependencias..."
"$PYTHON" -m pip install --upgrade pip --quiet
"$PYTHON" -m pip install -r "$RAIZ/requirements.txt"

echo
echo "== Verificación =="
cd "$RAIZ"
"$PYTHON" -m common.config
echo
"$PYTHON" -m training.device

cat <<EOF

== Listo ==
Activá el entorno con:
    source $RUTA/bin/activate

Y después, desde cero:
    python -m ingestion.run        # baja Bronze (~27 MB, sin credenciales)
    python -m transform.silver     # arma las 4 tablas Silver
    python -m features.gold_tp     # arma Gold (1.520 x 165)
    python -m training.run --todos # entrena y evalúa
    pytest                         # la suite completa
EOF
