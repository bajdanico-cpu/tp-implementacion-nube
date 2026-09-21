#!/usr/bin/env bash
# Qué hay en el bucket, qué falta, y qué conviene subir.
#
#   bash scripts/verificar_bucket.sh
#
# No toca nada: sólo mira. Existe para contestar la pregunta de siempre cuando ya hay
# cosas a medio subir — *¿borro todo y arranco de cero?* — y la respuesta casi siempre es
# no. Bronze es append-only por diseño: los snapshots viejos y los nuevos conviven, así
# que `rsync` suma sin pisar y lo que ya bajaste sigue valiendo.

set -uo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
BUCKET="${BUCKET:-${PROJECT_ID}-premier-ml}"
PREFIJO="${TP_GCS_PREFIX:-}"
[ -n "${PREFIJO}" ] && PREFIJO="${PREFIJO%/}/"

titulo() { printf '\n\033[1;35m▶ %s\033[0m\n' "$*"; }
ok()     { printf '  \033[0;32m✓\033[0m %s\n' "$*"; }
falta()  { printf '  \033[0;31m✗\033[0m %s\n' "$*"; }
nota()   { printf '  \033[0;33m!\033[0m %s\n' "$*"; }

titulo "Bucket gs://${BUCKET}"
if ! gcloud storage buckets describe "gs://${BUCKET}" >/dev/null 2>&1; then
  falta "no existe (o no tenés permiso). Se crea con:"
  echo "      gcloud storage buckets create gs://${BUCKET} --location=us-central1"
  exit 1
fi
ok "existe"

# ---------------------------------------------------------------- qué hay
titulo "Qué hay arriba"

contar() { gcloud storage ls -r "gs://${BUCKET}/${PREFIJO}$1/**" 2>/dev/null | grep -c "^gs://" ; }
pesar()  { gcloud storage du -s "gs://${BUCKET}/${PREFIJO}$1" 2>/dev/null | awk '{printf "%.1f MB", $1/1048576}'; }

FALTA_SERVICIO=""
for capa in gold predicciones models; do
  N="$(contar "$capa")"
  if [ "${N:-0}" -gt 0 ]; then
    ok "$(printf '%-14s %4s archivos  %s' "$capa/" "$N" "$(pesar "$capa")")"
  else
    falta "$(printf '%-14s vacío  <- lo necesita el SERVICIO' "$capa/")"
    FALTA_SERVICIO="${FALTA_SERVICIO} ${capa}"
  fi
done

for capa in silver bronze; do
  N="$(contar "$capa")"
  if [ "${N:-0}" -gt 0 ]; then
    ok "$(printf '%-14s %4s archivos  %s' "$capa/" "$N" "$(pesar "$capa")")"
  else
    nota "$(printf '%-14s vacío  (sólo lo usa el JOB; sin esto su primera corrida tarda más)' "$capa/")"
  fi
done

# ---------------------------------------------------------------- lo mínimo para servir
titulo "Lo mínimo para que el servicio arranque"

if gcloud storage ls "gs://${BUCKET}/${PREFIJO}gold/gold_tp_match.parquet" >/dev/null 2>&1; then
  ok "gold/gold_tp_match.parquet"
else
  falta "gold/gold_tp_match.parquet  <- sin esto, /health queda en degraded"
fi

PROD="$(gcloud storage ls "gs://${BUCKET}/${PREFIJO}models/*/PRODUCTION.json" 2>/dev/null | head -1)"
if [ -n "${PROD}" ]; then
  VER="$(gcloud storage cat "${PROD}" 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["version"])' 2>/dev/null)"
  ok "PRODUCTION.json -> ${VER:-?}"
  NUBJ="$(gcloud storage ls "gs://${BUCKET}/${PREFIJO}models/*/${VER}/*.ubj" 2>/dev/null | grep -c "ubj$")"
  if [ "${NUBJ:-0}" -gt 0 ]; then
    ok "${NUBJ} boosters de esa versión"
  else
    falta "la versión de producción NO tiene .ubj arriba: el servicio va a dar 503"
  fi
else
  falta "no hay models/<modelo>/PRODUCTION.json  <- el servicio no sabe qué servir"
fi

# ---------------------------------------------------------------- bronze parcial
titulo "Bronze: hasta dónde llega"
ULTIMO="$(gcloud storage ls "gs://${BUCKET}/${PREFIJO}bronze/fpl/**/" 2>/dev/null \
          | grep -o 'ingested_at=[0-9TZ]*' | sort -u | tail -1)"
if [ -n "${ULTIMO}" ]; then
  ok "último snapshot de FPL: ${ULTIMO#ingested_at=}"
  nota "Bronze es append-only: lo que ya está NO se pisa ni hay que volver a bajarlo."
  nota "Las temporadas CERRADAS no se re-descargan si ya tienen snapshot, así que"
  nota "aunque llegue sólo hasta la fecha 2, te ahorra lo que más pesa."
else
  nota "sin snapshots de FPL: la primera corrida del Job baja las cinco temporadas"
fi

# ---------------------------------------------------------------- veredicto
titulo "Qué hacer"
if [ -n "${FALTA_SERVICIO}" ]; then
  echo "  Falta subir:${FALTA_SERVICIO}. Desde la máquina que tiene el dato:"
  echo
  for capa in ${FALTA_SERVICIO}; do
    case "$capa" in
      models) echo "      gcloud storage rsync -r models gs://${BUCKET}/${PREFIJO}models" ;;
      *)      echo "      gcloud storage rsync -r data/${capa} gs://${BUCKET}/${PREFIJO}${capa}" ;;
    esac
  done
  echo
  echo "  O, si no tenés gcloud local:  python -m scripts.bundle_demo"
else
  echo "  Está todo lo que el servicio necesita. Para desplegar:"
  echo
  echo "      bash scripts/preparar_demo.sh"
fi
echo
echo "  NO hace falta borrar nada: rsync suma y actualiza, no pisa lo que ya está."
echo
