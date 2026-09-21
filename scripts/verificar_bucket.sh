#!/usr/bin/env bash
# Qué hay en el bucket, qué falta, y qué conviene subir.
#
#   bash scripts/verificar_bucket.sh
#   BUCKET=otro-nombre bash scripts/verificar_bucket.sh
#
# No toca nada: sólo mira. Existe para contestar la pregunta de siempre cuando ya hay
# cosas a medio subir — *¿borro todo y arranco de cero?* — y la respuesta casi siempre es
# no. Bronze es append-only por diseño: los snapshots viejos y los nuevos conviven, así
# que `rsync` suma sin pisar y lo que ya bajaste sigue valiendo.
#
# Mira dos cosas distintas, y la segunda es la que salva la demo:
#
#   1. Que los archivos ESTEN.
#   2. Que sean los que corresponden. Un Gold de hace tres semanas ocupa lo mismo que
#      uno de hoy, deja todos los tildes en verde, y no tiene la fila de la fecha que
#      viene: el servicio arranca igual y no puede predecir nada.

set -uo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
BUCKET="${BUCKET:-${PROJECT_ID}-premier-ml}"
PREFIJO="${TP_GCS_PREFIX:-}"
[ -n "${PREFIJO}" ] && PREFIJO="${PREFIJO%/}/"

titulo() { printf '\n\033[1;35m▶ %s\033[0m\n' "$*"; }
ok()     { printf '  \033[0;32m✓\033[0m %s\n' "$*"; }
falta()  { printf '  \033[0;31m✗\033[0m %s\n' "$*"; }
nota()   { printf '  \033[0;33m!\033[0m %s\n' "$*"; }

PROBLEMAS=""
anotar() { PROBLEMAS="${PROBLEMAS}\n  - $1"; }

titulo "Bucket gs://${BUCKET}"
if ! gcloud storage buckets describe "gs://${BUCKET}" >/dev/null 2>&1; then
  falta "no existe (o no tenés permiso). Se crea con:"
  echo "      gcloud storage buckets create gs://${BUCKET} --location=us-central1"
  exit 1
fi
ok "existe"

# Se lista UNA vez y se reusa: `gcloud storage ls` contra un bucket grande tarda, y antes
# se lo llamaba una vez por capa más otra por Bronze.
TODO="$(mktemp)"
trap 'rm -f "${TODO}"' EXIT
gcloud storage ls -r "gs://${BUCKET}/${PREFIJO}**" 2>/dev/null | grep '^gs://' > "${TODO}"

# ---------------------------------------------------------------- qué hay
titulo "Qué hay arriba"

contar() { grep -c "^gs://${BUCKET}/${PREFIJO}$1/" "${TODO}"; }
pesar()  { gcloud storage du -s "gs://${BUCKET}/${PREFIJO}$1" 2>/dev/null \
             | awk '{printf "%.1f MB", $1/1048576}'; }

for capa in gold predicciones models; do
  N="$(contar "$capa")"
  if [ "${N:-0}" -gt 0 ]; then
    ok "$(printf '%-14s %4s archivos  %s' "$capa/" "$N" "$(pesar "$capa")")"
  else
    falta "$(printf '%-14s vacío  <- lo necesita el SERVICIO' "$capa/")"
    anotar "falta subir ${capa}/"
  fi
done

for capa in silver bronze; do
  N="$(contar "$capa")"
  if [ "${N:-0}" -gt 0 ]; then
    ok "$(printf '%-14s %4s archivos  %s' "$capa/" "$N" "$(pesar "$capa")")"
  else
    nota "$(printf '%-14s vacío  (sólo lo usa el JOB; su primera corrida tarda más)' "$capa/")"
  fi
done

# ---------------------------------------------------------------- lo mínimo para servir
titulo "Lo mínimo para que el servicio arranque"

GOLD="gs://${BUCKET}/${PREFIJO}gold/gold_tp_match.parquet"
if grep -qx "${GOLD}" "${TODO}"; then
  CUANDO="$(gcloud storage ls -l "${GOLD}" 2>/dev/null | awk 'NR==1{print $2}')"
  ok "gold/gold_tp_match.parquet   (subido: ${CUANDO:-?})"
else
  falta "gold/gold_tp_match.parquet  <- sin esto, /health queda en degraded"
  anotar "falta gold/gold_tp_match.parquet"
fi

PROD="$(grep -m1 "/${PREFIJO}models/.*/PRODUCTION.json$" "${TODO}")"
if [ -n "${PROD}" ]; then
  VER="$(gcloud storage cat "${PROD}" 2>/dev/null \
         | python3 -c 'import json,sys; print(json.load(sys.stdin)["version"])' 2>/dev/null)"
  ok "PRODUCTION.json -> ${VER:-?}"
  NUBJ="$(grep -c "/${VER}/.*\.ubj$" "${TODO}")"
  if [ "${NUBJ:-0}" -gt 0 ]; then
    ok "${NUBJ} boosters de esa versión"
  else
    falta "la versión de producción NO tiene .ubj arriba: el servicio va a dar 503"
    anotar "faltan los .ubj de ${VER}"
  fi
else
  falta "no hay models/<modelo>/PRODUCTION.json  <- el servicio no sabe qué servir"
  anotar "falta PRODUCTION.json (se crea con: python -m training.registry --promover <VER> --motivo '...')"
  NUBJ_TOT="$(grep -c "\.ubj$" "${TODO}")"
  [ "${NUBJ_TOT:-0}" -gt 0 ] && nota "hay ${NUBJ_TOT} .ubj arriba, pero nadie eligió cuál sirve"
fi

# ---------------------------------------------------------------- ¿es el Gold correcto?
titulo "¿Es el Gold que la demo necesita?"
if grep -qx "${GOLD}" "${TODO}" && python3 -c "import pandas" 2>/dev/null; then
  TMP="$(mktemp --suffix=.parquet)"
  if gcloud storage cp "${GOLD}" "${TMP}" >/dev/null 2>&1; then
    python3 - "$TMP" <<'PY'
import sys
import pandas as pd

g = pd.read_parquet(sys.argv[1])
print(f"  · {len(g)} filas, {len(g.columns)} columnas")
if "gold_built_at" in g:
    print(f"  · construido: {g['gold_built_at'].max()}")
if "split" not in g:
    print("  \033[0;31m✗\033[0m sin columna `split`: es un Gold ANTERIOR al serving desde Gold")
    sys.exit(3)
inf = g[g["split"] == "inferencia"]
if inf.empty:
    print("  \033[0;31m✗\033[0m NO tiene fila de inferencia: no hay ninguna fecha para predecir")
    print("      el servicio va a contestar 409 a todo. Subí el Gold nuevo.")
    sys.exit(3)
gw = int(inf["gameweek"].min())
print(f"  \033[0;32m✓\033[0m la próxima predecible es la GW{gw} ({len(inf)} partidos)")
jug = g[(g["season"] == g["season"].max()) & (g["target_1x2"].notna())]
if not jug.empty:
    print(f"  · última fecha con resultado: GW{int(jug['gameweek'].max())}")
PY
    [ $? -eq 3 ] && anotar "el Gold de arriba es viejo: subí data/gold"
    rm -f "${TMP}"
  else
    nota "no se pudo descargar para verificar el contenido"
  fi
else
  nota "no se pudo verificar el contenido (falta el archivo, o pandas en este entorno)"
  nota "mirá la fecha de subida: si es de hace semanas, seguramente sea el Gold del lab"
fi

# ---------------------------------------------------------------- predicciones
titulo "Registro de predicciones"
NPRED="$(grep -c "/${PREFIJO}predicciones/.*\.parquet$" "${TODO}")"
if [ "${NPRED:-0}" -ge 1 ]; then
  ok "${NPRED} predicciones registradas"
  [ "${NPRED}" -lt 5 ] && nota "son pocas: las fechas sin registro contestan 409 en vez de mostrar la congelada"
else
  falta "ninguna: toda fecha jugada va a contestar 409"
  anotar "falta subir data/predicciones"
fi

# ---------------------------------------------------------------- bronze
titulo "Bronze: hasta dónde llega"
FUENTES="$(sed -n "s|^gs://${BUCKET}/${PREFIJO}bronze/\([^/]*\)/.*|\1|p" "${TODO}" | sort -u | tr '\n' ' ')"
ULTIMO="$(grep -o 'ingested_at=[0-9A-Za-z]*' "${TODO}" | sort -u | tail -1)"
if [ -n "${FUENTES// /}" ]; then
  ok "fuentes: ${FUENTES}"
  [ -n "${ULTIMO}" ] && ok "snapshot más nuevo: ${ULTIMO#ingested_at=}"
  nota "Bronze es append-only: lo que está NO se pisa ni hay que volver a bajarlo."
  nota "Las temporadas CERRADAS no se re-descargan si ya tienen snapshot, así que aun"
  nota "un Bronze parcial ahorra lo que más pesa."
else
  nota "sin Bronze: la primera corrida del Job baja las cinco temporadas"
fi

# ---------------------------------------------------------------- veredicto
titulo "Qué hacer"
if [ -n "${PROBLEMAS}" ]; then
  printf "  Falta resolver:%b\n\n" "${PROBLEMAS}"
  echo "  Desde la máquina que tiene el dato al día:"
  echo
  echo "      gcloud storage rsync -r data/gold         gs://${BUCKET}/${PREFIJO}gold"
  echo "      gcloud storage rsync -r data/predicciones gs://${BUCKET}/${PREFIJO}predicciones"
  echo "      gcloud storage rsync -r models            gs://${BUCKET}/${PREFIJO}models"
  echo
  echo "  O, si no tenés gcloud local:  python -m scripts.bundle_demo"
else
  echo "  Está todo lo que el servicio necesita. Para desplegar:"
  echo
  echo "      BUCKET=${BUCKET} bash scripts/preparar_demo.sh"
fi
echo
echo "  NO hace falta borrar nada: rsync suma y actualiza, no pisa lo que ya está."
echo
