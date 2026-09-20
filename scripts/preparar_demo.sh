#!/usr/bin/env bash
# Deja todo listo en GCP para la defensa, en un comando.
#
#   bash scripts/preparar_demo.sh              # provisiona todo y deja el estado inicial
#   bash scripts/preparar_demo.sh --reset      # SOLO vuelve el dato al estado inicial
#
# Es idempotente: se puede correr dos veces sin romper nada. Cada paso dice qué hace y
# sigue de largo si el recurso ya existe.
#
# `--reset` existe para poder PROBAR el boton sin gastar la demo. Apretarlo ingesta los
# resultados de la GW5 y Gold avanza a la GW6; despues de verificar que funciono, este
# modo vuelve a subir el estado congelado y queda todo como antes. Son unos segundos:
# son 2 MB de Gold y predicciones.
#
# ─────────────────────────────────────────────────────────────────────────────
# QUE DEJA ARMADO, Y POR QUE ASI
#
# El estado inicial de la demo es: Gold hasta la **GW5 predicha y registrada**, con la
# GW6 todavía sin features. Ese estado NO se regenera acá: se SUBE tal cual está en tu
# máquina. Si el script corriera el pipeline, ingestaría los resultados de la GW5 —que ya
# se jugó— y la demo en vivo se quedaría sin nada que mostrar.
#
# La demo es esa transición:
#
#     antes del botón          después del botón
#     GW1-4 con resultado      GW1-5 con resultado
#     GW5 predicha             GW6 predicha  <- aparece sola
#     GW6 sin datos            GW7 sin datos
#
# Por eso el script sube el dato congelado y NO dispara el Job. El Job lo disparás vos,
# desde la página, cuando estés grabando.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

MODO="completo"
[ "${1:-}" = "--reset" ] && MODO="reset"

# ---------------------------------------------------------------- variables
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-central1}"
BUCKET="${BUCKET:-${PROJECT_ID}-premier-ml}"
REPO="${REPO:-mlops-2026}"
SERVICE="${SERVICE:-premier-ml-api}"
JOB="${JOB:-premier-ml-pipeline}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:latest"
SA_API="premier-api@${PROJECT_ID}.iam.gserviceaccount.com"
SA_JOB="premier-job@${PROJECT_ID}.iam.gserviceaccount.com"

titulo() { printf '\n\033[1;35m▶ %s\033[0m\n' "$*"; }
ok()     { printf '  \033[0;32m✓\033[0m %s\n' "$*"; }
aviso()  { printf '  \033[0;33m!\033[0m %s\n' "$*"; }
morir()  { printf '\n\033[0;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 0 · control previo
titulo "0 · Control previo"

command -v gcloud >/dev/null || morir "No hay gcloud. Corré esto en Cloud Shell."
[ -n "${PROJECT_ID}" ] || morir "No hay proyecto. Usá: gcloud config set project TU_PROYECTO"
gcloud auth list --filter=status:ACTIVE --format='value(account)' | grep -q . \
  || morir "No hay sesión activa. Corré: gcloud auth login"

echo "  proyecto : ${PROJECT_ID}"
echo "  región   : ${REGION}"
echo "  bucket   : gs://${BUCKET}"
echo "  servicio : ${SERVICE}"
echo "  job      : ${JOB}"

FALTAN=""
for d in data/gold data/predicciones models; do
  [ -d "$d" ] || FALTAN="${FALTAN} $d"
done
if [ -n "${FALTAN}" ]; then
  aviso "Faltan en esta máquina:${FALTAN}"
  aviso "El dato y los modelos NO están en git (pesan y se regeneran)."
  aviso "Opciones:"
  aviso "  a) subirlos desde tu PC:  gcloud storage rsync -r data/gold gs://${BUCKET}/gold"
  aviso "  b) regenerarlos acá:      python -m pipeline.pre_deadline"
  aviso "     OJO con (b): ingesta la GW5 ya jugada y te come la demo en vivo."
  read -r -p "  ¿Seguir igual, asumiendo que el bucket ya tiene el dato? [s/N] " r
  [ "${r:-n}" = "s" ] || exit 1
  SUBIR_DATO="no"
else
  SUBIR_DATO="si"
fi

# ---------------------------------------------------------------- 1 · APIs
titulo "1 · Habilitando APIs (gratis; se paga el uso)"
gcloud services enable \
  storage.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  logging.googleapis.com \
  --quiet
ok "storage, artifactregistry, cloudbuild, run, logging"

# ---------------------------------------------------------------- 2 · bucket
titulo "2 · Bucket"
if gcloud storage buckets describe "gs://${BUCKET}" >/dev/null 2>&1; then
  ok "gs://${BUCKET} ya existe"
else
  gcloud storage buckets create "gs://${BUCKET}" --location="${REGION}" --quiet
  ok "gs://${BUCKET} creado"
fi

# ---------------------------------------------------------------- 3 · identidades
titulo "3 · Dos identidades: la que lee y la que escribe"

crear_sa() {
  local nombre="$1" desc="$2"
  if gcloud iam service-accounts describe "${nombre}@${PROJECT_ID}.iam.gserviceaccount.com" \
       >/dev/null 2>&1; then
    ok "${nombre} ya existe"
  else
    gcloud iam service-accounts create "${nombre}" --display-name "${desc}" --quiet
    ok "${nombre} creada"
  fi
}
crear_sa premier-api "Premier ML API (solo lee)"
crear_sa premier-job "Premier ML pipeline (escribe)"

# Un bug en el camino de lectura no puede corromper Gold. Es gratis, y es la diferencia
# entre "no debería escribir" y "no puede".
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${SA_API}" --role="roles/storage.objectViewer" --quiet >/dev/null
ok "la API puede LEER el bucket"

gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${SA_JOB}" --role="roles/storage.objectAdmin" --quiet >/dev/null
ok "el Job puede ESCRIBIR el bucket"

# ---------------------------------------------------------------- 4 · el dato
titulo "4 · Subiendo el dato congelado (NO se corre el pipeline)"
if [ "${SUBIR_DATO}" = "si" ]; then
  # En reset se BORRA lo que sobra: el Job dejo un Gold mas nuevo y la prediccion de la
  # GW6, y un reset a medias deja el bucket contando dos historias distintas.
  BORRAR=""
  [ "${MODO}" = "reset" ] && BORRAR="--delete-unmatched-destination-objects"

  # shellcheck disable=SC2086
  gcloud storage rsync -r data/gold         "gs://${BUCKET}/gold"         ${BORRAR} --quiet
  # shellcheck disable=SC2086
  gcloud storage rsync -r data/predicciones "gs://${BUCKET}/predicciones" ${BORRAR} --quiet
  gcloud storage rsync -r models            "gs://${BUCKET}/models"       --quiet
  ok "gold, predicciones y models subidos${BORRAR:+ (con borrado de lo que sobraba)}"

  # Silver y Bronze los necesita el JOB para no re-descargar cinco temporadas enteras.
  # Sin Bronze, la primera corrida baja todo de nuevo: son minutos de más, justo en vivo.
  if [ -d data/silver ]; then
    gcloud storage rsync -r data/silver "gs://${BUCKET}/silver" --quiet
    ok "silver subido"
  fi
  if [ -d data/bronze ]; then
    echo "  subiendo bronze (~300 MB, tarda)…"
    gcloud storage rsync -r data/bronze "gs://${BUCKET}/bronze" --quiet
    ok "bronze subido: el Job no va a re-descargar las temporadas cerradas"
  else
    aviso "sin bronze local: la primera corrida del Job va a bajar todo (varios minutos)"
  fi
else
  aviso "salteado; se asume que el bucket ya tiene el dato"
fi

# ---------------------------------------------------------------- reset: hasta acá
if [ "${MODO}" = "reset" ]; then
  # Se fuerza al servicio a releer Gold en vez de esperar a que venza el TTL de 5 min.
  SERVICE_URL="$(gcloud run services describe "${SERVICE}" --region "${REGION}" \
                 --format='value(status.url)' 2>/dev/null || echo '')"
  if [ -n "${SERVICE_URL}" ]; then
    gcloud run services update "${SERVICE}" --region "${REGION}" \
      --update-env-vars "TP_RESET_AT=$(date +%s)" --quiet >/dev/null
    ok "servicio reiniciado: relee Gold desde el bucket"
    sleep 5
    PROX="$(curl -s --max-time 90 "${SERVICE_URL}/health" \
            | python3 -c "import json,sys; print(json.load(sys.stdin)['proxima_predecible'])" \
            2>/dev/null || echo '?')"
    if [ "${PROX}" = "5" ]; then
      ok "estado inicial restaurado: la próxima predecible vuelve a ser la GW5"
    else
      aviso "la próxima predecible quedó en '${PROX}', esperaba 5"
    fi
    echo
    echo "  La página: ${SERVICE_URL}"
  else
    aviso "el servicio no existe todavía; corré el script sin --reset"
  fi
  exit 0
fi

# ---------------------------------------------------------------- 5 · imagen
titulo "5 · Construyendo la imagen (sólo código, sin datos ni modelos)"
gcloud artifacts repositories describe "${REPO}" --location="${REGION}" >/dev/null 2>&1 \
  || gcloud artifacts repositories create "${REPO}" \
       --repository-format=docker --location="${REGION}" --quiet
ok "Artifact Registry ${REPO}"

gcloud builds submit --tag "${IMAGE}" . --quiet
ok "imagen publicada"

# ---------------------------------------------------------------- 6 · el Job
titulo "6 · Cloud Run Job (el pipeline)"
gcloud run jobs deploy "${JOB}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --service-account "${SA_JOB}" \
  --set-env-vars "TP_STORAGE_BACKEND=gcs,TP_GCS_BUCKET=${BUCKET}" \
  --memory 2Gi --cpu 2 --task-timeout 30m --max-retries 1 \
  --command python --args "-m,pipeline.pre_deadline" \
  --quiet
ok "${JOB} desplegado (NO se ejecuta: eso es la demo)"

# ---------------------------------------------------------------- 7 · el servicio
titulo "7 · Cloud Run Service (la API y la página)"

# El POST que dispara el pipeline gasta cómputo y el servicio es público para que la
# página lo sea. Sin token el endpoint queda apagado; con token, sólo quien lo tenga.
ADMIN_TOKEN="${ADMIN_TOKEN:-$(openssl rand -hex 16)}"

gcloud run deploy "${SERVICE}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --service-account "${SA_API}" \
  --set-env-vars "TP_STORAGE_BACKEND=gcs,TP_GCS_BUCKET=${BUCKET},TP_GCP_PROJECT=${PROJECT_ID},TP_REGION=${REGION},TP_JOB_NAME=${JOB},TP_ADMIN_TOKEN=${ADMIN_TOKEN}" \
  --memory 1Gi --cpu 1 \
  --quiet
ok "${SERVICE} desplegado"

# Puede PEDIR que el Job corra. Sigue sin poder escribir en el bucket.
gcloud run jobs add-iam-policy-binding "${JOB}" --region "${REGION}" \
  --member="serviceAccount:${SA_API}" --role="roles/run.invoker" --quiet >/dev/null
ok "la API puede disparar el Job (y nada más)"

SERVICE_URL="$(gcloud run services describe "${SERVICE}" \
  --region "${REGION}" --format='value(status.url)')"

# ---------------------------------------------------------------- 8 · verificación
titulo "8 · Verificando que quedó como tiene que quedar"

esperar_json() {   # url, jq, esperado, descripcion
  local got
  got="$(curl -s --max-time 60 "$1" | python3 -c "import json,sys; print($2)" 2>/dev/null || echo "?")"
  if [ "${got}" = "$3" ]; then ok "$4  ($got)"; else aviso "$4  -> obtuve '${got}', esperaba '$3'"; fi
}

curl -s --max-time 90 "${SERVICE_URL}/health" >/dev/null || aviso "el primer request paga el arranque en frío"

esperar_json "${SERVICE_URL}/health" "json.load(sys.stdin)['status']" "ok" "/health"
esperar_json "${SERVICE_URL}/health" "json.load(sys.stdin)['proxima_predecible']" "5" "la próxima predecible es la GW5"

CODIGO_5="$(curl -s -o /dev/null -w '%{http_code}' "${SERVICE_URL}/predict/2026-27/5")"
CODIGO_6="$(curl -s -o /dev/null -w '%{http_code}' "${SERVICE_URL}/predict/2026-27/6")"
[ "${CODIGO_5}" = "200" ] && ok "GW5 se predice (200)" || aviso "GW5 dio ${CODIGO_5}"
[ "${CODIGO_6}" = "409" ] && ok "GW6 todavía no (409) — esto es lo que va a cambiar en vivo" \
                          || aviso "GW6 dio ${CODIGO_6}, esperaba 409"

esperar_json "${SERVICE_URL}/actualizar" "str(json.load(sys.stdin)['hace_falta'])" "True" \
  "el sistema sabe que hay datos nuevos para incorporar"

# ---------------------------------------------------------------- listo
titulo "Listo"
cat <<FIN

  La página:     ${SERVICE_URL}
  El token:      ${ADMIN_TOKEN}
                 (guardalo: lo pide el POST /actualizar desde afuera de la página)

  Para la demo en vivo:

    1. Mostrá la página. La GW5 está predicha; la 6 en adelante, en gris.
    2. El panel "Estado del dato" ya dice que conviene actualizar.
    3. Apretá "Actualizar datos". Dispara el Cloud Run Job.
    4. Mientras corre, mostrá los logs:

         gcloud run jobs executions list --job ${JOB} --region ${REGION}
         gcloud logging read 'jsonPayload.evento="prediccion"' --limit 10 \\
           --format='table(jsonPayload.gameweek, jsonPayload.estado, jsonPayload.latencia_ms)'

    5. Cuando termina, la página se recarga sola: la GW5 pasa a tener resultado y
       aparece la GW6 predicha.

  Para PROBAR que el botón funciona sin gastar la demo:

    1. Apretá "Actualizar datos". El Job ingesta la GW5 y Gold avanza a la GW6.
    2. Verificá que haya andado:

         curl -s ${SERVICE_URL}/health | python3 -m json.tool | grep proxima
         # tiene que decir 6

    3. Volvé al estado inicial:

         bash scripts/preparar_demo.sh --reset

       Vuelve a subir el Gold y las predicciones congeladas, y reinicia el servicio para
       que las relea. La próxima predecible vuelve a ser la GW5 y el botón queda otra vez
       con algo para mostrar.

  Para apagar todo después:

    gcloud run services delete ${SERVICE} --region ${REGION} --quiet
    gcloud run jobs delete ${JOB} --region ${REGION} --quiet
    gcloud artifacts repositories delete ${REPO} --location ${REGION} --quiet
    gcloud storage rm -r gs://${BUCKET}

FIN
