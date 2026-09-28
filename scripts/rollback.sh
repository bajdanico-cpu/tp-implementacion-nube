#!/usr/bin/env bash
# Rollback del servicio por tráfico: sin rebuild, sin redeploy, en segundos.
#
#   bash scripts/rollback.sh                    # lista revisiones: modelo, temporada, tráfico
#   bash scripts/rollback.sh --anterior         # 100 % a la revisión lista anterior a la actual
#   bash scripts/rollback.sh --a premier-ml-api-00012-abc   # 100 % a esa revisión
#
# Qué vuelve atrás y qué no (detalle en gcp/ROLLBACK.md):
#   - el CÓDIGO y las VARIABLES de la revisión: siempre.
#   - el MODELO: sólo si la revisión lo fija con TP_MODEL_VERSION. Si no, todas leen el
#     mismo PRODUCTION.json del bucket y el rollback no lo toca.
#   - la TEMPORADA: sólo si la revisión la fija con TP_SEASON.
#   - el DATO (Gold, predicciones): nunca. Vive en el bucket, compartido por todas las
#     revisiones. Se vuelve atrás aparte, con `python -m common.versiones --restaurar`.
#
# Después de mover el tráfico, verifica con /health que la revisión nueva sirve lo que
# dice servir. Un rollback que no se verifica es una esperanza.

set -euo pipefail

REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-premier-ml-api}"

command -v gcloud >/dev/null || { echo "No hay gcloud. Corré esto en Cloud Shell." >&2; exit 1; }

listar() {
  # Una sola llamada por recurso y el formateo en Python: el --format de gcloud no sabe
  # sacar una variable de entorno por nombre de una lista.
  local revs trafico
  revs="$(gcloud run revisions list --service "${SERVICE}" --region "${REGION}" --format=json)"
  trafico="$(gcloud run services describe "${SERVICE}" --region "${REGION}" --format=json)"
  REVS="${revs}" TRAFICO="${trafico}" python3 - <<'PY'
import json, os
revs = json.loads(os.environ["REVS"])
svc = json.loads(os.environ["TRAFICO"])
pct = {t.get("revisionName"): t.get("percent", 0) for t in svc.get("status", {}).get("traffic", [])}
print(f"{'revisión':<34} {'creada':<20} {'lista':<6} {'tráfico':>7}  {'TP_MODEL_VERSION':<18} TP_SEASON")
for r in sorted(revs, key=lambda r: r["metadata"]["creationTimestamp"], reverse=True):
    nombre = r["metadata"]["name"]
    env = {e["name"]: e.get("value", "") for e in r["spec"]["containers"][0].get("env", [])}
    lista = any(c.get("type") == "Ready" and c.get("status") == "True"
                for c in r.get("status", {}).get("conditions", []))
    print(f"{nombre:<34} {r['metadata']['creationTimestamp'][:19]:<20} "
          f"{'sí' if lista else 'NO':<6} {str(pct.get(nombre, 0)) + ' %':>7}  "
          f"{env.get('TP_MODEL_VERSION', '(PRODUCTION.json)'):<18} {env.get('TP_SEASON', '(config.yaml)')}")
PY
}

anterior() {
  # La revisión LISTA más nueva entre las que no reciben tráfico hoy y son anteriores a
  # la que más tráfico recibe. Es "la última buena" del instructivo de la clase 7.
  local revs trafico
  revs="$(gcloud run revisions list --service "${SERVICE}" --region "${REGION}" --format=json)"
  trafico="$(gcloud run services describe "${SERVICE}" --region "${REGION}" --format=json)"
  REVS="${revs}" TRAFICO="${trafico}" python3 - <<'PY'
import json, os, sys
revs = json.loads(os.environ["REVS"])
svc = json.loads(os.environ["TRAFICO"])
trafico = [t for t in svc.get("status", {}).get("traffic", []) if t.get("percent")]
if not trafico:
    sys.exit("No hay revisiones con tráfico.")
actual = max(trafico, key=lambda t: t["percent"])["revisionName"]
creada = {r["metadata"]["name"]: r["metadata"]["creationTimestamp"] for r in revs}
lista = {r["metadata"]["name"] for r in revs
         if any(c.get("type") == "Ready" and c.get("status") == "True"
                for c in r.get("status", {}).get("conditions", []))}
candidatas = sorted((n for n in lista if creada[n] < creada[actual]),
                    key=lambda n: creada[n], reverse=True)
if not candidatas:
    sys.exit(f"No hay ninguna revisión lista anterior a {actual}.")
print(candidatas[0])
PY
}

verificar() {
  local url salud
  url="$(gcloud run services describe "${SERVICE}" --region "${REGION}" --format='value(status.url)')"
  salud="$(curl -s --max-time 90 "${url}/health" || echo '{}')"
  echo
  echo "  /health después del rollback:"
  echo "${salud}" | python3 -c "
import json, sys
h = json.load(sys.stdin) if sys.stdin else {}
for k in ('status', 'model_version', 'season_actual', 'proxima_predecible', 'gold_built_at', 'detail'):
    print(f'    {k:<20} {h.get(k)}')
sys.exit(0 if h.get('status') == 'ok' else 1)
" || { echo "  ✗ la revisión no quedó sana: elegí otra con --a" >&2; exit 1; }
  echo "  ✓ sirviendo ok"
  echo
  echo "  Evidencia del lado del servidor (qué revisión atendió cada request):"
  echo "    python scripts/smoke_load.py --n 10 && python scripts/logs_servidor.py --desde 10m"
}

case "${1:-}" in
  "")
    listar
    ;;
  --anterior)
    DESTINO="$(anterior)"
    echo "Revisión anterior lista: ${DESTINO}"
    gcloud run services update-traffic "${SERVICE}" --region "${REGION}" \
      --to-revisions "${DESTINO}=100" --quiet
    verificar
    ;;
  --a)
    [ -n "${2:-}" ] || { echo "Uso: bash scripts/rollback.sh --a <revisión>" >&2; exit 1; }
    gcloud run services update-traffic "${SERVICE}" --region "${REGION}" \
      --to-revisions "${2}=100" --quiet
    verificar
    ;;
  *)
    sed -n 2,6p "$0"
    exit 1
    ;;
esac
