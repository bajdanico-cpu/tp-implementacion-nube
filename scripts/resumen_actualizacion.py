"""Resumen de la última actualización del dato: del botón al Gold nuevo, en un comando.

Un "Actualizar datos" deja rastro en cinco lugares distintos. Este script los junta:

    1. el pedido     jsonPayload.evento="pipeline_disparo"   (log del Service: quién, cuándo)
    2. el cierre     jsonPayload.evento="pipeline_fin"       (log del Service: ok/error, segundos)
    3. la ejecución  gcloud run jobs executions              (Cloud Run: estado y duración)
    4. los pasos     gs://BUCKET/pipeline/runs/<stamp>.json  (el diario del pipeline, paso por paso)
    5. el resultado  GET /health                             (qué fecha quedó próxima, cuántas filas)

Más los WARNING/ERROR que haya dejado esa ejecución del Job.

Sólo librería estándar + `gcloud` (Cloud Shell ya lo tiene autenticado).

Uso:
    python scripts/resumen_actualizacion.py
    python scripts/resumen_actualizacion.py --desde 1d
    python scripts/resumen_actualizacion.py | tee evidencia_actualizacion.txt
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request

REGION = os.environ.get("REGION", "us-central1")
SERVICE = os.environ.get("SERVICE", "premier-ml-api")
JOB = os.environ.get("JOB", "premier-ml-pipeline")
BUCKET = os.environ.get("TP_GCS_BUCKET", "tp-mlops-premier-2026-bucket")


def gcloud(*args: str) -> str:
    exe = shutil.which("gcloud")
    if not exe:
        raise SystemExit("No hay gcloud. Corré esto en Cloud Shell.")
    r = subprocess.run([exe, *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:500])
    return r.stdout


def gcloud_json(*args: str):
    try:
        return json.loads(gcloud(*args, "--format=json") or "null")
    except (RuntimeError, json.JSONDecodeError) as exc:
        print(f"  (no se pudo leer: {exc})")
        return None


def eventos_servicio(desde: str) -> list[dict]:
    filtro = (f'resource.type="cloud_run_revision" AND resource.labels.service_name="{SERVICE}" '
              f'AND (jsonPayload.evento="pipeline_disparo" OR jsonPayload.evento="pipeline_fin")')
    return gcloud_json("logging", "read", filtro, f"--freshness={desde}", "--limit=50") or []


def resumir_eventos(eventos: list[dict]) -> tuple[dict | None, dict | None]:
    """(el último disparo, su fin si ya llegó). Puro: lo usan los tests."""
    disparos = [e for e in eventos if e.get("jsonPayload", {}).get("evento") == "pipeline_disparo"]
    if not disparos:
        return None, None
    ultimo = max(disparos, key=lambda e: e.get("timestamp", ""))
    tarea = ultimo["jsonPayload"].get("tarea")
    fines = [e for e in eventos if e.get("jsonPayload", {}).get("evento") == "pipeline_fin"
             and e["jsonPayload"].get("tarea") == tarea]
    return ultimo, (fines[0] if fines else None)


def resumir_corrida(corrida: dict) -> list[str]:
    """Las líneas de la tabla de pasos de un `pipeline/runs/<stamp>.json`. Puro."""
    lineas = [f"  corrida {corrida.get('corrida')} · {corrida.get('at')} · "
              f"{'OK' if corrida.get('ok') else 'FALLÓ'}",
              f"  {'paso':<16} {'estado':<9} {'seg':>7}  detalle"]
    total = 0.0
    for p in corrida.get("pasos", []):
        seg = float(p.get("segundos") or 0)
        total += seg
        if p.get("estado") == "error":
            detalle = str(p.get("error", ""))[:90]
        else:
            s = p.get("salida")
            detalle = (f"{s.get('filas')} filas" if isinstance(s, dict) and "filas" in s
                       else str(s)[:60] if s not in (None, "None") else "")
        lineas.append(f"  {p.get('paso', '?'):<16} {p.get('estado', '?'):<9} {seg:>7.1f}  {detalle}")
    lineas.append(f"  {'total':<16} {'':<9} {total:>7.1f}")
    return lineas


def url_servicio() -> str | None:
    if os.environ.get("SERVICE_URL"):
        return os.environ["SERVICE_URL"].rstrip("/")
    try:
        return gcloud("run", "services", "describe", SERVICE, "--region", REGION,
                      "--format=value(status.url)").strip() or None
    except RuntimeError:
        return None


def get_json(url: str):
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return json.load(r)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Resumen de la última actualización del dato.")
    ap.add_argument("--desde", default="6h", help="Ventana para buscar el disparo (6h, 1d).")
    args = ap.parse_args()

    print(f"=== Última actualización del dato · servicio {SERVICE} · job {JOB} ===\n")

    # 1-2. El pedido y su cierre, desde el log del Service.
    print("1. El pedido (log del servicio)")
    disparo, fin = resumir_eventos(eventos_servicio(args.desde))
    if disparo is None:
        print(f"  No hay ningún pipeline_disparo en las últimas {args.desde}. "
              f"¿Se apretó el botón? Probá --desde 1d.")
    else:
        d = disparo["jsonPayload"]
        print(f"  {disparo['timestamp'][:19]}Z  tarea {d.get('tarea')}  "
              f"estado={d.get('estado')}  destino={d.get('destino')}")
        print(f"  motivo: {d.get('motivo')}")
        if fin:
            f = fin["jsonPayload"]
            print(f"  {fin['timestamp'][:19]}Z  FIN estado={f.get('estado')}  "
                  f"segundos={f.get('segundos')}")
        else:
            print("  todavía sin pipeline_fin (corriendo, o nadie consultó el estado desde la página)")

    # 3. La ejecución del Job.
    print("\n2. La ejecución del Job (Cloud Run)")
    ejecuciones = gcloud_json("run", "jobs", "executions", "list", "--job", JOB,
                              "--region", REGION, "--limit=1") or []
    exe = ejecuciones[0] if ejecuciones else None
    if exe:
        nombre = exe["metadata"]["name"]
        st = exe.get("status", {})
        print(f"  {nombre}")
        print(f"  arrancó {st.get('startTime', '?')[:19]}Z · terminó "
              f"{(st.get('completionTime') or 'todavía no')[:19]} · "
              f"ok={st.get('succeededCount', 0)} fallidas={st.get('failedCount', 0)}")
    else:
        nombre = None
        print("  no hay ejecuciones")

    # 4. El diario de pasos del bucket.
    print(f"\n3. Los pasos (gs://{BUCKET}/pipeline/runs/)")
    try:
        runs = sorted(x for x in gcloud("storage", "ls", f"gs://{BUCKET}/pipeline/runs/").split()
                      if x.endswith(".json"))
        corrida = json.loads(gcloud("storage", "cat", runs[-1])) if runs else None
    except (RuntimeError, json.JSONDecodeError) as exc:
        corrida = None
        print(f"  (no se pudo leer: {exc})")
    if corrida:
        print("\n".join(resumir_corrida(corrida)))

    # 5. Advertencias y errores de esa ejecución.
    if nombre:
        print("\n4. WARNING / ERROR de esa ejecución")
        filtro = (f'resource.type="cloud_run_job" AND resource.labels.job_name="{JOB}" '
                  f'AND labels."run.googleapis.com/execution_name"="{nombre}" AND severity>=WARNING')
        malas = gcloud_json("logging", "read", filtro, "--freshness=7d", "--limit=20") or []
        if not malas:
            print("  ninguna")
        for e in malas:
            msg = (e.get("jsonPayload") or {}).get("message") or e.get("textPayload", "")
            print(f"  {e.get('severity'):<8} {str(msg)[:110]}")

    # 6. Cómo quedó el servicio.
    print("\n5. Cómo quedó el servicio")
    base = url_servicio()
    if base:
        h = get_json(base + "/health")
        print(f"  /health: status={h.get('status')} proxima_predecible={h.get('proxima_predecible')} "
              f"gold_filas={h.get('gold_filas')} gold_built_at={h.get('gold_built_at')}")
        a = get_json(base + "/actualizar")
        print(f"  /actualizar: hace_falta={a.get('hace_falta')} · {a.get('motivo')}")
    else:
        print("  no se pudo resolver la URL del servicio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
