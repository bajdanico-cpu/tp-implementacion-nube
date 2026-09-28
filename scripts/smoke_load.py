"""Genera carga contra la API y mide latencia DESDE EL CLIENTE.

Dispara N requests `GET` a `/predict/{season}/{gameweek}` y reporta la distribución de
latencia (p50/p95/p99) y los errores. Es la mitad de la medición: la otra mitad la da el
servidor, en sus logs, y la junta `scripts/logs_servidor.py --corrida <id>`.

  1. **Latencia vista por quien consulta.** Incluye todo: red, TLS, el borde de Cloud Run,
     el arranque en frío si la instancia estaba apagada, y la app. El `max` suele ser el
     cold start.
  2. **Tráfico marcado.** Cada corrida manda `X-Corrida: <id>` en todos sus requests. El
     servicio lo copia en cada línea de log (`jsonPayload.corrida`), así que después se
     pueden pedir al servidor exactamente los eventos de ESTA corrida y comparar.

Sólo librería estándar: se corre en Cloud Shell sin instalar nada. No hay payload: la API
arma las features sola a partir de la temporada y la fecha de la ruta.

**La URL no tiene default fijo, a propósito.** Hasta septiembre de 2026 caía en silencio en
una URL escrita en el script, que era el servicio de OTRO proyecto: el smoke "andaba" y
después en Cloud Logging del proyecto propio no había nada. Ahora se resuelve así:
`--url`, la variable `SERVICE_URL`, y si no hay ninguna, se le pregunta a `gcloud` por el
servicio `premier-ml-api` del proyecto activo. Si nada de eso resuelve, corta con error.

Uso:
    python scripts/smoke_load.py                                  # 10 requests a la próxima fecha
    python scripts/smoke_load.py --n 30 --concurrency 4
    python scripts/smoke_load.py --endpoint /predict/2026-27/5
    python scripts/smoke_load.py --url http://127.0.0.1:8080      # contra la API local
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

SERVICE = os.environ.get("SERVICE", "premier-ml-api")
REGION = os.environ.get("REGION", "us-central1")
# Donde queda el detalle de cada corrida, para que `logs_servidor.py` lo lea.
SALIDA = Path(__file__).resolve().parents[1] / "monitoring" / "output" / "smoke"


def percentile(values: list[float], pct: float) -> float:
    """Percentil por interpolación lineal (sin numpy). `pct` en [0, 100]."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def url_de_gcloud(service: str = SERVICE, region: str = REGION) -> str | None:
    """La URL del servicio en el proyecto activo de gcloud, o None si no se puede saber."""
    gcloud = shutil.which("gcloud")
    if not gcloud:
        return None
    try:
        r = subprocess.run([gcloud, "run", "services", "describe", service,
                            "--region", region, "--format=value(status.url)"],
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    url = r.stdout.strip()
    return url if r.returncode == 0 and url.startswith("http") else None


def resolver_url(arg: str | None, env=None, desde_gcloud=url_de_gcloud) -> tuple[str, str]:
    """(url, de dónde salió). Corta con SystemExit si no hay forma de saberla."""
    env = os.environ if env is None else env
    if arg:
        return arg, "--url"
    if env.get("SERVICE_URL"):
        return env["SERVICE_URL"], "SERVICE_URL"
    url = desde_gcloud()
    if url:
        return url, f"gcloud run services describe {SERVICE}"
    raise SystemExit(
        "No sé contra qué servicio correr. Pasá --url, o exportá SERVICE_URL:\n"
        f"  export SERVICE_URL=\"$(gcloud run services describe {SERVICE} "
        f"--region {REGION} --format='value(status.url)')\"")


def proxima_fecha(base: str, timeout: float) -> str | None:
    """`/predict/<temporada>/<próxima>` según el /health del servicio.

    La próxima es la que corre el modelo; una jugada sale del registro congelado y no lo
    toca. Si el default fuera una fecha fija, en un mes estaría midiendo el camino liviano.
    """
    try:
        with urllib.request.urlopen(base + "/health", timeout=timeout) as r:
            h = json.load(r)
        if h.get("proxima_predecible"):
            return f"/predict/{h['season_actual']}/{h['proxima_predecible']}"
    except Exception:  # noqa: BLE001
        pass
    return None


def one_request(url: str, timeout: float, corrida: str = "") -> tuple[float, int]:
    """Devuelve (latencia_ms, status). status 0 si ni siquiera hubo respuesta HTTP."""
    cabeceras = {"X-Corrida": corrida, "User-Agent": f"smoke_load/{corrida}"} if corrida else {}
    request = urllib.request.Request(url, method="GET", headers=cabeceras)
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        status = exc.code
    except Exception:  # noqa: BLE001 (red caída, timeout, etc.)
        status = 0
    latency_ms = (time.perf_counter() - start) * 1000
    return latency_ms, status


def main() -> None:
    parser = argparse.ArgumentParser(description="Carga y latencia contra la API del TP Premier ML.")
    parser.add_argument("--url", help="Base URL del servicio (default: $SERVICE_URL, "
                                      "o la de gcloud para el proyecto activo).")
    parser.add_argument("--endpoint", help="Ruta a golpear (default: la próxima fecha "
                                           "predecible, según /health).")
    parser.add_argument("--n", type=int, default=10, help="Cantidad de requests.")
    parser.add_argument("--concurrency", type=int, default=1, help="Requests en paralelo.")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="Segundos de espera por request (el cold start puede ser largo).")
    parser.add_argument("--corrida", help="Id de la corrida (default: smoke-<timestamp UTC>).")
    args = parser.parse_args()

    base, origen = resolver_url(args.url)
    base = base.rstrip("/")
    endpoint = args.endpoint or proxima_fecha(base, args.timeout) or "/health"
    corrida = args.corrida or "smoke-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = base + endpoint

    print(f"Servicio : {base}   (de {origen})")
    print(f"Corrida  : {corrida}")
    print(f"Disparando {args.n} requests GET a {endpoint} (concurrencia {args.concurrency})...")
    desde = datetime.now(timezone.utc)
    wall_start = time.perf_counter()
    if args.concurrency > 1:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            results = list(pool.map(lambda _: one_request(target, args.timeout, corrida),
                                    range(args.n)))
    else:
        results = [one_request(target, args.timeout, corrida) for _ in range(args.n)]
    wall_s = time.perf_counter() - wall_start

    latencies = [latency for latency, _ in results]
    statuses = [status for _, status in results]
    ok = sum(1 for s in statuses if 200 <= s < 300)
    errors = len(statuses) - ok

    print("")
    print(f"  requests   : {len(results)}")
    print(f"  ok (2xx)   : {ok}")
    print(f"  errores    : {errors}")
    print(f"  throughput : {len(results) / wall_s:.1f} req/s  ({wall_s:.2f}s total)")
    print("  latencia CLIENTE (ms):")
    print(f"    min  : {min(latencies):8.1f}")
    print(f"    p50  : {statistics.median(latencies):8.1f}")
    print(f"    p95  : {percentile(latencies, 95):8.1f}")
    print(f"    p99  : {percentile(latencies, 99):8.1f}")
    print(f"    max  : {max(latencies):8.1f}   <- suele ser el arranque en frio (cold start)")
    if errors:
        codes = sorted({s for s in statuses if not (200 <= s < 300)})
        print(f"  codigos de error: {codes}")

    SALIDA.mkdir(parents=True, exist_ok=True)
    archivo = SALIDA / f"{corrida}.json"
    archivo.write_text(json.dumps({
        "corrida": corrida, "url": base, "endpoint": endpoint, "desde": desde.isoformat(),
        "n": len(results), "concurrency": args.concurrency,
        "latencias_ms": [round(x, 2) for x in latencies], "status": statuses,
    }, indent=1), encoding="utf-8")

    print(f"\n  detalle    : {archivo}")
    print("\nLa misma corrida vista desde el SERVIDOR (esperar ~30 s a que llegue a Cloud Logging):")
    print(f"  python scripts/logs_servidor.py --corrida {corrida}")


if __name__ == "__main__":
    main()
