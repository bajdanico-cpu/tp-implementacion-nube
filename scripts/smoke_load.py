"""Genera carga contra la API y mide latencia.

Dispara N requests `GET` a `/predict/{season}/{gameweek}` y reporta la distribución de
latencia (p50/p95/p99) y los errores. Sirve para dos cosas:

  1. **Ver latencia de verdad**, no un número teórico: el primer request paga el arranque en
     frío de Cloud Run (cold start) y los siguientes salen tibios.

     Desde que el servicio sirve desde Gold, los requests tibios dan **decenas de
     milisegundos** (p50 ~47 ms medido en local). Antes tardaban ~25 segundos porque cada
     uno reconstruía las 279 features desde Silver y recargaba los cinco boosters; hoy la
     fila ya está calculada y el modelo queda cacheado en el proceso. El `max` sigue siendo
     el cold start, que ahora incluye bajar Gold y los modelos del bucket.
  2. **Producir tráfico** para después leerlo en los logs (`gcloud run services logs read`).

Solo librería estándar: se corre en Cloud Shell sin instalar nada. No hay payload: la API
arma las features sola a partir de la temporada y la fecha de la ruta.

La URL base se resuelve en este orden: `--url`, la variable de entorno SERVICE_URL, y por
último la URL fija DEFAULT_URL.

Uso:
    python scripts/smoke_load.py                                  # 10 requests a SERVICE_URL (o la URL por defecto)
    python scripts/smoke_load.py --n 30 --concurrency 4
    python scripts/smoke_load.py --endpoint /predict/2026-27/5
    python scripts/smoke_load.py --url http://127.0.0.1:8080      # contra la API local
"""
from __future__ import annotations

import argparse
import os
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DEFAULT_URL = "https://premier-ml-api-tz75rnogkq-uc.a.run.app"
# La fecha 4 ya se jugó, así que la respuesta sale del registro congelado: es el camino
# más liviano. Para medir el camino que corre el modelo, apuntar a la próxima predecible
# (`--endpoint /predict/2026-27/5`), que es el que importa para el cold start.
DEFAULT_ENDPOINT = "/predict/2026-27/4"


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


def one_request(url: str, timeout: float) -> tuple[float, int]:
    """Devuelve (latencia_ms, status). status 0 si ni siquiera hubo respuesta HTTP."""
    request = urllib.request.Request(url, method="GET")
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
    parser.add_argument("--url", default=os.environ.get("SERVICE_URL") or DEFAULT_URL,
                        help="Base URL del servicio (default: $SERVICE_URL si existe, "
                             "si no la URL fija del script).")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Ruta a golpear.")
    parser.add_argument("--n", type=int, default=10, help="Cantidad de requests.")
    parser.add_argument("--concurrency", type=int, default=1, help="Requests en paralelo.")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="Segundos de espera por request (el cold start puede ser largo).")
    args = parser.parse_args()

    target = args.url.rstrip("/") + args.endpoint

    print(f"Disparando {args.n} requests GET a {target} (concurrencia {args.concurrency})...")
    wall_start = time.perf_counter()
    if args.concurrency > 1:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            results = list(pool.map(lambda _: one_request(target, args.timeout), range(args.n)))
    else:
        results = [one_request(target, args.timeout) for _ in range(args.n)]
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
    print("  latencia (ms):")
    print(f"    min  : {min(latencies):8.1f}")
    print(f"    p50  : {statistics.median(latencies):8.1f}")
    print(f"    p95  : {percentile(latencies, 95):8.1f}")
    print(f"    p99  : {percentile(latencies, 99):8.1f}")
    print(f"    max  : {max(latencies):8.1f}   <- suele ser el arranque en frio (cold start)")
    if errors:
        codes = sorted({s for s in statuses if not (200 <= s < 300)})
        print(f"  codigos de error: {codes}")


if __name__ == "__main__":
    main()
