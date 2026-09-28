"""La latencia y los errores vistos DESDE EL SERVIDOR, leídos de Cloud Logging.

`scripts/smoke_load.py` mide desde afuera: lo que tarda en volver la respuesta, con la red y
el arranque en frío adentro. Este script pregunta lo mismo al servicio, a partir de lo que
quedó en sus logs, en dos capas:

    borde   run.googleapis.com/requests   lo escribe Cloud Run: status y `httpRequest.latency`,
                                          desde que el request entra a Google hasta que sale
    app     jsonPayload.evento            lo escribe nuestra app: `latencia_ms` del handler,
                                          el estado de la fecha, el modelo, el error

Con `--corrida <id>` (el que imprime `smoke_load.py`) filtra exactamente esos requests y
arma la tabla de tres capas. La resta es lo que cada capa agrega:

    cliente - borde  = red + TLS entre Cloud Shell y Google
    borde   - app    = cola, arranque en frío, FastAPI/serialización

Además cuenta qué REVISIÓN de Cloud Run atendió cada request. Después de un rollback es la
prueba de que el tráfico volvió a la revisión buena (ver `gcp/ROLLBACK.md`).

Sólo librería estándar + `gcloud` (Cloud Shell ya lo tiene autenticado).

Uso:
    python scripts/logs_servidor.py                         # la última hora del servicio
    python scripts/logs_servidor.py --corrida smoke-20260928T213000Z
    python scripts/logs_servidor.py --desde 6h --errores    # detalle de cada error
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

SERVICE = os.environ.get("SERVICE", "premier-ml-api")
SMOKE = Path(__file__).resolve().parents[1] / "monitoring" / "output" / "smoke"
EVENTOS_APP = ("prediccion", "prediccion_error", "health_degradado")


def percentil(valores: list[float], pct: float) -> float:
    """Mismo cálculo que `smoke_load.percentile`, para comparar peras con peras."""
    if not valores:
        return float("nan")
    v = sorted(valores)
    if len(v) == 1:
        return v[0]
    rank = (pct / 100) * (len(v) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (rank - lo)


def latencia_borde_ms(entrada: dict) -> float | None:
    """`httpRequest.latency` viene como texto: "0.261346s"."""
    lat = (entrada.get("httpRequest") or {}).get("latency")
    if not lat or not str(lat).endswith("s"):
        return None
    try:
        return float(str(lat)[:-1]) * 1000
    except ValueError:
        return None


def filtro(service: str, corrida: str | None) -> str:
    base = (f'resource.type="cloud_run_revision" '
            f'AND resource.labels.service_name="{service}"')
    eventos = " OR ".join(f'jsonPayload.evento="{e}"' for e in EVENTOS_APP)
    if corrida:
        # La app copia `X-Corrida` en cada línea; Cloud Run guarda el User-Agent que
        # manda smoke_load ("smoke_load/<corrida>"). `:` es "contiene".
        return (f'{base} AND (jsonPayload.corrida="{corrida}" '
                f'OR httpRequest.userAgent:"smoke_load/{corrida}")')
    return (f'{base} AND (({eventos}) '
            f'OR logName:"run.googleapis.com%2Frequests")')


def leer(service: str, corrida: str | None, desde: str, limite: int,
         proyecto: str | None) -> list[dict]:
    gcloud = shutil.which("gcloud")
    if not gcloud:
        raise SystemExit("No hay gcloud. Corré esto en Cloud Shell.")
    cmd = [gcloud, "logging", "read", filtro(service, corrida), "--format=json",
           f"--limit={limite}", f"--freshness={desde}"]
    if proyecto:
        cmd.append(f"--project={proyecto}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"gcloud logging read falló:\n{r.stderr.strip()}")
    return json.loads(r.stdout or "[]")


def separar(entradas: list[dict]) -> tuple[list[dict], list[dict]]:
    """(líneas del borde, eventos de la app)."""
    borde, app = [], []
    for e in entradas:
        if "httpRequest" in e and str(e.get("logName", "")).endswith("%2Frequests"):
            borde.append(e)
        elif (e.get("jsonPayload") or {}).get("evento") in EVENTOS_APP:
            app.append(e)
    return borde, app


def resumir(entradas: list[dict]) -> dict:
    """Lo que el servidor sabe de esos requests. Puro: los tests lo llaman directo."""
    borde, app = separar(entradas)
    lat_borde = [x for x in (latencia_borde_ms(e) for e in borde) if x is not None]
    pred = [e["jsonPayload"] for e in app if e["jsonPayload"]["evento"] == "prediccion"]
    errs = [e["jsonPayload"] for e in app if e["jsonPayload"]["evento"] == "prediccion_error"]

    por_estado: dict[str, list[float]] = {}
    for p in pred:
        clave = f"{p.get('estado', '?')}/{p.get('origen', '?')}"
        por_estado.setdefault(clave, []).append(float(p.get("latencia_ms", 0)))

    return {
        "n_borde": len(borde),
        "status_borde": Counter(int((e.get("httpRequest") or {}).get("status", 0)) for e in borde),
        "lat_borde": lat_borde,
        "lat_app": [float(e["jsonPayload"].get("latencia_ms", 0)) for e in app
                    if e["jsonPayload"]["evento"] in ("prediccion", "prediccion_error")],
        "por_estado": por_estado,
        "errores": Counter((x.get("status"), x.get("error_type")) for x in errs),
        "detalle_errores": errs,
        "revisiones": Counter((e.get("resource") or {}).get("labels", {}).get("revision_name", "?")
                              for e in borde or app),
        "modelos": Counter(p.get("model_version") for p in pred),
        "degradado": [e["jsonPayload"] for e in app
                      if e["jsonPayload"]["evento"] == "health_degradado"],
    }


def _fila(nombre: str, v: list[float]) -> str:
    if not v:
        return f"  {nombre:<22} {'-':>7} {'-':>9} {'-':>9} {'-':>9}"
    return (f"  {nombre:<22} {len(v):>7} {percentil(v, 50):>9.1f} "
            f"{percentil(v, 95):>9.1f} {max(v):>9.1f}")


def imprimir(r: dict, cliente: dict | None, ver_errores: bool) -> None:
    print("\nLatencia (ms)            n       p50       p95       max")
    if cliente:
        print(_fila("cliente (smoke_load)", cliente["latencias_ms"]))
    print(_fila("borde   (Cloud Run)", r["lat_borde"]))
    print(_fila("app     (latencia_ms)", r["lat_app"]))
    for clave, v in sorted(r["por_estado"].items()):
        print(_fila(f"  └ {clave}", v))

    if cliente and r["lat_borde"] and r["lat_app"]:
        c, b, a = (percentil(cliente["latencias_ms"], 50), percentil(r["lat_borde"], 50),
                   percentil(r["lat_app"], 50))
        print(f"\n  En la mediana: red {c - b:+.1f} ms · Cloud Run + FastAPI {b - a:+.1f} ms · "
              f"app {a:.1f} ms")
        if len(r["lat_borde"]) < cliente["n"]:
            print(f"  Ojo: el cliente mandó {cliente['n']} y el servidor registra "
                  f"{len(r['lat_borde'])}. Esperá unos segundos y volvé a correr: "
                  f"Cloud Logging tarda en indexar.")

    print(f"\nStatus en el borde : {dict(sorted(r['status_borde'].items())) or '-'}")
    if r["errores"]:
        print("Errores de la app  : " + ", ".join(
            f"{s} {t} x{n}" for (s, t), n in r["errores"].most_common()))
    print(f"Revisión que atendió: {dict(r['revisiones']) or '-'}")
    print(f"Modelo servido      : {dict(r['modelos']) or '-'}")
    if r["degradado"]:
        print(f"health_degradado    : {len(r['degradado'])} — {r['degradado'][0].get('reason')}")
    if ver_errores:
        for e in r["detalle_errores"][:20]:
            print(f"  GW{e.get('gameweek')} -> {e.get('status')} {e.get('error_type')}: "
                  f"{e.get('reason', '')[:120]}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Latencia y errores del servicio, desde Cloud Logging.")
    ap.add_argument("--corrida", help="Id que imprimió smoke_load.py. Filtra esos requests.")
    ap.add_argument("--desde", default="1h", help="Ventana hacia atrás (1h, 6h, 1d). Default 1h.")
    ap.add_argument("--limite", type=int, default=2000)
    ap.add_argument("--service", default=SERVICE)
    ap.add_argument("--project", default=os.environ.get("PROJECT_ID"))
    ap.add_argument("--errores", action="store_true", help="Lista cada error con su motivo.")
    args = ap.parse_args()

    entradas = leer(args.service, args.corrida, args.desde, args.limite, args.project)
    cliente = None
    if args.corrida and (SMOKE / f"{args.corrida}.json").exists():
        cliente = json.loads((SMOKE / f"{args.corrida}.json").read_text(encoding="utf-8"))

    alcance = f"corrida {args.corrida}" if args.corrida else f"última(s) {args.desde}"
    print(f"Servicio {args.service} · {alcance} · {len(entradas)} entradas de log")
    if not entradas:
        print("\nNo hay nada. Revisá: proyecto activo (gcloud config get-value project), la "
              "ventana (--desde) y que el tráfico haya ido a ESTE servicio.")
        return 1
    imprimir(resumir(entradas), cliente, args.errores)
    return 0


if __name__ == "__main__":
    sys.exit(main())
