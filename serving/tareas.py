"""Disparar el pipeline desde la API, sin que la API lo ejecute.

El servicio **no corre el pipeline**: lo pide. La distinción no es formal. El pipeline baja
~27 MB, reconstruye Silver y Gold y tarda minutos; meterlo adentro de un request lo
pondría contra el timeout de Cloud Run, bloquearía el worker y obligaría a darle permiso
de escritura a la identidad que sólo tiene que leer. Acá se dispara un **Cloud Run Job**,
que tiene su propia identidad, su propio timeout y su propio registro de corridas.

    POST /actualizar        -> 202, con el id de la tarea
    GET  /actualizar/{id}   -> en qué anda

**En local no hay Job**, así que se lanza `python -m pipeline.pre_deadline` como
subproceso. Es el mismo comando que corre el Job: lo que cambia es quién lo ejecuta, no
qué se ejecuta.

**Quién puede dispararlo.** Un POST anónimo que gasta cómputo es un problema, y el
servicio está desplegado con `--allow-unauthenticated` para que la página sea pública. La
regla: en Cloud Run hace falta el header `X-Admin-Token` y tiene que coincidir con
`TP_ADMIN_TOKEN`; si la variable no está definida, el endpoint queda **apagado** en vez de
abierto. En local, sin `K_SERVICE`, se permite sin token para poder trabajar.
"""

from __future__ import annotations

import os
import secrets
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field

import pandas as pd

from common.config import CFG, utc_stamp
from common.logging_setup import evento, get_logger

log = get_logger(__name__)

TIMEOUT_METADATA_S = 3
TIMEOUT_ADMIN_S = 20

# Cada cuanto se le vuelve a preguntar a Cloud Run por una corrida en vuelo. La
# pagina consulta cada 4 s; sin esto serian cuatro llamadas a la Admin API por
# minuto y por pestania abierta, para un dato que cambia una vez.
REFRESCO_S = 10

# Si la Admin API no contesta nunca, igual hay que soltar: el Job tiene
# --task-timeout 30m, asi que a los 35 minutos una corrida viva es una mentira.
MAX_EN_VUELO_S = 35 * 60

# Cuánto se recuerda una tarea terminada. Alcanza para que la página muestre el final.
RETENCION_S = 3600


class NoAutorizado(Exception):
    """Falta el token, o el endpoint está apagado porque nadie configuró uno."""


class YaHayUna(Exception):
    """Ya hay una corrida en curso. Dos pipelines a la vez se pisan escribiendo Gold."""

    def __init__(self, tarea: "Tarea"):
        self.tarea = tarea
        super().__init__(f"Ya hay una actualización en curso ({tarea.id}).")


@dataclass
class Tarea:
    id: str
    estado: str                 # lanzada | corriendo | ok | error
    motivo: str
    detalle: str = ""
    lanzada_at: float = field(default_factory=time.time)
    terminada_at: float | None = None
    destino: str = "local"      # local | cloud-run-job
    referencia: str | None = None   # nombre de la operación en GCP, o el PID
    consultada_at: float = 0.0      # última vez que se le preguntó a Cloud Run

    def como_dict(self) -> dict:
        seg = (self.terminada_at or time.time()) - self.lanzada_at
        return {"id": self.id, "estado": self.estado, "motivo": self.motivo,
                "detalle": self.detalle, "destino": self.destino,
                "referencia": self.referencia, "segundos": round(seg, 1)}


_TAREAS: dict[str, Tarea] = {}
_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Autorización
# ---------------------------------------------------------------------------

def en_la_nube() -> bool:
    return bool(os.getenv("K_SERVICE"))


def autorizar(token: str | None) -> None:
    esperado = os.getenv("TP_ADMIN_TOKEN")
    if not en_la_nube():
        return                                  # en local, libre
    if not esperado:
        raise NoAutorizado(
            "El disparo del pipeline está apagado: no hay TP_ADMIN_TOKEN configurado en "
            "el servicio. Se prefiere apagado a abierto.")
    if not token or not secrets.compare_digest(token, esperado):
        raise NoAutorizado("Falta el header X-Admin-Token, o no coincide.")


# ---------------------------------------------------------------------------
# ¿Tiene sentido disparar?
# ---------------------------------------------------------------------------

def _construido_hace(gold: pd.DataFrame) -> float | None:
    """Segundos desde que se construyó Gold, o None si no se puede saber."""
    try:
        stamp = str(gold["gold_built_at"].max())
        t = pd.Timestamp(stamp.replace("Z", ""), tz="UTC")
        return max(0.0, (pd.Timestamp.now(tz="UTC") - t).total_seconds())
    except Exception:                        # noqa: BLE001 — es informativo, no crítico
        return None

def diagnostico(gold: pd.DataFrame | None) -> dict:
    """Hasta dónde llega Gold y si conviene actualizar.

    Se mira el artefacto, no el calendario: el servicio no lee Silver. La señal de que
    hay algo nuevo para incorporar es que Gold tenga partidos jugados **después** del
    corte de la fecha que hoy figura como próxima.
    """
    if gold is None or gold.empty:
        return {"hace_falta": True, "motivo": "no hay Gold cargado"}

    actual = gold[gold["season"] == CFG.current_season]
    if actual.empty:
        return {"hace_falta": True, "motivo": f"Gold no tiene {CFG.current_season}"}

    inf = actual[actual["split"] == "inferencia"]
    jugados = actual[actual["target_1x2"].notna()]
    ultimo = None if jugados.empty else pd.Timestamp(jugados["kickoff_time"].max())

    if inf.empty:
        return {"hace_falta": True,
                "motivo": "Gold no tiene ninguna fecha lista para predecir",
                "gold_built_at": str(actual["gold_built_at"].max()),
                "ultimo_partido_en_gold": None if ultimo is None else str(ultimo)}

    gw = int(inf["gameweek"].min())
    corte = pd.Timestamp(inf["corte"].iloc[0])
    ahora = pd.Timestamp.now(tz="UTC")

    # Dos señales distintas, y las dos importan:
    #
    #   (a) Gold tiene partidos jugados DESPUÉS del corte de la próxima. Es prueba dura
    #       de que la fila está vencida.
    #   (b) El corte ya pasó en tiempo real. Gold puede no saberlo todavía —es
    #       exactamente lo que pasa entre que se juega una fecha y alguien la ingesta— y
    #       por eso hay que mirar el reloj además del artefacto. Sin esto, el sistema
    #       contesta "está todo bien" mientras sirve la predicción de una fecha que ya
    #       terminó.
    vencida = ultimo is not None and ultimo >= corte
    paso_el_corte = corte <= ahora

    # Si Gold se reconstruyó hace poco y la próxima sigue siendo la misma fecha ya
    # arrancada, el que falta no es el pipeline: es la fuente. football-data publica el
    # CSV de la temporada con retraso, así que insistir con el botón no cambia nada.
    # Decir "conviene actualizar" ahí sería mandar a alguien a apretar en el vacío.
    reciente = _construido_hace(actual)
    if paso_el_corte and not vencida and reciente is not None and reciente < 3 * 3600:
        return {
            "hace_falta": False,
            "motivo": (f"la GW{gw} ya se jugó, pero sus resultados todavía no fueron "
                       f"publicados por la fuente. Gold se actualizó hace "
                       f"{reciente / 60:.0f} min y no había nada nuevo: hay que esperar "
                       f"a que football-data suba la fecha"),
            "proxima_predecible": gw,
            "corte_proxima": str(corte),
            "gold_built_at": str(actual["gold_built_at"].max()),
            "ultimo_partido_en_gold": None if ultimo is None else str(ultimo),
        }

    if vencida:
        motivo = (f"hay partidos jugados posteriores al corte de la GW{gw} ({corte}): "
                  f"la fila está vencida y corresponde recalcular")
    elif paso_el_corte:
        horas = (ahora - corte).total_seconds() / 3600
        motivo = (f"la GW{gw} arrancó hace {horas:.0f} h y sus resultados todavía no "
                  f"están en Gold: conviene actualizar para poder predecir la siguiente")
    else:
        faltan = (corte - ahora).total_seconds() / 3600
        motivo = (f"la GW{gw} está lista y su corte ({corte}) es en {faltan:.0f} h")

    return {
        "hace_falta": bool(vencida or paso_el_corte),
        "motivo": motivo,
        "proxima_predecible": gw,
        "corte_proxima": str(corte),
        "gold_built_at": str(actual["gold_built_at"].max()),
        "ultimo_partido_en_gold": None if ultimo is None else str(ultimo),
    }


# ---------------------------------------------------------------------------
# Disparo
# ---------------------------------------------------------------------------

def _limpiar() -> None:
    viejas = [k for k, t in _TAREAS.items()
              if t.terminada_at and (time.time() - t.terminada_at) > RETENCION_S]
    for k in viejas:
        _TAREAS.pop(k, None)


def _refrescar(t: Tarea) -> Tarea:
    """Le pregunta a Cloud Run como viene una corrida que se lanzo y no se espera.

    `_lanzar_job` dispara y vuelve: no puede quedarse esperando, porque el pipeline
    tarda minutos y un request de Cloud Run no dura eso. Pero entonces **nadie**
    actualizaba el estado, y una corrida que fallaba a los cuatro minutos dejaba la
    pagina girando para siempre. Peor: `en_curso()` seguia viendo esa tarea muerta,
    asi que el proximo intento contestaba 409 y el boton quedaba inutilizable hasta
    reiniciar el servicio.

    En local no hace falta —hay un hilo esperando al subproceso— y por eso el agujero
    no se veia hasta desplegar.
    """
    if t.destino != "cloud-run-job" or t.estado not in ("lanzada", "corriendo"):
        return t

    ahora = time.time()
    if ahora - t.consultada_at < REFRESCO_S:
        return t
    t.consultada_at = ahora

    if ahora - t.lanzada_at > MAX_EN_VUELO_S:
        t.estado, t.terminada_at = "error", ahora
        t.detalle = ("Sin noticias de Cloud Run pasados los 35 minutos. Mirá: "
                     "gcloud run jobs executions list --job premier-ml-pipeline")
        return t

    if not t.referencia:
        return t

    import requests

    try:
        # `:run` devuelve una operación de larga duración, no la ejecución: se
        # consulta esa, y cuando está `done` trae el resultado o el error.
        r = requests.get(
            f"https://run.googleapis.com/v2/{t.referencia}",
            headers={"Authorization": f"Bearer {_token_metadata()}"},
            timeout=TIMEOUT_ADMIN_S)
        if r.status_code >= 300:
            log.warning("Cloud Run devolvió %s al consultar %s", r.status_code, t.referencia)
            return t
        op = r.json()
    except Exception as exc:  # noqa: BLE001 — una consulta que falla no mata la tarea
        log.warning("No se pudo consultar la corrida %s: %s", t.referencia, exc)
        return t

    if not op.get("done"):
        return t

    t.terminada_at = ahora
    if "error" in op:
        t.estado = "error"
        t.detalle = str(op["error"].get("message", op["error"]))[:400]
    else:
        # La ejecución terminó, pero "terminó" no es "salió bien": una tarea que
        # devuelve exit(1) deja la operación `done` y sin `error`.
        ejec = op.get("response", {})
        fallidas = int(ejec.get("failedCount", 0) or 0)
        ok = int(ejec.get("succeededCount", 0) or 0)
        t.estado = "ok" if ok and not fallidas else "error"
        t.detalle = (f"{ok} tarea(s) ok, {fallidas} fallida(s). "
                     f"Logs: gcloud run jobs executions list "
                     f"--job {os.getenv('TP_JOB_NAME', 'premier-ml-pipeline')}")

    evento(log, "pipeline_fin", f"tarea {t.id}: {t.estado}",
           tarea=t.id, estado=t.estado, destino=t.destino,
           segundos=round(t.terminada_at - t.lanzada_at, 1))
    return t


def en_curso() -> Tarea | None:
    viva = next((t for t in _TAREAS.values()
                 if t.estado in ("lanzada", "corriendo")), None)
    return _refrescar(viva) if viva is not None else None


def _token_metadata() -> str:
    import requests

    r = requests.get(
        "http://metadata.google.internal/computeMetadata/v1/instance/"
        "service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"}, timeout=TIMEOUT_METADATA_S)
    r.raise_for_status()
    return r.json()["access_token"]


def _lanzar_job(t: Tarea) -> None:
    """Le pide a la Admin API de Cloud Run que ejecute el Job. No espera a que termine."""
    import requests

    proyecto = CFG.gcp_project or os.getenv("GOOGLE_CLOUD_PROJECT")
    region = os.getenv("TP_REGION", "us-central1")
    job = os.getenv("TP_JOB_NAME", "premier-ml-pipeline")
    if not proyecto:
        raise RuntimeError("No sé contra qué proyecto disparar: falta TP_GCP_PROJECT.")

    url = (f"https://run.googleapis.com/v2/projects/{proyecto}/locations/{region}"
           f"/jobs/{job}:run")
    r = requests.post(url, headers={"Authorization": f"Bearer {_token_metadata()}"},
                      timeout=TIMEOUT_ADMIN_S)
    if r.status_code >= 300:
        raise RuntimeError(f"Cloud Run devolvió {r.status_code}: {r.text[:300]}")

    t.destino = "cloud-run-job"
    t.referencia = r.json().get("name", "")
    t.estado = "corriendo"
    t.detalle = (f"Job {job} lanzado. Seguimiento: "
                 f"gcloud run jobs executions list --job {job} --region {region}")


def _lanzar_subproceso(t: Tarea) -> None:
    """En local: el mismo comando que corre el Job, como subproceso."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "pipeline.pre_deadline"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd=str(CFG.data_root.parent))
    t.destino = "local"
    t.referencia = str(proc.pid)
    t.estado = "corriendo"
    t.detalle = "pipeline.pre_deadline corriendo como subproceso"

    def esperar() -> None:
        salida, _ = proc.communicate()
        t.terminada_at = time.time()
        t.estado = "ok" if proc.returncode == 0 else "error"
        t.detalle = (salida or "").strip().splitlines()[-12:] and \
            "\n".join((salida or "").strip().splitlines()[-12:]) or t.detalle
        evento(log, "pipeline_fin", f"tarea {t.id}: {t.estado}",
               tarea=t.id, estado=t.estado, codigo=proc.returncode)

    threading.Thread(target=esperar, daemon=True).start()


def disparar(motivo: str = "pedido desde la API") -> Tarea:
    with _LOCK:
        _limpiar()
        corriendo = en_curso()
        if corriendo is not None:
            raise YaHayUna(corriendo)

        t = Tarea(id=utc_stamp() + "-" + secrets.token_hex(3), estado="lanzada",
                  motivo=motivo)
        _TAREAS[t.id] = t

    try:
        if en_la_nube():
            _lanzar_job(t)
        else:
            _lanzar_subproceso(t)
    except Exception as exc:                     # noqa: BLE001 — se reporta al cliente
        t.estado, t.terminada_at = "error", time.time()
        t.detalle = str(exc)
        log.error("No se pudo disparar el pipeline: %s", exc)

    evento(log, "pipeline_disparo", f"tarea {t.id}: {t.estado}",
           tarea=t.id, estado=t.estado, destino=t.destino, motivo=motivo)
    return t


def consultar(tid: str) -> Tarea | None:
    t = _TAREAS.get(tid)
    return _refrescar(t) if t is not None else None


def listar() -> list[Tarea]:
    for t in list(_TAREAS.values()):
        _refrescar(t)
    return sorted(_TAREAS.values(), key=lambda t: t.lanzada_at, reverse=True)
