"""API HTTP sobre `serving/predict.py`.

    uvicorn serving.main:app --reload --port 8080

Capa fina a propósito: la lógica (features, validación de columnas, control
anti-leakage, decisión) vive en `serving/predict.py`, `serving/registro.py` y
`serving/decision.py`, con sus propios tests. Acá sólo se traduce eso a HTTP.

**El servicio no lee Silver.** Lee la tabla Gold y el registro de predicciones, nada más.
Las features ya se construyeron en el pipeline con el mismo código del entrenamiento; el
servicio hace un lookup. Antes reconstruía las 279 features en cada request —25 segundos—
y por eso la imagen tenía que llevar las ocho tablas de Silver adentro.

**Una fecha se responde según su estado**, y la respuesta lo dice:

    jugada / en_curso  -> la predicción CONGELADA del registro + el resultado real
    proxima            -> se predice en vivo
    todavía no lista   -> 409, con cuál es la próxima predecible

Un GET sigue sin escribir: `predict.guardar()` no se llama desde acá. Registrar es
responsabilidad del job pre-deadline, no de quien consulta.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from common.config import CFG
from common.logging_setup import evento, get_logger
from serving import predict, registro, tareas
from serving.predict import FechaNoPreparada
from serving.schemas import (CalendarioResponse, Diagnostico, FechaCalendario,
                             FechaNoPreparadaResponse, GameweekPredictionResponse,
                             HealthResponse, MatchPrediction, Tarea)

log = get_logger(__name__)

# Gold lo reescribe el job pre-deadline por afuera del proceso, así que la copia en
# memoria tiene vencimiento. Cinco minutos es de sobra: entre dos fechas pasan días.
GOLD_TTL_S = 300


@dataclass
class Estado:
    """Lo que el servicio mantiene cargado entre requests.

    Existe por la latencia: cargar cinco boosters y la tabla Gold en cada request es lo
    que hacía que `/predict` tardara. Y si el modelo no cargó, **no se cachea el error**:
    se reintenta en el request siguiente. Una instancia que arrancó antes de que el
    bucket tuviera el objeto tiene que poder recuperarse sola, sin un redeploy.
    """

    boosters: list = field(default_factory=list)
    meta: dict | None = None
    error_modelo: str | None = None
    gold: pd.DataFrame | None = None
    gold_at: float = 0.0
    error_gold: str | None = None

    def modelo(self):
        if self.meta is None:
            try:
                self.boosters, self.meta = predict.cargar_modelo()
                self.error_modelo = None
            except Exception as exc:  # noqa: BLE001 — se reporta, no se traga
                self.error_modelo = str(exc)
                log.error("No se pudo cargar el modelo: %s", exc)
        return self.boosters, self.meta

    def tabla(self, forzar: bool = False) -> pd.DataFrame | None:
        vencida = (time.monotonic() - self.gold_at) > GOLD_TTL_S
        if self.gold is None or vencida or forzar:
            try:
                self.gold = predict.cargar_gold()
                self.gold_at = time.monotonic()
                self.error_gold = None
            except Exception as exc:  # noqa: BLE001
                self.error_gold = str(exc)
                log.error("No se pudo cargar Gold: %s", exc)
        return self.gold


ESTADO = Estado()


@asynccontextmanager
async def lifespan(app: FastAPI):
    ESTADO.modelo()
    ESTADO.tabla()
    yield


app = FastAPI(
    title="TP Premier ML — API",
    version="0.2.0",
    description="Predicción 1X2 de la Premier League, servida desde la tabla Gold.",
    lifespan=lifespan,
)


@app.exception_handler(FechaNoPreparada)
async def _fecha_no_preparada(request: Request, exc: FechaNoPreparada) -> JSONResponse:
    """409 y no 404: la fecha existe, lo que no existe todavía es su fila de features."""
    cuerpo = FechaNoPreparadaResponse(detail=str(exc), season=exc.season,
                                      gameweek=exc.gameweek, proxima_predecible=exc.proxima)
    return JSONResponse(status_code=409, content=cuerpo.model_dump())


# ---------------------------------------------------------------------------
# Salud
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Nunca tira: informa. Un /health que falla no sirve para monitorear nada."""
    _, meta = ESTADO.modelo()
    gold = ESTADO.tabla()

    detalle = "; ".join(x for x in (ESTADO.error_modelo, ESTADO.error_gold) if x) or None
    ok = meta is not None and gold is not None
    return HealthResponse(
        status="ok" if ok else "degraded",
        model_name=None if meta is None else meta["model_name"],
        model_version=None if meta is None else meta["model_version"],
        feature_set_version=None if meta is None else meta["feature_set_version"],
        n_seeds=None if meta is None else meta.get("n_seeds"),
        backend=CFG.backend,
        gold_built_at=None if gold is None else str(gold["gold_built_at"].max()),
        gold_filas=None if gold is None else int(len(gold)),
        season_actual=CFG.current_season,
        proxima_predecible=None if gold is None else predict.proxima_en_gold(gold),
        detail=detalle,
    )


# ---------------------------------------------------------------------------
# Calendario
# ---------------------------------------------------------------------------

def _gold_o_503() -> pd.DataFrame:
    gold = ESTADO.tabla()
    if gold is None:
        raise HTTPException(status_code=503, detail=ESTADO.error_gold or "Gold no disponible.")
    return gold


def _estado_de(d: pd.DataFrame) -> str:
    inf = d["split"] == "inferencia"
    if inf.all():
        return "proxima"
    if not inf.any():
        return "jugada"
    return "en_curso"


@app.get("/calendario/{season}", response_model=CalendarioResponse)
def calendario(season: str) -> CalendarioResponse:
    """El índice de la temporada: qué fechas hay, en qué estado y con qué resultado."""
    gold = _gold_o_503()
    de_la_temporada = gold[gold["season"] == season]
    if de_la_temporada.empty:
        raise HTTPException(status_code=404, detail=f"No hay datos de la temporada {season}.")

    registrados = set(registro.listar(season)["gameweek"])
    en_gold = dict(tuple(de_la_temporada.groupby("gameweek", sort=True)))
    hasta = max(en_gold) if en_gold else 0

    # Se listan TODAS las fechas de la temporada, no sólo las que tienen features. Las que
    # no están en Gold salen como `no_preparada`: el límite tiene que verse.
    fechas = []
    for gw in range(1, CFG.gameweeks + 1):
        d = en_gold.get(gw)
        if d is None:
            fechas.append(FechaCalendario(
                gameweek=gw, corte="", n_partidos=0, estado="no_preparada",
                tiene_registro=gw in registrados))
            continue
        fechas.append(FechaCalendario(
            gameweek=gw,
            corte=str(d["corte"].iloc[0]),
            n_partidos=int(len(d)),
            estado=_estado_de(d),
            tiene_registro=gw in registrados,
            n_con_resultado=int(d["target_1x2"].notna().sum()),
        ))

    return CalendarioResponse(
        season=season,
        proxima_predecible=predict.proxima_en_gold(gold, season),
        gold_built_at=str(de_la_temporada["gold_built_at"].max()),
        hasta_gameweek=int(hasta) or None,
        fechas=fechas,
    )


# ---------------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------------

def _filas(pred: pd.DataFrame) -> list[MatchPrediction]:
    def _int(v):
        return None if pd.isna(v) else int(v)

    def _bool(v):
        return None if v is None or pd.isna(v) else bool(v)

    def _str(v):
        return None if pd.isna(v) else str(v)

    def _float(v):
        return None if v is None or pd.isna(v) else float(v)

    return [
        MatchPrediction(
            fixture_id=int(r.fixture_id),
            home_short=r.home_short, away_short=r.away_short,
            kickoff_time=pd.Timestamp(r.kickoff_time).isoformat(),
            p_home=float(r.p_home), p_draw=float(r.p_draw), p_away=float(r.p_away),
            prediccion=r.prediccion, confianza=float(r.confianza),
            target_1x2=_str(getattr(r, "target_1x2", None)),
            home_goals=_int(getattr(r, "home_goals", None)),
            away_goals=_int(getattr(r, "away_goals", None)),
            acierto=_bool(getattr(r, "acierto", None)),
            cuota_home=_float(getattr(r, "cuota_home", None)),
            cuota_draw=_float(getattr(r, "cuota_draw", None)),
            cuota_away=_float(getattr(r, "cuota_away", None)),
        )
        for r in pred.itertuples()
    ]


@app.get("/predict/{season}/{gameweek}", response_model=GameweekPredictionResponse)
def predict_gameweek(season: str, gameweek: int) -> GameweekPredictionResponse:
    inicio = time.perf_counter()
    gold = _gold_o_503()

    # `filas_gold` levanta FechaNoPreparada -> 409 con la próxima predecible.
    d = predict.filas_gold(season, gameweek, gold=gold)
    estado = _estado_de(d)

    if estado == "proxima":
        boosters, meta = ESTADO.modelo()
        if meta is None:
            raise HTTPException(status_code=503, detail=ESTADO.error_modelo)
        try:
            pred = predict.predecir(season, gameweek, gold=gold,
                                    modelo=(boosters, meta))
        except AssertionError as exc:          # los dos controles de historia
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except ValueError as exc:              # contrato de features roto
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        origen, pre_deadline = "en_vivo", None
    else:
        pred = registro.congelada(season, gameweek)
        if pred is None:
            raise HTTPException(
                status_code=409,
                detail=(f"{season} GW{gameweek} ya se jugó y no quedó ninguna predicción "
                        f"registrada: no hay nada honesto que devolver."))
        pred = registro.con_resultado(pred, d)
        origen = "registro"
        pre_deadline = bool(pred["registro_pre_deadline"].iloc[0])

    filas = _filas(pred)
    con_resultado = pred["target_1x2"].notna() if "target_1x2" in pred else pd.Series(dtype=bool)
    n_res = int(con_resultado.sum())
    acc = float(pred.loc[con_resultado, "acierto"].mean()) if n_res else None

    # Un evento por request, consultable por campo en Cloud Logging. Se guarda la
    # decisión y la métrica: qué fecha, en qué estado, con qué modelo y cuánto tardó.
    # Nada sobre quién consultó — acá es trivial porque son equipos de fútbol, pero la
    # regla vale igual y conviene que esté escrita donde se decide.
    latencia = round((time.perf_counter() - inicio) * 1000, 2)
    evento(log, "prediccion",
           f"{season} GW{gameweek} [{estado}/{origen}] {len(filas)} partidos, {latencia} ms",
           season=season, gameweek=gameweek, estado=estado, origen=origen,
           n_partidos=len(filas), latencia_ms=latencia,
           model_version=pred["model_version"].iloc[0],
           feature_set_version=pred["feature_set_version"].iloc[0],
           accuracy=acc, n_con_resultado=n_res)

    return GameweekPredictionResponse(
        season=season, gameweek=gameweek, count=len(filas),
        estado=estado, origen=origen, corte=str(d["corte"].iloc[0]),
        model_name=pred["model_name"].iloc[0],
        model_version=pred["model_version"].iloc[0],
        feature_set_version=pred["feature_set_version"].iloc[0],
        predicted_at=str(pred["predicted_at"].iloc[0]),
        gold_built_at=str(d["gold_built_at"].iloc[0]) if "gold_built_at" in d else None,
        pre_deadline=pre_deadline,
        n_con_resultado=n_res, accuracy=acc,
        predictions=filas,
    )


# ---------------------------------------------------------------------------
# Actualizar el dato
#
# El servicio no corre el pipeline: lo pide. Ver `serving/tareas.py` — ahí está por qué,
# y cómo se evita que un POST anónimo queme cómputo.
# ---------------------------------------------------------------------------

def _tarea(t) -> Tarea:
    return Tarea(**t.como_dict())


@app.get("/actualizar", response_model=Diagnostico)
def diagnostico() -> Diagnostico:
    """¿Hace falta actualizar? Se contesta mirando Gold, sin tocar Silver."""
    d = tareas.diagnostico(ESTADO.tabla())
    corriendo = tareas.en_curso()
    return Diagnostico(**d, en_curso=None if corriendo is None else _tarea(corriendo),
                       puede_disparar=corriendo is None)


@app.post("/actualizar", response_model=Tarea, status_code=202)
def actualizar(request: Request) -> Tarea:
    """Dispara el pipeline. Responde 202 y no espera: la corrida tarda minutos."""
    try:
        tareas.autorizar(request.headers.get("X-Admin-Token"))
    except tareas.NoAutorizado as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    d = tareas.diagnostico(ESTADO.tabla())
    try:
        t = tareas.disparar(motivo=d["motivo"])
    except tareas.YaHayUna as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    if t.estado == "error":
        raise HTTPException(status_code=502, detail=t.detalle)
    return _tarea(t)


@app.get("/actualizar/{tarea_id}", response_model=Tarea)
def estado_tarea(tarea_id: str) -> Tarea:
    t = tareas.consultar(tarea_id)
    if t is None:
        raise HTTPException(status_code=404, detail=f"No existe la tarea {tarea_id}.")
    if t.estado == "ok":
        # Terminó y reescribió Gold por afuera del proceso: se relee ya mismo en vez de
        # esperar a que venza el TTL.
        ESTADO.tabla(forzar=True)
    return _tarea(t)


# ---------------------------------------------------------------------------
# La página
#
# Se monta al final y en la raíz, DESPUÉS de las rutas: FastAPI resuelve por orden de
# declaración, así que `/health`, `/calendario/...` y `/predict/...` siguen ganando y
# ninguna URL existente cambia.
#
# Va en el mismo servicio y no en un bucket ni en Firebase a propósito: una sola URL, un
# solo despliegue, un solo rollback, y cero CORS. La página le pega a `/predict/...`
# relativo, que es el mismo origen. Separarla sería un recurso más para configurar,
# monitorear y apagar, a cambio de nada.
# ---------------------------------------------------------------------------

WEB = Path(__file__).resolve().parent.parent / "web"

if WEB.is_dir():
    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
else:                                           # pragma: no cover
    log.warning("No existe %s: la API funciona igual, sin la página.", WEB)
