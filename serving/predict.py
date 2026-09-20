"""Predecir una fecha: el camino de producción.

    python -m serving.predict --gw 2                    # la fecha que viene
    python -m serving.predict --gw 1 --evaluar          # una ya jugada, contra el resultado

Es el primer eslabón de la Fase 6. Todavía no hay HTTP ni contenedor, pero la lógica que
va adentro del endpoint es ésta, y ya cumple las tres cosas que el bloque 8 del canvas
pide de una predicción de producción:

1. **Las features se calculan con el mismo código que el entrenamiento.**
   `features.gold_tp.construir(objetivos=...)` es literalmente la misma función; lo único
   que cambia es de dónde salen los objetivos. Dos implementaciones paralelas de las
   features es como se produce el train/serve skew.

2. **El orden de las columnas se valida contra el metadata del modelo.** XGBoost recibe un
   ndarray: si las columnas vienen en otro orden no se queja, predice cualquier cosa. Es
   un fallo silencioso, así que se chequea explícitamente.

3. **Cada predicción se registra** con el `fixture_id`, el momento en que se predijo, la
   versión del modelo y del feature set, y las tres probabilidades. Sin ese registro no hay
   monitoreo después: sólo un endpoint que responde.

La conversión de las tres probabilidades a una clase es un paso **aparte del modelo** y
vive en `serving/decision.py`: además de `prediccion` (la regla de producción) se registra
una columna por cada regla **candidata**, que corre en paralelo sobre los mismos partidos
sin cambiar lo que el sistema anuncia.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import CFG, PROJECT_ROOT, utc_stamp
from common.logging_setup import get_logger, setup
from common.storage import read_table
from eda.baselines import CLASES_ORD
from features import spec
from serving import decision, registro
from serving.registro import PREDICCIONES  # noqa: F401  (se reexporta por compatibilidad)
from training import registry

log = get_logger(__name__)

GOLD = "gold_tp_match"

# Cuanto puede tener la historia usada respecto del corte antes de considerar la fila
# rancia. El umbral tiene que tolerar el paron FIFA: entre la GW5 y la GW6 de 2026-27 hay
# 22 dias de calendario, y eso es normal, no es un dato viejo.
MAX_DIAS_HISTORIA = 60


class FechaNoPreparada(Exception):
    """La fecha existe en el calendario pero todavia no tiene fila en Gold.

    Es el caso de pedir la 7 estando en la 5. Antes devolvia 200 con features rancias
    --historia hasta la 4, `dias_descanso` de 35-- y nada en la respuesta lo decia.
    """

    def __init__(self, season: str, gameweek: int, proxima: int | None):
        self.season, self.gameweek, self.proxima = season, gameweek, proxima
        if proxima is None:
            detalle = ("no hay ninguna fecha predecible: o termino la temporada, o falta "
                       "correr el pipeline (python -m features.gold_tp)")
        else:
            detalle = f"la proxima predecible es la GW{proxima}"
        super().__init__(f"{season} GW{gameweek} todavia no esta preparada; {detalle}.")


# ---------------------------------------------------------------------------
# Objetivos: los partidos a predecir
# ---------------------------------------------------------------------------

def cargar_gold() -> pd.DataFrame:
    return read_table(GOLD, layer="gold")


def proxima_en_gold(gold: pd.DataFrame, season: str | None = None) -> int | None:
    """La fecha que se puede predecir, leida de Gold. Sin tocar Silver.

    Es `min(gameweek)` entre las filas marcadas como inferencia. Quien decidio que esa
    fecha ya estaba lista es problema de `features/calendario.py`, que corre en el
    pipeline; aca solo se lee el resultado de esa decision.
    """
    season = season or CFG.current_season
    inf = gold[(gold["season"] == season) & (gold["split"] == "inferencia")]
    return None if inf.empty else int(inf["gameweek"].min())


def filas_gold(season: str, gameweek: int,
               gold: pd.DataFrame | None = None) -> pd.DataFrame:
    """Las filas de Gold de esa fecha, ordenadas por kickoff. Es un LOOKUP, no un calculo.

    Antes aca se llamaba a `gold_tp.construir(objetivos=...)` y se reconstruian las 279
    features desde Silver en cada request: 25 segundos, y la obligacion de meter las ocho
    tablas de Silver dentro de la imagen. Las features ya se calcularon --con el mismo
    codigo del entrenamiento, que es lo que evita el train/serve skew-- cuando corrio el
    pipeline. Recalcularlas era hacer dos veces el mismo trabajo y arriesgar que diera
    distinto.
    """
    gold = cargar_gold() if gold is None else gold
    d = gold[(gold["season"] == season) & (gold["gameweek"] == gameweek)]
    if d.empty:
        raise FechaNoPreparada(season, gameweek, proxima_en_gold(gold, season))
    return d.sort_values("kickoff_time").reset_index(drop=True)


# ---------------------------------------------------------------------------
# El modelo
# ---------------------------------------------------------------------------

def cargar_modelo(nombre: str | None = None, version: str | None = None):
    """Carga los boosters de la versión pedida (o la de producción) y su metadata.

    Devuelve `(boosters, metadata)`. Son varios porque el entrenamiento promedia semillas:
    la predicción tiene que promediar las mismas.
    """
    nombre = nombre or CFG.modelo
    if version:
        ruta = registry.RAIZ / nombre / version
    else:
        v = registry.produccion(nombre)
        if v is not None:
            # Una versión puede tener toda su trazabilidad y ningún binario: los `.ubj`
            # están en `.gitignore` y `metadata.json` no, así que un `git checkout` deja
            # la carpeta llena de papeles y vacía de modelo. Si la de producción está
            # así hay que decirlo con ese nombre, no caer a otra por las dudas: servir
            # en silencio un modelo que nadie eligió es peor que no servir.
            if not registry.tiene_boosters(v):
                raise FileNotFoundError(
                    f"La versión de producción {nombre}/{v.version} no tiene archivos "
                    f".ubj. Reentrená con `python -m training.run`, bajá los binarios "
                    f"del bucket, o promové otra: `python -m training.registry --listar`.")
            ruta = v.ruta
        else:
            # El fallback elige la última versión SERVIBLE, no la última a secas. En
            # septiembre de 2026 la última por nombre era una que había llegado por git
            # sin binarios: dejó todo /predict en 503 y nada lo explicaba.
            candidatas = registry.servibles(nombre)
            if not candidatas:
                raise FileNotFoundError(
                    f"No hay ningún modelo servible en models/{nombre}/. "
                    f"Corré: python -m training.run --model {nombre}")
            ruta = candidatas[-1].ruta
            log.warning("No hay PRODUCTION.json; se usa la última versión servible: %s. "
                        "Fijala con `python -m training.registry --promover %s`.",
                        ruta.name, ruta.name)

    meta = registry.cargar_metadata(registry.Version(nombre, ruta.name, ruta))
    archivos = registry.boosters_de(registry.Version(nombre, ruta.name, ruta))
    if not archivos:
        raise FileNotFoundError(f"No hay archivos .ubj en {ruta}")

    # Se sirve en CPU aunque se haya entrenado en GPU: el .ubj es portable y el bloque 7
    # del canvas dice explícitamente que la inferencia va sin GPU. La carga la hace
    # `registry.cargar_booster`, que es el unico lugar del repo que sabe hacerlo.
    boosters = [registry.cargar_booster(f, device="cpu") for f in archivos]

    log.info("Modelo %s versión %s — %d semillas, %d features",
             nombre, ruta.name, len(boosters), meta["n_features"])
    return boosters, meta


def _validar_features(meta: dict) -> list[str]:
    """El contrato con el entrenamiento. Un desajuste acá es un fallo silencioso."""
    esperadas = meta["feature_names"]
    if meta.get("feature_set_version") != spec.FEATURE_SET_VERSION:
        raise ValueError(
            f"El modelo se entrenó con el feature set {meta.get('feature_set_version')} "
            f"y el código está en {spec.FEATURE_SET_VERSION}. Reentrená antes de servir.")
    faltan = [c for c in esperadas if c not in spec.FEATURES]
    if faltan:
        raise ValueError(f"El spec ya no tiene {len(faltan)} features del modelo: {faltan[:5]}")
    return esperadas


def predecir_proba(boosters, X: np.ndarray) -> np.ndarray:
    """Probabilidades promediadas entre semillas, en el orden de CLASES_ORD."""
    import xgboost as xgb

    dm = xgb.DMatrix(X)
    P = np.mean([b.predict(dm) for b in boosters], axis=0)
    return P / P.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------------

def predecir(season: str, gameweek: int, nombre: str | None = None,
             version: str | None = None,
             gold: pd.DataFrame | None = None,
             modelo: tuple | None = None) -> pd.DataFrame:
    """Las tres probabilidades de cada partido de la fecha, mas la decision de apuesta.

    La fila se resuelve ANTES de cargar el modelo. Al reves, un modelo roto devolvia 503
    hasta para una fecha que no existia, y ese error tapaba al que importaba.
    """
    feats = filas_gold(season, gameweek, gold)
    log.info("Prediciendo %s GW%d - %d partidos, corte %s (Gold del %s)",
             season, gameweek, len(feats), feats["corte"].iloc[0],
             feats["gold_built_at"].iloc[0] if "gold_built_at" in feats else "?")

    # `modelo` permite inyectar los boosters ya cargados. Sin eso, el servicio releía
    # cinco `.ubj` del disco en cada request y se comía casi un segundo por pedido, que
    # era justo lo que el lookup venía a eliminar.
    boosters, meta = cargar_modelo(nombre, version) if modelo is None else modelo
    features = _validar_features(meta)

    X = feats[features].to_numpy(dtype=np.float32)
    P = predecir_proba(boosters, X)

    out = feats[["season", "gameweek", "fixture_id", "kickoff_time",
                 "home_short", "away_short"]].copy()
    for i, c in enumerate(CLASES_ORD):
        out[f"p_{c}"] = P[:, i]
    # `prediccion` sale de la regla de PRODUCCION, y cada candidata deja su propia columna.
    # Todas leen las mismas probabilidades, asi que el registro queda listo para la
    # comparacion pareada del monitoreo sin volver a predecir nada.
    out = decision.etiquetar(out)
    elegida = [CLASES_ORD.index(c) for c in out["prediccion"]]
    out["confianza"] = P[np.arange(len(P)), elegida]

    # Trazabilidad: sin esto no hay monitoreo, sólo un endpoint que responde.
    out["predicted_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out["model_name"] = meta["model_name"]
    out["model_version"] = meta["model_version"]
    out["feature_set_version"] = meta["feature_set_version"]

    # Auditoría anti-leakage: la historia usada es anterior al corte, también acá.
    for lado in spec.LADOS:
        out[f"hist_kickoff_{lado}"] = feats[f"hist_kickoff_{lado}"]
    _assert_sin_leakage(out, feats)
    _assert_historia_fresca(feats)
    return out


def _assert_sin_leakage(out: pd.DataFrame, feats: pd.DataFrame) -> None:
    corte = feats["corte"]
    for lado in spec.LADOS:
        hk = feats[f"hist_kickoff_{lado}"]
        malas = hk.notna() & (hk >= corte)
        if malas.any():
            raise AssertionError(
                f"{int(malas.sum())} predicciones usan historia posterior al corte ({lado}).")


def _assert_historia_fresca(feats: pd.DataFrame,
                            max_dias: int = MAX_DIAS_HISTORIA) -> None:
    """El control simetrico del anti-leakage: historia demasiado VIEJA.

    `_assert_sin_leakage` mira una sola direccion --que no entre informacion del futuro--
    y esa asimetria era un agujero real: predecir una fecha lejana usaba la historia que
    hubiera, sin error y sin aviso. Ahora que el servicio sirve un artefacto pre-calculado
    en vez de construirlo, este es el unico control que queda entre una fila rancia y el
    usuario.
    """
    corte = feats["corte"]
    hk = feats[[f"hist_kickoff_{lado}" for lado in spec.LADOS]].max(axis=1)
    edad = (corte - hk).dt.days
    viejas = edad.notna() & (edad > max_dias)
    if viejas.any():
        raise AssertionError(
            f"{int(viejas.sum())} filas usan historia de hace mas de {max_dias} dias "
            f"(maximo: {int(edad.max())}). Gold se construyo con un Silver "
            f"desactualizado: corre el pipeline antes de predecir.")


def guardar(pred: pd.DataFrame, si_existe: str = "saltar") -> Path | None:
    """Delega en `serving/registro.py`, que es donde vive el registro y su deduplicacion."""
    return registro.guardar(pred, si_existe=si_existe)


# ---------------------------------------------------------------------------
# Evaluación contra el resultado real
# ---------------------------------------------------------------------------

def evaluar(pred: pd.DataFrame) -> dict:
    """Compara contra el resultado real, si ya se jugó. Es el cierre del ciclo."""
    from training import metrics

    m = read_table("fact_match")
    real = m[["season", "home_short", "away_short", "target_1x2",
              "home_goals", "away_goals"]]
    d = pred.merge(real, on=["season", "home_short", "away_short"], how="left")
    jugados = d[d["target_1x2"].notna()]
    if jugados.empty:
        return {"nota": "todavía no hay resultados para esta fecha"}

    P = jugados[[f"p_{c}" for c in CLASES_ORD]].to_numpy()
    rep = metrics.reporte(jugados["target_1x2"].to_numpy(),
                          jugados["prediccion"].to_numpy(), P, con_ic=False)
    rep["acierta_siempre_local"] = float((jugados["target_1x2"] == "home").mean())

    # Cada regla de decision sobre las MISMAS filas. Con una fecha sola no alcanza para
    # concluir nada (n=10 -> error estandar +-15,7 puntos); el veredicto lo da el
    # acumulado de `monitoring.temporada_actual`. Aca es para verlo pasar.
    season, gw = jugados["season"].iloc[0], int(jugados["gameweek"].iloc[0])
    rep["por_regla"] = {
        r.nombre: {
            "columna": r.columna,
            "accuracy": float((jugados[r.columna] == jugados["target_1x2"]).mean()),
            "empates_predichos": int((jugados[r.columna] == "draw").sum()),
            "prospectiva": r.cuenta_para(season, gw),
        }
        for r in decision.todas() if r.columna in jugados.columns
    }
    rep["detalle"] = d
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description="Predice una fecha.")
    ap.add_argument("--season", default=CFG.current_season)
    ap.add_argument("--gw", type=int, required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--version", default=None)
    ap.add_argument("--evaluar", action="store_true",
                    help="compara contra el resultado real (si la fecha ya se jugó)")
    ap.add_argument("--no-guardar", action="store_true")
    args = ap.parse_args()

    setup(CFG.log_level, CFG.log_format)
    pred = predecir(args.season, args.gw, args.model, args.version)

    print(f"\n{'=' * 78}")
    print(f"{args.season} — FECHA {args.gw}    modelo {pred['model_name'].iloc[0]} "
          f"({pred['model_version'].iloc[0]})")
    print(f"{'=' * 78}\n")
    cands = decision.candidatos()
    cabecera = "".join(f"{c.nombre[:20]:>22s}" for c in cands)
    print(f"{'kickoff':<17}{'partido':<16}{'local':>7}{'empate':>8}{'visita':>8}"
          f"   {'predice':<8}{cabecera}")
    for r in pred.itertuples():
        partido = f"{r.home_short}-{r.away_short}"
        # Se marca con `*` donde el candidato discrepa: es lo unico que despues aporta
        # informacion al McNemar, asi que conviene verlo de un vistazo.
        extra = "".join(
            f"{getattr(r, c.columna) + ('  *' if getattr(r, c.columna) != r.prediccion else ''):>22s}"
            for c in cands)
        print(f"{str(r.kickoff_time)[:16]:<17}{partido:<16}"
              f"{r.p_home:>7.3f}{r.p_draw:>8.3f}{r.p_away:>8.3f}   {r.prediccion:<8}{extra}")

    if not args.no_guardar:
        guardar(pred)

    if args.evaluar:
        ev = evaluar(pred)
        if "nota" in ev:
            print(f"\n{ev['nota']}")
            return
        print(f"\n{'-' * 78}\nCONTRA EL RESULTADO REAL\n{'-' * 78}\n")
        d = ev["detalle"]
        print(f"{'partido':<16}{'predijo':<10}{'p':>7}   {'real':<8}{'resultado':<10} ok")
        for r in d.itertuples():
            if pd.isna(r.target_1x2):
                continue
            marca = "OK" if r.prediccion == r.target_1x2 else "--"
            print(f"{r.home_short + '-' + r.away_short:<16}{r.prediccion:<10}"
                  f"{r.confianza:>7.3f}   {r.target_1x2:<8}"
                  f"{int(r.home_goals)}-{int(r.away_goals):<8} {marca}")
        print(f"\n  accuracy   {ev['accuracy']:.3f}  ({int(ev['accuracy'] * ev['n'])} de {ev['n']})")
        print(f"  log-loss   {ev['log_loss']:.4f}")
        print(f"  'siempre local' habria acertado: {ev['acierta_siempre_local']:.3f}")

        if len(ev["por_regla"]) > 1:
            print(f"\n  por regla de decision (mismas {ev['n']} filas, mismo modelo):")
            for nombre, r in ev["por_regla"].items():
                marca = "" if r["prospectiva"] else "   (retrospectivo: no cuenta)"
                print(f"    {nombre:<26s} accuracy {r['accuracy']:.3f}   "
                      f"empates predichos {r['empates_predichos']:>2d}{marca}")


if __name__ == "__main__":
    main()
