"""Cuánto cambia la performance al cambiar la REGLA DE DECISIÓN, y no el modelo.

    python -m training.decision_eval                  # holdout: cada modelo x cada regla
    python -m training.decision_eval --walk-forward   # + el ciclo operativo, 38 fechas
    python -m training.decision_eval --barrido        # + el umbral completo, por modelo
    python -m training.decision_eval --semillas       # + el control de ruido de semilla

`serving/decision.py` define las reglas; `training/empate.py` explica por qué el empate es
el único lugar donde una regla puede mover algo. Este módulo contesta la pregunta que
faltaba: **si aplicamos la regla a las predicciones ya hechas, qué performance da — y no
sólo para el xgb, sino para todos los modelos.**

---

## Lo primero, porque cambia cómo se lee todo lo demás

**El log-loss no se mueve.** La regla no toca las probabilidades, sólo la etiqueta que se
anuncia. Cualquier tabla de acá que muestre el log-loss lo muestra idéntico entre reglas, y
eso no es un bug: es la definición. Lo único que una regla puede mover es lo que depende de
la clase anunciada — accuracy, precision, recall, F1.

Esto tiene una consecuencia incómoda para el bloque 5 del canvas, que pide *accuracy, F1,
recall y precision*: **esas cuatro métricas se pueden mover sin que el modelo aprenda
nada**. El umbral es la prueba viva. Por eso el reporte pone el AUC al lado.

## Lo segundo: por qué la comparación es McNemar y no dos accuracies

Las dos reglas deciden sobre las **mismas probabilidades** de los **mismos partidos**. No
hay dos muestras, hay una sola con dos etiquetados. Comparar `0,5079` contra `0,5000` como
si fueran dos experimentos independientes ignora que difieren en un puñado de filas y que
todas las demás son idénticas. McNemar mira sólo los pares discordantes, que es donde está
toda la información disponible.

Con el umbral 0,30 sobre el holdout de 380 partidos, los discordantes son ~20: los partidos
que pasan de `home`/`away` a `draw`. La pregunta entera del A/B es cuántos de esos 20 eran
realmente empates contra cuántos ya estaban bien.

---

## El resultado, medido (05/09/2026)

**El umbral 0,30 no mejora la accuracy. El efecto es más chico que el ruido de semilla.**

Holdout fijo, `xgb_gbt`, cambiando **sólo** la cantidad de semillas que se promedian:

| semillas | argmax | umbral 0,30 | delta | discordantes | McNemar p |
|---|---|---|---|---|---|
| 1 | 0,4947 | 0,4947 | 0,0000 | 24 | 12 / 12 · 1,000 |
| 3 | 0,5000 | 0,5000 | 0,0000 | 22 | 11 / 11 · 1,000 |
| 5 (producción) | 0,4947 | 0,5026 | **+0,0079** | 21 | 12 / 9 · 0,664 |
| 7 | 0,4974 | 0,4947 | −0,0026 | 21 | 10 / 11 · 1,000 |

El **+0,0079** que quedó documentado en `training/README.md` es la fila de 5 semillas: real,
reproducible, y **una moneda**. Los pares discordantes salen 12/9, 11/11, 12/12 — nunca se
despegan del 50 %.

Y el walk-forward, que es el protocolo más parecido a producción (reentrena en cada fecha),
cambiando **sólo** la semilla de entrenamiento:

| seed | argmax | umbral 0,30 | delta | discordantes | p |
|---|---|---|---|---|---|
| 42 | 0,5026 | 0,5289 | **+0,0263** | 26 | 18 / 8 · 0,076 |
| 7 | 0,4974 | 0,5079 | +0,0105 | 18 | 11 / 7 · 0,481 |
| 123 | 0,4947 | 0,4895 | −0,0053 | 14 | 6 / 8 · 0,791 |
| 2024 | 0,4974 | 0,4921 | −0,0053 | 22 | 10 / 12 · 0,832 |
| 999 | 0,4947 | 0,5132 | +0,0184 | 29 | 18 / 11 · 0,265 |

Media **+0,0089**, desvío **0,0141**. La corrida con `seed=42` —la que el proyecto usa por
defecto y la que habría quedado en el informe— es **la mejor de las cinco**. Ese `p=0,076`
es lo más cerca que estuvo el candidato de parecer real, y desaparece al cambiar un número
que no tiene nada que ver con la regla.

## Y sobre los demás modelos: la regla les hace mal

Aplicando el mismo umbral 0,30 a los nueve modelos del holdout, sólo `xgb_rf` sube (+1,05
puntos, 6 discordantes, ns). `logreg` cae **6,3 puntos con p=0,0002** — el único resultado
significativo de toda la tabla, y es en contra. `ordinal` y `mlp` caen 3-4 puntos. Al
mercado le hace perder medio punto.

La razón es la misma en todos: el umbral fijo interactúa con **cuánta masa de probabilidad
pone cada modelo en el empate**. `logreg` y `ordinal` ya ponen mucha (59 y 68 empates por
argmax), así que 0,30 los desborda a 113 y 149. Un umbral no es transferible entre modelos:
es un parámetro de la calibración de *ese* modelo.

## Lo único que sí se mueve, y por qué no alcanza

`f1_macro` sube de 0,39 a 0,44 y `f1_draw` de 0,06 a 0,20, en toda corrida y todo protocolo.
Pero eso es **exactamente la trampa que `training/empate.py` documenta para `ordinal`**: el
F1 del empate sube porque se anuncian más empates, no porque se acierten más. Con AUC_draw
0,51 no hay ranking que explotar. Si el TP reporta el F1 macro como métrica del bloque 5,
este umbral lo "mejora" sin que el modelo haya aprendido nada — y esa es la razón de que la
tabla de acá muestre el AUC al lado.
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from common.config import CFG, PROJECT_ROOT
from common.logging_setup import get_logger, setup
from eda.baselines import CLASES_ORD, odds_a_probabilidades, COLUMNAS_CUOTAS
from features import spec
from serving import decision
from training import compare_models as cm
from training import dataset, evaluate, metrics
from training.device import resolve

log = get_logger(__name__)

SALIDA = PROJECT_ROOT / "training" / "output"

# El barrido que ya usa `training/empate.py`, para que las dos tablas se puedan cruzar.
UMBRALES = (0.50, 0.40, 0.35, 0.32, 0.30, 0.28, 0.26, 0.24, 0.22, 0.20)

I_DRAW = list(CLASES_ORD).index("draw")


# --------------------------------------------------------------------------- #
# Probabilidades: se calculan UNA vez por modelo y después se re-etiquetan
# --------------------------------------------------------------------------- #

def probabilidades_holdout(modelos=cm.MODELOS, n_seeds: int = 3,
                           datos: str | None = None) -> tuple[dict, np.ndarray, pd.DataFrame]:
    """Entrena cada modelo con la variante de datos de producción y predice el holdout.

    Devuelve `(probas_por_modelo, y_true, filas_holdout)`. Entrenar es lo caro y la regla
    de decisión no lo toca: se paga una vez y después se etiqueta tantas veces como reglas
    haya. Es la misma idea que hace barato el A/B en producción.
    """
    datos = datos or CFG.datos_entrenamiento
    filtro = cm.VARIANTES_DATOS[datos]
    info = resolve("auto")
    gold = dataset.cargar()
    features = spec.FEATURES

    val_season, test_season = CFG.valid_season, CFG.holdout_season
    train_seasons = [s for s in CFG.seasons_for_training() if s != val_season]
    tr = gold[gold["season"].isin(train_seasons)]
    tr = tr if filtro is None else tr[filtro(tr)]
    va = gold[gold["season"] == val_season]
    te = gold[gold["season"] == test_season]
    Xte = dataset.matriz(te, features)

    probas = {}
    for nombre in modelos:
        log.info("%-12s | train %d + val %d -> holdout %d", nombre, len(tr), len(va), len(te))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                probas[nombre] = cm._fit_predict(nombre, info, tr, va, Xte,
                                                 features, n_seeds)
        except Exception as exc:  # noqa: BLE001
            log.error("  %s falló: %s", nombre, str(exc)[:120])

    # El mercado entra como vara: tiene probabilidades propias, asi que la regla se le
    # puede aplicar igual. Es la comparacion mas interesante del bloque.
    for cols in COLUMNAS_CUOTAS:
        if all(c in te.columns for c in cols):
            p = odds_a_probabilidades(te, cols)
            if p.notna().all(axis=None):
                probas["mercado"] = p[CLASES_ORD].to_numpy()
            break

    return probas, te["target_1x2"].to_numpy(), te


# --------------------------------------------------------------------------- #
# Evaluar una matriz de probabilidades bajo varias reglas
# --------------------------------------------------------------------------- #

def evaluar_reglas(P: np.ndarray, y: np.ndarray, reglas=None) -> pd.DataFrame:
    """Todas las métricas del bloque 5, una fila por regla, sobre las MISMAS filas."""
    reglas = reglas or decision.todas()
    filas = []
    for r in reglas:
        pred = r.aplicar(P)
        rep = metrics.reporte(y, pred, P, con_ic=True)
        filas.append({
            "regla": r.nombre,
            "accuracy": rep["accuracy"],
            "ic_bajo": rep["accuracy_ic95"][0], "ic_alto": rep["accuracy_ic95"][1],
            "f1_macro": rep["f1_macro"],
            "f1_draw": rep["f1_draw"],
            "precision_draw": rep["precision_draw"],
            "recall_draw": rep["recall_draw"],
            "empates_pred": int((pred == "draw").sum()),
            # El log-loss va en la tabla A PROPOSITO, para que se vea que no se mueve:
            # la regla no toca las probabilidades.
            "log_loss": rep["log_loss"],
            "rps": rep["rps"],
        })
    return pd.DataFrame(filas)


def comparar_contra_produccion(P: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """McNemar de cada candidata contra la regla de producción, sobre las mismas filas."""
    prod = decision.produccion()
    base = prod.aplicar(P) == y
    filas = []
    for r in decision.candidatos():
        cand = r.aplicar(P) == y
        mc = metrics.mcnemar(cand, base)
        filas.append({
            "regla": r.nombre,
            "n": int(len(y)),
            "acc_candidato": float(cand.mean()),
            "acc_produccion": float(base.mean()),
            "delta": float(cand.mean() - base.mean()),
            "discordantes": mc["n_discordantes"],
            "gana_candidato": mc["n01"], "gana_produccion": mc["n10"],
            "p_valor": mc["p_valor"],
            "significativo": mc["p_valor"] < CFG.training.get("promocion", {}).get("alpha", 0.05),
        })
    return pd.DataFrame(filas)


def barrido(P: np.ndarray, y: np.ndarray, umbrales=UMBRALES) -> pd.DataFrame:
    """El umbral completo, para ver la forma de la curva y no sólo el punto elegido."""
    reglas = [decision.produccion()] + [
        decision.Regla(f"u{int(u * 100):02d}", "umbral_empate", {"umbral": u})
        for u in umbrales]
    d = evaluar_reglas(P, y, reglas)
    d.insert(1, "umbral", [None] + list(umbrales))
    return d


def estabilidad_semillas(seeds=(42, 7, 123, 2024, 999)) -> pd.DataFrame:
    """El mismo walk-forward con distintas semillas de entrenamiento. **El control.**

    Existe porque la corrida con `seed=42` —el default del proyecto— da +2,6 puntos con
    McNemar p=0,076, que es justo el numero que uno estaria tentado de reportar. Repetirlo
    con otras cuatro semillas lo baja a una media de +0,9 con desvio 1,4: el efecto de la
    regla es mas chico que el ruido de la semilla.

    Es el mismo argumento que ya hace `training/promotion.py` con las fechas, aplicado a
    otra fuente de varianza: antes de creerle a una diferencia, hay que ver cuanto se mueve
    sola.
    """
    filas = []
    original = CFG.raw["training"].get("seed")
    try:
        for s in seeds:
            CFG.raw["training"]["seed"] = s   # walk_forward entrena con n_seeds=1
            _, P, y = sobre_walk_forward()
            comp = comparar_contra_produccion(P, y)
            comp.insert(0, "seed", s)
            filas.append(comp)
    finally:
        CFG.raw["training"]["seed"] = original
    return pd.concat(filas, ignore_index=True)


def auc_empate(P: np.ndarray, y: np.ndarray) -> float:
    """La pregunta que el umbral NO puede contestar: ¿ordena los empates?"""
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score((y == "draw").astype(int), P[:, I_DRAW]))


# --------------------------------------------------------------------------- #
# El ciclo operativo: la regla sobre el walk-forward
# --------------------------------------------------------------------------- #

def sobre_walk_forward(nombre: str | None = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Reentrena fecha a fecha y evalúa cada regla sobre las 38 fechas del holdout.

    Es la medición más parecida a lo que va a pasar en producción: el holdout de una sola
    pieza mide un modelo entrenado una vez, y acá el modelo se reentrena en cada fecha —
    que es lo que hace el ciclo real. Si la regla ayuda, tiene que ayudar también acá.
    """
    nombre = nombre or CFG.modelo
    wf = evaluate.walk_forward(nombre, resolve("auto"), guardar_proba=True)
    P = np.concatenate([np.asarray(p) for p in wf["proba"]])
    y = np.concatenate([np.asarray(v) for v in wf["y"]])

    filas = []
    for r in decision.todas():
        pred = r.aplicar(P)
        aciertos = pred == y
        # Por fecha, para el "% de fechas en que gana", que es la metrica del bloque 10.
        por_fecha, i = [], 0
        for n in wf["n"]:
            por_fecha.append(aciertos[i:i + n].mean())
            i += n
        filas.append({"regla": r.nombre, "fechas": len(wf), "partidos": int(len(y)),
                      "accuracy": float(aciertos.mean()),
                      "accuracy_media_por_fecha": float(np.mean(por_fecha)),
                      "empates_pred": int((pred == "draw").sum())})
    return pd.DataFrame(filas), P, y


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="La regla de decisión, medida sobre todo.")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--walk-forward", action="store_true")
    ap.add_argument("--barrido", action="store_true")
    ap.add_argument("--semillas", action="store_true",
                    help="repite el walk-forward con varias semillas: el control de ruido")
    ap.add_argument("--modelos", nargs="*", default=None)
    args = ap.parse_args()

    setup(CFG.log_level, CFG.log_format)
    SALIDA.mkdir(parents=True, exist_ok=True)

    reglas = decision.todas()
    print(f"\n{'=' * 96}")
    print(f"LA REGLA DE DECISION, SOBRE EL HOLDOUT {CFG.holdout_season} (380 partidos)")
    print(f"Reglas: {' | '.join(r.etiqueta for r in reglas)}")
    print("=" * 96)
    print("\nEl log-loss NO se mueve entre reglas: la regla no toca las probabilidades.")
    print("Lo unico que puede moverse es lo que depende de la clase anunciada.\n")

    probas, y, _ = probabilidades_holdout(args.modelos or cm.MODELOS, n_seeds=args.seeds)

    tablas, comparaciones = [], []
    for nombre, P in probas.items():
        d = evaluar_reglas(P, y)
        d.insert(0, "modelo", nombre)
        d["auc_draw"] = auc_empate(P, y)
        tablas.append(d)

        c = comparar_contra_produccion(P, y)
        c.insert(0, "modelo", nombre)
        comparaciones.append(c)

    holdout = pd.concat(tablas, ignore_index=True)
    comp = pd.concat(comparaciones, ignore_index=True)
    holdout.to_csv(SALIDA / "decision_holdout.csv", index=False)
    comp.to_csv(SALIDA / "decision_mcnemar.csv", index=False)

    print(holdout.round(4).to_string(index=False))
    print(f"\n{'-' * 96}\nMcNEMAR PAREADO — candidata contra produccion, mismas filas\n{'-' * 96}\n")
    print(comp.round(4).to_string(index=False))
    print("\n  `discordantes` es toda la evidencia que hay: en el resto las dos reglas")
    print("  dicen lo mismo y no aportan nada al test.")

    if args.barrido:
        print(f"\n{'-' * 96}\nBARRIDO DEL UMBRAL, POR MODELO\n{'-' * 96}")
        for nombre, P in probas.items():
            b = barrido(P, y)
            b.insert(0, "modelo", nombre)
            print(f"\n{b.round(4).to_string(index=False)}")
            b.to_csv(SALIDA / f"decision_barrido_{nombre}.csv", index=False)

    if args.walk_forward:
        print(f"\n{'-' * 96}")
        print(f"EL CICLO OPERATIVO — walk-forward de {CFG.modelo}, reentrenando cada fecha")
        print("-" * 96 + "\n")
        wf, Pw, yw = sobre_walk_forward()
        print(wf.round(4).to_string(index=False))
        cw = comparar_contra_produccion(Pw, yw)
        print(f"\n{cw.round(4).to_string(index=False)}")
        wf.to_csv(SALIDA / "decision_walkforward.csv", index=False)

    if args.semillas:
        print(f"\n{'-' * 96}")
        print("EL CONTROL — el mismo walk-forward, cambiando solo la semilla")
        print("-" * 96 + "\n")
        est = estabilidad_semillas()
        print(est.round(4).to_string(index=False))
        d = est["delta"]
        print(f"\n  delta medio {d.mean():+.4f}   desvio {d.std():.4f}   "
              f"rango [{d.min():+.4f}, {d.max():+.4f}]")
        print("  Si el rango del delta cruza el cero, la regla no se distingue del ruido.")
        est.to_csv(SALIDA / "decision_semillas.csv", index=False)

    print(f"\nCSVs en {SALIDA}")


if __name__ == "__main__":
    main()
