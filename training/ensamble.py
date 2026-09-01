"""¿Sirve combinar el clasificador con el modelo de goles? La evidencia, entera.

    python -m training.ensamble
    python -m training.ensamble --device cpu

La idea es razonable y vale la pena escribirla antes de medirla: en vez de predecir la
clase, predecir **cuántos goles hace cada equipo** y derivar el 1X2 de la distribución
conjunta. El empate deja de ser una etiqueta y pasa a ser lo que realmente es —los dos
marcan lo mismo— y encima el modelo mira el problema desde un ángulo distinto al del
clasificador, que es la condición para que un ensamble aporte.

Este módulo responde tres preguntas con números, y ninguna de las tres da que sí. Existe
por el mismo motivo que `training/ablacion.py`: **una medición que sólo vive en una
conversación no se puede auditar**, y un resultado negativo medido vale tanto como uno
positivo.

Todo con el modelo de **evaluación** (`incluir_holdout=False`): entrena hasta 2024-25 y se
mide contra 2025-26.

---

## 1. ¿El ensamble aporta?

No. Ningún peso le gana al clasificador solo.

## 2. ¿Por qué no?

Porque los dos modelos se equivocan casi en los mismos partidos: coinciden en el argmax
alrededor del 90 % de las veces y sus probabilidades de local y visitante correlacionan por
encima de 0,90. Un ensamble sólo ayuda cuando los errores están decorrelacionados.

La excepción, y el único hallazgo aprovechable: **para el empate la correlación es ~0,34**.
Ahí sí piensan distinto.

## 3. ¿Y si se corrige la independencia?

El modelo de goles multiplica las dos distribuciones, o sea asume que los goles de un equipo
no dicen nada de los del otro. La corrección clásica para eso es Dixon-Coles (1997): un
parámetro `rho` que reajusta las cuatro celdas de marcador bajo.

**Tampoco aporta, y el diagnóstico dice exactamente por qué.** La tabla de celdas que
imprime este módulo muestra que 0-0 aparece *menos* de lo que predice la independencia y 1-1
*más*. La `tau` de Dixon-Coles tiene **un solo parámetro**, que empuja esas dos celdas en la
**misma** dirección: no existe un `rho` capaz de bajar una y subir la otra. El MLE lo
resuelve quedándose en ~0, que es la forma de decir "esta corrección no aplica acá".

No es que el modelo acierte los empates: los sigue subestimando. Es que **la desviación no
tiene la forma que Dixon-Coles corrige**.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import poisson as poisson_dist

from common.config import CFG, PROJECT_ROOT
from common.logging_setup import get_logger, setup
from eda.baselines import CLASES_ORD
from features import spec
from training import dataset, evaluate, metrics, models_alt as ma
from training.device import resolve

log = get_logger(__name__)

SALIDA = PROJECT_ROOT / "training" / "output"

PESOS = (0.25, 0.40, 0.50, 0.60, 0.75)      # peso del clasificador
CELDAS = ((0, 0), (0, 1), (1, 0), (1, 1))   # las que toca Dixon-Coles


def _reporte(nombre: str, P: np.ndarray, y) -> dict:
    pred = np.array(CLASES_ORD)[P.argmax(1)]
    r = metrics.reporte(y, pred, P, con_ic=False)
    return {"modelo": nombre, "accuracy": r["accuracy"], "f1_macro": r["f1_macro"],
            "f1_draw": r["f1_draw"], "log_loss": r["log_loss"],
            "p_draw_media": float(P[:, 1].mean()),
            "p_draw_max": float(P[:, 1].max()),
            "empates_argmax": int((P.argmax(1) == 1).sum())}


def _goles(X_train, gl, gv, X_test, info, dixon_coles: bool):
    """Promedia semillas, igual que el resto del proyecto. Devuelve (P, rho medio)."""
    probas, rhos = [], []
    for i in range(CFG.n_seeds):
        m = ma.PoissonBivariado(device=info.used, seed=CFG.seed + i,
                                dixon_coles=dixon_coles)
        m.fit(X_train, gl, gv)
        probas.append(m.predict_proba(X_test))
        rhos.append(m.rho)
    return ma.ensamble(probas), float(np.mean(rhos))


def diagnostico_celdas(X_train, gl, gv, info) -> pd.DataFrame:
    """Observado contra esperado bajo independencia, en las celdas que Dixon-Coles toca.

    Es la tabla que explica por qué la corrección no aplica: si 0-0 y 1-1 se desvían en
    direcciones opuestas, ningún `rho` único puede arreglar las dos.
    """
    m = ma.PoissonBivariado(device=info.used, seed=CFG.seed).fit(X_train, gl, gv)
    lam = np.clip(m.m_local.predict(X_train), 1e-6, None)
    mu = np.clip(m.m_visita.predict(X_train), 1e-6, None)

    filas = []
    for x, y in CELDAS:
        obs = int(((gl == x) & (gv == y)).sum())
        esp = float((poisson_dist.pmf(x, lam) * poisson_dist.pmf(y, mu)).sum())
        filas.append({"celda": f"{x}-{y}", "observado": obs, "esperado_indep": esp,
                      "obs_sobre_esp": obs / esp if esp else np.nan,
                      "es_empate": x == y})

    emp_obs = int((gl == gv).sum())
    emp_esp = float(sum((poisson_dist.pmf(k, lam) * poisson_dist.pmf(k, mu)).sum()
                        for k in range(ma.MAX_GOLES + 1)))
    filas.append({"celda": "empates (todos)", "observado": emp_obs,
                  "esperado_indep": emp_esp,
                  "obs_sobre_esp": emp_obs / emp_esp if emp_esp else np.nan,
                  "es_empate": True})
    return pd.DataFrame(filas)


def correr(device: str | None = None) -> dict:
    info = resolve(device)
    log.info("device: %s (%s)", info.used, info.reason)

    gold = dataset.cargar()
    sp = dataset.preparar(gold, spec.FEATURES)
    y_te = sp.y_test_txt

    # El clasificador, con su protocolo completo (early stopping + refit).
    P_clf = evaluate.evaluar_holdout(CFG.modelo, info, spec.FEATURES, gold,
                                     incluir_holdout=False)["proba"]

    # Los de goles, sobre exactamente el mismo train.
    full = dataset.filtrar_train(gold[gold["season"].isin(CFG.seasons_for_training())])
    X_f = dataset.matriz(full, spec.FEATURES)
    gl = full["home_goals"].to_numpy().astype(int)
    gv = full["away_goals"].to_numpy().astype(int)

    P_poi, _ = _goles(X_f, gl, gv, sp.X_test, info, dixon_coles=False)
    P_dc, rho = _goles(X_f, gl, gv, sp.X_test, info, dixon_coles=True)

    filas = [_reporte(CFG.modelo, P_clf, y_te),
             _reporte("poisson", P_poi, y_te),
             _reporte("poisson_dc", P_dc, y_te)]
    for w in PESOS:
        filas.append(_reporte(f"ensamble {w:.2f}/{1 - w:.2f}",
                              ma.ensamble([P_clf, P_poi], [w, 1 - w]), y_te))

    return {"tabla": pd.DataFrame(filas), "rho": rho,
            "celdas": diagnostico_celdas(X_f, gl, gv, info),
            "P_clf": P_clf, "P_poi": P_poi, "y": y_te, "n_train": len(full)}


def decorrelacion(P_a: np.ndarray, P_b: np.ndarray, y) -> None:
    ok_a = np.array(CLASES_ORD)[P_a.argmax(1)] == y
    ok_b = np.array(CLASES_ORD)[P_b.argmax(1)] == y

    print(f"  coinciden en el argmax        {(P_a.argmax(1) == P_b.argmax(1)).mean():7.1%}")
    print(f"  aciertan los dos              {(ok_a & ok_b).mean():7.1%}")
    print(f"  acierta solo el clasificador  {(ok_a & ~ok_b).mean():7.1%}")
    print(f"  acierta solo el de goles      {(~ok_a & ok_b).mean():7.1%}")
    print(f"  fallan los dos                {(~ok_a & ~ok_b).mean():7.1%}")
    print()
    for i, c in enumerate(CLASES_ORD):
        r = float(np.corrcoef(P_a[:, i], P_b[:, i])[0, 1])
        nota = "   <- aca SI piensan distinto" if r < 0.6 else ""
        print(f"  correlacion de p_{c:5s}         {r:7.3f}{nota}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="El clasificador contra el modelo de goles, y su ensamble.")
    ap.add_argument("--device", default=None, choices=("auto", "cuda", "cpu"))
    args = ap.parse_args()

    setup(CFG.log_level, CFG.log_format)
    SALIDA.mkdir(parents=True, exist_ok=True)

    res = correr(args.device)
    res["tabla"].to_csv(SALIDA / "ensamble_clf_goles.csv", index=False)

    print("\n" + "=" * 84)
    print(f"CLASIFICADOR vs MODELO DE GOLES — holdout {CFG.holdout_season} "
          f"(380 partidos, {res['n_train']} de train)")
    print("=" * 84 + "\n")
    print(res["tabla"].round(4).to_string(index=False))

    mejor_solo = res["tabla"].iloc[0]["accuracy"]
    mejor_ens = res["tabla"][res["tabla"].modelo.str.startswith("ensamble")]["accuracy"].max()
    print(f"\n  mejor ensamble {mejor_ens:.4f}  contra  clasificador solo {mejor_solo:.4f}")
    if mejor_ens <= mejor_solo:
        print("  -> ningun peso le gana al clasificador solo.")

    print("\n" + "-" * 84)
    print("POR QUE: los dos modelos se equivocan en los mismos partidos")
    print("-" * 84 + "\n")
    decorrelacion(res["P_clf"], res["P_poi"], res["y"])

    print("\n" + "-" * 84)
    print("Y POR QUE DIXON-COLES NO LO ARREGLA")
    print("-" * 84 + "\n")
    print(res["celdas"].round(3).to_string(index=False))
    print(f"\n  rho ajustado por maxima verosimilitud: {res['rho']:+.4f}")
    print("\n  La tau de Dixon-Coles tiene UN solo parametro, que empuja 0-0 y 1-1 en la")
    print("  MISMA direccion. Si una aparece de menos y la otra de mas, no hay rho que")
    print("  arregle las dos: el MLE se queda en ~0, que es la forma de decir que esta")
    print("  correccion no aplica a estos datos.")

    print(f"\nCSV en {SALIDA / 'ensamble_clf_goles.csv'}")


if __name__ == "__main__":
    main()
