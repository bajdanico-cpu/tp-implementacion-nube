"""pi-ratings: un rating de LOCAL y otro de VISITANTE para cada equipo.

    python -m features.pi_ratings            # ajusta lambda y gamma, y muestra la tabla

Constantinou & Fenton (2013). El mejor modelo de ML de la Soccer Prediction Challenge 2023
fue CatBoost sobre estas features, y el ganador de la de 2017 usó la familia vecina (Berrar
ratings). Es, junto con el Elo, la feature clásica del fútbol — pero arregla dos cosas que
el Elo del proyecto no puede.

## Qué arregla

**1 · La localía deja de ser una constante.** En `features/elo.py` la ventaja de local es
`VENTAJA_LOCAL = 65.0`: el mismo número para los veinte equipos, todas las temporadas. Es
falso y es medible que es falso — hay clubes que se transforman en su cancha y otros a los
que les da igual. Acá cada equipo tiene **dos ratings**, y la diferencia entre ellos *es* su
ventaja de localía, aprendida de sus resultados.

**2 · Se aprende de la DIFERENCIA DE GOLES, no del resultado.** El Elo actualiza con
`1 / 0,5 / 0` y después amplifica por el margen; los pi-ratings predicen directamente una
diferencia de goles y corrigen por el error de esa predicción. Ganar 1-0 al que ibas a
golear es información negativa, y así queda registrada.

## Cómo funciona

Cada equipo tiene `R_H` (su nivel jugando de local) y `R_A` (de visitante), los dos en 0.
Para un partido entre `H` y `A`:

    diferencia de goles esperada    gd_esp = f(R_H de H) - f(R_A de A)
    donde                           f(r)   = signo(r) * (10^(|r|/b) - 1)

`f` convierte el rating a goles con una curva suave: cerca de 0 es casi lineal y se estira
en los extremos, que es como se comportan las diferencias de goles reales.

Con el resultado, el error `e = |gd_real - gd_esp|` se atenúa con un logaritmo —igual que el
margen en el Elo, y por la misma razón: un 6-0 no debe mover el rating seis veces más que un
1-0— y se aplica:

    peso = c * log10(1 + e)
    R_H de H  +=  peso * lambda * signo(gd_real - gd_esp)
    R_A de H  +=  (lo que se movio R_H) * gamma        <- la transferencia cruzada

**`gamma` es la pieza interesante.** Dice cuánto de lo que aprendiste sobre un equipo
jugando de local aplica a cómo juega de visitante. Con `gamma = 1` los dos ratings son el
mismo número y el sistema colapsa a un Elo; con `gamma = 0` son dos equipos distintos que
comparten nombre. La verdad está en el medio, y se ajusta con datos.

## Los parámetros se ajustan DENTRO del train

`lambda` y `gamma` salen de `ajustar()`, que barre una grilla chica y elige por **error
absoluto medio de la diferencia de goles predicha** sobre las temporadas de entrenamiento.
Nunca toca el holdout: `docs/PLAN-MEJORAS.md` es explícito en que al Banco A tiene que
llegar **una sola** configuración, y que el holdout decide *si* la fase entra, no *cuál* de
sus variantes.

El error de goles es el objetivo correcto acá porque es lo que el rating predice. Elegir por
accuracy del 1X2 metería la conversión a probabilidades —que es trabajo del modelo, no del
rating— adentro del ajuste del rating.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.config import CFG
from common.logging_setup import get_logger, setup

log = get_logger(__name__)

# Escala de la conversion rating -> goles. Con b=10, un rating de 1,0 vale ~0,26 goles y uno
# de 3,0 vale ~1,0: el rango util queda en [-3, 3], que es donde viven las diferencias de
# goles esperadas del futbol.
B = 10.0
# Atenuacion del error. Es el analogo del `1 + log1p(margen)` del Elo.
C = 3.0

INICIAL = 0.0

# Grilla del ajuste. La primera version llegaba hasta lambda=0,10 y gamma=0,7, y el optimo
# cayo JUSTO en esa esquina -- un minimo en el borde no es un minimo, es una grilla corta.
# Extendida, el optimo queda adentro: lambda=0,20, gamma=0,70.
#
# `gamma = 1,0` esta incluido a proposito porque es la hipotesis nula de esta fase: con
# gamma=1 los dos ratings colapsan en uno y el sistema vuelve a ser un Elo. Que el ajuste lo
# rechace (1,3956 contra 1,3839) es la primera evidencia de que separar local y visitante
# sirve.
GRILLA_LAMBDA = (0.06, 0.10, 0.15, 0.20, 0.30, 0.45)
GRILLA_GAMMA = (0.3, 0.5, 0.7, 0.85, 1.0)

COLUMNAS = ["pi_home", "pi_away", "pi_ventaja"]


def a_goles(r: float | np.ndarray) -> float | np.ndarray:
    """Rating -> diferencia de goles esperada. Suave cerca de 0, se estira en los extremos."""
    r = np.asarray(r, dtype=float)
    return np.sign(r) * (10.0 ** (np.abs(r) / B) - 1.0)


def _peso(e: float) -> float:
    """El error, atenuado. Un 6-0 no vale seis veces un 1-0."""
    return C * np.log10(1.0 + e)


def _partidos(largo: pd.DataFrame) -> pd.DataFrame:
    return (largo[largo["es_local"]]
            [["season", "fixture_id", "kickoff_time", "team_short", "rival_short",
              "gf", "gc"]]
            .sort_values(["kickoff_time", "fixture_id"])
            .reset_index(drop=True))


def calcular(largo: pd.DataFrame, lam: float | None = None,
             gamma: float | None = None, emitir: bool = True) -> pd.DataFrame:
    """Los dos ratings de cada equipo DESPUÉS de cada partido, tageados con su kickoff.

    Mismo contrato que `features.elo.calcular`: secuencial, sólo mira hacia atrás, y el
    `merge_asof` del corte se encarga de que un partido nunca vea el suyo propio.

    Con `emitir=False` devuelve el error de predicción partido a partido en vez de las
    features. Es lo que consume `ajustar()`.
    """
    lam = CFG.pi_lambda if lam is None else lam
    gamma = CFG.pi_gamma if gamma is None else gamma

    casa: dict[str, float] = {}
    fuera: dict[str, float] = {}
    filas, errores = [], []

    for r in _partidos(largo).itertuples(index=False):
        rh = casa.setdefault(r.team_short, INICIAL)
        ra = fuera.setdefault(r.rival_short, INICIAL)
        # Y los ratings del otro lado, que existen aunque este partido no los use.
        fuera.setdefault(r.team_short, INICIAL)
        casa.setdefault(r.rival_short, INICIAL)

        gd_esp = a_goles(rh) - a_goles(ra)
        gd_real = float(r.gf - r.gc)
        e = abs(gd_real - gd_esp)
        errores.append({"season": r.season, "fixture_id": r.fixture_id,
                        "gd_real": gd_real, "gd_esp": float(gd_esp), "error": float(e)})

        if emitir:
            ajuste = _peso(e) * lam * np.sign(gd_real - gd_esp)

            # El local ajusta su rating de LOCAL, y transfiere `gamma` al de visitante.
            casa[r.team_short] = rh + ajuste
            fuera[r.team_short] += ajuste * gamma
            # El visitante, al reves: si el local rindio de mas, el visitante rindio de menos.
            fuera[r.rival_short] = ra - ajuste
            casa[r.rival_short] -= ajuste * gamma

            for equipo in (r.team_short, r.rival_short):
                filas.append({
                    "season": r.season, "fixture_id": r.fixture_id,
                    "team_short": equipo, "kickoff_time": r.kickoff_time,
                    "pi_home": casa[equipo], "pi_away": fuera[equipo],
                    # La ventaja de localia DE ESTE EQUIPO. Es lo que el Elo no puede tener:
                    # ahi es una constante de 65 puntos igual para los veinte.
                    "pi_ventaja": casa[equipo] - fuera[equipo]})
        else:
            ajuste = _peso(e) * lam * np.sign(gd_real - gd_esp)
            casa[r.team_short] = rh + ajuste
            fuera[r.team_short] += ajuste * gamma
            fuera[r.rival_short] = ra - ajuste
            casa[r.rival_short] -= ajuste * gamma

    if not emitir:
        return pd.DataFrame(errores)
    out = pd.DataFrame(filas)
    out["fixture_id"] = out["fixture_id"].astype("int64")
    return out


def gd_esperado(pi_home_local: pd.Series, pi_away_visita: pd.Series) -> pd.Series:
    """La predicción del sistema, a nivel partido: `f(R_H del local) - f(R_A del visitante)`.

    Es la única feature de pi-ratings que NO se puede derivar con el `dif_` automático de
    Gold: ése resta la misma columna de los dos lados (`local_pi_home - visita_pi_home`), y
    lo que hace falta es cruzar el rating de local del uno con el de visitante del otro.
    """
    return pd.Series(a_goles(pi_home_local.to_numpy()) - a_goles(pi_away_visita.to_numpy()),
                     index=pi_home_local.index)


def ajustar(largo: pd.DataFrame, temporadas_fit: list[str] | None = None) -> dict:
    """Elige `lambda` y `gamma` por error de goles, SOLO sobre temporadas de entrenamiento.

    El holdout no participa. Se recorre siempre la serie completa —un rating necesita su
    historia— pero el error se promedia únicamente sobre las temporadas permitidas.
    """
    temporadas_fit = temporadas_fit or CFG.seasons_for_training()
    filas = []
    for lam in GRILLA_LAMBDA:
        for gamma in GRILLA_GAMMA:
            err = calcular(largo, lam, gamma, emitir=False)
            err = err[err["season"].isin(temporadas_fit)]
            filas.append({"lambda": lam, "gamma": gamma,
                          "mae_goles": float(err["error"].mean()),
                          "n": int(len(err))})
    d = pd.DataFrame(filas).sort_values("mae_goles").reset_index(drop=True)
    mejor = d.iloc[0]
    log.info("pi-ratings ajustado sobre %s: lambda=%.3f gamma=%.2f (MAE %.4f goles)",
             ", ".join(temporadas_fit), mejor["lambda"], mejor["gamma"], mejor["mae_goles"])
    return {"lambda": float(mejor["lambda"]), "gamma": float(mejor["gamma"]),
            "mae_goles": float(mejor["mae_goles"]), "grilla": d}


def construir(largo: pd.DataFrame) -> pd.DataFrame:
    """Las features de este módulo, listas para el `merge_asof`."""
    out = calcular(largo)
    return out.rename(columns={"kickoff_time": "hist_kickoff"})


def main() -> None:
    from common.storage import read_table
    from features import player_agg, team_form as tf

    ap = argparse.ArgumentParser(description="pi-ratings: ajuste y tabla.")
    ap.add_argument("--ajustar", action="store_true")
    args = ap.parse_args()
    setup(CFG.log_level, CFG.log_format)

    largo = tf.construir_largo(
        read_table("fact_match"), read_table("fact_fixture"),
        player_agg.team_stats_by_fixture(read_table("fact_player_gw")))

    if args.ajustar:
        res = ajustar(largo)
        print(f"\n{'=' * 70}\nAJUSTE DE pi-ratings (solo temporadas de train)\n{'=' * 70}\n")
        print(res["grilla"].round(4).to_string(index=False))
        print(f"\n  elegido: lambda={res['lambda']}  gamma={res['gamma']}  "
              f"MAE {res['mae_goles']:.4f} goles")
        print("  -> ponerlos en config.yaml (features.pi_lambda / features.pi_gamma)\n")
        return

    r = calcular(largo)
    ult = (r.sort_values("kickoff_time").groupby("team_short").last()
            .sort_values("pi_home", ascending=False))
    print(f"\n{'=' * 70}\npi-ratings al ultimo partido de cada equipo\n{'=' * 70}\n")
    print(ult[["pi_home", "pi_away", "pi_ventaja"]].round(3).to_string())
    print(f"\n  ventaja de localia: media {ult['pi_ventaja'].mean():.3f}, "
          f"desvio {ult['pi_ventaja'].std():.3f}, "
          f"rango [{ult['pi_ventaja'].min():.3f}, {ult['pi_ventaja'].max():.3f}]")
    print("  (en el Elo esto es una constante igual para los veinte equipos)\n")


if __name__ == "__main__":
    main()
