"""Ratings separados de ATAQUE y DEFENSA, aprendidos de goles (familia Berrar).

    python -m features.ataque_defensa --ajustar

Berrar, Lopes & Dubitzky (2019). Un k-NN sobre estas features ganó la Soccer Prediction
Challenge 2017. La idea: un equipo no tiene *un* nivel, tiene **dos** — cuánto genera y
cuánto concede— y no tienen por qué moverse juntos.

## Qué agrega sobre el Elo y sobre los pi-ratings

`elo` colapsa todo en un escalar. `dif_elo` dice quién es mejor, no *por qué*: un 1-0 y un
4-3 mueven el rating parecido si el margen es parecido, y son equipos distintos.

Acá cada equipo tiene `ataque` y `defensa`, y el sistema predice **los goles de cada lado**:

    lambda_local  = exp(mu + ataque_local  - defensa_visita + ventaja_local)
    lambda_visita = exp(mu + ataque_visita - defensa_local)

`mu` es el log del promedio de goles de la liga, así que con ratings en cero el sistema
predice el promedio. La forma exponencial no es capricho: es la del modelo de Poisson, que
es como se modelan goles desde Maher (1982), y garantiza que la prediccion sea positiva.

Con el resultado se corrige por el error de cada lado —el gradiente online de una Poisson—:

    ataque_local   += k * (gf_local - lambda_local)
    defensa_visita -= k * (gf_local - lambda_local)

Si el local metió más de lo esperado, **o él atacó mejor o el otro defendió peor**, y el
sistema reparte el crédito entre las dos cosas. Eso es exactamente lo que un rating escalar
no puede hacer.

## Y una feature que las otras fases no podían dar

`af_lambda_total` = los goles esperados del partido, sumando los dos lados. Es la primera
feature del proyecto que apunta **directo al empate**: un partido de pocos goles esperados
tiene más chances de terminar igualado que uno de muchos, y eso no es lo mismo que "los dos
equipos son parejos" —que es lo que `dif_elo ≈ 0` ya decía—. Dos equipos parejos y goleadores
empatan menos que dos parejos y aburridos.

## Los parámetros se ajustan DENTRO del train

`k` y la ventaja de local salen de `ajustar()`, que minimiza la **deviance de Poisson** de
los goles predichos sobre las temporadas de entrenamiento. Nunca toca el holdout: al Banco A
llega una sola configuración.

La deviance y no el error absoluto porque el modelo es de conteos: `lambda` predice una
distribución, no un número, y la deviance es la pérdida que le corresponde.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.config import CFG
from common.logging_setup import get_logger, setup

log = get_logger(__name__)

K = 0.02                 # tasa de aprendizaje
VENTAJA_LOCAL = 0.25     # en escala log: exp(0,25) ~ 1,28 veces mas goles de local
REGRESION_TEMPORADA = 0.25
INICIAL = 0.0
# Techo del rating. Sin el, un equipo que golea seis fechas seguidas se dispara y la
# exponencial explota: exp(mu + 3) son 29 goles esperados.
TOPE = 1.5

GRILLA_K = (0.005, 0.01, 0.02, 0.035, 0.05, 0.08)
GRILLA_VENTAJA = (0.10, 0.18, 0.25, 0.32)

COLUMNAS = ["af_ataque", "af_defensa"]
# Las de partido: el `dif_` automatico no puede cruzar el ataque de uno con la defensa del
# otro, que es justamente lo que el sistema predice.
COLUMNAS_PARTIDO = ["af_lambda_local", "af_lambda_visita", "af_lambda_dif", "af_lambda_total"]


def mu_de(largo: pd.DataFrame) -> float:
    """El log del promedio de goles por equipo y partido.

    Lo comparten `calcular` y `partido`: si cada uno usara el suyo, las lambdas de las
    features de partido quedarian en otra escala que la que calibro los ratings.
    """
    p = _partidos(largo)
    return float(np.log(max(pd.concat([p["gf"], p["gc"]]).mean(), 0.1)))


def _partidos(largo: pd.DataFrame) -> pd.DataFrame:
    return (largo[largo["es_local"]]
            [["season", "fixture_id", "kickoff_time", "team_short", "rival_short",
              "gf", "gc"]]
            .sort_values(["kickoff_time", "fixture_id"])
            .reset_index(drop=True))


def lambdas(ataque_l: float, defensa_l: float, ataque_v: float, defensa_v: float,
            mu: float, ventaja: float) -> tuple[float, float]:
    """Goles esperados de cada lado. La forma de Poisson, positiva por construccion."""
    return (float(np.exp(mu + ataque_l - defensa_v + ventaja)),
            float(np.exp(mu + ataque_v - defensa_l)))


def calcular(largo: pd.DataFrame, k: float | None = None,
             ventaja: float | None = None, emitir: bool = True) -> pd.DataFrame:
    """Los dos ratings de cada equipo DESPUÉS de cada partido, tageados con su kickoff.

    Mismo contrato que `elo.calcular` y `pi_ratings.calcular`: secuencial, sólo hacia atrás,
    y el `merge_asof` del corte impide que un partido vea el suyo propio.

    Con `emitir=False` devuelve los goles esperados partido a partido, que es lo que
    consume `ajustar()`.
    """
    k = K if k is None else k
    ventaja = VENTAJA_LOCAL if ventaja is None else ventaja

    p = _partidos(largo)
    # `mu` es el log del promedio de goles por equipo y partido, calculado sobre lo que se
    # recorre. Con los ratings en cero el sistema predice ese promedio, que es el punto de
    # partida honesto para un equipo del que no se sabe nada.
    mu = mu_de(largo)

    ataque: dict[str, float] = {}
    defensa: dict[str, float] = {}
    temporada_previa: str | None = None
    filas, esperados = [], []

    for r in p.itertuples(index=False):
        if r.season != temporada_previa:
            for d in (ataque, defensa):
                for eq in d:
                    d[eq] += (INICIAL - d[eq]) * REGRESION_TEMPORADA
            temporada_previa = r.season

        al = ataque.setdefault(r.team_short, INICIAL)
        dl = defensa.setdefault(r.team_short, INICIAL)
        av = ataque.setdefault(r.rival_short, INICIAL)
        dv = defensa.setdefault(r.rival_short, INICIAL)

        lam_l, lam_v = lambdas(al, dl, av, dv, mu, ventaja)
        esperados.append({"season": r.season, "fixture_id": r.fixture_id,
                          "gf": float(r.gf), "gc": float(r.gc),
                          "lam_l": lam_l, "lam_v": lam_v})

        # Gradiente online de la Poisson: el error de cada lado se reparte entre el ataque
        # del que hizo los goles y la defensa del que los recibio.
        e_l = float(r.gf) - lam_l
        e_v = float(r.gc) - lam_v
        ataque[r.team_short] = np.clip(al + k * e_l, -TOPE, TOPE)
        defensa[r.rival_short] = np.clip(dv - k * e_l, -TOPE, TOPE)
        ataque[r.rival_short] = np.clip(av + k * e_v, -TOPE, TOPE)
        defensa[r.team_short] = np.clip(dl - k * e_v, -TOPE, TOPE)

        if emitir:
            for equipo in (r.team_short, r.rival_short):
                filas.append({"season": r.season, "fixture_id": r.fixture_id,
                              "team_short": equipo, "kickoff_time": r.kickoff_time,
                              "af_ataque": ataque[equipo], "af_defensa": defensa[equipo]})

    if not emitir:
        return pd.DataFrame(esperados)
    out = pd.DataFrame(filas)
    out["fixture_id"] = out["fixture_id"].astype("int64")
    return out


def _deviance(d: pd.DataFrame) -> float:
    """Deviance de Poisson de los goles predichos. La perdida que le toca a un conteo."""
    def dev(y, lam):
        y, lam = np.asarray(y, float), np.clip(np.asarray(lam, float), 1e-9, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(y > 0, y * np.log(y / lam), 0.0)
        return float(np.mean(2 * (t - (y - lam))))
    return (dev(d["gf"], d["lam_l"]) + dev(d["gc"], d["lam_v"])) / 2


def ajustar(largo: pd.DataFrame, temporadas_fit: list[str] | None = None) -> dict:
    """Elige `k` y la ventaja de local por deviance, SOLO sobre temporadas de train."""
    temporadas_fit = temporadas_fit or CFG.seasons_for_training()
    filas = []
    for k in GRILLA_K:
        for v in GRILLA_VENTAJA:
            e = calcular(largo, k, v, emitir=False)
            e = e[e["season"].isin(temporadas_fit)]
            filas.append({"k": k, "ventaja": v, "deviance": _deviance(e), "n": len(e)})
    d = pd.DataFrame(filas).sort_values("deviance").reset_index(drop=True)
    mejor = d.iloc[0]
    log.info("ataque/defensa ajustado: k=%.3f ventaja=%.2f (deviance %.4f)",
             mejor["k"], mejor["ventaja"], mejor["deviance"])
    return {"k": float(mejor["k"]), "ventaja": float(mejor["ventaja"]),
            "deviance": float(mejor["deviance"]), "grilla": d}


def construir(largo: pd.DataFrame) -> pd.DataFrame:
    """Las features por equipo, listas para el `merge_asof`."""
    out = calcular(largo, CFG.af_k, CFG.af_ventaja)
    return out.rename(columns={"kickoff_time": "hist_kickoff"})


def partido(gold: pd.DataFrame, mu: float | None = None) -> pd.DataFrame:
    """Los goles esperados del partido, cruzando ataque de uno con defensa del otro.

    El `dif_` automatico de Gold no puede: resta la misma columna de los dos lados.
    """
    out = gold.copy()
    cols = ("local_af_ataque", "local_af_defensa", "visita_af_ataque", "visita_af_defensa")
    if any(c not in out.columns for c in cols):
        for c in COLUMNAS_PARTIDO:
            out[c] = np.nan
        return out

    m = float(np.log(1.45)) if mu is None else mu
    v = CFG.af_ventaja
    out["af_lambda_local"] = np.exp(m + out["local_af_ataque"]
                                    - out["visita_af_defensa"] + v)
    out["af_lambda_visita"] = np.exp(m + out["visita_af_ataque"] - out["local_af_defensa"])
    out["af_lambda_dif"] = out["af_lambda_local"] - out["af_lambda_visita"]
    # Los goles esperados del PARTIDO. Apunta directo al empate: un partido de pocos goles
    # esperados empata mas que uno de muchos, y eso NO es lo mismo que "son parejos".
    out["af_lambda_total"] = out["af_lambda_local"] + out["af_lambda_visita"]
    return out


def main() -> None:
    from common.storage import read_table
    from features import player_agg, team_form as tf

    ap = argparse.ArgumentParser(description="Ratings de ataque y defensa.")
    ap.add_argument("--ajustar", action="store_true")
    args = ap.parse_args()
    setup(CFG.log_level, CFG.log_format)

    largo = tf.construir_largo(
        read_table("fact_match"), read_table("fact_fixture"),
        player_agg.team_stats_by_fixture(read_table("fact_player_gw")))

    if args.ajustar:
        res = ajustar(largo)
        print(f"\n{'=' * 70}\nAJUSTE (solo temporadas de train)\n{'=' * 70}\n")
        print(res["grilla"].head(10).round(4).to_string(index=False))
        print(f"\n  elegido: k={res['k']}  ventaja={res['ventaja']}  "
              f"deviance {res['deviance']:.4f}")
        base = calcular(largo, res["k"], res["ventaja"], emitir=False)
        base = base[base["season"].isin(CFG.seasons_for_training())]
        plano = base.assign(lam_l=base["gf"].mean(), lam_v=base["gc"].mean())
        print(f"  vara (predecir siempre el promedio): {_deviance(plano):.4f}\n")
        return

    r = calcular(largo)
    u = r.sort_values("kickoff_time").groupby("team_short").last()
    u["neto"] = u["af_ataque"] + u["af_defensa"]
    print(f"\n{'=' * 70}\nataque / defensa al ultimo partido\n{'=' * 70}\n")
    print(u[["af_ataque", "af_defensa", "neto"]].sort_values("neto", ascending=False)
          .round(3).to_string())
    print(f"\n  correlacion ataque-defensa: {u['af_ataque'].corr(u['af_defensa']):+.3f}")
    print("  (si diera ~1, los dos ratings serian el mismo numero y la fase no aportaria)\n")


if __name__ == "__main__":
    main()
