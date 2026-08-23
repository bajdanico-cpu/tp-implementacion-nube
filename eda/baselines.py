"""Las dos varas contra las que se mide el modelo.

Se calculan sobre los datos, nunca se citan de memoria. Si el modelo no le gana al
baseline trivial, no aprendió nada; si le gana al de cuotas, algo está mal (o se
encontró oro, pero primero hay que sospechar de un leakage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

CLASES = ["home", "draw", "away"]

# `log_loss` de sklearn asume las etiquetas en orden lexicográfico y alinea las
# columnas de probabilidad con ese orden. Pasarle ['home','draw','away'] desalinea
# las columnas y devuelve un número silenciosamente incorrecto (avisa por warning,
# pero no falla). Todo lo que va a log_loss se ordena con esta constante.
CLASES_ORD = sorted(CLASES)  # ['away', 'draw', 'home']

# Terna de cuotas de cierre, en orden de preferencia. `avg` promedia todas las
# casas y es más estable que una sola; `b365` es el respaldo.
COLUMNAS_CUOTAS = [
    ("odds_avg_close_home", "odds_avg_close_draw", "odds_avg_close_away"),
    ("odds_b365_close_home", "odds_b365_close_draw", "odds_b365_close_away"),
    ("odds_avg_home", "odds_avg_draw", "odds_avg_away"),
]


def odds_a_probabilidades(df: pd.DataFrame, columnas: tuple[str, str, str]) -> pd.DataFrame:
    """Convierte cuotas decimales a probabilidades y les saca el margen.

    La inversa de la cuota (1/cuota) es la probabilidad implícita, pero las tres
    suman más de 1: esa diferencia es el margen de la casa (el *overround*, típico
    5-7%). Normalizando por la suma se reparte proporcionalmente y quedan
    probabilidades legítimas.

    Es la normalización más simple y la estándar. Métodos como Shin o el de
    Margin-Weighted Proportional reparten mejor el favorito-longshot bias, pero
    para un benchmark no cambian la conclusión.
    """
    h, d, a = columnas
    inv = pd.DataFrame({
        "home": 1.0 / df[h],
        "draw": 1.0 / df[d],
        "away": 1.0 / df[a],
    })
    overround = inv.sum(axis=1)
    return inv.div(overround, axis=0)


def elegir_columnas_cuotas(df: pd.DataFrame) -> tuple[str, str, str] | None:
    """Primera terna de cuotas con cobertura suficiente."""
    for cols in COLUMNAS_CUOTAS:
        if all(c in df.columns for c in cols) and df[list(cols)].notna().all(axis=1).mean() > 0.9:
            return cols
    return None


def baseline_siempre_local(df: pd.DataFrame) -> dict:
    """El baseline trivial: apostar siempre al local.

    No tiene log-loss porque no es probabilístico — es una regla fija.
    """
    y = df["target_1x2"]
    pred = pd.Series("home", index=df.index)
    return {
        "nombre": "siempre local",
        "n": len(df),
        "accuracy": accuracy_score(y, pred),
        "log_loss": np.nan,
    }


def baseline_prior_de_clase(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Predecir siempre la distribución histórica de clases.

    Es el baseline probabilístico honesto: la versión de "siempre local" que sí
    tiene log-loss, y el piso que cualquier modelo tiene que superar.
    """
    prior = train["target_1x2"].value_counts(normalize=True).reindex(CLASES_ORD).fillna(0.0)
    probas = np.tile(prior.values, (len(test), 1))
    pred = np.full(len(test), prior.idxmax())

    return {
        "nombre": "prior de clase (frecuencias del train)",
        "n": len(test),
        "accuracy": accuracy_score(test["target_1x2"], pred),
        "log_loss": log_loss(test["target_1x2"], probas, labels=CLASES_ORD),
        "prior": prior.to_dict(),
    }


def baseline_cuotas(df: pd.DataFrame) -> dict:
    """El baseline duro: las cuotas de cierre de las casas de apuestas.

    Incorporan toda la información pública hasta minutos antes del partido —
    alineaciones, lesiones, clima — más el dinero apostado. Es el techo realista.
    """
    cols = elegir_columnas_cuotas(df)
    if cols is None:
        return {"nombre": "cuotas de cierre", "n": 0,
                "accuracy": np.nan, "log_loss": np.nan,
                "error": "sin cobertura de cuotas suficiente"}

    sub = df.dropna(subset=list(cols) + ["target_1x2"])
    probas = odds_a_probabilidades(sub, cols)
    pred = probas.idxmax(axis=1)

    return {
        "nombre": f"cuotas de cierre ({cols[0].replace('_home', '')})",
        "n": len(sub),
        "accuracy": accuracy_score(sub["target_1x2"], pred),
        "log_loss": log_loss(sub["target_1x2"], probas[CLASES_ORD].values, labels=CLASES_ORD),
        "overround_medio": float(
            (1 / sub[cols[0]] + 1 / sub[cols[1]] + 1 / sub[cols[2]]).mean()
        ),
    }


def por_temporada(matches: pd.DataFrame) -> pd.DataFrame:
    """Baselines temporada por temporada.

    La accuracy de las cuotas depende mucho de cuántos empates hubo: el empate casi
    nunca es el resultado más probable para el mercado, así que una temporada con
    muchos empates la hunde sin que el mercado haya estado peor informado. Mirar una
    sola temporada puede dar una idea equivocada del techo.
    """
    filas = []
    for season, grp in matches.groupby("season"):
        if grp.empty:
            continue
        cuotas = baseline_cuotas(grp)
        dist = grp["target_1x2"].value_counts(normalize=True)
        filas.append({
            "season": season,
            "n": len(grp),
            "% empates": dist.get("draw", float("nan")),
            "acc. siempre local": baseline_siempre_local(grp)["accuracy"],
            "acc. cuotas": cuotas["accuracy"],
            "log-loss cuotas": cuotas["log_loss"],
        })
    return pd.DataFrame(filas)


def calcular_todos(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Los tres baselines sobre el mismo conjunto de test."""
    filas = [
        baseline_siempre_local(test),
        baseline_prior_de_clase(train, test),
        baseline_cuotas(test),
    ]
    return pd.DataFrame([
        {k: v for k, v in f.items() if k in ("nombre", "n", "accuracy", "log_loss")}
        for f in filas
    ])
