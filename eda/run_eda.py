"""EDA — dimensionar los datos y fijar las varas antes de modelar.

Responde seis preguntas concretas:

1. ¿Qué cobertura y qué nulos tienen las tablas?
2. ¿Cómo se distribuye el 1X2 y cuánto vale el baseline trivial?
3. ¿Cuánto vale el baseline de cuotas — el techo realista?
4. ¿Cuánto se mueven las distribuciones entre temporadas (drift)?
5. ¿Qué historial tienen los equipos recién ascendidos (cold-start)?
6. ¿Alcanzan las filas para un split temporal?

Genera un reporte en consola, gráficos y un resumen markdown en `eda/output/`.

    python -m eda.run_eda
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin ventana: esto corre en batch
import matplotlib.pyplot as plt
import pandas as pd

from common.config import CFG, PROJECT_ROOT
from common.logging_setup import get_logger, setup
from common.storage import read_table
from eda import baselines

log = get_logger(__name__)

OUTPUT = PROJECT_ROOT / "eda" / "output"

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


def _seccion(titulo: str) -> str:
    linea = "=" * 78
    return f"\n{linea}\n{titulo}\n{linea}"


# --------------------------------------------------------------------------- #


def cobertura(matches: pd.DataFrame, players: pd.DataFrame, fixtures: pd.DataFrame) -> str:
    out = [_seccion("1. COBERTURA")]

    resumen = pd.DataFrame({
        "partidos": matches.groupby("season").size(),
        "fixtures": fixtures.groupby("season").size(),
        "filas jugador-fecha": players.groupby("season").size(),
    }).fillna(0).astype(int)
    out.append(resumen.to_string())

    out.append("\nNulos por columna en fact_match (sólo las que tienen):")
    nulos = matches.isna().sum()
    nulos = nulos[nulos > 0].sort_values(ascending=False)
    out.append(nulos.to_string() if not nulos.empty else "  (ninguna)")

    out.append("\nCobertura de xG por temporada (fact_player_gw):")
    xg = players.groupby("season")["expected_goals"].apply(lambda s: s.notna().mean())
    out.append(xg.map(lambda v: f"{v:.1%}").to_string())
    out.append(
        f"\n  La ventana arranca en {CFG.ingest_from} justamente porque antes de\n"
        f"  2022-23 el dataset de FPL no trae xG/xA/xGC."
    )
    return "\n".join(out)


def distribucion_1x2(matches: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    out = [_seccion("2. DISTRIBUCIÓN DEL TARGET 1X2")]

    dist = (
        matches.groupby("season")["target_1x2"]
        .value_counts(normalize=True)
        .unstack()
        .reindex(columns=baselines.CLASES)
    )
    out.append(dist.map(lambda v: f"{v:.1%}").to_string())

    glob = matches["target_1x2"].value_counts(normalize=True).reindex(baselines.CLASES)
    out.append(f"\nGlobal: " + " | ".join(f"{k} {v:.1%}" for k, v in glob.items()))
    out.append(
        f"\n  La ventaja de local ({glob['home']:.1%}) es el baseline trivial: "
        f"predecir\n  siempre 'gana el local' acierta esa fracción de las veces."
    )
    return "\n".join(out), dist


def reporte_baselines(matches: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    out = [_seccion("3. BASELINES")]

    train_seasons = CFG.seasons_for_training()
    test_season = CFG.holdout_season

    train = matches[matches["season"].isin(train_seasons)]
    test = matches[matches["season"] == test_season]

    out.append(f"Train : {train_seasons}  ({len(train):,} partidos)")
    out.append(f"Test  : {test_season}  ({len(test):,} partidos)\n")

    if test.empty:
        out.append("  Sin datos de test — no se pueden calcular baselines.")
        return "\n".join(out), pd.DataFrame()

    tabla = baselines.calcular_todos(train, test)
    out.append(tabla.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    cuotas = baselines.baseline_cuotas(test)
    if "overround_medio" in cuotas:
        out.append(
            f"\n  Overround medio de las casas: {cuotas['overround_medio']:.3f} "
            f"({(cuotas['overround_medio'] - 1) * 100:.1f}% de margen)."
        )

    triv = tabla.loc[tabla["nombre"] == "siempre local", "accuracy"].iloc[0]
    duro = tabla.loc[tabla["nombre"].str.startswith("cuotas"), "accuracy"]
    if not duro.empty:
        out.append(
            f"\n  Lectura: el modelo tiene que superar {triv:.1%} para valer algo, y\n"
            f"  {duro.iloc[0]:.1%} sería igualar al mercado. Quedar en el medio es el\n"
            f"  resultado esperable y no es un problema para el trabajo."
        )

    por_temp = baselines.por_temporada(matches)
    out.append("\nMismos baselines, temporada por temporada:")
    out.append(por_temp.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    out.append(
        "\n  El empate casi nunca es el resultado más probable para el mercado, así\n"
        "  que una temporada con muchos empates hunde la accuracy de las cuotas sin\n"
        "  que el mercado haya estado peor informado. Conviene mirar la serie, no una\n"
        "  sola temporada, antes de fijar el techo."
    )
    return "\n".join(out), tabla


def drift(matches: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    out = [_seccion("4. DRIFT ENTRE TEMPORADAS")]
    out.append(
        "Cuánto se mueven las distribuciones de una temporada a otra. Es el\n"
        "argumento directo para el monitoreo: si las features se corren solas,\n"
        "el modelo se degrada aunque nadie toque nada.\n"
    )

    m = matches.copy()
    m["goles_totales"] = m["home_goals"] + m["away_goals"]
    m["prob_local_implicita"] = 1 / m["odds_avg_close_home"]

    metricas = {
        "goles del local": "home_goals",
        "goles del visitante": "away_goals",
        "goles totales": "goles_totales",
        "tiros del local": "home_shots",
        "tiros al arco del local": "home_shots_target",
        "córners del local": "home_corners",
        "prob. implícita del local": "prob_local_implicita",
    }

    tabla = pd.DataFrame({
        nombre: m.groupby("season")[col].mean()
        for nombre, col in metricas.items() if col in m.columns
    })
    out.append(tabla.to_string(float_format=lambda v: f"{v:.3f}"))

    variacion = (tabla.max() - tabla.min()) / tabla.mean()
    out.append("\nVariación relativa (max-min)/media entre temporadas:")
    out.append(variacion.sort_values(ascending=False).map(lambda v: f"{v:.1%}").to_string())
    return "\n".join(out), tabla


def cold_start(matches: pd.DataFrame, dim: pd.DataFrame) -> str:
    out = [_seccion("5. COLD-START DE EQUIPOS ASCENDIDOS")]
    out.append(
        "Un equipo recién ascendido no tiene historial de Premier del cual sacar\n"
        "forma reciente. Es un caso real que hay que manejar en las features.\n"
    )

    partidos_por_equipo = (
        pd.concat([
            matches[["season", "home_short"]].rename(columns={"home_short": "short_name"}),
            matches[["season", "away_short"]].rename(columns={"away_short": "short_name"}),
        ])
        .groupby("short_name").size()
    )

    actual = dim[dim["season"] == CFG.current_season][["short_name", "team_name"]].copy()
    actual["partidos_en_la_ventana"] = actual["short_name"].map(partidos_por_equipo).fillna(0).astype(int)
    actual = actual.sort_values("partidos_en_la_ventana")

    out.append(f"Plantel de {CFG.current_season} y su historial dentro de la ventana ingestada:")
    out.append(actual.to_string(index=False))

    sin_historial = actual[actual["partidos_en_la_ventana"] == 0]
    if not sin_historial.empty:
        out.append(
            f"\n  {len(sin_historial)} de {len(actual)} equipos no tienen NINGÚN partido "
            f"en la ventana:\n  {', '.join(sin_historial['team_name'])}."
            f"\n  Para ellos hay que definir un prior explícito (p.ej. el promedio de los\n"
            f"  ascendidos históricos) en vez de dejar nulos que el modelo interprete mal."
        )
    return "\n".join(out)


def tamano_del_split(matches: pd.DataFrame) -> str:
    out = [_seccion("6. ¿ALCANZAN LOS DATOS?")]

    train = matches[matches["season"].isin(CFG.seasons_for_training())]
    test = matches[matches["season"] == CFG.holdout_season]

    out.append(f"Train : {len(train):,} partidos  ({', '.join(CFG.seasons_for_training())})")
    out.append(f"Test  : {len(test):,} partidos  ({CFG.holdout_season})")
    out.append(f"Total : {len(matches):,} partidos\n")

    # Con 3 clases, una regla de oro razonable es ~10 observaciones por parámetro.
    # Una logística multinomial con k features estima 2*(k+1) parámetros.
    for k in (10, 20, 30, 50):
        params = 2 * (k + 1)
        ratio = len(train) / params
        marca = "holgado" if ratio > 20 else ("aceptable" if ratio > 10 else "ajustado")
        out.append(f"  {k:>2} features -> {params:>3} parámetros -> {ratio:>5.1f} obs/parám  ({marca})")

    out.append(
        "\n  Con esta ventana conviene empezar por logística multinomial regularizada\n"
        "  y un gradient boosting chico. Una red profunda no tiene con qué."
    )
    if len(test) < 400:
        out.append(
            f"\n  El test tiene {len(test)} partidos: el error estándar de la accuracy\n"
            f"  ronda ±{(0.5 * (1 - 0.5) / len(test)) ** 0.5 * 1.96 * 100:.1f} puntos. "
            f"Diferencias chicas entre modelos no van a\n  ser distinguibles — conviene "
            f"validación temporal con varios cortes."
        )
    return "\n".join(out)


# --------------------------------------------------------------------------- #


def graficos(dist: pd.DataFrame, drift_tabla: pd.DataFrame, matches: pd.DataFrame) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    dist.plot(kind="bar", stacked=True, ax=axes[0],
              color=["#2a9d8f", "#e9c46a", "#e76f51"])
    axes[0].set_title("Distribución 1X2 por temporada")
    axes[0].set_ylabel("proporción")
    axes[0].set_xlabel("")
    axes[0].legend(title="", fontsize=8)
    axes[0].tick_params(axis="x", rotation=45)

    normalizada = drift_tabla / drift_tabla.iloc[0]
    normalizada.plot(ax=axes[1], marker="o", legend=False)
    axes[1].axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
    axes[1].set_title("Drift: métricas relativas a la 1ra temporada")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=45)

    probs = 1 / matches["odds_avg_close_home"].dropna()
    axes[2].hist(probs, bins=40, color="#264653")
    axes[2].set_title("Prob. implícita de victoria local (cuota de cierre)")
    axes[2].set_xlabel("probabilidad")

    fig.tight_layout()
    fig.savefig(OUTPUT / "eda_resumen.png", dpi=130)
    plt.close(fig)
    log.info("gráficos -> %s", OUTPUT / "eda_resumen.png")


def main() -> None:
    setup(CFG.log_level, CFG.log_format)

    matches = read_table("fact_match")
    players = read_table("fact_player_gw")
    fixtures = read_table("fact_fixture")
    dim = read_table("dim_team")

    partes = [
        cobertura(matches, players, fixtures),
    ]
    txt_dist, dist = distribucion_1x2(matches)
    partes.append(txt_dist)

    txt_base, tabla_base = reporte_baselines(matches)
    partes.append(txt_base)

    txt_drift, drift_tabla = drift(matches)
    partes.append(txt_drift)

    partes.append(cold_start(matches, dim))
    partes.append(tamano_del_split(matches))

    reporte = "\n".join(partes)
    print(reporte)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "eda_reporte.txt").write_text(reporte, encoding="utf-8")
    log.info("reporte -> %s", OUTPUT / "eda_reporte.txt")

    if not tabla_base.empty:
        tabla_base.to_csv(OUTPUT / "baselines.csv", index=False)
        log.info("baselines -> %s", OUTPUT / "baselines.csv")

    graficos(dist, drift_tabla, matches)


if __name__ == "__main__":
    main()
