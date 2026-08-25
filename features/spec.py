"""Contrato de la capa Gold: qué columnas existen, cómo se calculan y cuáles son features.

Este módulo es la ÚNICA fuente de verdad del esquema. Lo consumen `features/gold_tp.py`
para ordenar y validar la tabla, `training/dataset.py` para armar la matriz X, y el
serving para verificar que recibe las columnas en el mismo orden en que se entrenó.

`docs/FEATURES.md` se GENERA desde acá (`python -m features.spec --docs`), nunca se
escribe a mano: un diccionario de datos que se mantiene aparte del código queda
desfasado en la segunda semana. Hay un test que regenera y compara.

Estructura de los nombres, una sola regla:

    {lado}_{estadistica}_{ventana}

    lado      : local | visita          el equipo local y el visitante de ESE partido
    ventana   : u3 | u5                 media de los últimos N partidos del equipo,
                                        CRUZANDO el borde de temporada
                u5_temp                 igual pero sólo con partidos de la temporada actual
                cond_u5                 igual pero sólo con partidos en la misma condición
                                        (el local, de local; el visitante, de visitante)
                camp                    acumulado de la temporada al momento del corte

Ejemplos: `local_pts_def_u5` - `visita_xg_u3` - `local_pos_tabla_camp` - `visita_pts_u5_temp`
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from common.config import CFG, PROJECT_ROOT

FEATURE_SET_VERSION = "v2"   # v2 agrega Elo y estado del equipo (14 columnas)

LADOS = ("local", "visita")

DOCS_PATH = PROJECT_ROOT / "docs" / "FEATURES.md"


@dataclass(frozen=True)
class Feature:
    """Una columna de Gold, con todo lo necesario para documentarla."""

    nombre: str
    grupo: str
    fuente: str
    formula: str
    lado: str = "—"
    ventana: str = "—"


@dataclass(frozen=True)
class StatBase:
    nombre: str
    formula: str
    fuente: str


# ---------------------------------------------------------------------------
# Las 16 estadísticas base — grano equipo x partido
# ---------------------------------------------------------------------------
# Se calculan una vez por equipo y por partido sobre la tabla larga de 3.040 filas
# (4 temporadas x 380 partidos x 2 equipos). TODAS las features de forma salen de acá.

BASE_STATS: tuple[StatBase, ...] = (
    StatBase("pts", "3 si ganó, 1 si empató, 0 si perdió", "fact_match"),
    StatBase("gf", "goles a favor", "fact_match"),
    StatBase("gc", "goles en contra", "fact_match"),
    StatBase("tiros", "tiros totales", "fact_match"),
    StatBase("tiros_arco", "tiros al arco", "fact_match"),
    StatBase("corners", "córners", "fact_match"),
    StatBase("faltas", "faltas cometidas", "fact_match"),
    StatBase("tarjetas", "amarillas + 2 x rojas", "fact_match"),
    StatBase("xg", "suma de expected_goals de los jugadores del equipo", "fact_player_gw"),
    StatBase(
        "xgc",
        "el xg DEL RIVAL en ese mismo fixture. NO la suma de expected_goals_conceded, "
        "que se cuenta una vez por cada jugador del plantel e infla el valor x11 "
        "(medido: media 15,75 contra 1,47 goles concedidos reales)",
        "fact_player_gw",
    ),
    StatBase("xa", "suma de expected_assists", "fact_player_gw"),
    StatBase("pts_arq", "suma de total_points de los jugadores con position == GK", "fact_player_gw"),
    StatBase("pts_def", "suma de total_points de position == DEF", "fact_player_gw"),
    StatBase("pts_med", "suma de total_points de position == MID", "fact_player_gw"),
    StatBase("pts_del", "suma de total_points de position == FWD", "fact_player_gw"),
    StatBase("n_jugadores", "cantidad de jugadores con minutes > 0", "fact_player_gw"),
)

BASE_BY_NAME = {s.nombre: s for s in BASE_STATS}

# `dg` no es una estadística base: se deriva en la tabla larga como gf - gc.
DG = StatBase("dg", "diferencia de gol del partido (gf - gc)", "fact_match")

# Subconjuntos que usa cada grupo.
STATS_TEMP = ("pts", "gf", "gc", "xg", "xgc", "pts_def", "pts_med", "tiros_arco")
STATS_COND = ("pts", "gf", "gc", "dg", "xg", "xgc")
STATS_CAMP = (
    ("pts", "puntos acumulados en la temporada al momento del corte"),
    ("pj", "partidos jugados en la temporada"),
    ("ppp", "puntos por partido (pts / pj): el acumulado normalizado, comparable entre "
            "la fecha 3 y la 30"),
    ("gf", "goles a favor acumulados"),
    ("gc", "goles en contra acumulados"),
    ("dg", "diferencia de gol acumulada"),
    ("pos_tabla", "posición en la tabla, rankeando por (pts, dg, gf) dentro de "
                  "(temporada, corte): el criterio de desempate real de la Premier"),
)

NOTA_VENTANA = (
    "media de los últimos {n} partidos del equipo, min_periods=1, anclada al corte con "
    "merge_asof (nunca shift: shift cuenta partidos, merge_asof cuenta tiempo)"
)


def _forma() -> list[Feature]:
    """16 estadísticas x ventanas u3/u5 x 2 lados = 64. Cruzan el borde de temporada."""
    out = []
    for lado in LADOS:
        for n in CFG.rolling_windows:
            for s in BASE_STATS:
                out.append(Feature(
                    nombre=f"{lado}_{s.nombre}_u{n}",
                    grupo="Forma reciente",
                    fuente=s.fuente,
                    formula=f"{s.formula}. {NOTA_VENTANA.format(n=n)}. Cruza el borde de "
                            f"temporada, así que siempre hay dato, incluso en la GW1",
                    lado=lado,
                    ventana=f"u{n}",
                ))
    return out


def _forma_temp() -> list[Feature]:
    """8 principales x u5 restringida a la temporada actual x 2 lados = 16."""
    out = []
    for lado in LADOS:
        for nombre in STATS_TEMP:
            s = BASE_BY_NAME[nombre]
            out.append(Feature(
                nombre=f"{lado}_{nombre}_u5_temp",
                grupo="Forma intra-temporada",
                fuente=s.fuente,
                formula=f"{s.formula}. {NOTA_VENTANA.format(n=5)}, pero SÓLO con partidos "
                        f"de la temporada actual. NaN en la fecha 1: es el contrapunto a "
                        f"la ventana cruzada, porque acá el plantel sí es el de hoy",
                lado=lado,
                ventana="u5_temp",
            ))
    return out


def _forma_cond() -> list[Feature]:
    """6 estadísticas x u5 restringida a la condición x 2 lados = 12."""
    out = []
    for lado in LADOS:
        for nombre in STATS_COND:
            s = DG if nombre == "dg" else BASE_BY_NAME[nombre]
            cond = "de local" if lado == "local" else "de visitante"
            out.append(Feature(
                nombre=f"{lado}_{nombre}_cond_u5",
                grupo="Forma según condición",
                fuente=s.fuente,
                formula=f"{s.formula}. {NOTA_VENTANA.format(n=5)}, restringida a los "
                        f"partidos que el equipo jugó {cond}. Captura la ventaja de "
                        f"localía propia de cada equipo",
                lado=lado,
                ventana="cond_u5",
            ))
    return out


def _campeonato() -> list[Feature]:
    """7 acumulados de temporada x 2 lados = 14."""
    out = []
    for lado in LADOS:
        for nombre, desc in STATS_CAMP:
            out.append(Feature(
                nombre=f"{lado}_{nombre}_camp",
                grupo="Puntaje de campeonato",
                fuente="fact_match",
                formula=f"{desc}. expanding() en lugar de rolling(), mismo anclaje al "
                        f"corte por merge_asof",
                lado=lado,
                ventana="camp",
            ))
    return out


# El head-to-head NO es simétrico, así que se explicita la perspectiva. Los dos `pts` no
# son deducibles uno del otro: un empate da 1-1 (suma 2) y una victoria 3-0 (suma 3).
H2H_COLS = (
    ("pts_local", "puntos que sacó el equipo que HOY es local, en esos partidos"),
    ("pts_visita", "puntos que sacó el equipo que HOY es visitante, en esos partidos"),
    ("gf_local", "goles que hizo el equipo que hoy es local"),
    ("gc_local", "goles que recibió el equipo que hoy es local"),
    ("n", "cuántos enfrentamientos previos hay"),
)


def _h2h() -> list[Feature]:
    """Historial entre los dos equipos, en dos variantes = 10."""
    out = []
    for nombre, desc in H2H_COLS:
        out.append(Feature(
            nombre=f"h2h_{nombre}",
            grupo="Head-to-head",
            fuente="fact_match",
            formula=f"{desc}. Media sobre TODOS los enfrentamientos previos entre estos "
                    f"dos equipos anteriores al corte, en cualquier condición. Medido: "
                    f"media 2,83 antecedentes en la ventana, 5,15 en el holdout",
            ventana="h2h",
        ))
    for nombre, desc in H2H_COLS:
        out.append(Feature(
            nombre=f"h2h_cond_{nombre}",
            grupo="Head-to-head",
            fuente="fact_match",
            formula=f"{desc}. Igual que h2h_{nombre} pero SÓLO los partidos en que este "
                    f"mismo equipo fue local contra este mismo rival. Flaco por "
                    f"construcción: con 4 temporadas el máximo posible es 3 (media "
                    f"medida 1,16; 2,33 en el holdout)",
            ventana="h2h_cond",
        ))
    return out


def _continuidad() -> list[Feature]:
    """Cuánto vale la historia del equipo para el plantel de hoy = 2."""
    return [Feature(
        nombre=f"{lado}_continuidad_plantel_u5",
        grupo="Continuidad de plantel",
        fuente="fact_player_gw",
        formula="proporción de los minutos de los últimos 5 partidos que jugaron "
                "futbolistas que también jugaron el partido MÁS RECIENTE del equipo. "
                "100% leak-free: sólo mira partidos pasados. Se desploma cuando el "
                "plantel se renovó, que es justo cuando la forma pasada deja de "
                "representar al equipo actual. Medido: entre temporadas rota ~40% de "
                "los minutos",
        lado=lado,
        ventana="u5",
    ) for lado in LADOS]


CONTEXTO_POR_LADO = (
    ("n_hist", "cantidad de partidos previos disponibles para el equipo. Le dice al "
               "modelo cuán parcial es la media móvil"),
    ("dias_descanso", "días desde el partido anterior del equipo"),
    ("es_ascendido", "el equipo no estaba en la Premier la temporada anterior. Derivado "
                     "así, NO desde dim_team.promoted, que en realidad significa "
                     "'primera temporada dentro de la ventana ingestada' y marca 1 "
                     "equipo en 2024-25 cuando ascendieron 3"),
    ("mins_hhi", "índice de Herfindahl sobre el reparto de minutos en los últimos 5 "
                 "partidos: mide rotación de plantel. Reemplaza al grupo "
                 "'disponibilidad' del canvas, que no se puede construir porque la API "
                 "de FPL no sirve el `status` histórico"),
)


def _contexto() -> list[Feature]:
    """Metadatos que le dicen al modelo cuánto confiar en el resto = 10."""
    out = [Feature(nombre=f"{lado}_{n}", grupo="Contexto", fuente="derivada",
                   formula=f, lado=lado)
           for lado in LADOS for n, f in CONTEXTO_POR_LADO]
    out.append(Feature("gameweek", "Contexto", "fact_fixture",
                       "número de fecha; captura el efecto del momento de la temporada"))
    out.append(Feature("xg_available", "Contexto", "derivada",
                       "falso donde el xG viene hardcodeado en cero (2022-23 hasta la "
                       "GW15, los 20 equipos). Explica el NaN en vez de dejar que el "
                       "modelo lo interprete solo"))
    return out


ESTADO_POR_LADO = (
    ("elo", "rating Elo del equipo despues de su ultimo partido antes del corte. K=20, "
            "ventaja de localia 65 puntos, margen de victoria atenuado por logaritmo, y "
            "regresion del 25% a la media al cambiar de temporada. Resuelve lo que las "
            "medias moviles no pueden: ponderar cada resultado segun contra quien fue. "
            "Validado -- al cierre de 2024-25 da Liverpool/Arsenal/City arriba y "
            "Southampton/Ipswich/Leicester abajo, que son los tres descendidos"),
    ("elo_delta_u3", "cuanto GANO o PERDIO de rating Elo en sus ultimos 3 partidos. Es "
                     "informacion distinta del nivel y distinta de `racha`: `elo` dice "
                     "donde esta el equipo, `racha` compara puntos contra su propio "
                     "promedio tratando igual a todos los rivales, y esto dice hacia "
                     "donde va PONDERADO POR CONTRA QUIEN. Sumar 6 puntos contra dos "
                     "rivales de arriba sube mucho mas que sumarlos contra dos de abajo"),
    ("elo_delta_u5", "lo mismo sobre 5 partidos"),
    ("elo_delta_u10", "lo mismo sobre 10: menos ruidoso. Junto con `elo` le permite al "
                      "modelo distinguir cuatro situaciones que hoy se le mezclan: grande "
                      "en alza, grande en caida, chico en alza y chico en caida"),
    ("tiros_conc_u5", "tiros que le concedieron al equipo en sus ultimos 5 partidos. Las "
                      "ventanas de `tiros` miden lo que el equipo genera; esta mide lo "
                      "que regala, que es informacion distinta"),
    ("tiros_arco_conc_u5", "tiros al arco concedidos en los ultimos 5"),
    ("xg_diff_u5", "goles menos xG en los ultimos 5: suerte de definicion. Es fuertemente "
                   "reversible a la media, asi que un valor alto anticipa una caida. Ni "
                   "`gf_u5` ni `xg_u5` capturan esto por separado"),
    ("xgc_diff_u5", "goles recibidos menos xG concedido en los ultimos 5: lo mismo del "
                    "lado defensivo, incluye el rendimiento del arquero"),
    ("xg_por_tiro_u5", "xG dividido por tiros, promediado sobre los ultimos 5: la "
                       "CALIDAD de las situaciones, no la cantidad. 2,0 de xG en 3 "
                       "ocasiones claras y 2,0 en 20 remates de afuera son cosas "
                       "distintas y predicen distinto; el xG agregado no las separa. Es "
                       "la aproximacion gratis al xG a nivel tiro que daria Understat"),
    ("xgc_por_tiro_u5", "lo mismo del lado defensivo: que tan claras son las situaciones "
                        "que concede. Un equipo puede conceder muchos remates lejanos "
                        "(bajo riesgo) o pocas ocasiones claras (alto riesgo)"),
    ("prop_tiros_arco_u5", "proporcion de tiros que van al arco: punteria y seleccion "
                           "de remate"),
    ("prop_tiros_arco_conc_u5", "lo mismo entre los tiros que concede"),
    ("partidos_7d", "partidos de Premier jugados en los 7 dias previos. Detecta el "
                    "'jugo entre semana', que es lo mas cerca que se puede estar de "
                    "identificar un compromiso de copa o de Europa sin el calendario de "
                    "esas competencias"),
    ("partidos_14d", "partidos jugados en los 14 dias previos: la carga de dos semanas"),
    ("partidos_21d", "partidos en 21 dias: la carga acumulada. No es lo mismo un pico "
                     "aislado que tres semanas seguidas de partido cada tres dias"),
    ("racha", "puntos de los ultimos 3 partidos menos el promedio de lo que va de la "
              "temporada. Captura si el equipo esta por encima o por debajo de su nivel"),
    ("sorpresa_u5", "cuanto se apartaron los ultimos 5 resultados de lo que el Elo "
                    "esperaba: |real - esperado| promediado. Mide que tan IMPREDECIBLE "
                    "viene siendo el equipo, no en que direccion. Es informacion sobre la "
                    "confiabilidad de la prediccion. No usa las predicciones del modelo "
                    "-- eso seria un bucle de realimentacion -- sino la expectativa del "
                    "Elo, que sale solo de resultados pasados"),
    ("sorpresa_u10", "lo mismo sobre 10 partidos: menos ruidoso, mas estructural. Medido "
                     "en 2025-26, los mas impredecibles fueron CHE, NEW y AVL; los mas "
                     "predecibles BUR y BRE (ser consistentemente malo tambien es "
                     "predecible)"),
)


def _estado() -> list[Feature]:
    """Elo y estado del equipo: 7 x 2 lados = 14."""
    return [Feature(nombre=f"{lado}_{n}", grupo="Elo y estado", fuente="derivada",
                    formula=f, lado=lado)
            for lado in LADOS for n, f in ESTADO_POR_LADO]


def _dificultad() -> list[Feature]:
    return [
        Feature("fdr_local", "Dificultad", "fact_fixture",
                "FDR de FPL para el local (team_h_difficulty), escala 1-5", lado="local"),
        Feature("fdr_visita", "Dificultad", "fact_fixture",
                "FDR de FPL para el visitante (team_a_difficulty), escala 1-5", lado="visita"),
        Feature("fdr_dif", "Dificultad", "derivada", "fdr_local - fdr_visita"),
    ]


DIFERENCIALES = (
    "pts_u5", "gf_u5", "gc_u5", "xg_u5", "xgc_u5", "pts_def_u5", "pts_med_u5",
    "pts_camp", "pos_tabla_camp", "ppp_camp", "dias_descanso", "n_hist",
    "elo", "elo_delta_u5", "racha", "sorpresa_u10", "xg_por_tiro_u5",
)


def _diferenciales() -> list[Feature]:
    """12 restas explícitas. El árbol podría derivarlas, pero con 1.140 filas ayuda."""
    return [Feature(
        nombre=f"dif_{c}",
        grupo="Diferenciales",
        fuente="derivada",
        formula=f"local_{c} - visita_{c}",
    ) for c in DIFERENCIALES]


def _build() -> list[Feature]:
    out: list[Feature] = []
    for fn in (_forma, _forma_temp, _forma_cond, _campeonato, _h2h,
               _continuidad, _estado, _contexto, _dificultad, _diferenciales):
        out.extend(fn())
    nombres = [f.nombre for f in out]
    dupes = {n for n in nombres if nombres.count(n) > 1}
    if dupes:
        raise ValueError(f"Features duplicadas en el spec: {sorted(dupes)}")
    return out


FEATURE_SPECS: list[Feature] = _build()
FEATURES: list[str] = [f.nombre for f in FEATURE_SPECS]

# ---------------------------------------------------------------------------
# Columnas que NO son features
# ---------------------------------------------------------------------------

# `fixture_id` NO es único entre temporadas: FPL numera 1..380 cada año. La clave del
# partido es el par (season, fixture_id), y así se usa en todos los merges.
CLAVE_PARTIDO = ["season", "fixture_id"]

# `gameweek` no está acá porque ya es feature (grupo Contexto): captura el momento de la
# temporada, y duplicarla en las claves la metería dos veces en GOLD_COLUMNS.
CLAVES = ["season", "fixture_id", "match_date", "kickoff_time",
          "home_short", "away_short"]

TARGET = ["target_1x2", "home_goals", "away_goals", "goal_diff"]

AUDITORIA = ["corte", "hist_kickoff_local", "hist_kickoff_visita", "split",
             "feature_set_version", "gold_built_at"]

# Las cuotas viven en Gold porque la simulación de ROI las necesita, pero NUNCA son
# feature: si el modelo las usa aprende a copiar al mercado y el valor esperado da ~0
# por construcción, así que jamás encontraría una apuesta con valor. Las dos
# estimaciones tienen que ser independientes. Hay un test que lo verifica.
MERCADO = ["odds_avg_close_home", "odds_avg_close_draw", "odds_avg_close_away",
           "p_mercado_home", "p_mercado_draw", "p_mercado_away"]

GOLD_COLUMNS = CLAVES + TARGET + FEATURES + AUDITORIA + MERCADO

NO_FEATURES = set(CLAVES) | set(TARGET) | set(AUDITORIA) | set(MERCADO)


def grupos() -> dict[str, list[Feature]]:
    out: dict[str, list[Feature]] = {}
    for f in FEATURE_SPECS:
        out.setdefault(f.grupo, []).append(f)
    return out


# ---------------------------------------------------------------------------
# Documentación generada
# ---------------------------------------------------------------------------

def _esc(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


ENCABEZADO = """# Diccionario de features — Gold-TP

> **Archivo generado.** No lo edites a mano: se produce con `python -m features.spec --docs`
> a partir de `features/spec.py`, y hay un test (`test_docs_features_esta_sincronizado_con_spec`)
> que falla si quedan desfasados. Para cambiar una feature, tocá el spec.

## Cómo leer los nombres

```
{lado}_{estadística}_{ventana}

lado     local | visita        el equipo local y el visitante de ESE partido
ventana  u3 | u5               media de los últimos N partidos del equipo,
                               CRUZANDO el borde de temporada
         u5_temp               igual, pero sólo partidos de la temporada actual
         cond_u5               igual, pero sólo partidos en la misma condición
         camp                  acumulado de la temporada al momento del corte
```

**Todo es del equipo, mirando hacia atrás.** Ningún número describe el partido a predecir.
La construcción pasa por una tabla intermedia con grano equipo × partido (3.040 filas),
donde cada equipo tiene su propia fila; las ventanas se calculan ahí y recién al final se
pivotea a ancho, pegando la historia del local con prefijo `local_` y la del visitante con
`visita_`.

## La regla anti-leakage

```
corte(partido) = min(kickoff_time) de todos los partidos de su (temporada, gameweek)
```

Toda feature usa **únicamente partidos terminados antes del corte**. El corte es el inicio
de la fecha, así que todos los partidos de una misma gameweek se predicen con la misma
información — incluidas las 85 dobles fechas, donde un equipo juega dos veces en la misma
gameweek.

El mecanismo es `merge_asof`, no `shift(1)`: **shift cuenta partidos, merge_asof cuenta
tiempo.** Con shift, el segundo partido de una doble fecha vería el resultado del primero,
que se jugó después del corte.

Las columnas `hist_kickoff_local` / `hist_kickoff_visita` guardan el kickoff del último
partido efectivamente usado: son la **prueba auditable** de que no se miró el futuro, y hay
un assert que las verifica antes de escribir la tabla.

## Pases de jugadores y cambios de temporada

`fact_player_gw` se agrega primero a `(temporada, fixture, equipo)` y recién después se
calculan las ventanas. Como el primer paso agrega **por partido**, cada jugador ya quedó
atribuido al equipo con el que efectivamente jugó. Si un delantero pasa de A a B en enero,
sus goles de agosto quedan para siempre en el historial de A y los de febrero suman al de B:
**no hay doble conteo ni puntos huérfanos**, y un jugador nuevo suma a su equipo desde su
primer partido sin necesitar pasado.

El límite real: la forma pasada incluye gente que ya se fue. Medido, entre temporadas sólo
el **61,3 % / 66,0 % / 57,3 %** de los minutos los juega gente que ya estaba en el mismo
equipo. Por eso existen `continuidad_plantel_u5`, las ventanas `u5_temp` y `pj_camp`.

> ⚠️ **`fpl_player_id` NO es estable entre temporadas.** El 90 % de los ids (725 de 808)
> apunta a un futbolista distinto según el año: el id 1 es Cédric Soares en 2022-23 y David
> Raya en 2025-26. La clave entre temporadas es `player_name`.
"""


def render_docs() -> str:
    """Genera docs/FEATURES.md a partir del spec. Nunca se escribe a mano."""
    L = [ENCABEZADO, "", "## Resumen", "",
         f"- **Versión del feature set:** `{FEATURE_SET_VERSION}`",
         f"- **Features:** {len(FEATURES)}",
         f"- **Columnas totales:** {len(GOLD_COLUMNS)} ({len(FEATURES)} features + "
         f"{len(GOLD_COLUMNS) - len(FEATURES)} no-features)",
         "- **Grano:** un partido = una fila. 1.520 filas (4 temporadas × 380).",
         "", "| Grupo | Columnas |", "|---|---|"]
    for g, feats in grupos().items():
        L.append(f"| {g} | {len(feats)} |")
    L += [f"| **TOTAL DE FEATURES** | **{len(FEATURES)}** |", ""]

    L += ["## Las 16 estadísticas base", "",
          "Se calculan una vez por equipo y por partido, sobre la tabla larga de 3.040 "
          "filas. **Todas las features de forma salen de estas 16.**", "",
          "| # | Nombre | Fórmula | Fuente |", "|---|---|---|---|"]
    for i, s in enumerate(BASE_STATS, 1):
        L.append(f"| {i} | `{s.nombre}` | {_esc(s.formula)} | `{s.fuente}` |")
    L += ["", f"Más `dg`, derivada en la tabla larga: {DG.formula}.", "",
          "> **`total_points` está en `banned_columns` y esto no la viola.** Está "
          "prohibida como columna cruda del partido a predecir, donde es el resultado. "
          "Una media de partidos **anteriores** es una feature de forma, que es "
          "literalmente lo que pide el bloque 4 del canvas. Lo garantiza el `merge_asof`, "
          "y los nombres finales (`local_pts_def_u5`) no colisionan con la lista de "
          "prohibidas.", ""]

    L += ["## Features, una por una", ""]
    for g, feats in grupos().items():
        L += [f"### {g} — {len(feats)} columnas", "",
              "| Columna | Lado | Ventana | Fuente | Cálculo |", "|---|---|---|---|---|"]
        for f in feats:
            L.append(f"| `{f.nombre}` | {f.lado} | {f.ventana} | `{f.fuente}` | "
                     f"{_esc(f.formula)} |")
        L.append("")

    L += ["## Columnas que NO son features", "",
          "| Grupo | Columnas | Por qué están en Gold |", "|---|---|---|",
          f"| Claves | {', '.join(f'`{c}`' for c in CLAVES)} | identificación del partido |",
          f"| Target | {', '.join(f'`{c}`' for c in TARGET)} | `target_1x2` es la "
          "etiqueta; los goles se guardan para no cerrarle la puerta a un modelo Poisson |",
          f"| Auditoría | {', '.join(f'`{c}`' for c in AUDITORIA)} | `hist_kickoff_*` es "
          "la prueba de que no se miró el futuro |",
          f"| Mercado | {', '.join(f'`{c}`' for c in MERCADO)} | las necesita la "
          "simulación de ROI. **Nunca son feature**: si el modelo copia al mercado, el "
          "valor esperado da ~0 por construcción y jamás encontraría una apuesta con "
          "valor |", ""]
    return "\n".join(L) + "\n"


def write_docs(path: Path | None = None) -> Path:
    path = path or DOCS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_docs(), encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Contrato de la capa Gold.")
    ap.add_argument("--docs", action="store_true", help="regenera docs/FEATURES.md")
    args = ap.parse_args()
    if args.docs:
        p = write_docs()
        print(f"Escrito {p}")
        print(f"{len(FEATURES)} features, {len(GOLD_COLUMNS)} columnas.")
        return
    print(f"feature_set_version : {FEATURE_SET_VERSION}")
    print(f"features            : {len(FEATURES)}")
    print(f"columnas de Gold    : {len(GOLD_COLUMNS)}")
    print()
    for g, feats in grupos().items():
        print(f"  {g:26s} {len(feats):3d}")


if __name__ == "__main__":
    main()
