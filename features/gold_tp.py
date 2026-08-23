"""Gold-TP: una fila por partido, lista para entrenar.

Orquesta todo lo demás y —esto es lo importante— **corre los controles anti-leakage antes
de escribir**, no en los tests. La regla del proyecto es que el pipeline falle en el
momento de generar el dato malo; un test que corre después ya llegó tarde para producción.

Uso:

    python -m features.gold_tp
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common.config import CFG, PROJECT_ROOT, utc_stamp
from common.logging_setup import get_logger, setup
from common.storage import read_table, write_table
from eda.baselines import odds_a_probabilidades
from features import cold_start, h2h, player_agg, spec, team_form as tf
from transform import leakage

log = get_logger(__name__)

TABLA = "gold_tp_match"
CLAVE = spec.CLAVE_PARTIDO

ODDS_MERCADO = ("odds_avg_close_home", "odds_avg_close_draw", "odds_avg_close_away")


# ---------------------------------------------------------------------------
# Ensamblado
# ---------------------------------------------------------------------------

def _objetivos(largo: pd.DataFrame, cortes: pd.DataFrame) -> pd.DataFrame:
    """Un objetivo por partido: sus claves y su corte."""
    loc = largo[largo["es_local"]]
    obj = loc[["season", "gameweek", "fixture_id", "kickoff_time",
               "team_short", "rival_short"]].rename(
        columns={"team_short": "home_short", "rival_short": "away_short"})
    obj = obj.merge(cortes, on=["season", "gameweek"], how="left", validate="many_to_one")
    if obj["corte"].isna().any():
        raise ValueError("Hay partidos sin corte: falta su gameweek en fact_fixture.")
    return obj.reset_index(drop=True)


def _objetivos_por_lado(obj: pd.DataFrame) -> pd.DataFrame:
    """Dos filas por partido —local y visitante— para pegarles su propia historia."""
    partes = []
    for lado, yo in (("local", "home_short"), ("visita", "away_short")):
        d = obj[CLAVE + ["corte", "kickoff_time"]].copy()
        d["lado"] = lado
        d["team_short"] = obj[yo].to_numpy()
        d["es_local"] = lado == "local"
        partes.append(d)
    return pd.concat(partes, ignore_index=True)


def _pegar_bloque(obj_lado: pd.DataFrame, hist: pd.DataFrame, por: list[str],
                  cols: list[str], con_kickoff: bool = False) -> pd.DataFrame:
    """Un `merge_asof` y de vuelta a la clave. Nunca se alinea por posición."""
    res = tf.pegar_asof(obj_lado, hist, por, cols)
    quedan = CLAVE + ["lado"] + cols + (["hist_kickoff"] if con_kickoff else [])
    return res[quedan]


def _a_ancho(largo_feats: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """De dos filas por partido a una, con prefijos `local_` y `visita_`."""
    salida = None
    for lado in spec.LADOS:
        d = largo_feats[largo_feats["lado"] == lado].drop(columns="lado")
        d = d.rename(columns={c: f"{lado}_{c}" for c in cols})
        salida = d if salida is None else salida.merge(d, on=CLAVE, validate="one_to_one")
    return salida


def construir() -> pd.DataFrame:
    """Arma la tabla Gold completa desde Silver."""
    matches = read_table("fact_match")
    fixtures = read_table("fact_fixture")
    players = read_table("fact_player_gw")
    dim = read_table("dim_team")

    stats_jug = player_agg.team_stats_by_fixture(players)
    largo = tf.construir_largo(matches, fixtures, stats_jug)
    log.info("Tabla larga equipo-partido: %d filas", len(largo))

    cortes = tf.cortes_por_fecha(fixtures)
    obj = _objetivos(largo, cortes)
    obj_lado = _objetivos_por_lado(obj)

    # --- historias, cada una con su clave de agrupación ---
    h_gen = tf.historia_general(largo)
    h_tmp = tf.historia_temporada(largo)
    h_cnd = tf.historia_condicion(largo)
    h_cmp = tf.historia_campeonato(largo)

    orden = largo[["season", "fixture_id", "team_short", "k"]]
    plantel = player_agg.plantel_por_ventana(players, orden)
    h_pln = (plantel.merge(largo[["season", "fixture_id", "team_short", "kickoff_time"]],
                           on=["season", "fixture_id", "team_short"], validate="one_to_one")
                    .rename(columns={"kickoff_time": "hist_kickoff"}))

    cols_gen = [c for c in h_gen.columns if c not in ("team_short", "hist_kickoff")]
    cols_tmp = [c for c in h_tmp.columns if c not in ("season", "team_short", "hist_kickoff")]
    cols_cnd = [c for c in h_cnd.columns
                if c not in ("team_short", "es_local", "hist_kickoff")]
    cols_cmp = [c for c in h_cmp.columns if c not in ("season", "team_short", "hist_kickoff")]
    cols_pln = ["mins_hhi", "continuidad_plantel"]

    bloques = [
        _pegar_bloque(obj_lado, h_gen, ["team_short"], cols_gen, con_kickoff=True),
        _pegar_bloque(obj_lado, h_tmp, ["season", "team_short"], cols_tmp),
        _pegar_bloque(obj_lado, h_cnd, ["team_short", "es_local"], cols_cnd),
        _pegar_bloque(obj_lado, h_cmp, ["season", "team_short"], cols_cmp),
        _pegar_bloque(obj_lado, h_pln, ["team_short"], cols_pln),
    ]
    feats = bloques[0]
    for b in bloques[1:]:
        feats = feats.merge(b, on=CLAVE + ["lado"], validate="one_to_one")

    feats = feats.rename(columns={"continuidad_plantel": "continuidad_plantel_u5"})
    cols_lado = (cols_gen + cols_tmp + cols_cnd + cols_cmp
                 + ["mins_hhi", "continuidad_plantel_u5", "hist_kickoff"])

    # Descanso: días entre el último partido usado y el que se predice. El calendario se
    # conoce de antemano, así que usar el kickoff del propio partido no es leakage.
    feats = feats.merge(obj_lado[CLAVE + ["lado", "kickoff_time"]],
                        on=CLAVE + ["lado"], validate="one_to_one")
    feats["dias_descanso"] = (
        (feats["kickoff_time"] - feats["hist_kickoff"]).dt.total_seconds() / 86400)
    feats = feats.drop(columns="kickoff_time")
    cols_lado.append("dias_descanso")

    # --- ascendidos ---
    flags = cold_start.flags_ascendido(dim)
    feats = feats.merge(
        obj_lado[CLAVE + ["lado", "team_short"]], on=CLAVE + ["lado"], validate="one_to_one")
    feats = feats.merge(flags, on=["season", "team_short"], how="left")
    feats = feats.drop(columns="team_short")
    cols_lado.append("es_ascendido")

    ancho = _a_ancho(feats, cols_lado)
    # Las columnas de auditoría se leen mejor con el lado al final.
    ancho = ancho.rename(columns={f"{l}_hist_kickoff": f"hist_kickoff_{l}"
                                  for l in spec.LADOS})
    gold = obj.merge(ancho, on=CLAVE, validate="one_to_one")

    # `xg_available` es del partido, no de un lado: si el xG viene en cero, viene en cero
    # para los dos equipos. Se toma una vez.
    xga = largo[largo["es_local"]][CLAVE + ["xg_available"]]
    gold = gold.merge(xga, on=CLAVE, validate="one_to_one")

    gold = _posiciones(gold, h_cmp, cortes, largo, obj)
    gold = gold.merge(h2h.construir(largo, obj), on=CLAVE, validate="one_to_one")
    gold = _campeonato_al_arranque(gold)
    gold = _target_y_mercado(gold, matches, largo)
    gold = _dificultad(gold, fixtures)
    gold = _diferenciales(gold)

    prior = cold_start.ajustar_prior(largo, flags, CFG.seasons_for_training())
    gold = cold_start.aplicar_prior(gold, prior)
    gold.attrs["prior_ascendidos"] = prior

    gold["split"] = np.where(gold["season"] == CFG.holdout_season, "holdout", "train")
    gold["feature_set_version"] = spec.FEATURE_SET_VERSION
    gold["gold_built_at"] = utc_stamp()
    return gold


def _posiciones(gold: pd.DataFrame, h_cmp: pd.DataFrame, cortes: pd.DataFrame,
                largo: pd.DataFrame, obj: pd.DataFrame) -> pd.DataFrame:
    """Posición en la tabla de cada lado, al momento del corte."""
    equipos = largo[["season", "team_short"]].drop_duplicates()
    pos = tf.tabla_de_posiciones(h_cmp, cortes, equipos)
    for lado, col in (("local", "home_short"), ("visita", "away_short")):
        p = pos.rename(columns={"team_short": col,
                                "pos_tabla_camp": f"{lado}_pos_tabla_camp"})
        # muchos-a-uno, no uno-a-uno: en las 85 dobles fechas un mismo equipo aparece
        # dos veces en la misma gameweek, y las dos comparten la misma posición de tabla
        # porque comparten el corte. Es justamente el comportamiento que se busca.
        gold = gold.merge(p, on=["season", "gameweek", col], how="left",
                          validate="many_to_one")
    return gold


def _campeonato_al_arranque(gold: pd.DataFrame) -> pd.DataFrame:
    """En la fecha 1 el acumulado no es desconocido: es cero.

    El `merge_asof` devuelve NaN porque no hay historia previa, pero un equipo que no
    jugó ningún partido de la temporada tiene literalmente 0 puntos y 0 partidos. Eso sí
    se rellena. Lo que NO se rellena es `ppp` (0/0 no está definido) ni `pos_tabla`: con
    los 20 equipos en cero el ranking sería arbitrario, y un orden inventado es peor que
    un faltante declarado.
    """
    for lado in spec.LADOS:
        for c in ("pts", "pj", "gf", "gc", "dg"):
            col = f"{lado}_{c}_camp"
            gold[col] = gold[col].fillna(0.0)
        sin_jugar = gold[f"{lado}_pj_camp"] == 0
        gold.loc[sin_jugar, [f"{lado}_ppp_camp", f"{lado}_pos_tabla_camp"]] = np.nan
        gold[f"{lado}_n_hist"] = gold[f"{lado}_n_hist"].fillna(0)
    return gold


def _target_y_mercado(gold: pd.DataFrame, matches: pd.DataFrame,
                      largo: pd.DataFrame) -> pd.DataFrame:
    """El target y las cuotas de referencia (que NO son features)."""
    loc = largo[largo["es_local"]][["season", "fixture_id", "gf", "gc"]].rename(
        columns={"gf": "home_goals", "gc": "away_goals"})
    gold = gold.merge(loc, on=CLAVE, validate="one_to_one")
    gold["goal_diff"] = gold["home_goals"] - gold["away_goals"]
    gold["target_1x2"] = np.where(gold["goal_diff"] > 0, "home",
                                  np.where(gold["goal_diff"] == 0, "draw", "away"))

    # Las cuotas de CIERRE viven en Gold sólo para el baseline y la simulación de ROI.
    # Nunca son feature: se fijan minutos antes del kickoff (o sea, después del corte), y
    # además un modelo que las copia produce valor esperado ~0 y jamás encontraría una
    # apuesta con valor. `spec.FEATURES` no las incluye y hay un test que lo verifica.
    cuotas = matches[["season", "match_date", "home_short", "away_short",
                      *ODDS_MERCADO]]
    gold = gold.merge(cuotas, on=["season", "home_short", "away_short"], how="left",
                      validate="one_to_one")
    probs = odds_a_probabilidades(gold, ODDS_MERCADO)
    for c in ("home", "draw", "away"):
        gold[f"p_mercado_{c}"] = probs[c]
    return gold


def _dificultad(gold: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    fdr = fixtures[CLAVE + ["team_h_difficulty", "team_a_difficulty"]].rename(
        columns={"team_h_difficulty": "fdr_local", "team_a_difficulty": "fdr_visita"})
    gold = gold.merge(fdr, on=CLAVE, validate="one_to_one")
    gold["fdr_dif"] = gold["fdr_local"] - gold["fdr_visita"]
    return gold


def _diferenciales(gold: pd.DataFrame) -> pd.DataFrame:
    for c in spec.DIFERENCIALES:
        gold[f"dif_{c}"] = gold[f"local_{c}"] - gold[f"visita_{c}"]
    return gold


# ---------------------------------------------------------------------------
# Controles — corren ANTES de escribir, no en los tests
# ---------------------------------------------------------------------------

def _validar(gold: pd.DataFrame, fixtures: pd.DataFrame) -> None:
    leakage.assert_no_banned_columns(gold, context=TABLA)

    faltan = [c for c in spec.GOLD_COLUMNS if c not in gold.columns]
    sobran = [c for c in gold.columns if c not in spec.GOLD_COLUMNS]
    if faltan or sobran:
        raise ValueError(
            f"Gold no coincide con el contrato de features/spec.py.\n"
            f"  faltan ({len(faltan)}): {faltan[:12]}\n"
            f"  sobran ({len(sobran)}): {sobran[:12]}")

    if gold.duplicated(CLAVE).any():
        raise ValueError("Hay partidos duplicados en Gold.")

    # La prueba auditable: toda historia usada es anterior al corte.
    for lado in spec.LADOS:
        hk, corte = gold[f"hist_kickoff_{lado}"], gold["corte"]
        malas = (hk.notna()) & (hk >= corte)
        if malas.any():
            raise leakage.LeakageError(
                f"{int(malas.sum())} filas usan historia POSTERIOR al corte "
                f"({lado}). Ejemplo:\n{gold.loc[malas, CLAVE + ['corte']].head()}")

    # Y el control estricto contra el deadline de FPL, gameweek por gameweek.
    resumen = _auditar_por_fecha(gold, fixtures)
    salida = PROJECT_ROOT / "features" / "output"
    salida.mkdir(parents=True, exist_ok=True)
    resumen.to_csv(salida / "gold_audit.csv", index=False)
    log.info("Controles anti-leakage OK. Margen mínimo contra el deadline de FPL: %.1f h",
             resumen["margen_horas_min"].min())


def _auditar_por_fecha(gold: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Contrasta la historia usada contra el deadline de FPL, fecha por fecha.

    Es un control MÁS ESTRICTO que nuestro propio corte. El corte es el inicio de la
    fecha; el deadline de FPL cae 90 minutos antes. Si la historia que usamos resulta
    anterior incluso al deadline, entonces las features también habrían sido válidas para
    alguien que tuviera que decidir a esa hora.

    Devuelve el detalle por gameweek —que se guarda en `features/output/gold_audit.csv`
    para la defensa— y falla si alguna fecha lo viola.
    """
    dl = leakage.gameweek_deadlines(fixtures)
    g = gold.merge(dl, on=["season", "gameweek"], how="left", validate="many_to_one")

    usado = g[["hist_kickoff_local", "hist_kickoff_visita"]].max(axis=1)
    g = g.assign(hist_max=usado,
                 margen_h=(g["deadline"] - usado).dt.total_seconds() / 3600)

    resumen = (g.groupby(["season", "gameweek"], as_index=False)
                .agg(partidos=("fixture_id", "size"),
                     deadline=("deadline", "first"),
                     hist_mas_reciente=("hist_max", "max"),
                     margen_horas_min=("margen_h", "min")))

    malas = resumen[resumen["margen_horas_min"] < 0]
    if not malas.empty:
        raise leakage.LeakageError(
            f"{len(malas)} gameweeks usan historia posterior al deadline de FPL:\n"
            f"{malas.head(10).to_string(index=False)}")
    return resumen


def run(escribir: bool = True) -> pd.DataFrame:
    fixtures = read_table("fact_fixture")
    gold = construir()

    gold = gold[spec.GOLD_COLUMNS] if set(spec.GOLD_COLUMNS) <= set(gold.columns) else gold
    _validar(gold, fixtures)

    if escribir:
        ruta = write_table(gold, TABLA, layer="gold")
        log.info("Gold escrito en %s — %d filas x %d columnas",
                 ruta, len(gold), gold.shape[1])
        prior = gold.attrs.get("prior_ascendidos", {})
        if prior:
            pj = CFG.gold_root / "prior_ascendidos.json"
            pj.write_text(json.dumps(prior, indent=2), encoding="utf-8")
            log.info("Prior de ascendidos congelado en %s", pj)
    return gold


def main() -> None:
    ap = argparse.ArgumentParser(description="Construye la tabla Gold-TP.")
    ap.add_argument("--dry-run", action="store_true", help="no escribe, sólo valida")
    args = ap.parse_args()
    setup(CFG.log_level, CFG.log_format)
    gold = run(escribir=not args.dry_run)
    print(f"\nGold-TP: {len(gold)} filas x {gold.shape[1]} columnas "
          f"({len(spec.FEATURES)} features)")
    print(gold.groupby(["season", "split"]).size().to_string())


if __name__ == "__main__":
    main()
