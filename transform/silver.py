"""Silver — normalización de las tres fuentes a un modelo común.

Produce cuatro tablas:

    dim_team        (season, team_code)          dimensión de cambio lento
    fact_match      1 fila por partido            football-data: resultado + cuotas
    fact_fixture    1 fila por fixture            FPL: gameweek, kickoff y DEADLINE
    fact_player_gw  1 fila por jugador × fecha    vaastav: el detalle fino

`fact_player_gw` conserva la granularidad jugador-fecha a propósito. El TP la
colapsa a nivel partido en Gold, pero el proyecto de armado del equipo de FPL la
necesita entera, y una sola normalización sirve a los dos.

La clave de cruce entre fuentes es `(season, match_date, home_short, away_short)`.
"""

from __future__ import annotations

import io
import json

import pandas as pd

from common.config import CFG
from common.logging_setup import get_logger, setup
from common.storage import read_raw, write_table
from transform import fpl_live, team_mapping

log = get_logger(__name__)

# El deadline de FPL cae exactamente 90 minutos antes del primer partido de la
# fecha. Verificado sobre las 38 gameweeks de 2026-27 contra el `deadline_time`
# real de la API: diferencia constante, desvío estándar 0.
# Esto permite reconstruir el deadline de temporadas pasadas, donde vaastav no lo
# publica, sin inventar nada.
DEADLINE_OFFSET = pd.Timedelta(minutes=90)

# football-data -> nombres legibles. El renombre no es cosmético: `AS` (tiros del
# visitante) choca con la palabra reservada de Python en cualquier expresión.
FD_RENAME = {
    "Date": "match_date",
    "Time": "kickoff_local",
    "HomeTeam": "fd_home",
    "AwayTeam": "fd_away",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
    "HTHG": "home_goals_ht",
    "HTAG": "away_goals_ht",
    "HTR": "result_ht",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_target",
    "AST": "away_shots_target",
    "HC": "home_corners",
    "AC": "away_corners",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HY": "home_yellows",
    "AY": "away_yellows",
    "HR": "home_reds",
    "AR": "away_reds",
    # Cuotas de apertura
    "B365H": "odds_b365_home", "B365D": "odds_b365_draw", "B365A": "odds_b365_away",
    "AvgH": "odds_avg_home", "AvgD": "odds_avg_draw", "AvgA": "odds_avg_away",
    # Cuotas de CIERRE — las que valen como benchmark: incorporan toda la
    # información disponible hasta minutos antes del partido.
    "B365CH": "odds_b365_close_home", "B365CD": "odds_b365_close_draw",
    "B365CA": "odds_b365_close_away",
    "AvgCH": "odds_avg_close_home", "AvgCD": "odds_avg_close_draw",
    "AvgCA": "odds_avg_close_away",
    "MaxCH": "odds_max_close_home", "MaxCD": "odds_max_close_draw",
    "MaxCA": "odds_max_close_away",
}

# Columnas de merged_gw que se conservan. Son las comunes a las cuatro temporadas
# ingestadas; las que aparecen sólo en algunas (defensive_contribution, tackles,
# mng_*) quedan fuera para que el esquema no dependa de la temporada.
PLAYER_COLS = [
    "name", "position", "team", "element", "fixture", "opponent_team", "was_home",
    "GW", "round", "kickoff_time", "minutes", "starts",
    "goals_scored", "assists", "clean_sheets", "goals_conceded", "own_goals",
    "penalties_saved", "penalties_missed", "yellow_cards", "red_cards", "saves",
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded",
    "influence", "creativity", "threat", "ict_index",
    "total_points", "bonus", "bps", "value", "selected",
    "transfers_in", "transfers_out", "transfers_balance",
    "team_h_score", "team_a_score",
]

# `xP` se descarta ya en Silver. vaastav lo scrapea del campo `ep_this` DESPUÉS de
# que termina la fecha, y el propio repo advierte que puede reflejar información
# post-partido en lugar de la predicción que los managers veían antes del deadline.
# No se guarda para que no exista la tentación de usarlo.
DROPPED_LEAKY = ["xP"]

# Nombres de vaastav -> el esquema canonico de fact_player_gw, que es tambien el que
# produce `transform/fpl_live.py` desde los snapshots de la API.
RENOMBRE_VAASTAV = {"GW": "gameweek", "name": "player_name",
                    "element": "fpl_player_id", "fixture": "fixture_id"}

# `AM` (Assistant Manager) no es una posición: son los DT. FPL los hizo elegibles
# con un chip que existió sólo entre la GW23 de 2024-25 y el final de esa temporada
# — en 2025-26 ya no aparecen. Tienen `minutes = 0` y puntúan por resultados del
# equipo, así que contaminarían cualquier agregado por posición (y la media de
# minutos del plantel). Se excluyen.
PLAYER_POSITIONS = ("GK", "DEF", "MID", "FWD")


# --------------------------------------------------------------------------- #


def _read_bronze_csv(source: str, season: str, filename: str, **kwargs) -> pd.DataFrame | None:
    raw = read_raw(source, season, "raw", filename)
    if raw is None:
        return None
    return pd.read_csv(io.BytesIO(raw), **kwargs)


def build_fact_match(dim: pd.DataFrame) -> pd.DataFrame:
    """Una fila por partido, desde football-data: resultado, stats y cuotas."""
    registry = team_mapping.build_registry()
    frames = []

    for season in CFG.seasons_to_ingest():
        df = _read_bronze_csv(
            "football_data", season, "E0.csv",
            encoding="utf-8-sig", encoding_errors="ignore",
        )
        if df is None:
            log.info("[%s] sin E0.csv en Bronze — se omite de fact_match", season)
            continue

        keep = [c for c in FD_RENAME if c in df.columns]
        out = df[keep].rename(columns=FD_RENAME).copy()
        out["season"] = season

        # football-data usa dd/mm/yyyy y hora local del Reino Unido.
        out["match_date"] = pd.to_datetime(out["match_date"], format="%d/%m/%Y", errors="coerce")

        name_to_short = team_mapping.resolve_many(
            sorted(set(out["fd_home"]) | set(out["fd_away"])), registry
        )
        out["home_short"] = out["fd_home"].map(name_to_short)
        out["away_short"] = out["fd_away"].map(name_to_short)

        frames.append(out)

    if not frames:
        raise FileNotFoundError("No hay ningún E0.csv en Bronze para construir fact_match.")

    fact = pd.concat(frames, ignore_index=True)

    # Target 1X2 explícito, además de los goles (que quedan para un eventual Poisson).
    fact["target_1x2"] = fact["result"].map({"H": "home", "D": "draw", "A": "away"})

    codes = dim.drop_duplicates(["season", "short_name"]).set_index(["season", "short_name"])["team_code"]
    fact["home_code"] = fact.set_index(["season", "home_short"]).index.map(codes)
    fact["away_code"] = fact.set_index(["season", "away_short"]).index.map(codes)

    front = ["season", "match_date", "kickoff_local", "home_short", "away_short",
             "home_code", "away_code", "home_goals", "away_goals", "result", "target_1x2"]
    ordered = front + [c for c in fact.columns if c not in front]
    return fact[ordered].sort_values(["season", "match_date"]).reset_index(drop=True)


def build_fact_fixture(dim: pd.DataFrame) -> pd.DataFrame:
    """Una fila por fixture, con la gameweek y su **deadline**.

    Fuente por temporada:
    - pasadas  -> `fixtures.csv` de vaastav
    - actual   -> `fixtures.json` de la API (más fresco, se actualiza cada semana)

    `deadline_time` es el ancla del control anti-leakage: ninguna feature usada
    para predecir esta fecha puede provenir de un instante posterior.
    """
    frames = []

    for season in CFG.seasons_to_ingest():
        api_deadlines: dict[int, str] = {}

        if season == CFG.current_season:
            raw = read_raw("fpl", season, "fixtures", "fixtures.json")
            if raw is None:
                df = _read_bronze_csv("vaastav", season, "fixtures.csv")
            else:
                df = pd.DataFrame(json.loads(raw))

            boot = read_raw("fpl", season, "bootstrap", "bootstrap_static.json")
            if boot is not None:
                api_deadlines = {
                    e["id"]: e["deadline_time"] for e in json.loads(boot)["events"]
                }
        else:
            df = _read_bronze_csv("vaastav", season, "fixtures.csv")

        if df is None or df.empty:
            log.info("[%s] sin fixtures en Bronze — se omite de fact_fixture", season)
            continue

        cols = ["id", "event", "kickoff_time", "team_h", "team_a", "team_h_score",
                "team_a_score", "team_h_difficulty", "team_a_difficulty", "finished"]
        out = df[[c for c in cols if c in df.columns]].copy()
        out["season"] = season
        out["kickoff_time"] = pd.to_datetime(out["kickoff_time"], utc=True, errors="coerce")

        # id de FPL (que cambia todas las temporadas) -> short_name canónico.
        ids = dim[dim["season"] == season].set_index("fpl_team_id")["short_name"]
        out["home_short"] = out["team_h"].map(ids)
        out["away_short"] = out["team_a"].map(ids)

        missing = out[out["home_short"].isna() | out["away_short"].isna()]
        if not missing.empty:
            raise team_mapping.UnmappedTeam(
                f"[{season}] {len(missing)} fixtures con equipo sin mapear. "
                f"ids sin resolver: "
                f"{sorted(set(missing['team_h']) | set(missing['team_a']))}"
            )

        # Deadline: real si la API lo publica, derivado si no.
        derived = (
            out.groupby("event")["kickoff_time"].transform("min") - DEADLINE_OFFSET
        )
        if api_deadlines:
            real = out["event"].map(api_deadlines)
            real = pd.to_datetime(real, utc=True, errors="coerce")
            out["deadline_time"] = real.fillna(derived)
            out["deadline_source"] = pd.Series(real.notna()).map({True: "api", False: "derived"})
        else:
            out["deadline_time"] = derived
            out["deadline_source"] = "derived"

        frames.append(out)

    if not frames:
        raise FileNotFoundError("No hay fixtures en Bronze para construir fact_fixture.")

    fact = pd.concat(frames, ignore_index=True)
    fact = fact.rename(columns={"id": "fixture_id", "event": "gameweek"})
    fact["match_date"] = fact["kickoff_time"].dt.tz_convert("Europe/London").dt.normalize().dt.tz_localize(None)

    front = ["season", "gameweek", "fixture_id", "match_date", "kickoff_time",
             "deadline_time", "deadline_source", "home_short", "away_short"]
    ordered = front + [c for c in fact.columns if c not in front]
    return fact[ordered].sort_values(["season", "gameweek", "kickoff_time"]).reset_index(drop=True)


def build_fact_player_gw(dim: pd.DataFrame) -> pd.DataFrame:
    """Una fila por jugador × fecha. La tabla que alimenta los dos proyectos."""
    registry = team_mapping.build_registry()
    frames = []

    for season in CFG.seasons_to_ingest():
        df = _read_bronze_csv(
            "vaastav", season, "merged_gw.csv",
            encoding_errors="ignore", low_memory=False,
        )
        if df is None:
            log.info("[%s] sin merged_gw.csv — se omite de fact_player_gw "
                     "(normal si la temporada no arrancó)", season)
            continue

        present = [c for c in PLAYER_COLS if c in df.columns]
        out = df[present].copy()
        out["season"] = season
        out["kickoff_time"] = pd.to_datetime(out["kickoff_time"], utc=True, errors="coerce")

        # El equipo propio viene como nombre; el rival, como id de esa temporada.
        name_to_short = team_mapping.resolve_many(
            sorted(set(out["team"].dropna())), registry
        )
        out["team_short"] = out["team"].map(name_to_short)

        ids = dim[dim["season"] == season].set_index("fpl_team_id")["short_name"]
        out["opponent_short"] = out["opponent_team"].map(ids)

        unmapped = out["team_short"].isna() | out["opponent_short"].isna()
        if unmapped.any():
            raise team_mapping.UnmappedTeam(
                f"[{season}] {unmapped.sum()} filas de jugador con equipo sin mapear."
            )

        # El renombrado va ACA, no despues del concat: `fpl_live` ya produce estos
        # nombres, y renombrar al final dejaria dos columnas `gameweek`, dos
        # `player_name`, etc.
        out = out.rename(columns=RENOMBRE_VAASTAV)
        frames.append(out)

    # vaastav no publica la temporada en curso a tiempo: gap mediano de 10 dias, maximo
    # de 96. Los snapshots de /event/{GW}/live/ que guarda la ingesta cubren ese hueco y
    # traen las mismas estadisticas, xG incluido. Sin esto, las doce features derivadas
    # de jugadores llegan VACIAS a produccion.
    for season in CFG.seasons_to_ingest():
        if any(f["season"].iloc[0] == season for f in frames if len(f)):
            continue
        vivo = fpl_live.build(season, dim)
        if vivo is not None and len(vivo):
            frames.append(vivo)

    if not frames:
        raise FileNotFoundError(
            "No hay ningún merged_gw.csv ni snapshot de event_live en Bronze "
            "para construir fact_player_gw."
        )

    fact = pd.concat(frames, ignore_index=True)

    n_managers = (~fact["position"].isin(PLAYER_POSITIONS)).sum()
    if n_managers:
        otras = sorted(set(fact.loc[~fact["position"].isin(PLAYER_POSITIONS), "position"].dropna()))
        fact = fact[fact["position"].isin(PLAYER_POSITIONS)]
        log.info("excluidas %d filas que no son de jugadores (posiciones %s)", n_managers, otras)

    dropped = [c for c in DROPPED_LEAKY if c in fact.columns]
    if dropped:
        fact = fact.drop(columns=dropped)
    log.info("descartadas por riesgo de leakage: %s", DROPPED_LEAKY)

    front = ["season", "gameweek", "player_name", "position", "team_short",
             "opponent_short", "was_home", "kickoff_time", "minutes"]
    ordered = front + [c for c in fact.columns if c not in front]
    return fact[ordered].sort_values(["season", "gameweek", "team_short"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #


def report_join_rate(matches: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Cruza football-data con FPL y reporta cuántos partidos matchean.

    Si el cruce no da 100% en una temporada cerrada, hay un problema de mapeo o de
    fechas que conviene ver ahora y no cuando falten features.
    """
    key = ["season", "match_date", "home_short", "away_short"]
    merged = matches[key].merge(
        fixtures[key + ["gameweek"]], on=key, how="left", indicator=True
    )

    rows = []
    for season, grp in merged.groupby("season"):
        ok = (grp["_merge"] == "both").sum()
        rows.append({"season": season, "partidos": len(grp), "cruzados": ok,
                     "tasa": ok / len(grp) if len(grp) else 0.0})
    return pd.DataFrame(rows)


def run() -> dict[str, pd.DataFrame]:
    setup(CFG.log_level, CFG.log_format)
    log.info("=== Silver ===")

    dim = team_mapping.build_dim_team()
    write_table(dim, "dim_team")

    fixtures = build_fact_fixture(dim)
    write_table(fixtures, "fact_fixture")

    matches = build_fact_match(dim)
    write_table(matches, "fact_match")

    players = build_fact_player_gw(dim)
    write_table(players, "fact_player_gw")

    log.info("--- cruce football-data <-> FPL ---")
    rates = report_join_rate(matches, fixtures)
    for _, r in rates.iterrows():
        nivel = log.info if r["tasa"] == 1.0 else log.warning
        nivel("  %s: %d/%d partidos cruzados (%.1f%%)",
              r["season"], r["cruzados"], r["partidos"], r["tasa"] * 100)

    log.info("--- deadlines ---")
    for src, n in fixtures["deadline_source"].value_counts().items():
        log.info("  %s: %d fixtures", src, n)

    return {"dim_team": dim, "fact_fixture": fixtures,
            "fact_match": matches, "fact_player_gw": players}


if __name__ == "__main__":
    run()
