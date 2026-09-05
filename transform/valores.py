"""Silver — `fact_valor_jugador`: cuánto vale cada jugador, en qué club y desde cuándo.

    python -m transform.valores

Transfermarkt publica **valuaciones fechadas**: una fila por jugador cada vez que le
revisan el precio, con el club en el que estaba ese día. Esta tabla las convierte en
**intervalos** — `[desde, hasta)` — que es la forma en que el resto del pipeline puede
consumirlas sin mirar el futuro: para un partido con corte `t` se toma el intervalo que
contiene a `t`, exactamente igual que el `merge_asof` de cualquier otra ventana.

## Por qué intervalos y no un agregado por equipo

Podría agregarse acá directamente a "valor del plantel del equipo X en la fecha d", pero
entonces esta tabla tendría que saber qué cortes existen, y los cortes salen de Gold. El
grano jugador-intervalo es el hecho normalizado; agregarlo al corte es trabajo de la capa
de features. Es la misma división que ya hace `fact_player_gw` con las ventanas rodantes.

## Las tres trampas de la fuente

**1 · Filiales y homónimos.** Conviven `Manchester City` con `Manchester City U21` y
`... Reserves`, y `Newcastle United Jets` es un club **australiano**. El cruce va por nombre
exacto contra `team_mapping.TM_ALIASES`, nunca por `contains`.

**2 · El club sale de la valuación, no del jugador.** `players.current_club_id` dice dónde
está el jugador **hoy**; usarlo atribuiría los goles… perdón, el valor, al club equivocado
para todo el pasado. Cada fila de `player_valuations` trae su propio club, que es el de esa
fecha, y ése es el que vale. Es el mismo error que `player_agg` evita agregando por fixture.

**3 · El dataset se congeló el 06/07/2026.** Para 2022-23 a 2025-26 está completo; para
2026-27 el último intervalo de cada jugador queda abierto con el valor de julio, así que el
mercado de pases de agosto no existe. No es leakage —es dato viejo, no futuro— pero la
feature es más pobre justo en la temporada en curso.
"""

from __future__ import annotations

import io

import pandas as pd

from common.config import CFG
from common.logging_setup import get_logger, setup
from common.storage import read_raw, read_table, write_table
from transform import team_mapping

log = get_logger(__name__)

TABLA = "fact_valor_jugador"
FUENTE = "transfermarkt"

# Transfermarkt -> las cuatro lineas del proyecto, las mismas que usa `player_agg`.
POSICIONES = {"Goalkeeper": "arq", "Defender": "def", "Midfield": "med", "Attack": "del"}

# El intervalo abierto del final. Lejos, pero finito: un NaT obligaria a ramificar en cada
# comparacion de la capa de features.
FIN_ABIERTO = pd.Timestamp("2099-12-31")


def _leer(nombre: str) -> pd.DataFrame:
    crudo = read_raw(FUENTE, CFG.current_season, "raw", f"{nombre}.csv")
    if crudo is None:
        raise FileNotFoundError(
            f"No hay {nombre}.csv en Bronze. Corre `python -m ingestion.bronze_transfermarkt`.")
    return pd.read_csv(io.BytesIO(crudo))


def _mapa_clubes(nombres) -> dict[str, str]:
    """Nombre exacto de Transfermarkt -> `short_name`, solo para los que resuelven."""
    registry = team_mapping.build_registry()
    alias = {**team_mapping.FD_ALIASES, **team_mapping.TM_ALIASES}
    out = {}
    for n in sorted({x for x in nombres if isinstance(x, str)}):
        try:
            out[n] = team_mapping.resolve(n, registry, alias)
        except team_mapping.UnmappedTeam:
            continue          # club de otra liga o filial: esperado, no es error
    return out


def construir() -> pd.DataFrame:
    pv = _leer("player_valuations")
    jug = _leer("players")[["player_id", "name", "position", "sub_position"]]

    pv["date"] = pd.to_datetime(pv["date"], errors="coerce")
    pv = pv.dropna(subset=["date", "player_id", "market_value_in_eur"])

    mapa = _mapa_clubes(pv["current_club_name"])
    pv["team_short"] = pv["current_club_name"].map(mapa)
    antes = len(pv)
    pv = pv[pv["team_short"].notna()]
    log.info("Valuaciones de clubes de la ventana: %s de %s", f"{len(pv):,}", f"{antes:,}")

    d = pv.merge(jug, on="player_id", how="left")
    d["linea"] = d["position"].map(POSICIONES)

    # Intervalos: cada valuacion vale hasta la siguiente DEL MISMO JUGADOR, tenga o no el
    # mismo club. Si se agrupara por (jugador, club), un pase y vuelta dejaria huecos.
    d = d.sort_values(["player_id", "date"]).reset_index(drop=True)
    d["desde"] = d["date"]
    d["hasta"] = d.groupby("player_id")["date"].shift(-1).fillna(FIN_ABIERTO)

    out = d[["player_id", "name", "position", "sub_position", "linea", "team_short",
             "desde", "hasta", "market_value_in_eur"]].rename(
        columns={"name": "player_name", "market_value_in_eur": "valor_eur"})
    return out.sort_values(["team_short", "desde", "player_id"]).reset_index(drop=True)


def equipos_sin_valores(d: pd.DataFrame, esperados: list[str]) -> list[str]:
    """El control que la Fase 1 enseño a poner desde el principio, no al final."""
    return sorted(set(esperados) - set(d["team_short"].dropna()))


def run(escribir: bool = True) -> pd.DataFrame:
    d = construir()
    try:
        m = read_table("fact_match")
        esperados = sorted(set(m["home_short"]) | set(m["away_short"]))
        faltan = equipos_sin_valores(d, esperados)
        if faltan:
            log.warning("SIN valores de mercado y son de la ventana: %s — revisar "
                        "team_mapping.TM_ALIASES", faltan)
        else:
            log.info("Los %d equipos de la ventana tienen valores.", len(esperados))
    except FileNotFoundError:
        pass
    if escribir:
        write_table(d, TABLA, layer="silver")
    return d


def main() -> None:
    setup(CFG.log_level, CFG.log_format)
    d = run()
    print(f"\n{TABLA}: {len(d):,} valuaciones, "
          f"{d['desde'].min().date()} a {d['desde'].max().date()}")
    print(f"jugadores distintos: {d['player_id'].nunique():,} | "
          f"equipos: {d['team_short'].nunique()}")
    print(f"\nPor linea:\n{d['linea'].value_counts(dropna=False).to_string()}")
    print(f"\nvalor mediano: EUR {d['valor_eur'].median():,.0f} | "
          f"maximo: EUR {d['valor_eur'].max():,.0f}")
    ult = d[d["hasta"] == FIN_ABIERTO]
    print(f"\nintervalos abiertos (el ultimo de cada jugador): {len(ult):,}")
    print(f"ultima valuacion del dataset: {d['desde'].max().date()} "
          f"— de ahi en adelante la feature usa ese valor congelado")


if __name__ == "__main__":
    main()
