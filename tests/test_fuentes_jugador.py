"""Cómo se combinan las dos fuentes de `fact_player_gw`, y por qué así.

El histórico jugador-fecha tiene dos orígenes y **no son intercambiables**:

- **vaastav** archiva `merged_gw.csv` semanalmente. Es la versión **asentada**, pero
  llega tarde: medido sobre 2025-26, tocó el archivo doce veces en toda la temporada,
  con un gap mediano de 10 días y máximo de 96.
- **`/event/{GW}/live/`** de la API oficial llega **apenas termina la fecha**, pero
  puede agarrar un partido a medio asentar.

La regla es: **vaastav donde tenga la fecha, la API donde falte**, y la decisión va por
*(temporada, fecha)*, nunca por temporada entera.

Estos tests fijan las tres cosas que se rompieron una vez y no pueden volver a romperse.
"""

from __future__ import annotations

import pandas as pd
import pytest

from common.config import CFG
from common.storage import snapshot_at_or_before, snapshot_stamp


# --------------------------------------------------------------------------- #
#  El snapshot contemporáneo
# --------------------------------------------------------------------------- #

def test_snapshot_at_or_before_no_devuelve_uno_posterior(tmp_path, monkeypatch):
    """Es la garantía de la que depende la atribución por transferencia."""
    from common import storage

    raiz = tmp_path / "bronze" / "fpl" / "2026-27" / "bootstrap"
    stamps = ["20260817T232251Z", "20260824T234245Z", "20260830T150532Z"]
    for st in stamps:
        (raiz / f"ingested_at={st}").mkdir(parents=True)

    # CFG es un dataclass congelado: se sustituye el objeto entero, no un atributo.
    class _CfgFalso:
        @staticmethod
        def bronze_dataset_root(*_a, **_k):
            return raiz

    monkeypatch.setattr(storage, "CFG", _CfgFalso)

    # Justo el del medio: el más nuevo que no es posterior.
    elegido = storage.snapshot_at_or_before("fpl", "2026-27", "bootstrap",
                                            "20260824T234245Z")
    assert snapshot_stamp(elegido) == "20260824T234245Z"

    # Un momento intermedio cae en el anterior, nunca en el siguiente.
    elegido = storage.snapshot_at_or_before("fpl", "2026-27", "bootstrap",
                                            "20260826T000000Z")
    assert snapshot_stamp(elegido) == "20260824T234245Z"

    # Antes del primero no hay nada: None, no el más viejo.
    assert storage.snapshot_at_or_before("fpl", "2026-27", "bootstrap",
                                         "20260101T000000Z") is None


def test_los_stamps_ordenan_cronologicamente_como_texto():
    """`snapshot_at_or_before` compara strings; esto es lo que lo hace válido."""
    stamps = ["20260817T232251Z", "20260824T234245Z", "20260830T150532Z",
              "20270101T000000Z"]
    assert stamps == sorted(stamps)
    assert sorted(stamps) == sorted(stamps, key=lambda s: pd.Timestamp(s))


# --------------------------------------------------------------------------- #
#  La atribución por transferencia
# --------------------------------------------------------------------------- #

def test_ninguna_fila_pertenece_a_un_club_que_no_jugo_ese_partido(fact_player_gw,
                                                                   fact_fixture):
    """La invariante de fondo, sobre la tabla ya escrita.

    Si el equipo de una fila no es ninguno de los dos del fixture, `was_home` y el
    rival salen inventados, y las stats van a la historia de un club que no jugó.
    """
    fx = fact_fixture[["season", "fixture_id", "home_short", "away_short"]]
    m = fact_player_gw.merge(fx, on=["season", "fixture_id"], how="inner")
    assert len(m), "ninguna fila cruzó con fact_fixture"

    ajeno = m[(m.team_short != m.home_short) & (m.team_short != m.away_short)]
    assert ajeno.empty, (
        f"{len(ajeno)} filas atribuidas a un club que no jugó ese partido: "
        f"{ajeno[['season', 'player_name', 'team_short', 'home_short', 'away_short']].head().to_dict('records')}"
    )


def test_el_equipo_del_jugador_sale_del_bootstrap_de_esa_fecha():
    """Un jugador transferido no puede arrastrar sus goles al club nuevo.

    Baleba jugó la fecha 1 de 2026-27 en el Brighton y se fue al United. Leyendo el
    último bootstrap, sus estadísticas de esa fecha quedaban atribuidas al United —y
    como las features agregan por (temporada, fixture, equipo), ensuciaba la historia
    de los dos clubes.
    """
    from transform import fpl_live

    gws = fpl_live.gameweeks_disponibles(CFG.current_season)
    if not gws:
        pytest.skip("No hay snapshots de event_live. Corré `python -m ingestion.run`.")

    from common.storage import read_table, table_exists
    if not table_exists("dim_team"):
        pytest.skip("Falta silver.dim_team. Corré `python -m transform.silver`.")

    vivo = fpl_live.build(CFG.current_season, read_table("dim_team"))
    assert vivo is not None and len(vivo)

    # El equipo del jugador tiene que ser uno de los dos del fixture en el que jugó.
    fx = read_table("fact_fixture")
    fx = fx[fx.season == CFG.current_season][["fixture_id", "home_short", "away_short"]]
    m = vivo.merge(fx, on="fixture_id", how="inner")
    assert len(m), "ninguna fila de event_live cruzó con fact_fixture"

    ajeno = m[(m.team_short != m.home_short) & (m.team_short != m.away_short)]
    assert ajeno.empty, (
        f"{len(ajeno)} filas atribuyen al jugador un club que no jugó ese partido: "
        f"{ajeno[['player_name', 'team_short', 'home_short', 'away_short']].head().to_dict('records')}"
    )


# --------------------------------------------------------------------------- #
#  La preferencia por fecha, no por temporada
# --------------------------------------------------------------------------- #

def test_no_hay_filas_duplicadas_de_jugador_y_fecha(fact_player_gw):
    """Si las dos fuentes aportaran la misma fecha, habría doble conteo.

    Es el riesgo concreto de combinar por fecha en vez de por temporada: un jugador
    con dos filas en la misma gameweek duplicaría sus goles al agregar por equipo.
    """
    d = fact_player_gw

    # La clave lleva `fpl_player_id`, no `player_name`: hay homónimos reales en la
    # Premier (dos Ben Davies) y el nombre no identifica a nadie. El id sí es único
    # dentro de una temporada, que es el alcance de esta invariante.
    clave = ["season", "fpl_player_id", "gameweek", "fixture_id"]
    dup = d[d.duplicated(subset=clave, keep=False)]
    assert dup.empty, (
        f"{len(dup)} filas duplicadas por {clave}: se están contando dos veces al "
        f"agregar por equipo. {dup[clave + ['player_name']].head().to_dict('records')}"
    )


def test_la_temporada_en_curso_tiene_las_fechas_ya_jugadas(fact_player_gw, fact_fixture):
    """El hueco que este arreglo cierra: una fecha jugada sin una sola fila.

    Con la preferencia por temporada, alcanzaba con que vaastav publicara la fecha 1
    para descartar los snapshots en vivo de todo el resto del año.
    """
    actual = CFG.current_season
    fx = fact_fixture[fact_fixture.season == actual]
    if fx.empty:
        pytest.skip("La temporada en curso todavía no tiene fixtures.")

    # Fechas con al menos un partido terminado, según el marcador.
    jugadas = sorted(
        int(gw) for gw, g in fx.groupby("gameweek")
        if g[["team_h_score", "team_a_score"]].notna().all(axis=1).any()
    )
    if not jugadas:
        pytest.skip("La temporada en curso todavía no jugó ninguna fecha.")

    con_datos = set(fact_player_gw[fact_player_gw.season == actual].gameweek.dropna()
                    .astype(int))
    faltan = [gw for gw in jugadas if gw not in con_datos]
    assert not faltan, (
        f"Las fechas {faltan} de {actual} se jugaron pero no tienen ninguna fila en "
        "fact_player_gw. Las doce features derivadas de jugadores van a llegar vacías."
    )
