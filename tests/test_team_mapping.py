"""El mapeo de equipos tiene que cubrir el 100% de los nombres, en todas las fuentes.

Un equipo sin mapear no rompe nada de forma visible: simplemente sus partidos
desaparecen del join y aparecen mucho después como "faltan datos". Por eso acá se
exige cobertura total y no "casi total".
"""

from __future__ import annotations

import io

import pandas as pd
import pytest

from common.config import CFG
from common.storage import read_raw
from transform import team_mapping


@pytest.fixture(scope="module")
def registry() -> pd.DataFrame:
    return team_mapping.build_registry()


# --------------------------------------------------------------------------- #
#  La clave canónica
# --------------------------------------------------------------------------- #


def test_short_name_identifica_univocamente(registry):
    """`short_name` es la clave canónica: un short_name, un equipo."""
    assert registry["short_name"].is_unique
    assert registry["team_code"].is_unique


def test_el_id_de_fpl_no_sirve_como_clave(dim_team):
    """Documenta POR QUÉ no se usa `fpl_team_id`: se reasigna cada temporada.

    Si este test dejara de fallar en su premisa, se podría simplificar el mapeo —
    pero mientras tanto, usar el id para cruzar temporadas mezcla equipos.
    """
    cambios = dim_team.groupby("team_code")["fpl_team_id"].nunique()
    assert (cambios > 1).any(), (
        "Ningún equipo cambió de fpl_team_id entre temporadas. Verificar si la API "
        "cambió de comportamiento antes de simplificar team_mapping."
    )


# --------------------------------------------------------------------------- #
#  Cobertura
# --------------------------------------------------------------------------- #


def test_todos_los_nombres_de_football_data_resuelven(registry):
    """Cada nombre de cada temporada de football-data tiene que mapear."""
    revisadas = 0
    for season in CFG.seasons_to_ingest():
        raw = read_raw("football_data", season, "raw", "E0.csv")
        if raw is None:
            continue
        df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig", encoding_errors="ignore")
        nombres = sorted(set(df["HomeTeam"].dropna()) | set(df["AwayTeam"].dropna()))

        # Levanta UnmappedTeam listando TODOS los fallos si algo no resuelve.
        resueltos = team_mapping.resolve_many(nombres, registry)
        assert len(resueltos) == len(nombres)
        revisadas += 1

    if revisadas == 0:
        pytest.skip("No hay archivos de football-data en Bronze")


def test_todos_los_nombres_de_vaastav_resuelven(registry):
    revisadas = 0
    for season in CFG.seasons_to_ingest():
        raw = read_raw("vaastav", season, "raw", "teams.csv")
        if raw is None:
            continue
        nombres = sorted(pd.read_csv(io.BytesIO(raw))["name"].dropna())
        assert len(team_mapping.resolve_many(nombres, registry)) == len(set(nombres))
        revisadas += 1

    if revisadas == 0:
        pytest.skip("No hay teams.csv en Bronze")


def test_dim_team_tiene_20_equipos_por_temporada(dim_team):
    por_temporada = dim_team.groupby("season").size()
    malas = por_temporada[por_temporada != 20]
    assert malas.empty, f"Temporadas que no tienen 20 equipos:\n{malas.to_string()}"


def test_no_hay_short_name_repetido_dentro_de_una_temporada(dim_team):
    dup = dim_team[dim_team.duplicated(["season", "short_name"], keep=False)]
    assert dup.empty, f"short_name repetido dentro de una temporada:\n{dup.to_string()}"


# --------------------------------------------------------------------------- #
#  Los alias concretos que motivaron el módulo
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "nombre, esperado",
    [
        ("Man United", "MUN"),      # football-data
        ("Man Utd", "MUN"),         # FPL
        ("Tottenham", "TOT"),       # football-data
        ("Spurs", "TOT"),           # FPL
        ("Sheffield United", "SHU"),
        ("Sheffield Utd", "SHU"),
        ("Ipswich", "IPS"),         # FPL 2024-25
        ("Ipswich Town", "IPS"),    # FPL 2026-27 — la misma fuente, distinto nombre
        ("Nott'm Forest", "NFO"),
        ("Man City", "MCI"),
    ],
)
def test_alias_conocidos(registry, nombre, esperado):
    assert team_mapping.resolve(nombre, registry) == esperado


def test_man_city_y_man_united_no_colisionan(registry):
    """La normalización quita sufijos genéricos como `City` y `Town`.

    Hecha a lo bruto, colapsaría `Man City` y `Man United` en `man`. Este test fija
    que eso no pase, porque sería el peor error posible del módulo: dos clubes
    grandes intercambiados en silencio.
    """
    assert team_mapping.resolve("Man City", registry) != team_mapping.resolve("Man United", registry)


def test_un_nombre_desconocido_falla_ruidosamente(registry):
    """No se permite fallar en silencio: un equipo nuevo tiene que romper el build."""
    with pytest.raises(team_mapping.UnmappedTeam, match="no se pudo mapear"):
        team_mapping.resolve("Racing Club de Avellaneda", registry)


def test_resolve_many_reporta_todos_los_fallos_juntos(registry):
    """Informar de a un fallo por corrida haría el arreglo insoportablemente lento."""
    with pytest.raises(team_mapping.UnmappedTeam) as exc:
        team_mapping.resolve_many(["Arsenal", "Equipo Inventado", "Otro Inventado"], registry)

    assert "Equipo Inventado" in str(exc.value)
    assert "Otro Inventado" in str(exc.value)
