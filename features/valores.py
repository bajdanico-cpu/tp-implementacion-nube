"""Valor del plantel al corte: total, por línea, y relativo a la liga.

Es la única feature del proyecto que trae **información de afuera**. Todo lo demás —forma,
Elo, xG, puntos de FPL, Opta— se calcula con resultados pasados de la Premier. Un ascendido
no tiene resultados pasados de Premier, y ahí es donde el mercado de apuestas nos saca 9,3
puntos de accuracy contra 0,7 en el resto (`training/README.md`).

## Por qué no es un `merge_asof` como todos los demás bloques

Los otros bloques miran una **historia por equipo** y toman la fila más reciente antes del
corte. Acá el hecho está a nivel jugador-intervalo (`fact_valor_jugador`), y el valor del
plantel es una **agregación sobre los intervalos vigentes** en ese instante. No hay una fila
por equipo para traer: hay que sumar los jugadores que estaban.

Se resuelve por corte: los cortes son ~190 (una por fecha de cada temporada) y los
intervalos 24.595, así que la agregación directa es barata y explícita.

## El corte manda, igual que en todo el resto

Un intervalo `[desde, hasta)` entra si contiene al corte. Como las valuaciones se publican
en revisiones periódicas y el corte es el inicio de la fecha, la valuación vigente es
siempre anterior — no hay forma de que una revisión posterior al partido entre. El test lo
verifica sobre las filas reales.

## Lo que hay que decir al leer estas features

El dataset se congeló el **06/07/2026** (última valuación de nuestros clubes: 03/06/2026).
Para 2022-23 a 2025-26 está completo. Para 2026-27 el valor es el de junio, o sea **antes de
que cerrara el mercado de pases**: los refuerzos de agosto no están. No es leakage —es dato
viejo, no futuro— pero la feature es más pobre justo en la temporada en curso.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common.logging_setup import get_logger

log = get_logger(__name__)

LINEAS = ("arq", "def", "med", "del")
TOP_N = 11          # el once inicial: el plantel entero mezcla titulares con juveniles

COLUMNAS = (["valor_plantel", "valor_top11", "valor_rel", "valor_n"]
            + [f"valor_{l}" for l in LINEAS])


def _agregar(vig: pd.DataFrame) -> pd.DataFrame:
    """De intervalos vigentes a una fila por equipo."""
    g = vig.groupby("team_short")
    out = pd.DataFrame({
        "valor_plantel": g["valor_eur"].sum(),
        "valor_n": g["valor_eur"].size(),
        "valor_top11": g["valor_eur"].apply(lambda s: s.nlargest(TOP_N).sum()),
    })
    for l in LINEAS:
        out[f"valor_{l}"] = (vig[vig["linea"] == l].groupby("team_short")["valor_eur"]
                             .sum().reindex(out.index).fillna(0.0))
    # Relativo a la liga EN ESE MOMENTO. Es lo que hace la feature comparable entre
    # temporadas: el valor nominal de los planteles sube todos los años, y un arbol que ve
    # el crudo puede aprender a reconocer la temporada en vez del equipo -- la misma trampa
    # que `features/opta.py` documenta para las estadisticas recientes.
    total = out["valor_plantel"].sum()
    out["valor_rel"] = out["valor_plantel"] / total if total else np.nan
    return out.reset_index()


def construir(obj_lado: pd.DataFrame, valores: pd.DataFrame) -> pd.DataFrame:
    """Una fila por (partido, lado) con el valor del plantel de ese equipo en su corte.

    `obj_lado` trae `team_short` y `corte`; `valores` es `fact_valor_jugador`.
    """
    v = valores.dropna(subset=["team_short", "valor_eur"]).copy()
    v["desde"] = pd.to_datetime(v["desde"])
    v["hasta"] = pd.to_datetime(v["hasta"])

    # Los cortes vienen con timezone y las valuaciones no: se comparan en naive UTC.
    cortes = pd.to_datetime(obj_lado["corte"]).dt.tz_localize(None).unique()

    piezas = []
    for t in sorted(cortes):
        vig = v[(v["desde"] <= t) & (v["hasta"] > t)]
        if vig.empty:
            continue
        d = _agregar(vig)
        d["corte_naive"] = t
        piezas.append(d)

    if not piezas:
        raise ValueError("Ningun corte cae dentro de algun intervalo de valuacion.")

    tabla = pd.concat(piezas, ignore_index=True)

    obj = obj_lado.copy()
    obj["corte_naive"] = pd.to_datetime(obj["corte"]).dt.tz_localize(None)
    out = obj.merge(tabla, on=["team_short", "corte_naive"], how="left")

    faltan = out["valor_plantel"].isna().sum()
    if faltan:
        log.warning("%d filas sin valor de plantel (equipo sin valuaciones vigentes)", faltan)
    return out.drop(columns="corte_naive")
