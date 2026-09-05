"""Interacciones de matchup: no cuánto vale cada equipo, sino cómo se cruzan sus estilos.

La hipótesis del bloque es vieja y tiene nombre en el fútbol: **hay estilos que le ganan a
otros más de lo que predice la diferencia de nivel**. Un equipo que entra al área contra uno
que defiende lejos del arco; uno de posesión contra uno que roba arriba.

## Qué es un matchup y qué no

Gold ya tiene tres formas de mirar a los dos equipos, y **ninguna es un matchup**:

    local_posesion_u5      lo que hace uno
    visita_posesion_u5     lo que hace el otro
    dif_posesion_u5        la resta: LA MISMA estadística de los dos lados

Un matchup cruza **una estadística de un equipo con OTRA, distinta, del rival**: lo que yo
genero contra lo que vos concedés. Eso no se puede escribir como una resta de la misma
columna, y por eso estas features existen.

## Las advertencias, adelante y no al final

**1 · XGBoost ya modela interacciones.** Un árbol que parte primero por `local_posesion` y
después por `visita_quites` está representando ese producto. Así que la pregunta no es si el
modelo *puede* verlas, es si **alcanzan los datos para estimarlas** — y con 1.004 filas de
entrenamiento la respuesta no es obvia.

**2 · El historial del proyecto baja la expectativa.** Las 24 features de competencias y las
56 de Opta se usan (9,5 % y 19,8 % de la ganancia) y **no** se tradujeron en mejora. La
Fase 5 dejó el mismo patrón medido con más precisión: `dif_valor_top11` salió sexta de 274 y
el modelo no mejoró, porque correlacionaba 0,73 con `dif_elo`.

**3 · Por eso la versión barata, y no clusters.** Agrupar equipos en ~5 estilos y cruzarlos
da 25 celdas sobre 1.140 partidos: ~45 por celda, que es el régimen donde aparece un patrón
espurio. Siete términos elegidos por teoría se testean de a uno y se explican en una línea.

## Los siete

Cada uno se lee "lo que genera A contra lo que permite B". Los cuatro primeros van en las
dos direcciones porque el fútbol no es simétrico: que el local entre al área contra un
visitante que lo permite es un hecho distinto del recíproco.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common.logging_setup import get_logger

log = get_logger(__name__)

# nombre -> (columna del que ATACA, columna del que DEFIENDE/RESPONDE, descripcion)
#
# El sufijo `_l` significa que el termino se arma con el local como lado atacante, y `_v`
# con el visitante. Se guardan los dos porque el partido no es simetrico.
MATCHUPS: dict[str, tuple[str, str, str]] = {
    "mu_area_l": (
        "local_prop_tiros_area_u5", "visita_prop_tiros_area_conc_u5",
        "cuanto entra al area el local por cuanto deja entrar el visitante. El cruce mas "
        "directo de calidad de situacion: generar desde adentro contra permitir desde "
        "adentro"),
    "mu_area_v": (
        "visita_prop_tiros_area_u5", "local_prop_tiros_area_conc_u5",
        "el mismo cruce con el visitante como atacante"),
    "mu_posesion_quites_l": (
        "local_posesion_u5", "visita_quites_u5",
        "posesion del local por quites del visitante: el choque de estilos clasico, tener "
        "la pelota contra robarla"),
    "mu_posesion_quites_v": (
        "visita_posesion_u5", "local_quites_u5",
        "el mismo choque en la otra direccion"),
    "mu_pase_intercepciones_l": (
        "local_precision_pases_u5", "visita_intercepciones_u5",
        "precision de pase del local por intercepciones del visitante: elaborar contra "
        "cortar. Es distinto de quites -- interceptar es leer el pase, quitar es el duelo"),
    "mu_lejos_bloqueos_l": (
        "local_tiros_fuera_u5", "visita_bloqueos_u5",
        "tiros de afuera del area del local por bloqueos del visitante: patear de lejos "
        "contra un equipo que se tira a taparlos"),
    "mu_aereo_l": (
        "local_toques_area_rival_u5", "visita_prop_aereos_ganados_u5",
        "presencia del local en el area rival por cuanto gana por arriba el visitante. "
        "Volumen de ataque contra la forma de defenderlo"),
}

# La asimetria de estilo, que no es un producto sino una magnitud.
ASIMETRIA = "mu_asimetria_posesion"

COLUMNAS = list(MATCHUPS) + [ASIMETRIA]


def construir(gold: pd.DataFrame) -> pd.DataFrame:
    """Agrega las columnas de matchup a Gold. Requiere las de Opta ya pegadas.

    Si falta alguna columna base, la feature queda en NaN en vez de romper: las de Opta
    existen sólo desde que la API oficial las publica, y un NaN es información honesta que
    los árboles manejan.
    """
    out = gold.copy()
    faltantes = []
    for nombre, (a, b, _) in MATCHUPS.items():
        if a not in out.columns or b not in out.columns:
            faltantes.append(nombre)
            out[nombre] = np.nan
            continue
        out[nombre] = out[a] * out[b]

    if "local_posesion_u5" in out.columns and "visita_posesion_u5" in out.columns:
        # No es un `dif_`: es su VALOR ABSOLUTO. Dice cuan desparejo es el partido en
        # estilo, sin importar para que lado -- dos equipos que quieren la pelota producen
        # un partido distinto de uno que la quiere y otro que la cede, y el signo no
        # distingue esos dos casos.
        out[ASIMETRIA] = (out["local_posesion_u5"] - out["visita_posesion_u5"]).abs()
    else:
        faltantes.append(ASIMETRIA)
        out[ASIMETRIA] = np.nan

    if faltantes:
        log.warning("Matchups sin columnas base, quedan en NaN: %s", faltantes)
    return out
