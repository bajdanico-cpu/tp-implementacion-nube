"""De las tres probabilidades a UNA clase: la regla de decisión, explícita y versionada.

    python -m serving.decision                  # las reglas activas y en qué se diferencian
    python -m serving.decision --backfill       # etiqueta las predicciones ya registradas

## Por qué esto es un módulo y no un `argmax` suelto

El modelo devuelve **tres probabilidades**. Convertirlas en una predicción es un paso
aparte, y hasta ahora estaba implícito: `P.argmax(1)`, repetido en `serving/predict.py`,
en `monitoring/temporada_actual.py` y en cada script de evaluación.

`training/empate.py` demostró que ese paso es la **única palanca real** que queda sobre el
empate. Sobre el holdout de 380 partidos:

| regla | empates predichos | precisión | recall | accuracy global |
|---|---|---|---|---|
| `argmax` | 4 | 0,750 | 0,029 | 0,5000 |
| **`umbral_empate` 0,30** | **36** | 0,417 | 0,144 | **0,5079** |
| `umbral_empate` 0,26 | 103 | 0,301 | 0,298 | 0,4711 |

A 0,30 se recuperan 36 empates y la accuracy **sube**. Parece gratis — y ahí está la
trampa que este módulo no deja olvidar: **el AUC del empate es 0,515, IC95 [0,441-0,584]**.
Mover un umbral sobre una señal que no ordena cambia *cuántos* empates se anuncian, no
*cuáles*. Por eso el umbral no es una decisión de modelado sino **de negocio**: cuánto
cuesta perderse un empate contra cuánto cuesta anunciar uno que no fue.

> ⚠️ **Ese +0,0079 no sobrevivió a la medición** (05/09/2026). `training/decision_eval.py`
> repitió la comparación cambiando sólo la cantidad de semillas y sólo la semilla del
> walk-forward: el delta va de −0,005 a +0,026 sin que la regla cambie, y los pares
> discordantes salen siempre cerca de 50/50 (McNemar p entre 0,08 y 1,00). **El efecto del
> umbral es más chico que el ruido de semilla.** Lo que sí se mueve siempre es el F1 del
> empate (0,06 → 0,20), que es la trampa del volumen, no del acierto.
>
> Eso **no invalida** correr la regla en paralelo: la invalida como mejora demostrada. Sigue
> siendo la palanca de negocio correcta el día que haya un costo asimétrico entre perderse
> un empate y anunciar uno que no fue, y la temporada en curso es la única muestra que
> todavía no se miró.

## Lo que NO es

**No es un hiperparámetro del modelo.** Los boosters son los mismos y las probabilidades
son idénticas al bit; cambia sólo la función que va de `P` a la etiqueta. Eso tiene tres
consecuencias que hacen que medir en paralelo salga casi gratis:

1. **No hay que reentrenar ni guardar un segundo modelo.** El candidato no tiene `.ubj`
   propio: tiene una entrada en `config.yaml`.
2. **La comparación es pareada de la forma más fuerte posible** — mismos partidos, mismo
   modelo, misma información. McNemar (`training/promotion.py`) es exactamente el test que
   corresponde, y sólo cuentan los partidos donde las dos reglas discrepan.
3. **Las predicciones ya registradas se pueden etiquetar hacia atrás** sin inventar nada:
   el parquet guardó `p_away/p_draw/p_home`, y la regla es una función pura de esas tres
   columnas. `--backfill` agrega la columna del candidato sin tocar ni las probabilidades
   ni `predicted_at`.

## La honestidad del backfill

Etiquetar hacia atrás es reproducir, no predecir — pero **una fecha ya jugada, etiquetada
con una regla elegida después, no es evidencia prospectiva**. Por eso cada candidato
declara `desde` en `config.yaml`, y el monitoreo separa las fechas en dos: las anteriores
son *retrospectivas* (se muestran, no cuentan) y las posteriores son la medición de verdad.

El candidato `umbral_empate_030` se fijó el **2026-09-01**, con las fechas 1 y 2 ya jugadas
y la 3 todavía sin empezar. Su medición honesta arranca en la **fecha 3 de 2026-27**.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common.config import CFG, PROJECT_ROOT
from common.logging_setup import get_logger, setup
from eda.baselines import CLASES_ORD

log = get_logger(__name__)

PREDICCIONES = PROJECT_ROOT / "data" / "predicciones"

I_DRAW = list(CLASES_ORD).index("draw")
NO_DRAW = [i for i, c in enumerate(CLASES_ORD) if c != "draw"]

COL_PRODUCCION = "prediccion"       # el nombre histórico de la columna; no se toca
COL_REGLA = "regla_produccion"


# --------------------------------------------------------------------------- #
# Las reglas
# --------------------------------------------------------------------------- #

def _argmax(P: np.ndarray) -> np.ndarray:
    return np.asarray(CLASES_ORD)[P.argmax(axis=1)]


def _umbral_empate(P: np.ndarray, umbral: float) -> np.ndarray:
    """Empate si `p_draw >= umbral`; si no, el mejor entre local y visita.

    Ojo con la alternativa ingenua —enmascarar la columna del empate y hacer argmax—:
    cuando `p_draw` queda debajo del umbral pero sigue siendo el máximo de las tres, el
    argmax crudo devolvería `draw` igual. Hay que elegir explícitamente entre las OTRAS dos.
    """
    otros = np.asarray(CLASES_ORD)[np.asarray(NO_DRAW)[P[:, NO_DRAW].argmax(axis=1)]]
    return np.where(P[:, I_DRAW] >= umbral, "draw", otros)


TIPOS = {
    "argmax": lambda P, **kw: _argmax(P),
    "umbral_empate": lambda P, umbral, **kw: _umbral_empate(P, float(umbral)),
}


@dataclass(frozen=True)
class Regla:
    """Una regla de decisión con nombre propio. Es lo que se mide en paralelo."""

    nombre: str
    tipo: str
    params: dict[str, Any] = field(default_factory=dict)
    desde: str | None = None            # "<season> GW<n>": desde dónde la medición cuenta
    motivo: str = ""
    es_produccion: bool = False

    def __post_init__(self) -> None:
        if self.tipo not in TIPOS:
            raise ValueError(
                f"Regla de decisión desconocida: '{self.tipo}'. "
                f"Las disponibles son {sorted(TIPOS)}.")

    def aplicar(self, P: np.ndarray) -> np.ndarray:
        return TIPOS[self.tipo](np.asarray(P, dtype=float), **self.params)

    @property
    def columna(self) -> str:
        """Dónde vive esta regla en el parquet de predicciones."""
        return COL_PRODUCCION if self.es_produccion else f"pred_{self.nombre}"

    @property
    def etiqueta(self) -> str:
        ps = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.nombre} ({self.tipo}{': ' + ps if ps else ''})"

    def desde_partes(self) -> tuple[str, int] | None:
        """`"2026-27 GW3"` -> `("2026-27", 3)`. None si la regla no declara arranque."""
        if not self.desde:
            return None
        season, gw = self.desde.split()
        return season, int(gw.upper().removeprefix("GW"))

    def cuenta_para(self, season: str, gameweek: int) -> bool:
        """¿Esta fecha es medición prospectiva de la regla, o etiquetado hacia atrás?"""
        partes = self.desde_partes()
        if partes is None:
            return True
        s0, gw0 = partes
        return (season, gameweek) >= (s0, gw0)


# --------------------------------------------------------------------------- #
# Lo que dice config.yaml
# --------------------------------------------------------------------------- #

def produccion() -> Regla:
    """La regla que decide lo que el sistema **anuncia**. Por defecto, `argmax`."""
    raw = CFG.decision.get("produccion") or {}
    tipo = raw.get("regla", "argmax")
    params = {k: v for k, v in raw.items()
              if k not in ("regla", "nombre", "motivo", "desde")}
    return Regla(nombre=raw.get("nombre", tipo), tipo=tipo, params=params,
                 motivo=raw.get("motivo", ""), es_produccion=True)


def candidatos() -> list[Regla]:
    """Las reglas que corren **en paralelo**: se registran y se miden, no se anuncian."""
    out = []
    for raw in CFG.decision.get("candidatos") or []:
        params = {k: v for k, v in raw.items()
                  if k not in ("regla", "nombre", "motivo", "desde")}
        out.append(Regla(nombre=raw["nombre"], tipo=raw["regla"], params=params,
                         desde=raw.get("desde"), motivo=raw.get("motivo", "")))
    return out


def todas() -> list[Regla]:
    return [produccion(), *candidatos()]


# --------------------------------------------------------------------------- #
# Aplicar las reglas a un DataFrame de predicciones
# --------------------------------------------------------------------------- #

def matriz(df: pd.DataFrame) -> np.ndarray:
    """Las tres probabilidades en el orden de CLASES_ORD.

    El orden no es cosmético: un `argmax` sobre las columnas en otro orden devuelve la
    clase equivocada en silencio, que es la misma familia de error que el del log-loss.
    """
    faltan = [f"p_{c}" for c in CLASES_ORD if f"p_{c}" not in df.columns]
    if faltan:
        raise ValueError(f"Al DataFrame le faltan columnas de probabilidad: {faltan}")
    return df[[f"p_{c}" for c in CLASES_ORD]].to_numpy(dtype=float)


def etiquetar(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega una columna por regla activa. No toca las probabilidades ni `predicted_at`.

    La de producción sobrescribe `prediccion` — tiene que dar la misma cuenta que ya hacía
    `serving.predict`, y si algún día la regla de producción cambia, ésta es la única línea
    que lo decide.
    """
    P = matriz(df)
    out = df.copy()
    prod = produccion()
    out[COL_PRODUCCION] = prod.aplicar(P)
    out[COL_REGLA] = prod.etiqueta
    for r in candidatos():
        out[r.columna] = r.aplicar(P)
    return out


def discrepancias(df: pd.DataFrame) -> pd.DataFrame:
    """Los partidos donde algún candidato no dice lo mismo que producción."""
    d = etiquetar(df)
    cols = [r.columna for r in candidatos()]
    if not cols:
        return d.iloc[0:0]
    distinto = np.logical_or.reduce([(d[c] != d[COL_PRODUCCION]).to_numpy() for c in cols])
    return d[distinto]


# --------------------------------------------------------------------------- #
# Backfill del registro de predicciones
# --------------------------------------------------------------------------- #

def backfill(carpeta: Path | None = None, dry_run: bool = False) -> pd.DataFrame:
    """Etiqueta con todas las reglas las predicciones ya registradas.

    Es reproducir, no re-predecir: la regla es una función pura de `p_away/p_draw/p_home`,
    que quedaron congeladas en el parquet antes del deadline. Se preservan intactas esas
    columnas, `predicted_at`, `model_version` y `feature_set_version`; lo único que se
    agrega son las columnas de las reglas. Es idempotente.
    """
    carpeta = carpeta or PREDICCIONES
    filas = []
    for p in sorted(carpeta.glob("*.parquet")):
        antes = pd.read_parquet(p)
        despues = etiquetar(antes)
        nuevas = [c for c in despues.columns if c not in antes.columns]
        cambia_prod = (COL_PRODUCCION in antes.columns
                       and not antes[COL_PRODUCCION].equals(despues[COL_PRODUCCION]))
        if cambia_prod:
            # La regla de producción cambió de definición: eso no es un backfill, es otra
            # decisión. Se avisa fuerte en vez de reescribir el registro en silencio.
            log.warning("%s: la regla de producción cambia %d predicciones ya registradas",
                        p.name,
                        int((antes[COL_PRODUCCION] != despues[COL_PRODUCCION]).sum()))
        if nuevas and not dry_run:
            despues.to_parquet(p, index=False)
        filas.append({"archivo": p.name, "n": len(antes),
                      "columnas_agregadas": ", ".join(nuevas) or "-",
                      "produccion_cambia": cambia_prod})
    return pd.DataFrame(filas)


def ultimas_por_fecha(carpeta: Path | None = None) -> dict[tuple[str, int], Path]:
    """El registro más reciente de cada (temporada, gameweek). El nombre lleva el stamp."""
    carpeta = carpeta or PREDICCIONES
    out: dict[tuple[str, int], Path] = {}
    for p in sorted(carpeta.glob("*.parquet")):
        season, gw, _ = p.stem.split("_", 2)
        out[(season, int(gw.removeprefix("GW")))] = p    # sorted -> gana el stamp más nuevo
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Las reglas de decisión activas.")
    ap.add_argument("--backfill", action="store_true",
                    help="agrega las columnas de las reglas a las predicciones registradas")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    setup(CFG.log_level, CFG.log_format)

    print(f"\n{'=' * 78}\nREGLAS DE DECISIÓN\n{'=' * 78}\n")
    for r in todas():
        rol = "PRODUCCIÓN" if r.es_produccion else "candidato "
        print(f"  [{rol}] {r.etiqueta:<40s} -> columna `{r.columna}`")
        if r.desde:
            print(f"{'':15s}mide desde {r.desde}")
        if r.motivo:
            print(f"{'':15s}{r.motivo}")
    print()

    if args.backfill:
        res = backfill(dry_run=args.dry_run)
        print(f"{'-' * 78}\nBACKFILL{' (dry-run)' if args.dry_run else ''}\n{'-' * 78}")
        print(res.to_string(index=False) if not res.empty
              else "  no hay predicciones registradas")
        print()

    ultimas = ultimas_por_fecha()
    cands = candidatos()
    if not ultimas or not cands:
        return
    print(f"{'-' * 78}\nDÓNDE DISCREPAN, FECHA POR FECHA\n{'-' * 78}\n")
    for (season, gw), p in sorted(ultimas.items()):
        d = etiquetar(pd.read_parquet(p))
        marcas = []
        for r in cands:
            n = int((d[r.columna] != d[COL_PRODUCCION]).sum())
            marcas.append(f"{r.nombre}: {n}/{len(d)} "
                          f"({'mide' if r.cuenta_para(season, gw) else 'retrospectivo'})")
        print(f"  {season} GW{gw:<3d} {'  |  '.join(marcas)}")
        for r in cands:
            for x in d[d[r.columna] != d[COL_PRODUCCION]].itertuples():
                print(f"{'':16s}{x.home_short}-{x.away_short}  "
                      f"{getattr(x, COL_PRODUCCION)} -> {getattr(x, r.columna)}   "
                      f"(p_draw {x.p_draw:.3f})")
    print()


if __name__ == "__main__":
    main()
