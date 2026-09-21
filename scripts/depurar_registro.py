"""Dejar en el registro una sola predicción por fecha: la del modelo de producción.

    python -m scripts.depurar_registro              # muestra qué haría
    python -m scripts.depurar_registro --aplicar    # lo hace

El registro acumuló 18 parquets para cinco fechas. No es rastro de auditoría: son
sobras de desarrollo de dos clases distintas, y **ninguna de las dos es una predicción
del sistema tal como quedó**.

  * **De otros modelos.** Entre el 24 y el 25 de agosto el modelo se reentrenó seis
    veces, y cada corrida dejó su predicción de la GW2. Esas versiones no existen en
    `models/` —el TP fija una sola— así que sus predicciones apuntan a un modelo que
    nadie puede cargar ni auditar.
  * **Posteriores al corte.** Re-predecir una fecha ya jugada no es predecir: es
    reconstruir con el dato de hoy. `registro.congelada` ya las descarta al elegir, pero
    seguían ocupando lugar y contradiciendo el relato del entregable.

Lo que queda, por fecha, es **la que gobierna**: la última que el modelo de producción
emitió antes del corte. Si no hubo ninguna a tiempo —la GW1 se jugó antes de que el
sistema registrara— queda la más temprana, que la API sigue marcando con
`registro_pre_deadline=False`.

**Nada se borra.** Lo descartado se mueve a `data/_registro_desarrollo/`, fuera del
glob del registro y fuera del zip de la demo, pero ahí para quien quiera mirarlo.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

from common.config import CFG
from common.logging_setup import get_logger
from serving import registro
from training import registry

log = get_logger(__name__)

DESCARTADAS = CFG.data_root / "_registro_desarrollo"


def _emitida(ruta: Path) -> pd.Timestamp:
    return pd.Timestamp(pd.read_parquet(ruta)["predicted_at"].iloc[0])


def gobernante(rutas: list[Path]) -> tuple[Path, bool]:
    """Cuál de esas predicciones vale, y si llegó a tiempo.

    Misma regla que `registro.congelada`, aplicada acá sobre un subconjunto ya filtrado
    por modelo. Duplicarla sería invitar a que se separen: por eso el test compara que
    lo que deja este script sea exactamente lo que `congelada` elige después.
    """
    con_fecha = sorted(((_emitida(r), r) for r in rutas), key=lambda t: t[0])
    corte = pd.Timestamp(pd.read_parquet(con_fecha[0][1])["kickoff_time"].min())
    a_tiempo = [(t, r) for t, r in con_fecha if t < corte]
    if a_tiempo:
        return a_tiempo[-1][1], True
    return con_fecha[0][1], False


def plan(nombre: str | None = None) -> tuple[list[Path], list[Path], list[str]]:
    """(se quedan, se descartan, avisos). No toca nada."""
    nombre = nombre or CFG.modelo
    v = registry.produccion(nombre)
    if v is None:
        raise SystemExit(
            f"No hay modelo de producción para {nombre}. Promové uno primero:\n"
            f"    python -m training.registry --listar\n"
            f"    python -m training.registry --promover <VERSION> --motivo '...'")

    todos = registro.listar()
    if todos.empty:
        return [], [], ["El registro está vacío."]

    quedan: list[Path] = []
    fuera: list[Path] = []
    avisos: list[str] = []

    for (season, gw), grupo in todos.groupby(["season", "gameweek"]):
        rutas = list(grupo["ruta"])
        del_modelo = [r for r in rutas
                      if str(pd.read_parquet(r)["model_version"].iloc[0]) == v.version]

        if not del_modelo:
            # El modelo de producción nunca predijo esta fecha. Descartar lo que hay
            # dejaría un hueco (409) y quedárselo sería servir un modelo que no existe:
            # ninguna de las dos se decide sola, así que no se toca y se avisa.
            avisos.append(
                f"{season} GW{gw:02d}: ninguna predicción de {v.version}. "
                f"Se dejan los {len(rutas)} archivos como están; "
                f"regenerá con: python -m serving.predict --gw {gw}")
            quedan.extend(rutas)
            continue

        elegida, a_tiempo = gobernante(del_modelo)
        quedan.append(elegida)
        fuera.extend(r for r in rutas if r != elegida)
        if not a_tiempo:
            avisos.append(f"{season} GW{gw:02d}: la que queda es POSTERIOR al corte "
                          f"(reconstrucción). La API la marca como tal.")

    return quedan, fuera, avisos


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--aplicar", action="store_true",
                    help="mover de verdad (por defecto sólo muestra)")
    ap.add_argument("--modelo", default=None, help=f"por defecto {CFG.modelo}")
    args = ap.parse_args(argv)

    quedan, fuera, avisos = plan(args.modelo)

    print(f"\nSe quedan {len(quedan)}:")
    for r in sorted(quedan):
        print(f"   {r.name}")
    print(f"\nSe descartan {len(fuera)} -> {DESCARTADAS.name}/:")
    for r in sorted(fuera):
        print(f"   {r.name}")
    for a in avisos:
        print(f"\n  ! {a}")

    if not fuera:
        print("\nNada que hacer.")
        return 0

    if not args.aplicar:
        print("\nNo se movió nada. Para hacerlo: "
              "python -m scripts.depurar_registro --aplicar")
        return 0

    DESCARTADAS.mkdir(parents=True, exist_ok=True)
    for r in fuera:
        shutil.move(str(r), str(DESCARTADAS / r.name))
    print(f"\n{len(fuera)} archivos movidos a {DESCARTADAS}")
    print(f"Quedan {len(quedan)} predicciones, una por fecha.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
