"""El histórico de Silver y Gold: ver qué versiones hay, etiquetarlas y volver a una.

    python -m common.versiones                              # que hay guardado
    python -m common.versiones --snapshot "antes de fase 1" # congela lo vigente, con nombre
    python -m common.versiones --restaurar gold_tp_match 20260901T203100Z
    python -m common.versiones --diff gold_tp_match         # que cambio entre versiones

## Por qué existe

`data/` está en `.gitignore` —y tiene que estarlo: son 180 MB que se regeneran— así que
para Silver y Gold **no hay red de git**. Hasta ahora `write_table` escribía encima, lo que
significaba que una corrida distraída de `python -m features.gold_tp` destruía sin rastro
el Gold con el que se entrenó el modelo que está sirviendo.

Bronze nunca tuvo ese problema (es append-only desde el día uno) y los modelos tampoco
(cada corrida es una carpeta con su timestamp). Silver y Gold eran el agujero.

Ahora `common.storage.archivar` aparta la versión vigente antes de cada escritura, y este
módulo es la ventana a ese histórico. **Ninguna operación de acá borra nada**: `--restaurar`
archiva la versión que está viva antes de reemplazarla, así que restaurar también es
reversible.

## La disciplina, para que esto sirva de algo

Un Gold archivado sin contexto es un parquet más. Antes de una etapa de desarrollo que
cambie features, conviene:

    TP_VERSION_LABEL="antes de fase 1: pi-ratings" python -m features.gold_tp

o correr `--snapshot` con la etiqueta, que hace lo mismo sobre todo lo vigente. La etiqueta
queda en el manifiesto y es lo que después permite contestar "¿con qué Gold se entrenó el
modelo `20260825T024144Z`?" sin adivinar por fecha.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from common.config import CFG
from common.logging_setup import get_logger, setup
from common.storage import BACKEND, archivar, versiones, versiones_root

log = get_logger(__name__)

CAPAS = {"silver": lambda: CFG.silver_root, "gold": lambda: CFG.gold_root}


def vivas(layer: str) -> list:
    """Los archivos que están vigentes en una capa (parquet y json sueltos)."""
    root = CAPAS[layer]()
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir()
                  if p.is_file() and p.suffix in (".parquet", ".json"))


def inventario() -> pd.DataFrame:
    """Una fila por versión archivada, más una por cada tabla viva."""
    filas = []
    for layer in CAPAS:
        for p in vivas(layer):
            filas.append({"layer": layer, "tabla": p.stem, "stamp": "VIGENTE",
                          "filas": None, "columnas": None,
                          "MB": round(p.stat().st_size / 1e6, 2), "etiqueta": ""})
        raiz = versiones_root(layer)
        if not raiz.exists():
            continue
        for carpeta in sorted(raiz.iterdir()):
            for v in versiones(layer, carpeta.name):
                filas.append({"layer": layer, "tabla": v["tabla"], "stamp": v["stamp"],
                              "filas": v.get("filas"), "columnas": v.get("columnas"),
                              "MB": round(v.get("bytes", 0) / 1e6, 2),
                              "etiqueta": v.get("etiqueta", "")})
    return pd.DataFrame(filas)


def snapshot(etiqueta: str) -> pd.DataFrame:
    """Archiva TODO lo vigente con una etiqueta. Es el "duplicalo antes de tocar nada"."""
    filas = []
    for layer in CAPAS:
        for p in vivas(layer):
            destino = archivar(p, layer, etiqueta=etiqueta)
            filas.append({"layer": layer, "archivo": p.name,
                          "resultado": destino.name if destino
                          else "ya estaba archivado con este contenido"})
    return pd.DataFrame(filas)


def restaurar(layer: str, tabla: str, stamp: str) -> None:
    """Vuelve una versión archivada a ser la vigente. Archiva la actual antes.

    Restaurar tampoco destruye: lo que estaba vivo queda en el histórico, así que se puede
    ir y volver.
    """
    candidatas = [v for v in versiones(layer, tabla) if v["stamp"] == stamp]
    if not candidatas:
        disponibles = [v["stamp"] for v in versiones(layer, tabla)]
        raise ValueError(f"No hay version '{stamp}' de {layer}.{tabla}. "
                         f"Disponibles: {disponibles or '(ninguna)'}")
    v = candidatas[0]
    origen = versiones_root(layer, tabla) / v["archivo"]
    destino = CAPAS[layer]() / f"{tabla}{origen.suffix}"

    archivar(destino, layer, etiqueta=f"reemplazada al restaurar {stamp}")
    BACKEND.write_bytes(destino, BACKEND.read_bytes(origen))
    log.info("RESTAURADO %s.%s <- %s (%s filas, etiqueta '%s')",
             layer, tabla, stamp, v.get("filas"), v.get("etiqueta", ""))


def diff(layer: str, tabla: str) -> pd.DataFrame:
    """Qué cambió de una versión a la siguiente: filas, columnas y peso."""
    vs = versiones(layer, tabla)
    if not vs:
        return pd.DataFrame()
    d = pd.DataFrame([{"stamp": v["stamp"], "creada_at": v.get("creada_at"),
                       "filas": v.get("filas"), "columnas": v.get("columnas"),
                       "MB": round(v.get("bytes", 0) / 1e6, 2),
                       "etiqueta": v.get("etiqueta", "")} for v in vs])
    d["d_filas"] = d["filas"].diff()
    d["d_columnas"] = d["columnas"].diff()
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description="El histórico de Silver y Gold.")
    ap.add_argument("--snapshot", metavar="ETIQUETA",
                    help="archiva todo lo vigente con esta etiqueta")
    ap.add_argument("--restaurar", nargs=2, metavar=("TABLA", "STAMP"))
    ap.add_argument("--layer", default=None, choices=sorted(CAPAS))
    ap.add_argument("--diff", metavar="TABLA")
    args = ap.parse_args()
    setup(CFG.log_level, CFG.log_format)

    if args.snapshot:
        print(f"\nSNAPSHOT — etiqueta: {args.snapshot}\n")
        print(snapshot(args.snapshot).to_string(index=False))
        print()

    if args.restaurar:
        tabla, stamp = args.restaurar
        layer = args.layer or next(
            (c for c in CAPAS if versiones(c, tabla)), None)
        if layer is None:
            raise SystemExit(f"No hay versiones archivadas de '{tabla}' en ninguna capa.")
        restaurar(layer, tabla, stamp)
        print(f"\n{layer}.{tabla} restaurado a {stamp}. "
              f"La version que estaba viva quedo archivada.\n")

    if args.diff:
        layer = args.layer or next((c for c in CAPAS if versiones(c, args.diff)), None)
        if layer is None:
            raise SystemExit(f"No hay versiones archivadas de '{args.diff}'.")
        print(f"\nHISTORIA DE {layer}.{args.diff}\n")
        print(diff(layer, args.diff).round(2).to_string(index=False))
        print()
        return

    inv = inventario()
    if inv.empty:
        print("No hay nada en Silver ni en Gold todavia.")
        return
    print(f"\n{'=' * 92}\nSILVER Y GOLD: VIGENTE + HISTORICO\n{'=' * 92}\n")
    print(inv.to_string(index=False))
    archivadas = int((inv["stamp"] != "VIGENTE").sum())
    print(f"\n  {archivadas} versiones archivadas en {CFG.data_root / '_versiones'}")
    print("  Nada se pisa: `write_table` aparta la vigente antes de cada escritura.\n")


if __name__ == "__main__":
    main()
