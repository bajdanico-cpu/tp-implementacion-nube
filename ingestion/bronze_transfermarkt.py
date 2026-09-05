"""Bronze — valores de mercado y transferencias reales, de Transfermarkt.

    python -m ingestion.bronze_transfermarkt

## Por qué esta fuente y no scrapear Transfermarkt

Transfermarkt no tiene API pública y scrapear su sitio va contra sus términos. Lo que sí
existe es **[`dcaribou/transfermarkt-datasets`](https://github.com/dcaribou/transfermarkt-datasets)**:
un dataset derivado, publicado como CSV en un bucket público, con doce tablas. Se baja sin
credenciales y se cita.

## Por qué hace falta

Es la primera fuente del proyecto que trae **información de afuera** en vez de recombinar lo
que ya está. Todo lo demás —forma, Elo, xG, puntos de FPL, estadísticas de Opta— sale de
resultados pasados de la Premier. Un ascendido no tiene resultados pasados de Premier, y por
eso `training/README.md` deja medido que el mercado de apuestas le saca **9,3 puntos de
accuracy** al modelo en esos partidos contra 0,7 en el resto: ellos tienen altas, bajas y
precios; nosotros no.

Y esto **cubre el Championship**: un ascendido llega con valores de mercado reales de antes
de pisar la Premier. Ninguna otra fuente del proyecto puede dar eso.

## Las dos trampas de esta fuente

**1 · Los filiales y los homónimos.** En `player_valuations` conviven `Manchester City` con
`Manchester City U21`, `Manchester City Reserves` y `Manchester City U18` — y, peor,
`Newcastle United Jets`, que es un club **australiano**. El cruce va por nombre exacto contra
un mapa explícito, nunca por `contains`.

**2 · El dataset dejó de actualizarse el 06/07/2026.** Para la ventana de entrenamiento y el
holdout (2022-23 a 2025-26) está completo. Para 2026-27 queda **a mitad del mercado de
pases**: los refuerzos de agosto no están. No es leakage —los datos son viejos, no futuros—
pero sí es una feature más pobre justo en la temporada en curso, y hay que decirlo cuando se
lea el monitoreo.
"""

from __future__ import annotations

import argparse
import gzip
import io
import time

import pandas as pd

from common.config import CFG, utc_stamp
from common.logging_setup import get_logger, setup
from common.storage import latest_snapshot, sha256, write_manifest, write_raw
from ingestion.http_utils import fetch

log = get_logger(__name__)

SOURCE = "transfermarkt"
DATASET = "raw"
# Bucket publico del proyecto. Los archivos van comprimidos.
BASE = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"
# La temporada es una etiqueta: el dataset es global y no se particiona por temporada
# nuestra. Se usa la actual para que conviva con el resto de Bronze sin inventar un esquema.
TABLAS = ("competitions", "clubs", "players", "player_valuations", "transfers")
PAUSA_S = 0.5


def descargar(tabla: str, timeout: int = 180) -> bytes | None:
    """Baja y **descomprime** una tabla. Se guarda el CSV plano, no el .gz.

    Guardar descomprimido es a propósito: Bronze es la copia auditable de lo que se bajó, y
    un `.gz` obliga a descomprimir para mirarlo. El costo en disco es despreciable frente a
    los 180 MB que ya ocupa Bronze.
    """
    res = fetch(f"{BASE}/{tabla}.csv.gz", timeout=timeout)
    if not res.ok or not res.content:
        log.warning("[%s] no disponible (HTTP %s)", tabla, res.status)
        return None
    try:
        return gzip.decompress(res.content)
    except (OSError, EOFError) as exc:
        log.error("[%s] el archivo no es un gzip valido: %s", tabla, exc)
        return None


def ingest(tabla: str, stamp: str, force: bool = False) -> dict:
    season = CFG.current_season
    filename = f"{tabla}.csv"

    snap = latest_snapshot(SOURCE, season, DATASET)
    if not force and snap and (snap / filename).exists():
        return {"tabla": tabla, "skipped": True}

    crudo = descargar(tabla)
    if crudo is None:
        return {"tabla": tabla, "rejected": True}

    # Control minimo: que sea un CSV parseable y no una pagina de error.
    try:
        d = pd.read_csv(io.BytesIO(crudo), nrows=50)
    except Exception as exc:  # noqa: BLE001
        log.error("[%s] no se pudo parsear como CSV: %s", tabla, str(exc)[:120])
        return {"tabla": tabla, "rejected": True, "reason": "csv ilegible"}

    write_raw(SOURCE, season, DATASET, filename, crudo, stamp=stamp)
    log.info("[%s] %d columnas, %.1f MB", tabla, len(d.columns), len(crudo) / 1e6)
    return {"tabla": tabla, "file": filename, "bytes": len(crudo),
            "sha256": sha256(crudo), "cols": len(d.columns), "rejected": False}


def run(force: bool = False, tablas: tuple[str, ...] = TABLAS) -> pd.DataFrame:
    stamp = utc_stamp()
    log.info("=== Bronze Transfermarkt — %d tablas ===", len(tablas))
    filas = []
    for t in tablas:
        r = ingest(t, stamp, force=force)
        filas.append(r)
        if not r.get("skipped"):
            time.sleep(PAUSA_S)
    entradas = [r for r in filas if not r.get("skipped")]
    if entradas:
        write_manifest(SOURCE, CFG.current_season, DATASET, entradas, stamp)
    return pd.DataFrame(filas)


def main() -> None:
    ap = argparse.ArgumentParser(description="Transfermarkt: valores y transferencias.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    setup(CFG.log_level, CFG.log_format)
    d = run(force=args.force)
    print(f"\n{d.to_string(index=False)}\n")


if __name__ == "__main__":
    main()
