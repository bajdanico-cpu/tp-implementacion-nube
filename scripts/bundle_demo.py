"""Arma el paquete con el estado congelado de la demo, para subirlo a Cloud Shell.

    python -m scripts.bundle_demo

Existe por un problema concreto: **el dato no está en git**. `data/` y los `.ubj` están
en `.gitignore` a propósito —pesan y se regeneran— así que un `git pull` en Cloud Shell
trae el código y nada más.

Y regenerarlo allá no sirve para la demo: `pipeline.pre_deadline` ingestaría los
resultados de la GW5, que ya se jugó, y la transición que queremos mostrar en vivo
—apretar el botón y que aparezca la GW6— desaparecería. El estado tiene que viajar
congelado.

Lo que entra, y sólo eso:

    data/gold/          la tabla de features, con la GW5 como inferencia   ~1,5 MB
    data/predicciones/  lo que el sistema anunció en cada fecha            ~300 KB
    models/<produccion> el modelo que sirve, y su trazabilidad             ~2 MB
    data/silver/        opcional (--con-silver), para que el Job arranque      4 MB
                        sin re-derivar todo

Bronze NO entra: son ~300 MB. Sin él, la primera corrida del Job vuelve a bajar las
cinco temporadas y tarda bastante más. Si tenés `gcloud` en tu máquina conviene saltear
este paquete y subir todo directo:

    gcloud storage rsync -r data/bronze gs://TU-BUCKET/bronze
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from common.config import CFG, PROJECT_ROOT
from common.logging_setup import get_logger, setup
from training import registry

log = get_logger(__name__)

SALIDA = PROJECT_ROOT / "demo-premier-ml.zip"


def _agregar(z: zipfile.ZipFile, origen: Path, prefijo: str) -> tuple[int, int]:
    n = total = 0
    for f in sorted(origen.rglob("*")):
        if not f.is_file():
            continue
        z.write(f, f"{prefijo}/{f.relative_to(origen).as_posix()}")
        n += 1
        total += f.stat().st_size
    return n, total


def construir(con_silver: bool = True, salida: Path = SALIDA) -> Path:
    v = registry.produccion(CFG.modelo)
    if v is None:
        raise SystemExit(
            "No hay modelo de producción declarado. Corré:\n"
            "  python -m training.registry --listar\n"
            "  python -m training.registry --promover <VERSION> --motivo '...'")
    if not registry.tiene_boosters(v):
        raise SystemExit(f"La versión {v.version} no tiene .ubj: no se puede servir.")

    piezas = [
        (CFG.gold_root, "data/gold"),
        (CFG.data_root / "predicciones", "data/predicciones"),
        (v.ruta, f"models/{v.nombre}/{v.version}"),
    ]
    if con_silver:
        piezas.append((CFG.silver_root, "data/silver"))

    total = 0
    with zipfile.ZipFile(salida, "w", zipfile.ZIP_DEFLATED) as z:
        for origen, prefijo in piezas:
            if not origen.exists():
                log.warning("No existe %s, se saltea", origen)
                continue
            n, b = _agregar(z, origen, prefijo)
            total += b
            log.info("%-28s %3d archivos  %6.1f MB", prefijo, n, b / 1e6)

        # El PRODUCTION.json va aparte: vive un nivel arriba de la versión.
        prod = registry.RAIZ / v.nombre / registry.PRODUCCION
        if prod.exists():
            z.write(prod, f"models/{v.nombre}/{registry.PRODUCCION}")
            log.info("%-28s   1 archivo", f"models/{v.nombre}/PRODUCTION.json")

    log.info("Listo: %s  (%.1f MB comprimido, %.1f MB sin comprimir)",
             salida.name, salida.stat().st_size / 1e6, total / 1e6)
    return salida


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Paquete con el estado congelado de la demo.")
    p.add_argument("--sin-silver", action="store_true",
                   help="No incluye data/silver (el Job lo re-deriva).")
    p.add_argument("--salida", default=str(SALIDA))
    args = p.parse_args(argv)

    setup()
    ruta = construir(con_silver=not args.sin_silver, salida=Path(args.salida))

    print(f"""
  Subilo a Cloud Shell y desempaquetalo en la raíz del repo:

    1. En Cloud Shell: menú de tres puntos -> "Subir" -> elegí {ruta.name}
    2. cd ~/tp-implementacion-nube
    3. unzip -o ~/{ruta.name}
    4. bash scripts/preparar_demo.sh

  El paso 3 deja data/ y models/ exactamente como están acá, con la GW5 predicha y
  sin ingestar. Eso es lo que hace que el botón tenga algo para mostrar en vivo.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
