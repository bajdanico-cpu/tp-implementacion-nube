"""CLI unificada de ingesta Bronze.

Pensada para correr en local, a mano, cuando haga falta:

    python -m ingestion.run                          # todas las fuentes
    python -m ingestion.run --source fpl             # sólo el snapshot de la API
    python -m ingestion.run --source fpl --gw 1      # + resultados de la GW1
    python -m ingestion.run --season 2025-26         # una temporada puntual
    python -m ingestion.run --force                  # re-baja lo cacheado

Política de caché
-----------------
- Temporadas **cerradas**: si ya están en Bronze no se vuelven a bajar (no cambian).
- Temporada **actual**: siempre se re-baja (se actualiza cada fecha).
- API de FPL: nunca se cachea — cada corrida es un snapshot con valor propio.
"""

from __future__ import annotations

import argparse
import sys

from common.config import CFG
from common.logging_setup import get_logger, setup
from ingestion import bronze_footballdata, bronze_fpl, bronze_vaastav

log = get_logger(__name__)

SOURCES = ("fpl", "vaastav", "football_data")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ingestion.run",
        description="Ingesta Bronze del TP Premier ML.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--source", choices=(*SOURCES, "all"), default="all",
        help="Fuente a ingestar (default: all).",
    )
    p.add_argument(
        "--season", action="append", default=None,
        help="Temporada en formato 2025-26. Repetible. Default: todas las configuradas.",
    )
    p.add_argument(
        "--gw", type=int, default=None,
        help="Gameweek para /event/{GW}/live/ (sólo aplica a la fuente fpl).",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Ignora la caché y vuelve a bajar todo.",
    )
    p.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING...")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup(args.log_level or CFG.log_level, CFG.log_format)

    seasons = args.season or CFG.seasons_to_ingest()

    unknown = [s for s in seasons if s not in CFG.seasons_to_ingest()]
    if unknown:
        log.error("Temporadas fuera del rango configurado (%s..%s): %s",
                  CFG.ingest_from, CFG.current_season, unknown)
        log.error("Ampliá seasons.ingest_from en config.yaml si las necesitás.")
        return 2

    targets = SOURCES if args.source == "all" else (args.source,)

    log.info("Ingesta Bronze | fuentes=%s | temporadas=%s | force=%s",
             list(targets), seasons, args.force)
    log.info("Destino: %s", CFG.data_root / "bronze")

    failures: list[str] = []

    for source in targets:
        try:
            if source == "vaastav":
                bronze_vaastav.run(seasons=seasons, force=args.force)
            elif source == "football_data":
                bronze_footballdata.run(seasons=seasons, force=args.force)
            elif source == "fpl":
                bronze_fpl.run(gameweek=args.gw, force=args.force)
        except Exception as exc:  # noqa: BLE001 — una fuente caída no debe tumbar el resto
            log.exception("La fuente '%s' falló: %s", source, exc)
            failures.append(source)

    if failures:
        log.error("Ingesta terminada CON ERRORES en: %s", failures)
        return 1

    log.info("Ingesta Bronze completada.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
