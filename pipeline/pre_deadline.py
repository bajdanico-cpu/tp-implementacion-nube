"""La cadena que hay que correr antes del deadline de cada fecha.

    python -m pipeline.pre_deadline                 # detecta la próxima fecha sola
    python -m pipeline.pre_deadline --gw 6
    python -m pipeline.pre_deadline --desde gold    # retoma sin volver a bajar 27 MB
    python -m pipeline.pre_deadline --solo gold,predecir
    python -m pipeline.pre_deadline --dry-run       # no escribe Gold ni el registro

Reemplaza a la secuencia de siete comandos que vivía en el README. No es sólo comodidad:
saltearse un paso no falla, **degrada en silencio**. El control anti-leakage detecta
información del futuro y nunca información vieja, así que predecir con un Silver
desactualizado daba diez predicciones de aspecto normal calculadas con la historia que
hubiera. Encadenar los pasos es lo que convierte ese error en imposible.

Está escrito para correr como Cloud Run Job —`correr()` no llama a `sys.exit`, y cada
corrida deja su JSON— pero por ahora se invoca a mano.

**Qué pasa si un paso falla:**

    bronze_fpl      fixtures y deadlines nuevos      sigue con el snapshot previo;
                                                     corta si no hay ninguno
    bronze_vaastav  stats de jugador de la fecha     corta en `gold`: el criterio de
                                                     definitividad lo atrapa antes
    bronze_fd       cuotas de cierre                 OPCIONAL: no son features
    bronze_opta     ventanas de Opta                 corta: sería skew silencioso
    silver          todo                             corta
    gold            el control anti-leakage          no se escribe nada; el Gold
                                                     vigente queda intacto
    predecir        el registro de la fecha          corta; `--desde predecir` retoma
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from common.config import CFG
from common.logging_setup import get_logger, setup
from pipeline.pasos import Paso, Resultado, correr as correr_pasos

log = get_logger(__name__)


def _fecha_objetivo(gameweek: int | None, season: str) -> int | None:
    """La fecha que se va a predecir: la pedida, o la próxima predecible."""
    if gameweek is not None:
        return int(gameweek)

    from common.storage import read_table
    from features import calendario

    gw = calendario.proxima_predecible(read_table("fact_fixture"),
                                       read_table("fact_match"), season)
    if gw is None:
        log.warning("No hay ninguna fecha predecible en %s.", season)
    return gw


def _guard_pre_deadline(season: str, gameweek: int, forzar: bool) -> None:
    """Aborta si la fecha ya arrancó.

    El comando se llama `pre_deadline` por algo. Emitir con la fecha empezada y dejarlo
    en el registro contamina la evidencia: el monitoreo la contaría como predicción
    cuando en realidad es una reconstrucción hecha con información posterior.
    """
    from common.storage import read_table

    fx = read_table("fact_fixture")
    d = fx[(fx["season"] == season) & (fx["gameweek"] == gameweek)]
    if d.empty:
        return

    corte = d["kickoff_time"].min()
    ahora = pd.Timestamp.now(tz="UTC")
    if ahora < corte:
        horas = (corte - ahora).total_seconds() / 3600
        log.info("Faltan %.1f h para el inicio de %s GW%d (%s).", horas, season, gameweek, corte)
        return

    msg = (f"{season} GW{gameweek} arrancó el {corte} y ya pasó. Una predicción emitida "
           f"ahora no es una predicción: es una reconstrucción.")
    if not forzar:
        raise RuntimeError(msg + " Usá --forzar si aun así querés registrarla.")
    log.warning("%s Se continúa por --forzar.", msg)


def _pasos(season: str, gameweek: int | None, dry_run: bool,
           forzar: bool, force_ingesta: bool) -> list[Paso]:
    from features import gold_tp
    from ingestion import (bronze_footballdata as fd, bronze_fpl as fpl,
                           bronze_pulselive as pulse, bronze_vaastav as vaastav)
    from serving import predict
    from transform import competencias, opta_stats, silver

    def _predecir():
        gw = _fecha_objetivo(gameweek, season)
        if gw is None:
            raise RuntimeError("No hay fecha predecible: no hay nada que predecir.")
        _guard_pre_deadline(season, gw, forzar)

        pred = predict.predecir(season, gw)
        if dry_run:
            log.info("[dry-run] %s GW%d predicha, no se registra.", season, gw)
            return {"gameweek": gw, "partidos": len(pred), "registrado": False}
        ruta = predict.guardar(pred)
        return {"gameweek": gw, "partidos": len(pred),
                "registrado": ruta is not None, "archivo": None if ruta is None else ruta.name}

    return [
        # Se llaman los bronze por separado y no `ingestion.run`: ese CLI se traga toda
        # excepción y devuelve un único código, así que no distingue "se cayó FPL"
        # —fatal, sin fixtures no hay nada— de "se cayó football-data", que sólo cuesta
        # las cuotas. Y además orquesta tres de las seis fuentes.
        Paso("bronze_fpl", lambda: fpl.run(force=force_ingesta),
             descripcion="fixtures, deadlines y resultados en vivo"),
        Paso("bronze_vaastav", lambda: vaastav.run(force=force_ingesta),
             descripcion="histórico jugador-fecha"),
        Paso("bronze_fd", lambda: fd.run(force=force_ingesta), obligatorio=False,
             descripcion="cuotas de cierre (no son features)"),
        Paso("bronze_opta", lambda: pulse.run(force=force_ingesta),
             descripcion="copas, Europa y estadísticas de Opta"),

        Paso("silver", lambda: silver.run(), descripcion="normalización"),
        Paso("competencias", lambda: competencias.run(), descripcion="depende de dim_team"),
        Paso("opta", lambda: opta_stats.run(), descripcion="stats por equipo-partido"),

        Paso("gold", lambda: gold_tp.run(escribir=not dry_run),
             descripcion="features + la fila de la próxima fecha"),
        Paso("predecir", _predecir, descripcion="predice y registra"),
    ]


def correr(season: str | None = None, gameweek: int | None = None,
           desde: str | None = None, solo: tuple[str, ...] = (),
           dry_run: bool = False, forzar: bool = False,
           force_ingesta: bool = False) -> Resultado:
    season = season or CFG.current_season
    pasos = _pasos(season, gameweek, dry_run, forzar, force_ingesta)
    log.info("=== pre_deadline | %s | %s ===", season,
             f"GW{gameweek}" if gameweek else "próxima fecha")
    return correr_pasos(pasos, desde=desde, solo=solo, registrar=not dry_run)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Ingesta -> Silver -> Gold -> predicción registrada.")
    p.add_argument("--season", default=None, help=f"Por defecto: {CFG.current_season}")
    p.add_argument("--gw", type=int, default=None,
                   help="Fecha a predecir. Por defecto, la próxima predecible.")
    p.add_argument("--desde", default=None, help="Retoma desde este paso.")
    p.add_argument("--solo", default="", help="Corre sólo estos pasos, separados por coma.")
    p.add_argument("--dry-run", action="store_true",
                   help="No escribe Gold ni el registro.")
    p.add_argument("--forzar", action="store_true",
                   help="Predice aunque la fecha ya haya arrancado.")
    p.add_argument("--force-ingesta", action="store_true",
                   help="Ignora la caché de Bronze y re-descarga todo.")
    p.add_argument("--sin-ingesta", action="store_true", help="Atajo de --desde silver.")
    p.add_argument("--log-level", default=None)
    args = p.parse_args(argv)

    setup(args.log_level)
    desde = "silver" if args.sin_ingesta else args.desde
    solo = tuple(s.strip() for s in args.solo.split(",") if s.strip())

    res = correr(season=args.season, gameweek=args.gw, desde=desde, solo=solo,
                 dry_run=args.dry_run, forzar=args.forzar,
                 force_ingesta=args.force_ingesta)
    print("\n" + res.resumen() + "\n")
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(main())
