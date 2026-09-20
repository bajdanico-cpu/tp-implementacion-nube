"""Genera `web/escudos.json`: la tabla de equipos que usa la página.

    python -m scripts.escudos

Es **referencia de presentación**, no dato del modelo: nombre largo y código de escudo
por cada `short_name`. Existe como archivo generado y no como endpoint porque el servicio
no lee Silver —esa es la decisión de diseño que sostiene todo el despliegue— y `dim_team`
es Silver.

El `team_code` es el `code` de FPL, que es **estable entre temporadas**, a diferencia del
`id`, que FPL reasigna todos los años (cambia en 18 de 27 equipos). Por eso este archivo
se regenera sólo cuando ascienden clubes nuevos, no cada fecha.

Los escudos se sirven desde el CDN de la Premier:
`https://resources.premierleague.com/premierleague/badges/70/t{code}.png`
"""

from __future__ import annotations

import json
from pathlib import Path

from common.config import PROJECT_ROOT
from common.logging_setup import get_logger, setup
from common.storage import read_table

log = get_logger(__name__)

SALIDA = PROJECT_ROOT / "web" / "escudos.json"


def construir() -> dict[str, dict]:
    d = read_table("dim_team")
    # La temporada más reciente manda, por si un club cambió de nombre.
    d = d.sort_values("season").drop_duplicates("short_name", keep="last")
    return {
        str(r.short_name): {"nombre": str(r.team_name), "code": int(r.team_code)}
        for r in d.itertuples()
        if r.team_code == r.team_code          # descarta NaN
    }


def main() -> int:
    setup()
    equipos = construir()
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(json.dumps(equipos, ensure_ascii=False, indent=1, sort_keys=True),
                      encoding="utf-8")
    log.info("%d equipos en %s", len(equipos), SALIDA)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
