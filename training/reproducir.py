"""Qué hace falta para reproducir un modelo guardado.

    python -m training.reproducir              # lista todos, con su identidad exacta
    python -m training.reproducir --version 20260825T011618Z

Existe por un error concreto: durante varios días `FEATURE_SET_VERSION` se mantuvo a mano
y quedó pegada en `"v2"` mientras el set pasaba por 159, 164, 171, 175, 184 y 192 columnas.
Seis modelos distintos guardados con la misma etiqueta — justo lo que una versión tiene que
evitar. Ahora la versión se **deriva** de un hash de la lista de features
(`v2.a29de7c7.192`), así que no puede volver a desincronizarse.

Lo que salva a los modelos viejos es que el `metadata.json` guarda dos cosas que sí son
fiables: el **`git_sha`** del código con el que se entrenó, y la **lista ordenada completa
de `feature_names`**. Con eso alcanza para reproducir, aunque la etiqueta mienta.

Hay dos formas de "reproducir", y conviene no confundirlas:

- **Volver a usarlo** — cargar el `.ubj` y predecir. No necesita nada más que el modelo y
  su lista de features. Es lo que hace `serving/predict.py`, y es lo que se sube a la nube.
- **Volver a entrenarlo** — reconstruir el mismo artefacto desde cero. Requiere el mismo
  código (`git checkout <sha>`), la misma Silver y el **mismo device**: entre GPU y CPU
  cambian 18 de 380 predicciones.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.config import CFG
from common.logging_setup import get_logger, setup
from features import spec
from training import registry

log = get_logger(__name__)


def inventario() -> list[dict]:
    """Todos los modelos guardados, con su identidad real."""
    filas = []
    for meta_path in sorted(registry.RAIZ.glob("*/2*/metadata.json")):
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        nombres = m.get("feature_names") or []
        filas.append({
            "modelo": m.get("model_name"),
            "version": m.get("model_version"),
            "n_features": len(nombres),
            "feature_set_declarado": m.get("feature_set_version"),
            # Se recalcula desde la lista guardada: es la identidad real del set, sin
            # depender de que la etiqueta de entonces fuera correcta.
            "feature_set_real": spec._version_features(nombres) if nombres else None,
            "coincide_con_spec_actual": nombres == spec.FEATURES,
            "git_sha": m.get("git_sha"),
            "git_dirty": m.get("git_dirty"),
            "device": m.get("device_used"),
            "n_train": m.get("n_train"),
            "incluye_holdout": m.get("incluye_holdout"),
            "ruta": meta_path.parent,
        })
    return filas


def instrucciones(fila: dict) -> str:
    sha = fila.get("git_sha") or "?"
    dirty = fila.get("git_dirty")
    L = [f"Modelo {fila['modelo']} version {fila['version']}", ""]

    if fila["coincide_con_spec_actual"]:
        L.append("El feature set actual COINCIDE con el de este modelo.")
        L.append("Se puede reentrenar sin tocar el codigo:")
        L.append("")
        L.append("    python -m features.gold_tp")
        L.append(f"    python -m training.run --model {fila['modelo']}"
                 + ("" if fila.get("incluye_holdout") else " --sin-holdout"))
    else:
        L.append(f"El feature set CAMBIO desde entonces: el modelo tiene "
                 f"{fila['n_features']} features y el spec actual {len(spec.FEATURES)}.")
        L.append("Para reentrenarlo hay que volver al codigo de entonces:")
        L.append("")
        L.append(f"    git stash")
        L.append(f"    git checkout {sha}")
        L.append("    python -m features.gold_tp")
        L.append(f"    python -m training.run --model {fila['modelo']}"
                 + ("" if fila.get("incluye_holdout") else " --sin-holdout"))
        L.append("    git checkout -")
        if dirty:
            L.append("")
            L.append("    ATENCION: se entreno con el arbol SUCIO (git_dirty=True), asi que")
            L.append("    el sha no fija el codigo exacto. El resultado puede diferir.")

    L.append("")
    L.append(f"Para USARLO tal cual (que es lo que necesita el serving) no hace falta nada")
    L.append(f"de lo anterior: el .ubj y la lista de features del metadata alcanzan.")
    L.append("")
    L.append(f"    {fila['ruta']}")

    if fila.get("device") == "cuda":
        L.append("")
        L.append("Se entreno en GPU. Reentrenar en CPU NO da el mismo modelo: medido,")
        L.append("cambian 18 de 380 predicciones. Para servir en CPU, cargar el .ubj.")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description="Como reproducir un modelo guardado.")
    ap.add_argument("--version", default=None, help="version del modelo (timestamp)")
    args = ap.parse_args()

    setup(CFG.log_level, CFG.log_format)
    inv = inventario()
    if not inv:
        print("No hay modelos guardados.")
        return

    if args.version:
        elegido = [f for f in inv if f["version"] == args.version]
        if not elegido:
            print(f"No existe la version {args.version}. Disponibles:")
            for f in inv:
                print(f"   {f['modelo']:10s} {f['version']}")
            return
        print(instrucciones(elegido[0]))
        return

    print(f"\n{'=' * 96}")
    print(f"MODELOS GUARDADOS — el spec actual es {spec.FEATURE_SET_VERSION}")
    print("=" * 96 + "\n")
    print(f"{'modelo':10s} {'version':18s} {'feature set real':22s} {'declarado':11s} "
          f"{'git':9s} {'=spec':6s}")
    print("-" * 96)
    for f in inv:
        marca = "  si" if f["coincide_con_spec_actual"] else "  no"
        print(f"{f['modelo']:10s} {f['version']:18s} {str(f['feature_set_real']):22s} "
              f"{str(f['feature_set_declarado']):11s} {str(f['git_sha'])[:8]:9s}{marca}")

    print(f"\nLa columna 'declarado' es lo que el modelo dijo ser; 'feature set real' se")
    print(f"recalcula desde su lista de features guardada. Donde difieren, la etiqueta")
    print(f"mentia -- por eso ahora la version se deriva del contenido.")
    print(f"\nPara las instrucciones de uno: python -m training.reproducir --version <v>")


if __name__ == "__main__":
    main()
