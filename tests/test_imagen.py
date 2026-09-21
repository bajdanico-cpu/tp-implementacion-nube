"""Que la imagen se pueda importar a sí misma.

Existe por un `ModuleNotFoundError`. El `Dockerfile` copia paquete por paquete —es lo
que mantiene la imagen en código puro, sin `data/` ni `models/`— y `eda` no estaba en la
lista, aunque `serving.predict`, `serving.observability` y `features.gold_tp` importan
`eda.baselines`. El contenedor moría antes de abrir el puerto y Cloud Run sólo decía
*failed to start and listen on the port defined by PORT=8080*, que no nombra el módulo.

Arreglar el `COPY` no alcanzó: `.gcloudignore` excluía `eda/` entero, así que el
archivo ni llegaba al contexto y el build moría en *file not found in build context*.
Son dos listas que tienen que estar de acuerdo —qué copia el `Dockerfile` y qué sube
`gcloud`— y cada una falla en un momento distinto: la primera en runtime, la segunda en
el build.

Lo caro no fue el error sino el ciclo para verlo: `gcloud builds submit` (minutos),
`gcloud run deploy`, esperar el timeout del health check, y recién ahí abrir los logs.
Esto lo contesta en milisegundos, sin Docker y sin red: lee las líneas `COPY`, arma el
conjunto de archivos que van a existir en `/app`, verifica que todo import local que
sale de ahí resuelva ahí adentro, y que nada de eso esté excluido del contexto.

No reemplaza construir la imagen —no ve dependencias de PyPI que falten en
`requirements-serving.txt`, ni un import adentro de una función— pero cubre el caso que
de verdad pasó, y el que va a volver a pasar cada vez que alguien agregue un paquete.
"""

from __future__ import annotations

import ast
import re
from fnmatch import fnmatch
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
DOCKERFILE = RAIZ / "Dockerfile"

# Un paquete "local" es un directorio del repo con `__init__.py`: lo que un import puede
# resolver por estar el proyecto en el path, y no por estar instalado.
LOCALES = {d.name for d in RAIZ.iterdir() if d.is_dir() and (d / "__init__.py").exists()}


def copiados() -> set[str]:
    """Las rutas (relativas, POSIX) que van a existir dentro de la imagen.

    Se leen del `Dockerfile` en vez de fijarse en una lista acá, porque una lista
    duplicada se desactualiza y el test pasaría a mentir justo cuando importa.
    """
    dentro: set[str] = set()
    for linea in DOCKERFILE.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*COPY\s+(?!--)(.+)$", linea)
        if not m:
            continue
        partes = m.group(1).split()
        for origen in partes[:-1]:            # el último es el destino
            p = RAIZ / origen
            if p.is_dir():
                dentro |= {q.relative_to(RAIZ).as_posix() for q in p.rglob("*.py")}
            elif p.is_file():
                dentro.add(p.relative_to(RAIZ).as_posix())
    return dentro


def resuelve(modulo: str, dentro: set[str]) -> bool:
    """¿`import a.b.c` encuentra algo dentro de la imagen?"""
    ruta = modulo.replace(".", "/")
    return f"{ruta}.py" in dentro or f"{ruta}/__init__.py" in dentro


def importados(archivo: Path) -> set[str]:
    """Los módulos que este archivo importa, con los relativos ya resueltos."""
    arbol = ast.parse(archivo.read_text(encoding="utf-8"))
    paquete = archivo.relative_to(RAIZ).parent.as_posix().split("/")
    mods: set[str] = set()

    for n in ast.walk(arbol):
        if isinstance(n, ast.Import):
            mods |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            if n.level:                        # from . import x / from ..y import z
                base = paquete[: len(paquete) - n.level + 1]
                mods.add(".".join([*base, n.module] if n.module else base))
            elif n.module:
                mods.add(n.module)
    return mods


@pytest.fixture(scope="module")
def dentro() -> set[str]:
    d = copiados()
    assert d, "no se leyó ningún COPY del Dockerfile"
    return d


def test_todo_import_local_de_la_imagen_resuelve_dentro_de_la_imagen(dentro):
    """El caso real: `eda.baselines` importado desde `serving/`, y `eda` sin copiar."""
    faltan: dict[str, list[str]] = {}

    for rel in sorted(dentro):
        if not rel.endswith(".py"):
            continue
        for mod in importados(RAIZ / rel):
            if mod.split(".")[0] not in LOCALES:
                continue                       # PyPI: lo cubre requirements-serving.txt
            if not resuelve(mod, dentro):
                faltan.setdefault(mod, []).append(rel)

    assert not faltan, (
        "la imagen importa módulos que no copia — el contenedor va a morir con "
        "ModuleNotFoundError antes de abrir el puerto:\n"
        + "\n".join(f"  {m}  <- {', '.join(sorted(d))}" for m, d in sorted(faltan.items()))
        + "\nAgregá el COPY que falta en el Dockerfile.")


def test_el_arranque_del_servicio_esta_entero(dentro):
    """`serving.main` y lo que cuelga de él, cerrado transitivamente.

    El test de arriba ya lo cubre, pero este nombra el camino que Cloud Run ejecuta: si
    falla, el mensaje dice *el servicio no arranca* y no *hay un import suelto*.
    """
    pendientes = ["serving/main.py"]
    vistos: set[str] = set()

    while pendientes:
        rel = pendientes.pop()
        if rel in vistos:
            continue
        vistos.add(rel)
        assert rel in dentro, f"{rel} no está en la imagen (lo necesita serving.main)"

        for mod in importados(RAIZ / rel):
            if mod.split(".")[0] not in LOCALES:
                continue
            ruta = mod.replace(".", "/")
            for cand in (f"{ruta}.py", f"{ruta}/__init__.py"):
                if (RAIZ / cand).exists():
                    pendientes.append(cand)
                    break
            else:
                pytest.fail(f"{mod}, importado desde {rel}, no existe en el repo")


def _excluido(rel: str, patrones: list[str]) -> str | None:
    """El patrón de `.gcloudignore` que deja este archivo fuera del contexto, si hay uno.

    Es una aproximación de la sintaxis gitignore, no una implementación: cubre patrones
    de directorio (`eda/`), globs (`*.md`) y rutas literales, que es todo lo que el
    archivo usa hoy. Si algún día usa negaciones o `**`, este test va a quedarse corto
    —y el comentario está acá para que se note antes de confiar de más—.
    """
    partes = rel.split("/")
    for pat in patrones:
        if pat.startswith("!"):
            continue
        if pat.endswith("/"):
            d = pat.rstrip("/")
            # Sin barra interna, gitignore busca ese nombre de directorio en cualquier
            # nivel; con barra, ancla en la raiz del contexto.
            suelto = "/" not in d and d in partes[:-1]
            anclado = "/" in d and rel.startswith(d + "/")
            if suelto or anclado:
                return pat
        elif fnmatch(rel, pat) or fnmatch(partes[-1], pat):
            return pat
    return None


def test_lo_que_el_dockerfile_copia_llega_al_contexto_de_build(dentro):
    """La otra mitad del mismo error: el `COPY` estaba bien y el archivo no subía.

    `gcloud builds submit` arma el contexto con `.gcloudignore` y, al existir ese
    archivo, ignora `.gitignore` por completo. Un `COPY` de algo excluido no falla al
    escribirlo: falla minutos después, en el build, con *file not found in build
    context* — un mensaje que culpa al `Dockerfile` y no a la lista que lo causó.
    """
    ruta = RAIZ / ".gcloudignore"
    if not ruta.exists():
        pytest.skip("no hay .gcloudignore")

    patrones = [l.strip() for l in ruta.read_text(encoding="utf-8").splitlines()
                if l.strip() and not l.strip().startswith("#")]

    fuera = {rel: pat for rel in sorted(dentro)
             if (pat := _excluido(rel, patrones))}

    assert not fuera, (
        "el Dockerfile copia archivos que .gcloudignore deja fuera del contexto — "
        "el build va a morir con 'file not found in build context':\n"
        + "\n".join(f"  {r}  <- excluido por '{p}'" for r, p in fuera.items()))


def test_no_se_copian_datos_ni_modelos():
    """La decisión que hace que el dato viva en el bucket y no adentro de la imagen.

    Volver a meterlos sería el camino cómodo —un `COPY models` y listo— y desharía la
    separación entre los tres ciclos de vida: código, dato y modelo.
    """
    texto = DOCKERFILE.read_text(encoding="utf-8")
    copias = re.findall(r"^\s*COPY\s+(?!--)(.+)$", texto, flags=re.MULTILINE)
    origenes = [o for linea in copias for o in linea.split()[:-1]]

    prohibidos = [o for o in origenes
                  if o.split("/")[0] in {"data", "models"}]
    assert not prohibidos, (
        f"el Dockerfile volvió a hornear dato o modelo: {prohibidos}. "
        f"Eso se lee del bucket (TP_STORAGE_BACKEND=gcs).")
