"""Publica `web/` en la rama gh-pages.

    python web/desplegar.py            # ver qué haría, sin tocar nada
    python web/desplegar.py --hacer    # publicar de verdad

## Por qué una rama aparte y no `main`

GitHub Pages sirve el contenido tal cual, desde la raíz de la rama. Si se
sirviera `main`, la URL quedaría colgando de `/web/` y además quedaría
publicado todo el repositorio: los CSV de sesiones, los resultados de las
corridas, el corpus. Nada de eso es secreto, pero publicarlo sin decidirlo es
distinto de decidir publicarlo.

Con una rama huérfana se publica exactamente lo que se lista acá y nada más.

## Lo que NO se publica

Las pruebas y este script. No rompen nada si se publican, pero el sitio es lo
que ve alguien que abre el link, y menos superficie es menos que mantener.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
WEB = RAIZ / "web"
RAMA = "gh-pages"

# Lo que va al sitio. Explícito y no "todo menos": una lista de exclusiones
# publica por omisión cualquier archivo nuevo que alguien deje en la carpeta.
PUBLICAR = ["index.html", "css", "js", "datos"]


def git(*args: str, cwd: Path | None = None) -> str:
    r = subprocess.run(["git", *args], cwd=str(cwd or RAIZ),
                       capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}:\n{r.stderr.strip()}")
    return r.stdout.strip()


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="desplegar.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hacer", action="store_true",
                    help="publicar de verdad (sin esto solo muestra el plan)")
    ap.add_argument("--rama", default=RAMA)
    args = ap.parse_args(argv[1:])

    faltan = [p for p in PUBLICAR if not (WEB / p).exists()]
    if faltan:
        print(f"[error] faltan en web/: {', '.join(faltan)}")
        return 1

    archivos = []
    for nombre in PUBLICAR:
        origen = WEB / nombre
        if origen.is_dir():
            archivos += [p for p in origen.rglob("*") if p.is_file()]
        else:
            archivos.append(origen)
    total = sum(p.stat().st_size for p in archivos)

    print(f"Se publicarían {len(archivos)} archivo(s), "
          f"{total / 1024:.0f} KB, en la rama {args.rama}:")
    for p in sorted(archivos):
        print(f"  {p.relative_to(WEB)}  ({p.stat().st_size / 1024:.1f} KB)")

    url = git("config", "--get", "remote.origin.url")
    usuario, repo = _partes(url)
    print(f"\nQuedaría en:  https://{usuario}.github.io/{repo}/")

    if not args.hacer:
        print("\n(simulacro: nada se tocó. Agregá --hacer para publicar)")
        return 0

    if git("status", "--porcelain"):
        print("\n[error] hay cambios sin commitear. Commiteá primero: la rama "
              "del sitio\n        se arma desde lo que está guardado, no desde "
              "el directorio.")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        etapa = Path(tmp) / "sitio"
        etapa.mkdir()
        for nombre in PUBLICAR:
            origen = WEB / nombre
            destino = etapa / nombre
            if origen.is_dir():
                shutil.copytree(origen, destino)
            else:
                shutil.copy2(origen, destino)
        # Sin esto GitHub Pages pasa todo por Jekyll, que ignora los
        # directorios que empiezan con guion bajo y puede romper rutas.
        (etapa / ".nojekyll").write_text("", encoding="utf-8")

        git("init", "-q", cwd=etapa)
        git("checkout", "-q", "-b", args.rama, cwd=etapa)
        git("add", "-A", cwd=etapa)
        git("-c", "user.name=desplegar", "-c", "user.email=desplegar@local",
            "commit", "-q", "-m",
            f"sitio desde {git('rev-parse', '--short', 'HEAD')}", cwd=etapa)
        git("remote", "add", "origin", url, cwd=etapa)
        git("push", "-q", "--force", "origin", args.rama, cwd=etapa)

    print(f"\n[listo] publicado en la rama {args.rama}.")
    print(f"        Falta activarlo UNA vez en GitHub:")
    print(f"        Settings -> Pages -> Source: Deploy from a branch -> "
          f"{args.rama} / (root)")
    print(f"        Después queda en https://{usuario}.github.io/{repo}/")
    return 0


def _partes(url: str) -> tuple[str, str]:
    limpia = url.removesuffix(".git")
    if limpia.startswith("git@"):
        limpia = limpia.split(":", 1)[1]
    trozos = [t for t in limpia.replace("https://github.com/", "").split("/") if t]
    return (trozos[-2], trozos[-1]) if len(trozos) >= 2 else ("usuario", "repo")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
