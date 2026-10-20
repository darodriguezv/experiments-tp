"""Estudio de las tres condiciones del tablero CAA.

    python main.py servir              # el conductor, en el navegador
    python main.py plan --n 12         # el contrabalanceo y la asignación
    python main.py protocolo           # las condiciones, el caso y las frases
    python main.py bloque --participante 1 --condicion letra
    python main.py informe             # el ahorro, por persona y agregado
    python main.py smoke               # chequeos del instrumento

El conductor es la forma normal de correrlo. `bloque` existe para cuando algo
falla a mitad de una sesión y hay que rehacer uno solo sin pasar por la página.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _utf8_console() -> None:
    try:
        import ctypes

        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
    except Exception:
        pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_utf8_console()

from aac_estudio import diseno, protocolo  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="main.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_servir = sub.add_parser("servir", help="el conductor en el navegador")
    p_servir.add_argument("--puerto", type=int, default=8765)
    p_servir.add_argument("--no-abrir", action="store_true",
                          help="no abrir el navegador solo")

    p_plan = sub.add_parser("plan", help="contrabalanceo y asignación")
    p_plan.add_argument("--n", type=int, default=10)

    sub.add_parser("protocolo", help="condiciones, caso y conjuntos de frases")

    p_bloque = sub.add_parser(
        "bloque", help="correr UN bloque a mano (el conductor es lo normal)")
    p_bloque.add_argument("--participante", type=int, required=True)
    p_bloque.add_argument("--condicion", required=True,
                          choices=list(diseno.CONDICIONES))
    p_bloque.add_argument("--monitor", default="ask")
    p_bloque.add_argument("--camera", type=int, default=0)
    p_bloque.add_argument("--model", default=None)

    sub.add_parser("informe", help="el ahorro, por persona y agregado")
    sub.add_parser("smoke", help="chequeos sin cámara ni modelo")

    args = parser.parse_args(argv)

    if args.command == "smoke":
        from aac_estudio import smoke

        return smoke.run()

    if args.command == "plan":
        print(diseno.informe_balance(args.n))
        return 0

    if args.command == "protocolo":
        print(f"Caso: {protocolo.CASO.nombre} - {protocolo.CASO.resumen}\n")
        for clave in diseno.CONDICIONES:
            cfg = protocolo.CONFIG_CONDICION[clave]
            flags = [k for k, v in cfg.items() if v is True]
            print(f"  {diseno.NOMBRES[clave]:<26} "
                  f"{', '.join(flags) or 'tablero pelado'}")
        print()
        print(protocolo.resumen_conjuntos())
        print("Lo que se le dice al participante:\n")
        for clave in diseno.CONDICIONES:
            print(f"  {diseno.NOMBRES[clave]}:")
            print(f"    {protocolo.INSTRUCCIONES[clave]}\n")
        return 0

    if args.command == "servir":
        from aac_estudio import servidor

        return servidor.servir(args.puerto, abrir=not args.no_abrir)

    if args.command == "bloque":
        from aac_estudio import sesion

        res = sesion.correr_bloque(
            args.participante, args.condicion, monitor=args.monitor,
            camera=args.camera, model=args.model)
        if not res.get("csv"):
            print("[bloque] no se guardó ningún CSV. ¿Se cerró el tablero "
                  "antes de decir algo?")
            return 1
        print(f"[bloque] {res['enunciados']} enunciado(s) -> {res['csv']}")
        if res["incompleto"]:
            print("[bloque] INCOMPLETO: se cerró antes de las seis tareas. "
                  "Queda marcado, no descartado.")
        return 0

    if args.command == "informe":
        from aac_estudio import informe

        print(informe.texto())
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
