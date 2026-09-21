"""Predicción personalizada para el tablero CAA.

    python main.py demo                  # el tablero clickeable, para probarlo
    python main.py smoke                 # chequeos del instrumento
    python main.py sim                   # ahorro sobre el usuario sintético
    python main.py sweep                 # barre casillas y vida media
    python main.py logs RealsenseV1/logs # ahorro sobre sesiones REALES
    python main.py show --model user.json  # qué aprendió, en texto
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


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

from aac_prediccion import report, sessions  # noqa: E402
from aac_prediccion.replay import replay  # noqa: E402
from aac_prediccion.simulate import Persona  # noqa: E402
from aac_prediccion.store import UserModel  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="main.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("smoke", help="chequeos sin datos")

    p_sim = sub.add_parser("sim", help="ahorro sobre el usuario sintético")
    p_sim.add_argument("--days", type=int, default=60)
    p_sim.add_argument("--slots", type=int, default=5)
    p_sim.add_argument("--halflife", type=float, default=30.0)
    p_sim.add_argument("--novelty", type=float, default=0.15)
    p_sim.add_argument("--seed", type=int, default=7)

    p_sweep = sub.add_parser("sweep", help="barre casillas y vida media")
    p_sweep.add_argument("--days", type=int, default=60)
    p_sweep.add_argument("--seed", type=int, default=7)

    p_logs = sub.add_parser(
        "logs", help="ahorro sobre las sesiones CSV del tablero real")
    p_logs.add_argument("path", help="carpeta de logs o un CSV suelto")
    p_logs.add_argument("--slots", type=int, default=5)
    p_logs.add_argument("--halflife", type=float, default=30.0)
    p_logs.add_argument("--save", default=None,
                        help="guardar el modelo entrenado a este JSON")

    p_demo = sub.add_parser(
        "demo", help="el tablero clickeable con la franja de predicción")
    p_demo.add_argument("--model", default="mio.user.json",
                        help="dónde se guarda lo aprendido")
    p_demo.add_argument("--slots", type=int, default=5)

    p_show = sub.add_parser("show", help="qué aprendió el modelo")
    p_show.add_argument("--model", required=True)
    p_show.add_argument("--limit", type=int, default=15)

    args = parser.parse_args(argv)

    if args.command == "smoke":
        from aac_prediccion import smoke

        return smoke.run()

    if args.command == "sim":
        utts = Persona(days=args.days, novelty=args.novelty,
                       seed=args.seed).generate()
        result = replay(utts, UserModel(halflife_days=args.halflife),
                        slots=args.slots)
        print(report.summary(result, args.slots))
        print("\n  Usuario SINTÉTICO. Valida el mecanismo y sirve para barrer "
              "parámetros;\n  no predice el ahorro sobre una persona real. "
              "Para eso: `main.py logs`.")
        return 0

    if args.command == "sweep":
        rows = []
        for slots in (1, 3, 5, 8):
            utts = Persona(days=args.days, seed=args.seed).generate()
            rows.append((f"{slots} casillas, 30d",
                         replay(utts, UserModel(halflife_days=30.0),
                                slots=slots)))
        for hl in (3.0, 30.0, 3650.0):
            utts = Persona(days=args.days, seed=args.seed).generate()
            etiqueta = "sin olvido" if hl > 365 else f"vida media {hl:.0f}d"
            rows.append((f"5 casillas, {etiqueta}",
                         replay(utts, UserModel(halflife_days=hl), slots=5)))
        print(report.sweep_table(rows, slots=5))
        print("\n  Cada casilla extra suma una celda a TODA selección. El "
              "ahorro tiene que\n  pagar esa lectura, y por eso más casillas "
              "no es automáticamente mejor.")
        return 0

    if args.command == "logs":
        utts, skipped = sessions.load(Path(args.path))
        if not utts:
            print(f"No se encontró ningún enunciado en {args.path}.\n"
                  f"Se leen los CSV que escribe el tablero "
                  f"(aac-session-*.csv): cada HABLAR cierra un enunciado.\n"
                  f"Si no hay ninguno todavía, usá el tablero un rato y "
                  f"volvé.")
            return 1
        model = UserModel(halflife_days=args.halflife)
        result = replay(utts, model, slots=args.slots)
        print(report.summary(result, args.slots))
        print(f"\n  Sesiones REALES: {len(utts)} enunciados"
              + (f", {skipped} descartados por no llegar a HABLAR" if skipped
                 else ""))
        if len(utts) < 100:
            print("  Con menos de ~100 enunciados este número es indicativo, "
                  "no concluyente.")
        if args.save:
            model.save(args.save)
            print(f"  modelo guardado -> {args.save}")
        return 0

    if args.command == "demo":
        from aac_prediccion import demo

        return demo.run(model_path=args.model, slots=args.slots)

    if args.command == "show":
        model = UserModel.load(args.model)
        print(f"{model.utterances_seen} enunciados aprendidos, "
              f"vida media {model.halflife_days:.0f} días\n")
        ranked = model.phrase_completions((), limit=args.limit)
        if not ranked:
            import time as _t

            if not model.utterances_seen:
                print("Todavía no aprendió nada. Usá el tablero y volvé.")
            else:
                dias = (_t.time() - model.last_seen) / 86400.0
                print(f"No hay ninguna frase por encima del umbral de "
                      f"evidencia ({model.min_evidence:g}).")
                print(f"La observación más reciente es de hace {dias:.0f} "
                      f"días y la vida media es de "
                      f"{model.halflife_days:.0f}: todo decayó.")
                print("No es un fallo, es el olvido funcionando. Con datos "
                      "frescos vuelve a proponer.")
            return 0
        print("frases más probables ahora:")
        for tokens, w in ranked:
            print(f"  {w:6.2f}  {' '.join(tokens)}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
