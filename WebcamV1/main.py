"""Sistema CAA por mirada con GazeFollower (webcam del portátil).

    run.cmd smoke        # chequeos sin cámara
    run.cmd geometry     # qué geometría física está asumiendo - MIRALO ANTES
    run.cmd pipeline     # todo junto: colocar, calibrar, medir, tablero
    run.cmd calibrate    # solo la calibración
    run.cmd live         # cursor de mirada, para ver si quedó bien
    run.cmd accuracy     # error en grados, comparable con Eyetrax
    run.cmd board        # el tablero por permanencia

live, accuracy y board calibran antes de empezar: la postura exacta de una
sesión anterior no se reproduce. Es el arranque de sesión previsto, no un
trámite, así que van los 9 puntos completos. Las calibraciones se acumulan, de
modo que cada sesión le agrega posturas al modelo. `--skip-calibration` la
saltea; `--method 5point` la abrevia.

Mismo arnés que Eyetrax; lo único distinto es el estimador. Esa es la
condición para que `accuracy` de los dos módulos sea comparable.
"""

from __future__ import annotations

import argparse
import sys


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

from aac_webcam.config import Settings  # noqa: E402


def _pair(text: str) -> tuple[float, float]:
    parts = text.replace(",", " ").split()
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("se esperan dos números, p.ej. 34.4x19.3")
    return float(parts[0]), float(parts[1])


class _Cancelled(Exception):
    """La persona cerró el selector de pantalla."""


def _resolve_monitor(spec: str) -> int | str:
    """`--monitor ask` abre el selector visual; el resto pasa tal cual.

    Por defecto se pregunta, y con dos pantallas eso no es un lujo: acá la
    principal es el monitor externo y la webcam está en el portátil, así que
    cualquier default fijo -«primary» incluido- acierta la mitad de las veces.
    El selector pone un cartel en cada panel, así que se elige mirando cuál se
    enciende y no adivinando un índice.
    """
    from aac_eyetrax import display

    if str(spec).strip().lower() != "ask":
        return spec
    screen = display.pick_screen("¿En qué pantalla va a estar el tablero?")
    if screen is None:
        raise _Cancelled
    print(f"[pantalla] {screen.label()}")
    print(f"           (pasá --monitor {screen.index} para saltear el selector)")
    return screen.index


def _common(p) -> None:
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--monitor", default="ask",
                   help="índice de `main.py monitors`, primary, secondary, o "
                        "ask para elegir en pantalla (por defecto)")
    p.add_argument("--smoother", default="kalman_ema")
    p.add_argument("--screen-cm", type=_pair, default=None,
                   help="ancho y alto visibles en cm, p.ej. '34.4 19.3'")
    p.add_argument("--camera-cm", type=_pair, default=None,
                   help="posición de la cámara en cm desde la esquina "
                        "superior izquierda; y negativa si está por encima")
    p.add_argument("--no-position-check", action="store_true")


def _settings(args) -> Settings:
    s = Settings()
    s.camera_index = getattr(args, "camera", 0)
    s.monitor = _resolve_monitor(getattr(args, "monitor", "ask"))
    s.smoother = getattr(args, "smoother", "kalman_ema")
    if getattr(args, "screen_cm", None):
        s.screen_cm = args.screen_cm
    if getattr(args, "camera_cm", None):
        s.camera_cm = args.camera_cm
    if getattr(args, "no_position_check", False):
        s.check_position = False
    if getattr(args, "fresh", False):
        s.calibration_fresh = True
    if getattr(args, "dwell", None):
        s.dwell_ms = args.dwell
    if getattr(args, "stick", None) is not None:
        s.stick_margin_ratio = args.stick
    if getattr(args, "fixation_ms", None) is not None:
        s.fixation_window_ms = args.fixation_ms
    s.validate()
    return s


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="main.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("smoke", help="chequeos sin cámara")
    sub.add_parser("monitors", help="lista las pantallas")
    p_geo = sub.add_parser("geometry", help="la geometría física asumida")
    _common(p_geo)

    p_cal = sub.add_parser("calibrate", help="calibrar y guardar")
    _common(p_cal)
    p_cal.add_argument("--method", default="9point",
                       choices=("5point", "9point", "13point"))
    p_cal.add_argument("--fresh", action="store_true",
                       help="descartar las calibraciones anteriores. Por "
                            "defecto se acumulan, que es lo que hace al modelo "
                            "tolerar que uno se mueva")

    p_live = sub.add_parser("live", help="cursor de mirada en vivo")
    _common(p_live)
    p_live.add_argument("--skip-calibration", action="store_true",
                        help="no recalibrar: usar la guardada tal cual. Por defecto se hace una puesta al dia corta, porque la postura de una calibracion anterior no se reproduce")
    p_live.add_argument("--method", default="9point",
                        choices=("5point", "9point", "13point"))

    p_acc = sub.add_parser("accuracy", help="error en grados")
    _common(p_acc)
    p_acc.add_argument("--rows", type=int, default=3)
    p_acc.add_argument("--cols", type=int, default=3)
    p_acc.add_argument("--hold", type=float, default=2.0)
    p_acc.add_argument("--skip-calibration", action="store_true",
                       help="no recalibrar: usar la guardada tal cual. Por defecto se hace una puesta al dia corta, porque la postura de una calibracion anterior no se reproduce")
    p_acc.add_argument("--method", default="9point",
                       choices=("5point", "9point", "13point"))

    p_board = sub.add_parser("board", help="tablero por permanencia")
    _common(p_board)
    p_board.add_argument("--dwell", type=int, default=800)
    p_board.add_argument("--stick", type=float, default=None,
                         help="cuánto se agranda la celda sostenida, como fracción de su tamaño (0 = sin pegado, el comportamiento viejo). Subilo si la permanencia se reinicia sola")
    p_board.add_argument("--fixation-ms", type=int, default=None,
                         help="ventana en ms sobre la que se toma la mediana para decidir la celda. Más larga estabiliza más pero tarda más en cambiar")
    p_board.add_argument("--no-speech", action="store_true")
    p_board.add_argument("--skip-calibration", action="store_true",
                         help="no recalibrar: usar la guardada tal cual. Por defecto se hace una puesta al dia corta, porque la postura de una calibracion anterior no se reproduce")
    p_board.add_argument("--method", default="9point",
                         choices=("5point", "9point", "13point"))

    p_pipe = sub.add_parser("pipeline", help="flujo completo")
    _common(p_pipe)
    p_pipe.add_argument("--method", default="9point",
                        choices=("5point", "9point", "13point"))
    p_pipe.add_argument("--fresh", action="store_true",
                        help="descartar las calibraciones anteriores")
    p_pipe.add_argument("--dwell", type=int, default=800)
    p_pipe.add_argument("--stick", type=float, default=None,
                        help="cuánto se agranda la celda sostenida, como fracción de su tamaño (0 = sin pegado, el comportamiento viejo). Subilo si la permanencia se reinicia sola")
    p_pipe.add_argument("--fixation-ms", type=int, default=None,
                        help="ventana en ms sobre la que se toma la mediana para decidir la celda. Más larga estabiliza más pero tarda más en cambiar")
    p_pipe.add_argument("--skip-calibration", action="store_true")
    p_pipe.add_argument("--no-accuracy", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "smoke":
        from aac_webcam import smoke

        return smoke.run()

    if args.command == "monitors":
        from aac_eyetrax import display

        print(display.describe_screens())
        return 0

    if args.command == "geometry" and str(args.monitor).lower() == "ask":
        # Sin elegir: se muestran todas, que es lo que uno quiere al comparar
        # antes de decidir en cuál calibrar.
        from aac_eyetrax import display

        for screen in display.list_screens():
            s = _settings(argparse.Namespace(**{**vars(args),
                                                "monitor": screen.index}))
            print(s.describe_geometry())
            print(f"  calibración -> "
                  f"{s.calibration_dir_for(screen).relative_to(s.calibration_dir.parents[1])}")
            print()
        print("Elegí con --monitor <índice>. La webcam tiene que estar sobre "
              "la pantalla\nque elijas, o hay que medir --camera-cm respecto "
              "de ella.")
        return 0

    try:
        settings = _settings(args)
    except _Cancelled:
        print("[cancelado] no se eligió pantalla.")
        return 130

    if args.command == "geometry":
        print("geometría física asumida:\n")
        print(settings.describe_geometry())
        return 0

    from aac_eyetrax import aac_board, accuracy, calibrate, live_gaze, pipeline

    from aac_webcam import tracker as gf_tracker

    make = gf_tracker.make

    if args.command == "calibrate":
        print(settings.describe_geometry(), "\n")
        return 0 if calibrate.run(settings, args.method, make) else 1

    def _al_dia() -> bool:
        """Calibra al empezar. Es el arranque de sesión, no un trámite.

        La postura exacta de una calibración anterior **no se reproduce**: uno
        se sienta distinto, la silla cambia de altura, la pantalla se inclina.
        Y como el modelo aprende una postura, reusarlo sin recalibrar arranca
        con un error sistemático que ningún suavizado corrige - y encima
        invisible, porque el cursor se ve igual de firme estando corrido.

        Calibrar al empezar a usar el dispositivo es el modelo de uso previsto,
        no un costo a minimizar: es lo mismo que hacen los rastreadores
        comerciales, y una vez por sesión no se parece en nada a una vez por
        frase. Por eso se calibran los 9 puntos completos y no una versión
        abreviada: con el minuto ya pagado, darle menos cobertura a la sesión
        solo empeora el modelo.

        Encima las calibraciones se **acumulan**, así que cada sesión le agrega
        posturas a lo que el modelo ya sabía en vez de reemplazarlas.

        Con `--skip-calibration` se usa la guardada tal cual, que sirve para
        comparar dos configuraciones sin que la calibración cambie en el medio
        - y para medir si lo acumulado ya alcanza sin recalibrar.
        """
        if getattr(args, "skip_calibration", False):
            print("[calibración] se reusa la guardada (--skip-calibration). "
                  "Si te sentaste distinto, el error va a ser sistemático.")
            return True
        metodo = getattr(args, "method", None) or "9point"
        print(f"[calibración] calibrando ({metodo}) antes de empezar "
              f"(--skip-calibration para saltearla, --method 5point para "
              f"abreviar)")
        return calibrate.run(settings, metodo, make) is not None

    if args.command == "live":
        if not _al_dia():
            return 1
        live_gaze.run(settings, make)
        return 0

    if args.command == "accuracy":
        if not _al_dia():
            return 1
        stats = accuracy.run(settings, args.rows, args.cols, args.hold,
                             make_tracker=make)
        return 0 if stats is not None else 1

    if args.command == "board":
        if args.no_speech:
            settings.speak = False
        if not _al_dia():
            return 1
        aac_board.run(settings, make)
        return 0

    if args.command == "pipeline":
        print(settings.describe_geometry(), "\n")
        return pipeline.run(
            settings, method=args.method,
            skip_calibration=args.skip_calibration,
            run_accuracy=not args.no_accuracy,
            make_tracker=make)

    return 1


if __name__ == "__main__":
    sys.exit(main())
