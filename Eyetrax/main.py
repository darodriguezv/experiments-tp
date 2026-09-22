"""Sistema CAA por seguimiento ocular (EyeTrax).

    python main.py pipeline    # TODO junto: calibrar, medir y abrir el tablero
    python main.py monitors    # lista las pantallas disponibles
    python main.py smoke       # chequeos sin cámara ni pantalla
    python main.py calibrate   # calibración -> models/gaze_model.pkl
    python main.py live        # cursor de mirada, para verificar la calibración
    python main.py accuracy    # mide el error y diagnostica qué mejorar
    python main.py board       # tablero CAA con selección por permanencia
"""

from __future__ import annotations

import argparse
import sys


def _utf8_console() -> None:
    """Let accented Spanish survive the Windows console.

    The console defaults to a legacy codepage (850/1252 on a Spanish install),
    so 'ACÉRCATE' comes out as mojibake. Switching the output codepage to UTF-8
    and reconfiguring the streams fixes it; both steps are best-effort because
    neither works when output is redirected to a pipe on older Windows.
    """
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

from aac_eyetrax.config import CALIBRATIONS, GAZE_MODELS, SMOOTHERS, Settings  # noqa: E402


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--camera", type=int, default=0, help="índice de la cámara (por defecto 0)")
    p.add_argument(
        "--monitor", default="ask",
        help="pantalla a usar: un índice de `main.py monitors`, o "
             "primary / secondary / ask (por defecto ask: muestra un selector "
             "en cada pantalla; pasá un índice para saltearlo)",
    )
    p.add_argument(
        "--model", choices=GAZE_MODELS, default="ridge",
        help="modelo de regresión de la mirada (por defecto ridge)",
    )
    p.add_argument("--model-path", default=None,
                   help="ruta al archivo .pkl del modelo")
    p.add_argument(
        "--smoother", choices=SMOOTHERS, default="kalman_ema",
        help="suavizado de la mirada (por defecto kalman_ema)",
    )
    p.add_argument(
        "--ema-alpha", type=float, default=0.25,
        help="fuerza del EMA en kalman_ema: 0 = apagado, "
             "cerca de 1 = más estable pero con más retardo",
    )
    p.add_argument(
        "--capture", default="1280x720",
        help="resolución de captura, por ejemplo 1280x720 (por defecto) o "
             "640x480. Más resolución = más píxeles sobre el iris, que es de "
             "donde sale la precisión; 'default' deja la del driver",
    )
    p.add_argument("--no-camera-inset", action="store_true",
                   help="oculta la vista previa de la cámara")
    p.add_argument("--show-raw", action="store_true",
                   help="dibuja también el punto sin suavizar")
    p.add_argument("--no-position-check", action="store_true",
                   help="no mostrar la guía de colocación de la cara")


class Cancelled(Exception):
    """The user dismissed the monitor picker."""


def _resolve_monitor(spec: str) -> int | str:
    """Turn --monitor into something concrete, prompting on screen if asked.

    Done here rather than inside Settings so the library layer never opens a
    window as a side effect of being configured.
    """
    if str(spec).strip().lower() != "ask":
        return spec

    from aac_eyetrax import display

    screen = display.pick_screen()
    if screen is None:
        raise Cancelled
    print(
        f"[pantalla] usando [{screen.index}] {screen.name} "
        f"{screen.width}x{screen.height} "
        f"(pasá --monitor {screen.index} para saltear el selector)"
    )
    return screen.index


def _capture(spec: str) -> tuple[int, int]:
    """Parse --capture: 'WxH', or 'default'/'0' to leave the driver alone."""
    spec = str(spec).strip().lower()
    if spec in ("default", "native", "0", ""):
        return 0, 0
    try:
        w, h = spec.replace(" ", "").split("x")
        return int(w), int(h)
    except ValueError:
        raise ValueError(
            f"--capture debe tener la forma 1280x720 o ser 'default' "
            f"(recibí {spec!r})"
        ) from None


def _settings(args) -> Settings:
    cap_w, cap_h = _capture(args.capture)
    s = Settings(
        capture_width=cap_w,
        capture_height=cap_h,
        camera_index=args.camera,
        monitor=_resolve_monitor(args.monitor),
        model_name=args.model,
        smoother=args.smoother,
        ema_alpha=args.ema_alpha,
        show_camera=not args.no_camera_inset,
        show_raw_gaze=args.show_raw,
        check_position=not getattr(args, "no_position_check", False),
    )
    if args.model_path:
        from pathlib import Path

        s.model_path = Path(args.model_path)
    if hasattr(args, "dwell"):
        s.dwell_ms = args.dwell
    if getattr(args, "stick", None) is not None:
        s.stick_margin_ratio = args.stick
    if getattr(args, "fixation_ms", None) is not None:
        s.fixation_window_ms = args.fixation_ms
    if hasattr(args, "no_speech"):
        s.speak = not args.no_speech
    if hasattr(args, "no_log"):
        s.log_session = not args.no_log
    s.validate()
    return s


def _al_dia(settings, args) -> bool:
    """Calibra al empezar. SIEMPRE, y no hay forma de saltearlo.

    La postura exacta de una calibración anterior no se reproduce: uno se
    sienta distinto, la silla cambia de altura, la pantalla se inclina. Como el
    modelo aprende una postura, reusarlo sin recalibrar arranca con un error
    sistemático que ningún suavizado corrige - y encima invisible, porque el
    cursor se ve igual de firme estando corrido.

    Calibrar al empezar a usar el dispositivo es el modelo de uso previsto: lo
    mismo que hacen los rastreadores comerciales, y una vez por sesión no se
    parece en nada a una vez por frase.

    No hay bandera para saltearla. La había, y se sacó a pedido: un atajo que
    evita la calibración se termina usando, y entonces uno mide un modelo
    entrenado para una postura que ya no tiene sin enterarse.

    **Diferencia con WebcamV1, que importa al comparar.** Allá las muestras de
    todas las sesiones se acumulan, así que cada calibración le agrega posturas
    al modelo. Acá cada calibración REEMPLAZA a la anterior y el modelo nunca
    conoce más de una postura por vez.

    Eso es una diferencia del arnés, **no del estimador**, y es corregible: la
    afirmación anterior de que «las rutinas de eyetrax son dueñas del bucle y
    no dejan quedarse con las muestras» describía solo a sus rutinas de
    calibración, que no hace falta usar. `GazeEstimator` expone
    `extract_features(imagen)` y `train(X, y)`, que es exactamente lo que el
    recolector de WebcamV1 necesita. Mientras no se haga, cualquier ventaja de
    WebcamV1 medida tras varias sesiones incluye la acumulación y no solo el
    modelo.
    """
    from aac_eyetrax import calibrate

    metodo = getattr(args, "method", None) or "9point"
    print(f"[calibración] recalibrando ({metodo}) antes de empezar "
          f"(obligatoria: no hay forma de saltearla)")
    return calibrate.run(settings, metodo) is not None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Sistema CAA controlado por la mirada, sobre EyeTrax.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("monitors", help="lista las pantallas y sus índices")
    sub.add_parser("smoke", help="chequeos de instalación (sin cámara ni pantalla)")

    p_cal = sub.add_parser("calibrate", help="calibrar y guardar el modelo")
    _add_common(p_cal)
    p_cal.add_argument(
        "--method", choices=CALIBRATIONS, default="9point",
        help="patrón de calibración (por defecto 9point)",
    )

    p_pipe = sub.add_parser(
        "pipeline",
        help="flujo completo: calibrar, medir y abrir el tablero CAA",
    )
    _add_common(p_pipe)
    p_pipe.add_argument(
        "--method", choices=CALIBRATIONS, default="9point",
        help="patrón de calibración (por defecto 9point)",
    )
    p_pipe.add_argument(
        "--dwell", type=int, default=800,
        help="tiempo de permanencia en ms (por defecto 800)",
    )
    p_pipe.add_argument("--stick", type=float, default=None,
                        help="cuánto se agranda la celda sostenida, como fracción de su tamaño (0 = sin pegado, el comportamiento viejo). Subilo si la permanencia se reinicia sola")
    p_pipe.add_argument("--fixation-ms", type=int, default=None,
                        help="ventana en ms sobre la que se toma la mediana para decidir la celda. Más larga estabiliza más pero tarda más en cambiar")
    p_pipe.add_argument(
        "--no-accuracy", action="store_true",
        help="salta el paso de medición de precisión",
    )
    p_pipe.add_argument("--no-speech", action="store_true",
                        help="desactiva la voz")
    p_pipe.add_argument("--no-log", action="store_true",
                        help="no guardar el CSV de la sesión")

    p_live = sub.add_parser("live", help="cursor de mirada en vivo")
    _add_common(p_live)

    p_board = sub.add_parser("board", help="tablero CAA por permanencia")
    _add_common(p_board)
    p_board.add_argument(
        "--dwell", type=int, default=800,
        help="tiempo de permanencia en ms (por defecto 800)"
    )
    p_board.add_argument("--stick", type=float, default=None,
                         help="cuánto se agranda la celda sostenida, como fracción de su tamaño (0 = sin pegado, el comportamiento viejo). Subilo si la permanencia se reinicia sola")
    p_board.add_argument("--fixation-ms", type=int, default=None,
                         help="ventana en ms sobre la que se toma la mediana para decidir la celda. Más larga estabiliza más pero tarda más en cambiar")
    p_board.add_argument("--no-speech", action="store_true",
                         help="desactiva la voz")
    p_board.add_argument("--no-log", action="store_true",
                         help="no guardar el CSV de la sesión")

    p_acc = sub.add_parser("accuracy",
                           help="mide el error y diagnostica qué mejorar")
    _add_common(p_acc)
    p_acc.add_argument("--rows", type=int, default=3)
    p_acc.add_argument("--cols", type=int, default=3)
    p_acc.add_argument(
        "--hold", type=float, default=2.0,
        help="segundos que se miden por punto"
    )
    p_acc.add_argument(
        "--settle", type=float, default=0.7,
        help="segundos descartados al principio de cada punto",
    )

    # Sin bandera para saltear la calibracion: ver `_al_dia`.
    for _p in (p_live, p_board, p_acc):
        _p.add_argument('--method', choices=CALIBRATIONS, default='9point')

    args = parser.parse_args(argv)

    if args.command == "monitors":
        from aac_eyetrax import display

        print(display.describe_screens())
        return 0

    if args.command == "smoke":
        from aac_eyetrax import smoke

        return smoke.run()

    settings = _settings(args)

    if args.command == "pipeline":
        from aac_eyetrax import pipeline

        return pipeline.run(
            settings,
            method=args.method,
            skip_calibration=False,
            run_accuracy=not args.no_accuracy,
        )

    if args.command == "calibrate":
        from aac_eyetrax import calibrate

        return 0 if calibrate.run(settings, args.method) else 1

    if args.command == "live":
        from aac_eyetrax import live_gaze

        if not _al_dia(settings, args):
            return 1
        live_gaze.run(settings)
        return 0

    if args.command == "board":
        from aac_eyetrax import aac_board

        if not _al_dia(settings, args):
            return 1
        aac_board.run(settings)
        return 0

    if args.command == "accuracy":
        from aac_eyetrax import accuracy

        if not _al_dia(settings, args):
            return 1
        stats = accuracy.run(settings, args.rows, args.cols, args.hold, args.settle)
        return 0 if stats else 1

    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Cancelled:
        print("cancelado - no se eligió pantalla")
        sys.exit(130)
    except (FileNotFoundError, ValueError) as exc:
        # Missing calibration, or a bad --monitor / --ema-alpha: these carry a
        # useful message (monitor errors include the full listing), so show it
        # plainly instead of a traceback.
        print(f"\nerror: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\ninterrumpido")
        sys.exit(130)
