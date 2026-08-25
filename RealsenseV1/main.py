"""Sistema CAA por seguimiento ocular sobre Intel RealSense D435i.

    python main.py devices     # detecta cámaras RealSense y reporta sus capacidades
    python main.py smoke       # chequeos sin cámara ni pantalla (valida la geometría)
    python main.py monitors    # lista las pantallas disponibles
    python main.py preview     # vista en vivo de color + profundidad, sin calibrar
    python main.py mount       # mide la inclinación de la cámara con el IMU
    python main.py calibrate   # calibración -> models/gaze_model.json
    python main.py live        # cursor de mirada, con prueba de postura (tecla H)
    python main.py accuracy    # mide el error y diagnostica qué mejorar
    python main.py board       # tablero CAA con selección por permanencia
    python main.py pipeline    # todo junto: calibrar, medir y abrir el tablero
    python main.py record      # graba una sesión a .bag para analizarla después
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _utf8_console() -> None:
    """Let accented Spanish survive the Windows console.

    The console defaults to a legacy codepage (850/1252 on a Spanish install),
    so 'ACÉRCATE' comes out as mojibake. Both steps are best-effort because
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

from aac_realsense.config import (CALIBRATIONS, ESTIMATORS, RECORDINGS_DIR,  # noqa: E402
                                  SMOOTHERS, CameraMount, Settings)


class Cancelled(Exception):
    """The user dismissed the monitor picker."""


def _size(spec: str, what: str) -> tuple[int, int]:
    try:
        w, h = str(spec).replace(" ", "").lower().split("x")
        return int(w), int(h)
    except ValueError:
        raise ValueError(
            f"--{what} debe tener la forma 1280x720 (recibí {spec!r})") from None


def _resolve_monitor(spec: str) -> int | str:
    """Turn --monitor into something concrete, prompting on screen if asked.

    Done here rather than inside Settings so the library layer never opens a
    window as a side effect of being configured.
    """
    if str(spec).strip().lower() != "ask":
        return spec

    from aac_realsense import display

    screen = display.pick_screen()
    if screen is None:
        raise Cancelled
    print(f"[pantalla] usando [{screen.index}] {screen.name} "
          f"{screen.width}x{screen.height} "
          f"(pasá --monitor {screen.index} para saltear el selector)")
    return screen.index


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--serial", default=None,
                   help="número de serie de la cámara (por defecto la primera, "
                        "prefiriendo una con IMU)")
    p.add_argument("--monitor", default="ask",
                   help="pantalla a usar: un índice de `main.py monitors`, o "
                        "primary / secondary / ask (por defecto ask)")
    p.add_argument("--color", default="1280x720",
                   help="resolución de color. Los píxeles sobre el iris son de "
                        "donde sale la precisión angular: 1920x1080 da un 50%% "
                        "más de ancho y es la palanca más directa")
    p.add_argument("--depth", default="848x480",
                   help="resolución de profundidad. 848x480 es un modo nativo "
                        "de la D435 y conserva todo el campo de visión")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--estimator", choices=ESTIMATORS, default="geometric",
                   help="geometric = modelo 3D (recomendado); hybrid = "
                        "regresión sobre rasgos métricos, para comparar")
    p.add_argument("--model-path", default=None,
                   help="ruta del archivo .json del modelo")
    p.add_argument("--smoother", choices=SMOOTHERS, default="kalman_ema")
    p.add_argument("--ema-alpha", type=float, default=0.25,
                   help="fuerza del EMA: 0 = apagado, cerca de 1 = más estable "
                        "pero con más retardo")
    p.add_argument("--no-position-check", action="store_true",
                   help="no mostrar la guía de colocación")
    p.add_argument("--no-camera-inset", action="store_true")
    p.add_argument("--show-raw", action="store_true",
                   help="dibuja también el punto sin suavizar")
    p.add_argument("--no-imu", action="store_true",
                   help="no leer el acelerómetro (la inclinación de la cámara "
                        "se toma de --pitch)")
    p.add_argument("--no-depth-filters", action="store_true")
    p.add_argument("--laser-power", type=float, default=150.0,
                   help="potencia del emisor IR, 0-360 (por defecto 150)")
    p.add_argument("--visual-preset", default="medium density",
                   help="preset de profundidad: 'medium density' (por defecto), "
                        "'high accuracy', 'default', 'high density' o 'none'. "
                        "High Accuracy da menos cobertura sobre la cara (~62%% "
                        "medido vs ~96%%), y los puntos que pierde son los "
                        "periféricos, que son los que fijan el giro de la "
                        "cabeza")
    p.add_argument("--playback", default=None,
                   help="reproducir una grabación .bag en vez de abrir la cámara")

    mount = p.add_argument_group(
        "geometría del montaje",
        "Del centro de la pantalla al centro del lente, en milímetros. Solo "
        "hace falta que sea aproximado: la calibración lo refina. Ver el README.")
    mount.add_argument("--offset-x", type=float, default=0.0,
                       help="+ hacia el borde derecho de la pantalla")
    mount.add_argument("--offset-y", type=float, default=175.0,
                       help="+ hacia abajo (cámara en el borde inferior)")
    mount.add_argument("--offset-z", type=float, default=20.0,
                       help="+ hacia el usuario (cuánto sobresale el lente)")
    mount.add_argument("--pitch", type=float, default=8.0,
                       help="+ apunta hacia arriba. Se sobrescribe con el IMU "
                            "si hay uno")
    mount.add_argument("--yaw", type=float, default=0.0)
    mount.add_argument("--screen-mm", default=None,
                       help="tamaño físico del área activa, p.ej. 527x296. Por "
                            "defecto se toma del EDID de la pantalla")


def _settings(args) -> Settings:
    mount = CameraMount(
        offset_x_mm=args.offset_x, offset_y_mm=args.offset_y,
        offset_z_mm=args.offset_z, pitch_deg=args.pitch, yaw_deg=args.yaw,
        measured_by="medido a mano",
    )
    if getattr(args, "screen_mm", None):
        w, h = _size(args.screen_mm, "screen-mm")
        mount.screen_width_mm, mount.screen_height_mm = float(w), float(h)

    s = Settings(
        serial=args.serial,
        color_size=_size(args.color, "color"),
        depth_size=_size(args.depth, "depth"),
        fps=args.fps,
        estimator=args.estimator,
        mount=mount,
        smoother=args.smoother,
        ema_alpha=args.ema_alpha,
        monitor=_resolve_monitor(args.monitor),
        check_position=not args.no_position_check,
        show_camera=not args.no_camera_inset,
        show_raw_gaze=args.show_raw,
        use_imu=not args.no_imu,
        depth_filters=not args.no_depth_filters,
        laser_power=args.laser_power,
        visual_preset=args.visual_preset,
        playback=Path(args.playback) if args.playback else None,
    )
    if args.model_path:
        s.model_path = Path(args.model_path)
    # Only some subcommands define these, so they are copied over conditionally
    # rather than being added to every parser.
    if hasattr(args, "dwell"):
        s.dwell_ms = args.dwell
    if hasattr(args, "no_speech"):
        s.speak = not args.no_speech
    if hasattr(args, "no_log"):
        s.log_session = not args.no_log
    if getattr(args, "record", None):
        s.record = Path(args.record)
    s.validate()
    return s


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Sistema CAA controlado por la mirada, sobre Intel RealSense.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("devices", help="detecta cámaras RealSense y sus capacidades")
    sub.add_parser("monitors", help="lista las pantallas y sus índices")
    sub.add_parser("smoke", help="chequeos sin cámara (valida la geometría)")

    p_preview = sub.add_parser(
        "preview", help="vista en vivo de color y profundidad, sin calibrar")
    _add_common(p_preview)

    p_mount = sub.add_parser(
        "mount", help="mide la inclinación de la cámara con el acelerómetro")
    _add_common(p_mount)

    p_cal = sub.add_parser("calibrate", help="calibrar y guardar el modelo")
    _add_common(p_cal)
    p_cal.add_argument("--method", choices=CALIBRATIONS, default="9point")
    p_cal.add_argument(
        "--passes", type=int, default=2,
        help="pasadas de calibración (por defecto 2). Entre pasadas se pide "
             "cambiar de postura, que es lo único que separa la posición del "
             "ojo de la de la pantalla; con 1 sola pasada el ajuste queda "
             "válido solo cerca de esa postura")
    p_cal.add_argument("--no-refine-mount", action="store_true",
                       help="no ajustar la pose de la pantalla en el ajuste")
    p_cal.add_argument("--no-residual", action="store_true",
                       help="solo geometría, sin corrección polinómica")

    p_live = sub.add_parser("live", help="cursor de mirada en vivo")
    _add_common(p_live)

    p_acc = sub.add_parser("accuracy", help="mide el error y diagnostica")
    _add_common(p_acc)
    p_acc.add_argument("--rows", type=int, default=3)
    p_acc.add_argument("--cols", type=int, default=3)
    p_acc.add_argument("--hold", type=float, default=2.0)
    p_acc.add_argument("--settle", type=float, default=0.7)

    p_board = sub.add_parser("board", help="tablero CAA por permanencia")
    _add_common(p_board)
    p_board.add_argument("--dwell", type=int, default=800)
    p_board.add_argument("--no-speech", action="store_true")
    p_board.add_argument("--no-log", action="store_true")

    p_pipe = sub.add_parser("pipeline", help="flujo completo")
    _add_common(p_pipe)
    p_pipe.add_argument("--method", choices=CALIBRATIONS, default="9point")
    p_pipe.add_argument("--passes", type=int, default=2)
    p_pipe.add_argument("--dwell", type=int, default=800)
    p_pipe.add_argument("--skip-calibration", action="store_true")
    p_pipe.add_argument("--no-accuracy", action="store_true")
    p_pipe.add_argument("--no-speech", action="store_true")
    p_pipe.add_argument("--no-log", action="store_true")

    p_rec = sub.add_parser(
        "record",
        help="graba color+profundidad a .bag para reanalizar sin la persona")
    _add_common(p_rec)
    p_rec.add_argument("--out", default=None, help="ruta del .bag de salida")
    p_rec.add_argument("--seconds", type=float, default=20.0)

    args = parser.parse_args(argv)

    # --- commands that need no camera and no settings ------------------------
    if args.command == "devices":
        from aac_realsense import rsdevice

        print(rsdevice.report())
        return 0

    if args.command == "monitors":
        from aac_realsense import display

        print(display.describe_screens())
        return 0

    if args.command == "smoke":
        from aac_realsense import smoke

        return smoke.run()

    if args.command == "record":
        args.record = args.out or str(
            RECORDINGS_DIR / "session.bag")
    settings = _settings(args)

    if args.command == "preview":
        from aac_realsense import preview

        return preview.run(settings)

    if args.command == "mount":
        from aac_realsense import preview

        return preview.measure_mount(settings)

    if args.command == "record":
        from aac_realsense import preview

        return preview.record(settings, args.seconds)

    if args.command == "calibrate":
        from aac_realsense import calibrate

        settings.refine_mount = not args.no_refine_mount
        settings.residual_correction = not args.no_residual
        return 0 if calibrate.run(settings, args.method, args.passes) else 1

    if args.command == "live":
        from aac_realsense import live_gaze

        live_gaze.run(settings)
        return 0

    if args.command == "accuracy":
        from aac_realsense import accuracy

        stats = accuracy.run(settings, args.rows, args.cols, args.hold,
                             args.settle)
        return 0 if stats else 1

    if args.command == "board":
        from aac_realsense import aac_board

        aac_board.run(settings)
        return 0

    if args.command == "pipeline":
        from aac_realsense import pipeline

        return pipeline.run(settings, method=args.method, passes=args.passes,
                            skip_calibration=args.skip_calibration,
                            run_accuracy=not args.no_accuracy)

    parser.error(f"comando desconocido {args.command}")
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Cancelled:
        print("cancelado - no se eligió pantalla")
        sys.exit(130)
    except (FileNotFoundError, ValueError) as exc:
        # Missing calibration, or a bad --monitor / --color: these carry a
        # useful message, so show it plainly instead of a traceback.
        print(f"\nerror: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\ninterrumpido")
        sys.exit(130)
