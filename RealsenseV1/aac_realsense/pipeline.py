"""Flujo completo: de cero al tablero CAA funcionando, en un solo comando.

    1. Elegir pantalla        (si no se pasó --monitor)
    2. Verificar el sensor    (¿hay profundidad? ¿alcanza el iris?)
    3. Calibración            (9 puntos, 2 pasadas por defecto)
    4. Medición de precisión  (con diagnóstico y prueba de postura)
    5. Tablero CAA            (la aplicación real)

Cada paso se puede cancelar con ESC sin dejar nada a medio hacer: la calibración
solo se guarda si terminó, y los pasos siguientes se saltan si el anterior se
canceló.
"""

from __future__ import annotations

import time

import cv2

from . import aac_board, accuracy, calibrate, rsdevice, tts, ui
from .config import Settings
from .tracker import GazeTracker

WINDOW = "CAA - flujo completo"


def _step_screen(tracker: GazeTracker, number: int, total: int, title: str,
                 lines: list[str], *, accent=None) -> bool:
    """Pantalla de aviso entre pasos. False si la persona cancela con ESC."""
    accent = accent or ui.ACCENT
    w, h = tracker.screen_w, tracker.screen_h
    tracker.open_window(WINDOW)

    img = ui.canvas(w, h)
    ui.text_center(img, f"PASO {number} DE {total}", w // 2, int(h * 0.22), 30,
                   accent, bold=True)
    ui.text_center(img, title, w // 2, int(h * 0.33), 62, ui.TEXT, bold=True)
    for i, line in enumerate(lines):
        ui.text_center(img, line, w // 2, int(h * 0.46) + i * 40, 28, ui.TEXT_DIM)
    ui.text_center(img, "ENTER para continuar     ESC para salir", w // 2,
                   int(h * 0.86), 30, accent)

    while True:
        cv2.imshow(WINDOW, img)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 32):
            cv2.destroyWindow(WINDOW)
            cv2.waitKey(1)
            return True
        if key == 27:
            cv2.destroyWindow(WINDOW)
            cv2.waitKey(1)
            return False
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            return False


def _final_screen(tracker: GazeTracker, stats: dict | None,
                  number: int, total: int) -> None:
    w, h = tracker.screen_w, tracker.screen_h
    tracker.open_window(WINDOW)
    img = ui.canvas(w, h)
    ui.text_center(img, f"PASO {number} DE {total}", w // 2, int(h * 0.22), 30,
                   ui.GOOD, bold=True)
    ui.text_center(img, "TODO LISTO", w // 2, int(h * 0.33), 78, ui.GOOD,
                   bold=True)

    lines = ["El tablero se abre a continuación.",
             "Mire una celda y mantenga la mirada para elegirla."]
    if stats:
        lines += ["", f"Precisión medida: p95 = {stats['p95_px']:.0f} px "
                      f"-> celda mínima ~{2 * stats['p95_px']:.0f} px"]
    for i, line in enumerate(lines):
        ui.text_center(img, line, w // 2, int(h * 0.48) + i * 40, 28, ui.TEXT_DIM)
    ui.text_center(img, "ENTER para abrir el tablero", w // 2, int(h * 0.86),
                   30, ui.GOOD)

    while True:
        cv2.imshow(WINDOW, img)
        if (cv2.waitKey(30) & 0xFF) in (13, 32, 27):
            break
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyWindow(WINDOW)
    cv2.waitKey(1)


def _report_camera(settings: Settings) -> None:
    """Say what camera is in use, and warn about it, before anything else.

    Printed up front rather than buried: a USB 2 connection or a missing IMU
    silently costs accuracy for the whole session, and finding out afterwards
    means repeating it with the participant.
    """
    if settings.playback is not None:
        print(f"[flujo] reproduciendo {settings.playback}")
        return
    try:
        info = rsdevice.find(settings.serial)
    except rsdevice.RealSenseUnavailable as exc:
        print(f"[flujo] {exc}")
        return
    print(f"[flujo] cámara: {info.model} {info.serial}, firmware "
          f"{info.firmware}, USB {info.usb_descriptor}")
    for warning in info.warnings:
        print(f"[flujo] aviso: {warning}")


def run(settings: Settings, *, method: str = "9point", passes: int = 2,
        skip_calibration: bool = False, run_accuracy: bool = True,
        acc_rows: int = 3, acc_cols: int = 3) -> int:
    """Ejecuta el flujo completo. Devuelve un código de salida."""
    total = int(not skip_calibration) + int(run_accuracy) + 1
    step = 0
    t0 = time.time()

    print("=" * 70)
    print("  SISTEMA CAA POR SEGUIMIENTO OCULAR (RealSense) - flujo completo")
    print("=" * 70)
    _report_camera(settings)

    with GazeTracker(settings) as shell:
        print(f"[flujo] pantalla {shell.screen.index} ({shell.screen.name}) "
              f"{shell.screen_w}x{shell.screen_h}")
        print(f"[flujo] montaje: {settings.mount.describe()}")
        voice = tts.voice_name()
        print(f"[flujo] voz: {voice or 'ninguna en español (se imprime el texto)'}")

        if not skip_calibration:
            step += 1
            if not _step_screen(
                shell, step, total, "CALIBRACIÓN",
                ["Primero se verifica que la cámara le mida bien.",
                 "Después siga con la mirada cada punto, uno por uno.",
                 f"Son {9 if method == '9point' else 'varios'} puntos, "
                 f"{passes} vez/veces.",
                 "",
                 "Entre pasadas se le va a pedir cambiar un poco de postura.",
                 "Eso no es un capricho: es lo que le permite al sistema",
                 "separar dónde está su ojo de dónde está la pantalla,",
                 "y es lo que hace que después pueda moverse con libertad."],
            ):
                print("[flujo] cancelado antes de calibrar.")
                return 130
        else:
            print("[flujo] se salta la calibración (--skip-calibration)")

    if not skip_calibration:
        if calibrate.run(settings, method, passes) is None:
            print("[flujo] la calibración no se completó - se detiene acá.")
            return 1

    stats = None
    if run_accuracy:
        with GazeTracker(settings) as shell:
            shell.load()
            step += 1
            if not _step_screen(
                shell, step, total, "MEDICIÓN DE PRECISIÓN",
                ["Ahora se mide cuánto error quedó.",
                 f"Siga el punto blanco por la pantalla ({acc_rows}x{acc_cols}).",
                 "Con esto sabemos de qué tamaño tienen que ser las celdas.",
                 "",
                 "Si puede, muévase un poco entre puntos: así se mide",
                 "si el sistema aguanta los cambios de postura."],
            ):
                print("[flujo] cancelado antes de medir.")
                return 130
        stats = accuracy.run(settings, acc_rows, acc_cols)
        if stats is None:
            print("[flujo] la medición se canceló; seguimos igual al tablero.")

    with GazeTracker(settings) as shell:
        shell.load()
        _final_screen(shell, stats, total, total)

    aac_board.run(settings)

    print(f"[flujo] terminado en {(time.time() - t0) / 60.0:.1f} min")
    return 0
