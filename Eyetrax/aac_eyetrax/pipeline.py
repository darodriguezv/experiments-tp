"""Flujo completo: de cero al tablero CAA funcionando, en un solo comando.

Encadena los pasos en el orden en que importan, con una pantalla de aviso entre
cada uno para que la persona sepa qué viene:

    1. Elegir pantalla        (si no se pasó --monitor)
    2. Colocación de la cara  (fija la posición de referencia)
    3. Calibración            (9 puntos por defecto)
    4. Medición de precisión  (opcional, con diagnóstico)
    5. Tablero CAA            (la aplicación real)

Cada paso puede cancelarse con ESC sin dejar nada a medio hacer: la calibración
solo se guarda si terminó, y los pasos siguientes se saltan si el anterior se
canceló.
"""

from __future__ import annotations

import time

import cv2

from . import aac_board, accuracy, calibrate, positioning, tts, ui
from .config import Settings
from .tracker import GazeTracker

WINDOW = "CAA - flujo completo"


def _step_screen(
    tracker: GazeTracker,
    number: int,
    total: int,
    title: str,
    lines: list[str],
    *,
    accent=None,
) -> bool:
    """Pantalla de aviso entre pasos. False si la persona cancela con ESC."""
    accent = accent or ui.ACCENT
    w, h = tracker.screen_w, tracker.screen_h
    tracker.open_window(WINDOW)

    img = ui.canvas(w, h)
    ui.text_center(img, f"PASO {number} DE {total}", w // 2, int(h * 0.24), 30,
                   accent, bold=True)
    ui.text_center(img, title, w // 2, int(h * 0.36), 66, ui.TEXT, bold=True)
    for i, line in enumerate(lines):
        ui.text_center(img, line, w // 2, int(h * 0.50) + i * 44, 30, ui.TEXT_DIM)
    ui.text_center(img, "ENTER para continuar     ESC para salir", w // 2,
                   int(h * 0.84), 30, accent)

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


def _final_screen(
    tracker: GazeTracker, stats: dict | None, number: int = 0, total: int = 0
) -> None:
    w, h = tracker.screen_w, tracker.screen_h
    tracker.open_window(WINDOW)
    img = ui.canvas(w, h)
    if total:
        ui.text_center(img, f"PASO {number} DE {total}", w // 2, int(h * 0.24), 30,
                       ui.GOOD, bold=True)
    ui.text_center(img, "TODO LISTO", w // 2, int(h * 0.34), 80, ui.GOOD, bold=True)
    lines = ["El tablero se abre a continuación.",
             "Mire una celda y mantenga la mirada para elegirla."]
    if stats:
        tile = 2 * stats["p95_px"]
        lines.append("")
        lines.append(f"Precisión medida: p95 = {stats['p95_px']:.0f} px "
                     f"→ celda mínima ~{tile:.0f} px")
    for i, line in enumerate(lines):
        ui.text_center(img, line, w // 2, int(h * 0.50) + i * 44, 30, ui.TEXT_DIM)
    ui.text_center(img, "ENTER para abrir el tablero", w // 2, int(h * 0.84),
                   30, ui.GOOD)
    while True:
        cv2.imshow(WINDOW, img)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 32, 27):
            break
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyWindow(WINDOW)
    cv2.waitKey(1)


def run(
    settings: Settings,
    *,
    method: str = "9point",
    skip_calibration: bool = False,
    run_accuracy: bool = True,
    acc_rows: int = 3,
    acc_cols: int = 3,
) -> int:
    """Ejecuta el flujo completo. Devuelve un código de salida."""
    # Solo se numeran las pantallas que la persona realmente ve: los avisos
    # de calibración y de medición, más la pantalla final del tablero.
    total = int(not skip_calibration) + int(run_accuracy) + 1
    step = 0
    t0 = time.time()

    print("=" * 68)
    print("  SISTEMA CAA POR SEGUIMIENTO OCULAR - flujo completo")
    print("=" * 68)

    # Un solo tracker para las pantallas de aviso: crear el modelo de MediaPipe
    # cuesta cerca de un segundo, y los pasos internos ya arman el suyo.
    with GazeTracker(settings) as shell:
        print(f"[flujo] pantalla {shell.screen.index} ({shell.screen.name}) "
              f"{shell.screen_w}x{shell.screen_h}, "
              f"captura {settings.capture_width}x{settings.capture_height}")
        voice = tts.voice_name()
        print(f"[flujo] voz: {voice or 'ninguna en español (se imprimirá el texto)'}")

        # ---------------------------------------------------------- calibración
        if not skip_calibration:
            step += 1
            if not _step_screen(
                shell, step, total, "CALIBRACIÓN",
                ["Primero va a colocar su cara en una guía.",
                 "Después siga con la mirada cada punto verde",
                 "hasta que deje de latir. Son 9 puntos.",
                 "",
                 "Importante: no cambie de postura después de calibrar."],
            ):
                print("[flujo] cancelado antes de calibrar.")
                return 130
        else:
            print("[flujo] se salta la calibración (--skip-calibration)")

    if not skip_calibration:
        if calibrate.run(settings, method) is None:
            print("[flujo] la calibración no se completó - se detiene acá.")
            return 1

    # --------------------------------------------------------------- precisión
    stats = None
    if run_accuracy:
        with GazeTracker(settings) as shell:
            shell.load()
            step += 1
            if not _step_screen(
                shell, step, total, "MEDICIÓN DE PRECISIÓN",
                ["Ahora medimos cuánto error tiene la calibración.",
                 f"Siga el punto blanco por la pantalla ({acc_rows}x{acc_cols}).",
                 "Con esto sabemos de qué tamaño tienen que ser las celdas."],
            ):
                print("[flujo] cancelado antes de medir.")
                return 130
        stats = accuracy.run(settings, acc_rows, acc_cols)
        if stats is None:
            print("[flujo] la medición se canceló; seguimos igual al tablero.")

    # ----------------------------------------------------------------- tablero
    with GazeTracker(settings) as shell:
        shell.load()
        _final_screen(shell, stats, total, total)

    aac_board.run(settings)

    mins = (time.time() - t0) / 60.0
    print(f"[flujo] terminado en {mins:.1f} min")
    return 0
