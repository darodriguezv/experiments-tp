"""Live gaze overlay - the quickest way to judge whether calibration is usable.

Shows the smoothed gaze point (and optionally the raw model output next to it)
on a fullscreen canvas with a reference grid.
"""

from __future__ import annotations

import time

import cv2


from . import positioning, ui
from .video import camera, iter_frames
from .config import Settings
from .tracker import GazeTracker

WINDOW = "CAA - mirada en vivo"


def _grid(img, w: int, h: int) -> None:
    for i in range(1, 3):
        cv2.line(img, (w * i // 3, 0), (w * i // 3, h), (34, 34, 38), 1)
        cv2.line(img, (0, h * i // 3), (w, h * i // 3), (34, 34, 38), 1)


def run(settings: Settings) -> None:
    with GazeTracker(settings) as tracker:
        tracker.load()
        w, h = tracker.screen_w, tracker.screen_h
        print(f"[vivo] pantalla {tracker.screen.index} ({tracker.screen.name}) "
              f"{w}x{h} suavizado={settings.smoother}")
        print("[vivo] ESC o Q para salir, R muestra el punto crudo (sin suavizar)")

        # Volver a la posición de la calibración: sin esto el error es
        # sistemático y ningún suavizado lo corrige.
        if settings.check_position:
            if positioning.run(
                settings, tracker,
                title="VUELVA A LA POSICIÓN DE LA CALIBRACIÓN",
                subtitle="Alinee su cara con la guía antes de comenzar",
            ) is None:
                print("[live] cancelado en la colocación.")
                return

        show_raw = settings.show_raw_gaze
        frames = 0
        fps = 0.0
        t0 = time.time()

        tracker.open_window(WINDOW)
        with camera(settings.camera_index, *settings.capture_size) as cap:
            for frame in iter_frames(cap):
                sample = tracker.process(frame)

                img = ui.canvas(w, h)
                _grid(img, w, h)

                if sample.ok:
                    if show_raw:
                        ui.gaze_cursor(img, sample.raw_x, sample.raw_y, ui.RAW, 10)
                    ui.gaze_cursor(img, sample.x, sample.y)
                elif sample.blink:
                    ui.banner(img, "PARPADEO", ui.TEXT_DIM)
                else:
                    ui.banner(img, "NO SE DETECTA LA CARA")

                frames += 1
                if frames % 10 == 0:
                    now = time.time()
                    fps = 10.0 / max(now - t0, 1e-6)
                    t0 = now

                ui.status_bar(img, [
                    f"suavizado={settings.smoother}  modelo={settings.model_name}  "
                    f"{fps:4.1f} fps",
                    f"mirada=({sample.x:4d},{sample.y:4d})  "
                    f"crudo=({sample.raw_x:5d},{sample.raw_y:5d})",
                    "ESC/Q salir   R punto crudo",
                ])
                if settings.show_camera:
                    ui.camera_inset(img, frame)

                cv2.imshow(WINDOW, img)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("r"):
                    show_raw = not show_raw

        cv2.destroyAllWindows()
