"""Live gaze overlay - the quickest way to judge whether calibration is usable.

Shows the smoothed gaze point (and optionally the raw model output next to it)
on a fullscreen canvas with a reference grid.
"""

from __future__ import annotations

import time

from collections import deque

import cv2
import numpy as np


from . import positioning, ui
from .video import camera, iter_frames
from .config import Settings
from .accuracy import PX_PER_DEG
from .tracker import GazeTracker, MedianWindow

WINDOW = "CAA - mirada en vivo"


def _grid(img, w: int, h: int) -> None:
    for i in range(1, 3):
        cv2.line(img, (w * i // 3, 0), (w * i // 3, h), (34, 34, 38), 1)
        cv2.line(img, (0, h * i // 3), (w, h * i // 3), (34, 34, 38), 1)


def run(settings: Settings, make_tracker=None) -> None:
    with (make_tracker or GazeTracker)(settings) as tracker:
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

        estable = MedianWindow(settings.fixation_window_ms / 1000.0)
        # Medidor de temblor: distancia mediana entre el punto por cuadro y el
        # estabilizado, sobre el ultimo segundo. Convierte "vibra demasiado" en
        # un numero, que es lo unico con lo que se puede decidir si hay que
        # tocar el suavizado, la ventana, o si el problema es la calibracion.
        temblores: deque = deque(maxlen=60)
        tracker.open_window(WINDOW)
        with camera(settings.camera_index, *settings.capture_size) as cap:
            for frame in iter_frames(cap):
                sample = tracker.process(frame)

                img = ui.canvas(w, h)
                _grid(img, w, h)

                if sample.ok:
                    # Mismo punto estabilizado que usa el tablero. Antes `live`
                    # mostraba el punto por cuadro y el tablero decidia sobre la
                    # mediana, asi que `live` se veia mucho peor de lo que el
                    # tablero se comporta - y `live` es justamente la pantalla
                    # donde uno juzga si la calibracion sirve.
                    sx, sy = estable.push(sample.x, sample.y)
                    temblores.append(float(np.hypot(sample.x - sx,
                                                    sample.y - sy)))
                    if show_raw:
                        ui.gaze_cursor(img, sample.raw_x, sample.raw_y, ui.RAW, 10)
                    ui.gaze_cursor(img, sample.x, sample.y, ui.RAW, 7)
                    ui.gaze_cursor(img, sx, sy)
                elif sample.blink:
                    estable.clear()
                    ui.banner(img, "PARPADEO", ui.TEXT_DIM)
                else:
                    estable.clear()
                    ui.banner(img, "NO SE DETECTA LA CARA")

                frames += 1
                if frames % 10 == 0:
                    now = time.time()
                    fps = 10.0 / max(now - t0, 1e-6)
                    t0 = now

                # Deriva de cabeza: solo los trackers que la calculan gratis
                # la exponen (WebcamV1 reusa los puntos que ya detecto). Sin
                # esto, alejarse o ladearse degrada todo en silencio, que es
                # exactamente lo que hace pensar que el estimador no sirve.
                deriva = getattr(tracker, "pose_drift", None)
                deriva = deriva() if deriva else None
                if temblores:
                    tj = float(np.median(temblores))
                    linea_temblor = (f"temblor {tj:4.0f} px = "
                                     f"{tj / PX_PER_DEG:4.2f}°")
                else:
                    linea_temblor = "temblor --"
                ui.status_bar(img, [
                    f"suavizado={settings.smoother}  modelo={settings.model_name}  "
                    f"{fps:4.1f} fps   {linea_temblor}",
                    f"mirada=({sample.x:4d},{sample.y:4d})  "
                    f"crudo=({sample.raw_x:5d},{sample.raw_y:5d})",
                    "ESC/Q salir   R punto crudo",
                ])
                if deriva is not None:
                    etiqueta, grave = deriva
                    if grave:
                        ui.banner(img, etiqueta, ui.WARN)
                    elif etiqueta != "posición OK":
                        ui.text_center(img, etiqueta, w // 2, int(h * 0.06), 24,
                                       ui.ACCENT)
                if settings.show_camera:
                    ui.camera_inset(img, frame)

                cv2.imshow(WINDOW, img)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("r"):
                    show_raw = not show_raw

        cv2.destroyAllWindows()
