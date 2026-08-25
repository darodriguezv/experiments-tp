"""Live gaze overlay - the fastest way to judge whether a calibration is usable.

Adds one thing the webcam version could not offer: a **head-invariance check you
can feel**. Press `H` and the view stops showing gaze and starts showing what
happens to it as you move. Lean in, lean back, turn a little - the readout tracks
how far you have moved from the calibration posture and how much the gaze point
drifts as a result. With a working geometric model the cursor should stay on
whatever you are looking at while those numbers change.

That is the difference between believing the model is head-invariant and seeing
it, and it takes ten seconds rather than a full accuracy run.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from . import positioning, ui
from .config import Settings
from .tracker import GazeTracker

WINDOW = "CAA - mirada en vivo"


def _grid(img, w: int, h: int) -> None:
    for i in range(1, 3):
        cv2.line(img, (w * i // 3, 0), (w * i // 3, h), (34, 34, 38), 1)
        cv2.line(img, (0, h * i // 3), (w, h * i // 3), (34, 34, 38), 1)


def _confidence_ring(img, x: int, y: int, confidence: float) -> None:
    """Draw the cursor's own confidence around it.

    Worth the pixels: a user learning the system needs to distinguish "the
    system cannot see me right now" from "the system thinks I am looking there".
    The webcam pipeline could not tell them apart, so a bad frame and a bad aim
    looked identical.
    """
    colour = ui.GOOD if confidence > 0.6 else (
        ui.ACCENT if confidence > 0.3 else ui.WARN)
    cv2.ellipse(img, (x, y), (26, 26), 0, -90,
                -90 + int(360 * np.clip(confidence, 0.0, 1.0)),
                colour, 3, cv2.LINE_AA)


def run(settings: Settings) -> None:
    with GazeTracker(settings) as tracker:
        tracker.load()
        w, h = tracker.screen_w, tracker.screen_h
        print(f"[vivo] pantalla {tracker.screen.index} ({tracker.screen.name}) "
              f"{w}x{h} suavizado={settings.smoother}")
        print("[vivo] ESC/Q salir | R punto crudo | H prueba de postura | "
              "C cámara")

        show_raw = settings.show_raw_gaze
        head_test = False
        anchor: np.ndarray | None = None
        anchor_gaze: tuple[int, int] | None = None
        frames, fps, t0 = 0, 0.0, time.time()

        with tracker.open_source() as src:
            if settings.check_position:
                if positioning.run(settings, tracker, src) is None:
                    print("[vivo] cancelado en la colocación.")
                    return

            tracker.open_window(WINDOW)
            for frame in src.frames():
                sample = tracker.process(frame)

                img = ui.canvas(w, h)
                _grid(img, w, h)

                if sample.ok:
                    if show_raw:
                        ui.gaze_cursor(img, sample.raw_x, sample.raw_y, ui.RAW, 10)
                    ui.gaze_cursor(img, sample.x, sample.y)
                    _confidence_ring(img, sample.x, sample.y, sample.confidence)
                elif sample.blink:
                    ui.banner(img, "PARPADEO", ui.TEXT_DIM)
                else:
                    ui.banner(img, (sample.reason or "NO SE DETECTA LA CARA").upper())

                # -------------------------------------------- head-move test
                if head_test and sample.ok and tracker.last_head is not None:
                    here = tracker.last_head.translation
                    if anchor is None:
                        anchor = here.copy()
                        anchor_gaze = (sample.x, sample.y)
                    moved_cm = float(np.linalg.norm(here - anchor)) * 100.0
                    drift = float(np.hypot(sample.x - anchor_gaze[0],
                                           sample.y - anchor_gaze[1]))
                    cv2.circle(img, anchor_gaze, 10, ui.TEXT_DIM, 2, cv2.LINE_AA)
                    cv2.line(img, anchor_gaze, (sample.x, sample.y),
                             ui.TEXT_DIM, 1, cv2.LINE_AA)
                    ui.text_center(
                        img,
                        f"movimiento de cabeza {moved_cm:5.1f} cm    "
                        f"desplazamiento de la mirada {drift:5.0f} px",
                        w // 2, 60, 34,
                        ui.GOOD if drift < 80 else ui.WARN, bold=True)
                    ui.text_center(
                        img,
                        "Mire un punto fijo y muévase. Si el modelo es "
                        "invariante a la postura, la mirada casi no se corre.",
                        w // 2, 100, 22, ui.TEXT_DIM)

                frames += 1
                if frames % 10 == 0:
                    now = time.time()
                    fps = 10.0 / max(now - t0, 1e-6)
                    t0 = now

                ui.status_bar(img, [
                    f"{fps:4.1f} fps   conf {sample.confidence:.2f}   "
                    f"{sample.distance_cm:5.1f} cm   iris {sample.iris_px:4.1f} px   "
                    f"pose={sample.method or '-'}",
                    f"cabeza  giro {sample.head_yaw:+5.1f}°  "
                    f"inclinación {sample.head_pitch:+5.1f}°  "
                    f"rotación {sample.head_roll:+5.1f}°",
                    f"mirada=({sample.x:4d},{sample.y:4d})  "
                    f"cruda=({sample.raw_x:5d},{sample.raw_y:5d})"
                    f"{'' if sample.on_screen else '   [fuera de pantalla]'}",
                    "ESC/Q salir   R crudo   H prueba de postura   C cámara",
                ])
                if settings.show_camera:
                    ui.camera_inset(img, frame.color)

                cv2.imshow(WINDOW, img)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("r"):
                    show_raw = not show_raw
                if key == ord("c"):
                    settings.show_camera = not settings.show_camera
                if key == ord("h"):
                    head_test = not head_test
                    anchor = anchor_gaze = None

        cv2.destroyAllWindows()
