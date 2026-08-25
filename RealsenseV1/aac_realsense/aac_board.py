"""Gaze-driven AAC board: dwell on a tile to add it to the message, then SPEAK.

Ported from the EyeTrax harness, since the board itself was not the problem -
the gaze under it was. Three changes, all of them consequences of having
confidence and metric depth:

* **Dwell holds through low-confidence frames** instead of advancing on them.
  See `DwellSelector`: advancing causes mis-selections, resetting throws away the
  user's effort, holding does neither.
* **The tile under the cursor shows why it is not committing** when confidence is
  poor, so a stalled dwell is legible rather than mysterious. Being unable to
  tell "it is ignoring me" from "it cannot see me" is a real problem for someone
  whose only output channel is this board.
* **The session CSV records the sensor state** - confidence, distance, head
  angles, which pose path ran - alongside each selection. That is what turns a
  logged mis-selection into a diagnosable one.
"""

from __future__ import annotations

import csv
import time
from functools import lru_cache

import cv2
import numpy as np

from . import positioning, tts, ui
from .board_layout import ACTION, LETTER, WORD, Tile, layout, message_area
from .config import LOGS_DIR, Settings, ensure_dirs
from .tracker import DwellSelector, GazeTracker

WINDOW = "Tablero CAA"
CAMERA_INSET = (200, 150)


class Message:
    """Composed text as a stack of chunks, so DELETE removes a whole entry."""

    def __init__(self) -> None:
        self.chunks: list[str] = []

    @property
    def text(self) -> str:
        return "".join(self.chunks)

    def add_word(self, word: str) -> None:
        prefix = "" if not self.chunks or self.text.endswith(" ") else " "
        self.chunks.append(prefix + word.lower())

    def add_raw(self, s: str) -> None:
        self.chunks.append(s.lower())

    def delete(self) -> None:
        if self.chunks:
            self.chunks.pop()

    def clear(self) -> None:
        self.chunks.clear()


class SessionLog:
    """CSV of every selection - the raw material for a dwell-accuracy write-up."""

    def __init__(self, enabled: bool, settings: Settings):
        self.file = None
        self.writer = None
        self.settings = settings
        self.t0 = time.time()
        if not enabled:
            return
        ensure_dirs()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.path = LOGS_DIR / f"aac-session-{stamp}.csv"
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            # First eight match the EyeTrax harness for comparability.
            "t_since_start_s", "event", "label", "gaze_x", "gaze_y",
            "dwell_ms", "smoother", "estimator",
            # Sensor state, so a mis-selection can be explained afterwards.
            "confidence", "distance_cm", "iris_px",
            "head_yaw", "head_pitch", "head_roll", "pose_method",
        ])
        print(f"[tablero] registrando selecciones -> {self.path}")

    def selection(self, label: str, sample) -> None:
        if self.writer is None:
            return
        self.writer.writerow([
            f"{time.time() - self.t0:.3f}", "select", label,
            sample.x, sample.y, self.settings.dwell_ms,
            self.settings.smoother, self.settings.estimator,
            f"{sample.confidence:.3f}", f"{sample.distance_cm:.1f}",
            f"{sample.iris_px:.2f}", f"{sample.head_yaw:.2f}",
            f"{sample.head_pitch:.2f}", f"{sample.head_roll:.2f}",
            sample.method,
        ])
        self.file.flush()

    def close(self) -> None:
        if self.file is not None:
            self.file.close()


def _draw_message(img, msg: Message, screen_w: int, screen_h: int,
                  reserve_right: int = 0) -> None:
    x1, y1, x2, y2 = message_area(screen_w, screen_h)
    ui.rounded_rect(img, x1 + 10, y1 + 10, x2 - 10, y2 - 6, (30, 30, 34))

    text = msg.text.upper() or "..."
    # Keep the tail visible when the sentence outgrows the bar, and stay clear
    # of the camera inset parked on the right.
    avail_x1, avail_x2 = x1 + 30, x2 - 30 - reserve_right
    max_w = avail_x2 - avail_x1
    size = 44
    while len(text) > 1 and ui.text_size(text, size, bold=True)[0] > max_w:
        text = text[1:]
    color = ui.TEXT if msg.chunks else ui.TEXT_DIM
    ui.text_center(img, text, (avail_x1 + avail_x2) // 2, (y1 + y2) // 2 - 6,
                   size, color, bold=True)


@lru_cache(maxsize=256)
def _tile_bitmap(label: str, kind: str, w: int, h: int, hot: bool):
    """Pre-render a whole tile once.

    A tile's appearance depends only on these five things, and a board redraws
    the same 25-35 tiles every frame. Compositing text per frame cost ~35 ms at
    1080p in the webcam harness; caching the finished bitmap turns each tile into
    a memcpy and is what keeps the board at ~12 ms/frame.
    """
    base = ui.TILE_HOT if hot else ui.TILE
    if kind == ACTION:
        base = tuple(min(255, c + 14) for c in base)

    bmp = np.empty((h, w, 3), dtype=np.uint8)
    bmp[:] = ui.BG
    ui.rounded_rect(bmp, 0, 0, w - 1, h - 1, base)

    size = ui.fit_text_size(label, w - 24, h - 24, bold=True)
    color = ui.ACCENT if kind == ACTION else ui.TEXT
    ui.text_center(bmp, label, w // 2, h // 2, size, color, bold=True)
    return bmp


def _draw_tile(img, tile: Tile, progress: float, hot: bool,
               holding: bool = False) -> None:
    w, h = tile.x2 - tile.x1, tile.y2 - tile.y1
    if w <= 0 or h <= 0:
        return
    img[tile.y1:tile.y2, tile.x1:tile.x2] = _tile_bitmap(
        tile.key.label, tile.key.kind, w, h, hot)

    if progress > 0:
        # Amber while the dwell advances, dim while it is held because the
        # signal is poor - so a stall is visibly a sensor problem, not the
        # board ignoring the user.
        ui.progress_border(img, tile.x1, tile.y1, tile.x2, tile.y2, progress,
                           ui.TEXT_DIM if holding else ui.ACCENT, 6)


def run(settings: Settings) -> None:
    with GazeTracker(settings) as tracker:
        tracker.load()
        w, h = tracker.screen_w, tracker.screen_h

        page = "core"
        tiles = layout(page, w, h)
        by_id = {t.tile_id: t for t in tiles}

        msg = Message()
        dwell = DwellSelector(settings)
        log = SessionLog(settings.log_session, settings)
        speak_on = settings.speak and tts.available()

        print(f"[tablero] pantalla {tracker.screen.index} "
              f"({tracker.screen.name}) {w}x{h} "
              f"permanencia={settings.dwell_ms}ms "
              f"estimador={settings.estimator}")
        print("[tablero] ESC/Q salir | [ ] permanencia -/+100ms | S voz | "
              "C cámara | X limpiar")
        if settings.speak and not speak_on:
            print("[tablero] no hay voz SAPI disponible - las selecciones se "
                  "imprimen")

        last_fired, last_fired_at = "", 0.0
        frames, fps, t0 = 0, 0.0, time.time()

        with tracker.open_source() as src:
            if settings.check_position:
                if positioning.run(settings, tracker, src) is None:
                    print("[tablero] cancelado en la colocación.")
                    return

            tracker.open_window(WINDOW)
            for frame in src.frames():
                sample = tracker.process(frame)

                hovered: Tile | None = None
                if sample.ok:
                    for t in tiles:
                        if t.contains(sample.x, sample.y):
                            hovered = t
                            break

                progress, fired_id = dwell.update(
                    hovered.tile_id if hovered else None, sample.confidence)
                holding = (hovered is not None
                           and sample.confidence < dwell.min_confidence)

                if fired_id is not None:
                    tile = by_id[fired_id]
                    key = tile.key
                    last_fired, last_fired_at = key.label, time.time()
                    log.selection(key.label, sample)

                    if key.kind == WORD:
                        msg.add_word(key.payload)
                    elif key.kind == LETTER:
                        msg.add_raw(key.payload)
                    else:
                        action = key.payload
                        if action.startswith("page:"):
                            page = action.split(":", 1)[1]
                            tiles = layout(page, w, h)
                            by_id = {t.tile_id: t for t in tiles}
                            # Do not let the gaze that switched pages fire
                            # whatever tile now sits under it.
                            dwell.block(0.7)
                        elif action == "space":
                            msg.add_raw(" ")
                        elif action == "delete":
                            msg.delete()
                        elif action == "clear":
                            msg.clear()
                        elif action == "speak":
                            if speak_on:
                                tts.speak(msg.text)
                            else:
                                print(f"[voz] {msg.text}")

                # ------------------------------ render ------------------------
                img = ui.canvas(w, h)
                _draw_message(img, msg, w, h,
                              reserve_right=CAMERA_INSET[0] + 20
                              if settings.show_camera else 0)
                for t in tiles:
                    is_hot = hovered is not None and t.tile_id == hovered.tile_id
                    _draw_tile(img, t, progress if is_hot else 0.0, is_hot,
                               holding and is_hot)

                if sample.ok:
                    ui.gaze_cursor(img, sample.x, sample.y)
                    if settings.show_raw_gaze:
                        ui.gaze_cursor(img, sample.raw_x, sample.raw_y, ui.RAW, 9)

                frames += 1
                if frames % 10 == 0:
                    now = time.time()
                    fps = 10.0 / max(now - t0, 1e-6)
                    t0 = now

                if not sample.face:
                    cv2.rectangle(img, (0, 0), (w - 1, h - 1), ui.WARN, 6)
                elif holding:
                    # Amber frame: seen, but not trusted enough to commit.
                    cv2.rectangle(img, (0, 0), (w - 1, h - 1), ui.TILE_HOT, 4)

                state = ("OK" if sample.ok
                         else "PARPADEO" if sample.blink else "SIN CARA")
                recent = (f"  última: {last_fired}"
                          if time.time() - last_fired_at < 2 else "")
                # Status lives inside the message panel so it never covers a tile.
                ui.status_bar(img, [
                    f"{state}  {fps:4.1f} fps  conf {sample.confidence:.2f}  "
                    f"{sample.distance_cm:.0f} cm  perm={settings.dwell_ms}ms  "
                    f"pág={page}  voz={'sí' if speak_on else 'no'}"
                    f"  [ESC salir]{recent}"
                    f"{'   SEÑAL DÉBIL - manteniendo' if holding else ''}",
                ], y=int(h * 0.16) - 26, size=19)
                if settings.show_camera:
                    ui.camera_inset(img, frame.color, size=CAMERA_INSET,
                                    margin=14, anchor="tr")

                cv2.imshow(WINDOW, img)
                key_code = cv2.waitKey(1) & 0xFF
                if key_code in (27, ord("q")):
                    break
                if key_code == ord("["):
                    settings.dwell_ms = max(200, settings.dwell_ms - 100)
                    dwell = DwellSelector(settings)
                if key_code == ord("]"):
                    settings.dwell_ms = min(3000, settings.dwell_ms + 100)
                    dwell = DwellSelector(settings)
                if key_code == ord("s"):
                    speak_on = not speak_on and tts.available()
                if key_code == ord("c"):
                    settings.show_camera = not settings.show_camera
                if key_code == ord("x"):
                    msg.clear()

        log.close()
        cv2.destroyAllWindows()
        print(f"[tablero] mensaje final: {msg.text!r}")
