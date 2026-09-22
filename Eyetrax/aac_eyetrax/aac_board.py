"""Gaze-driven AAC board: dwell on a tile to add it to the message, then SPEAK.

This is the part that actually answers "is EyeTrax good enough for our AAC
system?" - it exercises accuracy, jitter and blink handling under the real
interaction, not just a cursor overlay.
"""

from __future__ import annotations

import csv
import time
from functools import lru_cache

import cv2
import numpy as np


from . import positioning, tts, ui
from .video import camera, iter_frames
from .board_layout import ACTION, LETTER, WORD, Tile, layout, message_area
from .config import LOGS_DIR, Settings, ensure_dirs
from .tracker import DwellSelector, GazeTracker

WINDOW = "Tablero CAA"


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
        self.writer.writerow(
            ["t_since_start_s", "event", "label", "gaze_x", "gaze_y",
             "dwell_ms", "smoother", "model"]
        )
        print(f"[tablero] registrando selecciones -> {self.path}")

    def selection(self, label: str, x: int, y: int) -> None:
        if self.writer is None:
            return
        self.writer.writerow([
            f"{time.time() - self.t0:.3f}", "select", label, x, y,
            self.settings.dwell_ms, self.settings.smoother, self.settings.model_name,
        ])
        self.file.flush()

    def close(self) -> None:
        if self.file is not None:
            self.file.close()


CAMERA_INSET = (200, 150)


def _draw_message(
    img, msg: Message, screen_w: int, screen_h: int, reserve_right: int = 0
) -> None:
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
    ui.text_center(
        img, text, (avail_x1 + avail_x2) // 2, (y1 + y2) // 2 - 6, size, color, bold=True
    )


@lru_cache(maxsize=256)
def _tile_bitmap(label: str, kind: str, w: int, h: int, hot: bool):
    """Pre-render a whole tile (background + label) once.

    A tile's appearance only depends on these five things, and a board redraws
    the same 25-35 tiles every frame. Compositing text per frame cost ~35 ms at
    1080p; caching the finished bitmap turns each tile into a plain memcpy.
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


def _draw_tile(img, tile: Tile, progress: float, hot: bool) -> None:
    w, h = tile.x2 - tile.x1, tile.y2 - tile.y1
    if w <= 0 or h <= 0:
        return
    img[tile.y1:tile.y2, tile.x1:tile.x2] = _tile_bitmap(
        tile.key.label, tile.key.kind, w, h, hot
    )

    if progress > 0:
        ui.progress_border(
            img, tile.x1, tile.y1, tile.x2, tile.y2, progress, ui.ACCENT, 6
        )


def run(settings: Settings, make_tracker=None) -> None:
    with (make_tracker or GazeTracker)(settings) as tracker:
        tracker.load()
        w, h = tracker.screen_w, tracker.screen_h

        # Volver a la posición de la calibración: sin esto el error es
        # sistemático y ningún suavizado lo corrige.
        if settings.check_position:
            if positioning.run(
                settings, tracker,
                title="VUELVA A LA POSICIÓN DE LA CALIBRACIÓN",
                subtitle="Alinee su cara con la guía antes de comenzar",
            ) is None:
                print("[tablero] cancelado en la colocación.")
                return

        page = "core"
        tiles = layout(page, w, h)
        by_id = {t.tile_id: t for t in tiles}

        msg = Message()
        dwell = DwellSelector(settings)
        log = SessionLog(settings.log_session, settings)
        speak_on = settings.speak and tts.available()

        print(f"[tablero] pantalla {tracker.screen.index} ({tracker.screen.name}) "
              f"{w}x{h} permanencia={settings.dwell_ms}ms suavizado={settings.smoother}")
        print("[tablero] ESC/Q salir | [ ] permanencia -/+100ms | S voz | C cámara | X limpiar")
        if settings.speak and not speak_on:
            print("[tablero] no hay voz SAPI disponible - las selecciones se imprimen")

        last_fired = ""
        last_fired_at = 0.0
        frames, fps, t0 = 0, 0.0, time.time()

        tracker.open_window(WINDOW)
        with camera(settings.camera_index, *settings.capture_size) as cap:
            for frame in iter_frames(cap):
                sample = tracker.process(frame)

                # La celda se decide sobre la mediana de una ventana corta y se
                # "pega" mientras esa mediana no salga de la celda agrandada.
                # Con el punto crudo de cada cuadro, una sola muestra que cruza
                # el borde reiniciaba la permanencia y seleccionar era
                # imposible: ver DwellSelector.step.
                def _hit(x, y):
                    for t in tiles:
                        if t.contains(x, y):
                            return t.tile_id
                    return None

                def _bounds(tile_id):
                    t = by_id.get(tile_id)
                    return None if t is None else (t.x1, t.y1, t.x2, t.y2)

                progress, fired_id, stable = dwell.step(sample, _hit, _bounds)
                # Se resalta la celda que ESTÁ acumulando, no la que queda bajo
                # el punto: con el pegado las dos pueden diferir, y mostrar la
                # segunda haría que la barra de progreso apareciera sobre una
                # celda distinta de la que se va a seleccionar.
                hovered = by_id.get(dwell.target) if dwell.target else None

                if fired_id is not None:
                    tile = by_id[fired_id]
                    key = tile.key
                    last_fired, last_fired_at = key.label, time.time()
                    log.selection(key.label, sample.x, sample.y)

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
                            # Don't let the gaze that switched pages immediately
                            # trigger whatever tile now sits under it.
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
                                print(f"[speak] {msg.text}")

                # ---------------- render ----------------
                img = ui.canvas(w, h)
                _draw_message(
                    img, msg, w, h,
                    reserve_right=CAMERA_INSET[0] + 20 if settings.show_camera else 0,
                )
                for t in tiles:
                    is_hot = hovered is not None and t.tile_id == hovered.tile_id
                    _draw_tile(img, t, progress if is_hot else 0.0, is_hot)

                if sample.ok:
                    # El cursor grande va en el punto ESTABILIZADO, que es el
                    # que decide la selección. Dibujarlo en el punto por cuadro
                    # mostraba un temblor que el sistema ya estaba ignorando, y
                    # eso desanima: parece que no entiende hacia dónde se mira
                    # cuando en realidad sí. El punto vivo queda como marca
                    # tenue, para no perder la sensación de respuesta.
                    if stable:
                        ui.gaze_cursor(img, sample.x, sample.y, ui.RAW, 7)
                        ui.gaze_cursor(img, stable[0], stable[1])
                    else:
                        ui.gaze_cursor(img, sample.x, sample.y)
                    if settings.show_raw_gaze:
                        ui.gaze_cursor(img, sample.raw_x, sample.raw_y, ui.RAW, 9)

                frames += 1
                if frames % 10 == 0:
                    now = time.time()
                    fps = 10.0 / max(now - t0, 1e-6)
                    t0 = now

                state = "OK" if sample.ok else ("PARPADEO" if sample.blink else "SIN CARA")
                if not sample.face:
                    cv2.rectangle(img, (0, 0), (w - 1, h - 1), ui.WARN, 6)

                recent = f"  última: {last_fired}" if time.time() - last_fired_at < 2 else ""
                # Status lives inside the message panel so it never covers a tile.
                ui.status_bar(img, [
                    f"{state}  {fps:4.1f} fps  perm={settings.dwell_ms}ms  "
                    f"pág={page}  voz={'sí' if speak_on else 'no'}"
                    f"  [ESC salir]{recent}",
                ], y=int(h * 0.16) - 26, size=19)
                if settings.show_camera:
                    ui.camera_inset(img, frame, size=CAMERA_INSET, margin=14, anchor="tr")

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
