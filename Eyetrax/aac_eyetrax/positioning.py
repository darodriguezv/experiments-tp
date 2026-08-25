"""Guía de colocación: le muestra a la persona dónde poner la cara.

Por qué existe: un modelo de mirada solo es válido para, más o menos, la
posición de cabeza con la que se calibró. Si se calibra sentado cerca y después
se usa recostado, el error es sistemático y no hay suavizado que lo corrija.

La pantalla muestra la cámara en espejo con un óvalo objetivo encima. El espejo
es deliberado: con la imagen invertida, moverse hacia el óvalo es intuitivo y no
hace falta razonar sobre izquierda y derecha.

Cómo se decide si la posición es correcta:

* **Antes de calibrar** no hay referencia, así que solo se exige lo que se puede
  definir en absoluto: estar centrado, estar a una distancia razonable (un rango
  amplio, no un valor exacto) y tener la cabeza derecha y de frente. Exigir una
  distancia exacta dejaría a la persona atascada si su cámara no la permite.
* **Después de calibrar** el objetivo es la posición registrada, y se compara
  contra ella. La inclinación vertical (pitch) solo se controla en este caso:
  su valor de base depende de la cara y de la altura de la cámara, así que no
  existe un "cero" absoluto con el que comparar.

Las medidas se filtran con una mediana temporal, porque los puntos de MediaPipe
oscilan cuadro a cuadro y sin filtrar los avisos parpadean.
"""

from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np

from . import ui
from .config import Settings
from .tracker import DEFAULT_POSE, FacePose, GazeTracker
from .video import camera, iter_frames

WINDOW = "CAA - colocación"

# Rangos aceptables cuando todavía no hay una calibración de referencia.
SPAN_MIN, SPAN_MAX = 0.115, 0.260
CX_MIN, CX_MAX = 0.34, 0.66
CY_MIN, CY_MAX = 0.26, 0.60
ROLL_MAX = 14.0
YAW_MAX = 0.22

# Tolerancias cuando se compara contra la posición de la calibración.
TOL_CENTRE = 0.060
TOL_SPAN = 0.045
TOL_ROLL = 14.0
TOL_YAW = 0.22
TOL_PITCH = 0.20

HYSTERESIS = 1.45
"""Una vez alineado se acepta algo más de holgura, para que el estado no
parpadee justo en el límite."""

HOLD_S = 1.2
"""Cuánto hay que mantener la posición correcta antes de continuar."""

FILTER_N = 7
"""Cuadros de la mediana temporal (~0.25 s a 30 fps)."""

LOST_RESET_S = 0.6
"""Tras este tiempo sin cara, se descarta el historial del filtro."""


class PoseFilter:
    """Mediana temporal de las medidas de pose.

    Mediana y no promedio: un cuadro con los puntos mal detectados produce un
    valor disparatado, y la mediana lo ignora en lugar de repartirlo.
    """

    def __init__(self, n: int = FILTER_N) -> None:
        self.buf: deque[FacePose] = deque(maxlen=n)

    def reset(self) -> None:
        self.buf.clear()

    def update(self, pose: FacePose) -> FacePose:
        self.buf.append(pose)
        if len(self.buf) < 3:
            return pose  # al principio conviene responder rápido

        def med(field: str) -> float:
            return float(np.median([getattr(p, field) for p in self.buf]))

        return FacePose(
            cx=med("cx"), cy=med("cy"), eye_span=med("eye_span"),
            roll_deg=med("roll_deg"), yaw_ratio=med("yaw_ratio"),
            pitch_ratio=med("pitch_ratio"),
        )


def _preview_rect(screen_w: int, screen_h: int) -> tuple[int, int, int, int]:
    """Zona centrada, en 16:9, donde se dibuja la cámara.

    Se deja libre el tercio inferior: ahí van las indicaciones, y si la vista
    previa llegara hasta ese punto el texto quedaría encima de la imagen.
    """
    pw = int(screen_w * 0.52)
    ph = int(pw * 9 / 16)
    max_h = int(screen_h * 0.52)
    if ph > max_h:
        ph = max_h
        pw = int(ph * 16 / 9)
    x1 = (screen_w - pw) // 2
    y1 = int(screen_h * 0.17)
    return x1, y1, x1 + pw, y1 + ph


# Proporciones de una cabeza respecto de la distancia entre las esquinas
# externas de los ojos, que es lo que se mide.
_HEAD_RX = 1.0      # semieje horizontal = 1.0 x separación de ojos
_HEAD_RATIO = 1.35  # semieje vertical respecto del horizontal
_EYE_OFFSET = 0.32  # los ojos quedan por encima del centro de la cabeza


def _head_ellipse(cx_px: int, eye_y: int, span_px: float):
    """Óvalo de cabeza a partir del punto medio de los ojos."""
    rx = max(8, int(span_px * _HEAD_RX))
    ry = max(8, int(rx * _HEAD_RATIO))
    return (cx_px, int(eye_y + _EYE_OFFSET * ry)), (rx, ry)


def _hints(
    pose: FacePose,
    target: FacePose,
    has_reference: bool = False,
    slack: float = 1.0,
) -> list[str]:
    """Qué corregir, en orden de importancia. Lista vacía = posición correcta."""
    out: list[str] = []

    # --- distancia: lo que más afecta a la precisión ---
    if has_reference:
        tol = TOL_SPAN * slack
        if pose.eye_span < target.eye_span - tol:
            out.append("ACÉRQUESE A LA CÁMARA")
        elif pose.eye_span > target.eye_span + tol:
            out.append("ALÉJESE UN POCO")
    else:
        if pose.eye_span < SPAN_MIN / slack:
            out.append("ACÉRQUESE A LA CÁMARA")
        elif pose.eye_span > SPAN_MAX * slack:
            out.append("ALÉJESE UN POCO")

    # --- centrado: las indicaciones se dan sobre la imagen en espejo, que es
    # lo que la persona está viendo ---
    if has_reference:
        tol = TOL_CENTRE * slack
        dx = target.cx - pose.cx
        if dx < -tol:
            out.append("MUÉVASE A LA DERECHA")
        elif dx > tol:
            out.append("MUÉVASE A LA IZQUIERDA")
        dy = pose.cy - target.cy
        if dy > tol:
            out.append("SUBA UN POCO")
        elif dy < -tol:
            out.append("BAJE UN POCO")
    else:
        margin = (CX_MAX - CX_MIN) * (slack - 1.0) / 2.0
        if pose.cx > CX_MAX + margin:
            out.append("MUÉVASE A LA DERECHA")
        elif pose.cx < CX_MIN - margin:
            out.append("MUÉVASE A LA IZQUIERDA")
        if pose.cy > CY_MAX + margin:
            out.append("SUBA UN POCO")
        elif pose.cy < CY_MIN - margin:
            out.append("BAJE UN POCO")

    # --- inclinación de la cabeza ---
    ref_roll = target.roll_deg if has_reference else 0.0
    if abs(pose.roll_deg - ref_roll) > ROLL_MAX * slack:
        out.append("ENDERECE LA CABEZA")

    ref_yaw = target.yaw_ratio if has_reference else 0.0
    if abs(pose.yaw_ratio - ref_yaw) > YAW_MAX * slack:
        out.append("MIRE DE FRENTE A LA PANTALLA")

    # El pitch no tiene un cero absoluto: su base depende de la cara y de la
    # altura de la cámara, así que solo se compara contra la referencia.
    if has_reference:
        dp = pose.pitch_ratio - target.pitch_ratio
        if dp > TOL_PITCH * slack:
            out.append("LEVANTE UN POCO LA BARBILLA")
        elif dp < -TOL_PITCH * slack:
            out.append("BAJE UN POCO LA BARBILLA")

    return out


def _draw_face_guide(img, rect, pose: FacePose | None, target: FacePose, aligned: bool):
    """Dibuja el óvalo objetivo y, si hay cara, la posición actual."""
    x1, y1, x2, y2 = rect
    pw, ph = x2 - x1, y2 - y1

    # El objetivo se dibuja en coordenadas espejadas, igual que la imagen.
    t_eye_x = x1 + int((1.0 - target.cx) * pw)
    t_eye_y = y1 + int(target.cy * ph)
    t_centre, t_axes = _head_ellipse(t_eye_x, t_eye_y, target.eye_span * pw)

    colour = ui.GOOD if aligned else ui.ACCENT
    # Guía objetivo: trazo discontinuo, para distinguirla del óvalo real.
    for a0 in range(0, 360, 22):
        cv2.ellipse(img, t_centre, t_axes, 0, a0, a0 + 12, colour, 4, cv2.LINE_AA)
    # Marca a la altura de los ojos: es lo que hay que hacer coincidir.
    cv2.line(img, (t_eye_x - t_axes[0] // 2, t_eye_y),
             (t_eye_x + t_axes[0] // 2, t_eye_y), colour, 2, cv2.LINE_AA)

    if pose is not None:
        f_eye_x = x1 + int((1.0 - pose.cx) * pw)
        f_eye_y = y1 + int(pose.cy * ph)
        f_centre, f_axes = _head_ellipse(f_eye_x, f_eye_y, pose.eye_span * pw)
        # En espejo la inclinación se ve invertida, de ahí el signo.
        cv2.ellipse(img, f_centre, f_axes, -pose.roll_deg, 0, 360,
                    ui.GOOD if aligned else ui.TEXT, 3, cv2.LINE_AA)
        # Línea de los ojos: hace visible la inclinación real.
        half = f_axes[0]
        a = np.radians(-pose.roll_deg)
        dx, dy = int(half * np.cos(a)), int(half * np.sin(a))
        cv2.line(img, (f_eye_x - dx, f_eye_y - dy), (f_eye_x + dx, f_eye_y + dy),
                 ui.GOOD if aligned else ui.TEXT, 2, cv2.LINE_AA)
        if not aligned:
            cv2.arrowedLine(img, (f_eye_x, f_eye_y), (t_eye_x, t_eye_y),
                            ui.ACCENT, 3, cv2.LINE_AA, tipLength=0.25)


def run(
    settings: Settings,
    tracker: GazeTracker,
    *,
    target: FacePose | None = None,
    title: str = "COLOQUE SU CARA EN LA GUÍA",
    subtitle: str = "",
    require_hold: bool = True,
) -> FacePose | None:
    """Muestra la guía hasta que la posición sea correcta.

    Devuelve la posición alcanzada, o None si se cancela con ESC.
    Con ENTER o ESPACIO se continúa con la posición que haya en ese momento.
    """
    has_reference = target is not None or tracker.reference_pose is not None
    target = target or tracker.reference_pose or DEFAULT_POSE
    w, h = tracker.screen_w, tracker.screen_h
    rect = _preview_rect(w, h)
    px1, py1, px2, py2 = rect

    if has_reference:
        print(f"[colocación] objetivo: posición de la calibración "
              f"(ojos={target.eye_span:.3f}, centro=({target.cx:.2f},{target.cy:.2f}), "
              f"inclinación={target.roll_deg:+.0f}°)")
    else:
        print(f"[colocación] objetivo: rango por defecto "
              f"(ojos {SPAN_MIN:.3f}-{SPAN_MAX:.3f}, centrado, cabeza derecha)")
    print("[colocación] ESC cancela, ENTER continúa con la posición actual")

    tracker.open_window(WINDOW)
    filt = PoseFilter()
    aligned_since: float | None = None
    was_aligned = False
    last_seen = time.time()
    last_pose: FacePose | None = None

    with camera(settings.camera_index, *settings.capture_size) as cap:
        for frame in iter_frames(cap):
            now = time.time()
            raw = tracker.face_pose(frame)

            if raw is None:
                # Una pérdida breve no debe borrar el historial del filtro.
                if now - last_seen > LOST_RESET_S:
                    filt.reset()
                    last_pose = None
                pose = None
            else:
                last_seen = now
                pose = filt.update(raw)
                last_pose = pose

            if pose is None:
                hints: list[str] = []
                aligned = False
            else:
                # Con holgura extra mientras ya está alineado: evita parpadeo.
                slack = HYSTERESIS if was_aligned else 1.0
                hints = _hints(pose, target, has_reference, slack)
                aligned = not hints
            was_aligned = aligned

            if aligned:
                if aligned_since is None:
                    aligned_since = now
            else:
                aligned_since = None

            held = 0.0 if aligned_since is None else (now - aligned_since)
            if require_hold and held >= HOLD_S and pose is not None:
                cv2.destroyWindow(WINDOW)
                print(f"[colocación] posición correcta (ojos={pose.eye_span:.3f}, "
                      f"inclinación={pose.roll_deg:+.0f}°, "
                      f"giro={pose.yaw_ratio:+.2f})")
                return pose

            # ---------------------------------------------------------- dibujo
            img = ui.canvas(w, h)
            ui.text_center(img, title, w // 2, int(h * 0.09), 46, ui.TEXT, bold=True)
            if subtitle:
                ui.text_center(img, subtitle, w // 2, int(h * 0.145), 24, ui.TEXT_DIM)

            preview = cv2.flip(cv2.resize(frame, (px2 - px1, py2 - py1)), 1)
            img[py1:py2, px1:px2] = (preview * 0.55).astype(np.uint8)
            cv2.rectangle(img, (px1 - 2, py1 - 2), (px2 + 1, py2 + 1), ui.TILE, 2)

            _draw_face_guide(img, rect, pose, target, aligned)

            if pose is None:
                ui.text_center(img, "NO SE DETECTA LA CARA", w // 2,
                               int(h * 0.74), 44, ui.WARN, bold=True)
                ui.text_center(img, "Verifique que haya luz de frente a su cara",
                               w // 2, int(h * 0.795), 24, ui.TEXT_DIM)
            elif hints:
                ui.text_center(img, hints[0], w // 2, int(h * 0.74), 46,
                               ui.ACCENT, bold=True)
                if len(hints) > 1:
                    ui.text_center(img, "   ".join(hints[1:]), w // 2,
                                   int(h * 0.795), 24, ui.TEXT_DIM)
            else:
                ui.text_center(img, "PERFECTO · NO SE MUEVA", w // 2,
                               int(h * 0.74), 46, ui.GOOD, bold=True)
                if require_hold:
                    frac = min(1.0, held / HOLD_S)
                    bw = int((px2 - px1) * 0.5)
                    bx = w // 2 - bw // 2
                    by = int(h * 0.83)
                    cv2.rectangle(img, (bx, by), (bx + bw, by + 16), ui.TILE, -1)
                    cv2.rectangle(img, (bx, by), (bx + int(bw * frac), by + 16),
                                  ui.GOOD, -1)

            if pose is not None:
                objetivo = (f"{target.eye_span:.3f}" if has_reference
                            else f"{SPAN_MIN:.2f}-{SPAN_MAX:.2f}")
                ui.status_bar(img, [
                    f"distancia(ojos) {pose.eye_span:.3f} / {objetivo}    "
                    f"centro ({pose.cx:.2f},{pose.cy:.2f})    "
                    f"inclinación {pose.roll_deg:+.0f}°    "
                    f"giro {pose.yaw_ratio:+.2f}",
                ], y=h - 40, size=20)

            ui.text_center(img, "ESC cancelar     ENTER continuar", w // 2,
                           int(h * 0.955), 20, ui.TEXT_DIM)

            cv2.imshow(WINDOW, img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                cv2.destroyWindow(WINDOW)
                print("[colocación] cancelada")
                return None
            if key in (13, 32):  # ENTER / ESPACIO: seguir con lo que haya
                cv2.destroyWindow(WINDOW)
                return last_pose

    cv2.destroyWindow(WINDOW)
    return last_pose
