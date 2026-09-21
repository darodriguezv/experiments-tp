"""La calibración por puntos, escrita acá porque no se podía reusar.

Eyetrax delega la calibración en las rutinas del propio paquete `eyetrax`, que
son dueñas del bucle: abren la cámara, dibujan sus puntos y entrenan **su**
estimador. Nada de eso sirve para GazeFollower, que necesita recolectar el
embedding de MGazeNet y ajustar una SVR sobre él.

Lo que sí se conserva es todo lo demás: la misma pantalla, el mismo suavizado,
la misma guía de colocación y la misma medición de precisión después. La
calibración es la única pieza que difiere, y difiere porque tiene que hacerlo.

## Dos decisiones que afectan el resultado

**Se descarta el primer tramo de cada punto.** Los ojos tardan en llegar y la
sacada misma produce muestras que apuntan a cualquier lado. Promediar eso
mete ruido de movimiento en un modelo que se supone estático. El descarte es
mayor en el primer punto, donde además hay que encontrar el objetivo.

**Un punto con pocas muestras buenas no se rellena, se avisa.** Si la cara se
perdió justo ahí, ese punto queda con menos peso, y forzar muestras malas para
completar el cupo enseñaría al modelo que esa zona de la pantalla se ve así.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

_EYETRAX = Path(__file__).resolve().parents[2] / "Eyetrax"
if str(_EYETRAX) not in sys.path:
    sys.path.insert(0, str(_EYETRAX))

from aac_eyetrax import ui, video  # noqa: E402

WINDOW = "Calibracion"
MARGIN = 0.10
"""Margen desde el borde, en fracción de pantalla. Los bordes son justamente
donde el error es peor, así que calibrar pegado al borde degrada el ajuste en
todas partes; pero calibrar demasiado adentro deja los bordes extrapolados."""

SETTLE_S = 0.65
FIRST_SETTLE_S = 1.4


def _grid(points: int) -> list[tuple[float, float]]:
    """Posiciones normalizadas, en el orden en que se van a mostrar."""
    lo, mid, hi = MARGIN, 0.5, 1.0 - MARGIN
    if points == 5:
        return [(lo, lo), (hi, lo), (mid, mid), (lo, hi), (hi, hi)]
    base = [(x, y) for y in (lo, mid, hi) for x in (lo, mid, hi)]
    if points == 9:
        return base
    extra = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
    return base + extra


def _draw_target(img, cx: int, cy: int, phase: float, collecting: bool) -> None:
    """Un punto que late mientras se asienta y se queda quieto al medir.

    El latido no es decorativo: dice «todavía no» sin texto, y cuando para, la
    persona ya está mirando fijo, que es exactamente lo que hay que capturar.
    """
    radius = 26 if collecting else int(20 + 10 * abs(np.sin(phase * 3.0)))
    color = ui.GOOD if collecting else ui.ACCENT
    cv2.circle(img, (cx, cy), radius + 10, (30, 30, 34), -1)
    cv2.circle(img, (cx, cy), radius, color, -1)
    cv2.circle(img, (cx, cy), 5, (12, 12, 14), -1)


def run(tracker, points: int = 9) -> bool:
    """Recolecta y ajusta. Devuelve True si la SVR quedó entrenada."""
    settings = tracker.settings
    w, h = tracker.screen_w, tracker.screen_h
    targets = _grid(points)

    print(f"[calibración] {points} puntos, "
          f"{settings.samples_per_point} muestras por punto")
    print("[calibración] mire el punto hasta que deje de latir. ESC cancela.")

    feats: list[np.ndarray] = []
    labels: list[tuple[float, float]] = []
    flojos: list[int] = []

    tracker.open_window(WINDOW)
    with video.camera(settings.camera_index, *settings.capture_size) as cap:
        frames = video.iter_frames(cap)
        for i, (nx, ny) in enumerate(targets):
            cx, cy = int(nx * w), int(ny * h)
            got = 0
            t0 = time.time()
            settle = FIRST_SETTLE_S if i == 0 else SETTLE_S
            while got < settings.samples_per_point:
                frame = next(frames)
                elapsed = time.time() - t0
                collecting = elapsed >= settle

                img = ui.canvas(w, h)
                _draw_target(img, cx, cy, elapsed, collecting)
                ui.text_center(img, f"{i + 1} / {len(targets)}", w // 2,
                               h - 60, 26, ui.TEXT_DIM)
                cv2.imshow(WINDOW, img)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    cv2.destroyWindow(WINDOW)
                    print("[calibración] cancelada.")
                    return False

                if not collecting:
                    continue
                f = tracker.features(frame)
                if f is None:
                    if elapsed > settle + 6.0:
                        break          # la cara no aparece: no colgarse acá
                    continue
                feats.append(f)
                labels.append(_target_cm(tracker, cx, cy))
                got += 1
            if got < settings.samples_per_point:
                flojos.append(i + 1)

    cv2.destroyWindow(WINDOW)

    if len(feats) < points * 3:
        print(f"[calibración] solo {len(feats)} muestras útiles: muy pocas. "
              f"¿Había luz suficiente y la cara entraba entera?")
        return False
    if flojos:
        print(f"[calibración] puntos con menos muestras de las pedidas: "
              f"{flojos}. Quedan con menos peso; si son los bordes, el error "
              f"ahí va a ser peor.")

    X = np.asarray(feats, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    ok, err_cm, _ = tracker.calibration.calibrate(X, y)
    if not ok:
        print("[calibración] la SVR no pudo entrenarse.")
        return False

    # El error de ajuste es sobre los mismos datos de entrenamiento, así que es
    # optimista por construcción. Se informa igual, pero nunca como "precisión":
    # eso lo mide `accuracy`, sobre puntos que el modelo no vio.
    px = err_cm * tracker.screen_w / tracker.screen_cm[0]
    print(f"[calibración] {len(feats)} muestras, {X.shape[1]} rasgos")
    print(f"[calibración] error de AJUSTE {err_cm:.2f} cm (~{px:.0f} px). "
          f"Es sobre los datos de entrenamiento: optimista. El número real "
          f"sale de `accuracy`.")
    return True


def _target_cm(tracker, px_x: int, px_y: int) -> tuple[float, float]:
    """Píxel de pantalla -> cm respecto de la cámara. Inverso de `cm_to_px`."""
    w_cm, h_cm = tracker.screen_cm
    cam_x, cam_y = tracker.camera_cm
    cm_x = px_x * w_cm / tracker.screen_w - cam_x
    cm_y = -(px_y * h_cm / tracker.screen_h - cam_y)
    return cm_x, cm_y
