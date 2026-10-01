"""Calibración en varias posturas. Es la mitad del arreglo del movimiento.

Agregar las columnas de cabeza al vector de features no sirve de nada si la
calibración ocurre en una sola postura: esas columnas quedan constantes, el
regresor no puede aprender nada de ellas, y la regularización las manda a cero.
El modelo vuelve a ser el de eyetrax con seis ceros al final.

Así que acá la persona **se mueve a propósito**, en tres bloques:

1. Nueve puntos en su postura natural. Es la referencia.
2. Cinco puntos corrido unos centímetros a un lado.
3. Cinco puntos un poco más cerca o más lejos.

Diecinueve puntos en vez de nueve, y alrededor de un minuto y medio. Es más
caro que la calibración vieja y vale la pena: lo que compra es que después uno
pueda acomodarse en la silla sin que el cursor se vaya a otro lado.

El orden importa. El bloque natural va primero porque es la postura que la
guía va a pedir después, y porque si alguien abandona a mitad, lo que quedó
guardado es lo más parecido a cómo se va a sentar de verdad.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from . import ui, video

WINDOW = "Calibracion (postura variable)"
MARGIN = 0.10
SETTLE_S = 0.6
FIRST_SETTLE_S = 1.3
SAMPLES_PER_POINT = 14


def _grid(n: int) -> list[tuple[float, float]]:
    lo, mid, hi = MARGIN, 0.5, 1.0 - MARGIN
    if n == 5:
        return [(lo, lo), (hi, lo), (mid, mid), (lo, hi), (hi, hi)]
    return [(x, y) for y in (lo, mid, hi) for x in (lo, mid, hi)]


BLOQUES = [
    (9, "SU POSTURA NATURAL",
     "Sientese como se sentaria para usar el tablero"),
    (5, "CORRASE UNOS CENTIMETROS",
     "A un lado, lo que haria sin pensarlo al acomodarse"),
    (5, "ACERQUESE O ALEJESE UN POCO",
     "Un cambio chico de distancia, no exagerado"),
]


def _aviso(tracker, titulo: str, bajada: str, w: int, h: int) -> bool:
    """Pantalla entre bloques. False si cancela."""
    while True:
        img = ui.canvas(w, h)
        ui.text_center(img, titulo, w // 2, int(h * 0.38), 60, ui.TEXT, bold=True)
        ui.text_center(img, bajada, w // 2, int(h * 0.48), 30, ui.TEXT_DIM)
        ui.text_center(img, "ENTER para seguir     ESC para cancelar",
                       w // 2, int(h * 0.62), 26, ui.ACCENT)
        cv2.imshow(WINDOW, img)
        k = cv2.waitKey(30) & 0xFF
        if k in (13, 32):
            return True
        if k == 27:
            return False


def run(tracker, method: str = "9point") -> bool:
    """Recolecta en los tres bloques y ajusta. True si quedó entrenado."""
    s = tracker.settings
    w, h = tracker.screen_w, tracker.screen_h

    bloques = BLOQUES if method != "5point" else [(5, t, b) for _, t, b in BLOQUES]
    total = sum(n for n, _, _ in bloques)
    print(f"[cabeza] {total} puntos en {len(bloques)} posturas, "
          f"{SAMPLES_PER_POINT} muestras por punto")

    X: list[np.ndarray] = []
    y: list[tuple[int, int]] = []
    poses: list[tuple] = []

    tracker.open_window(WINDOW)
    try:
        with video.camera(s.camera_index, *s.capture_size) as cap:
            frames = video.iter_frames(cap)
            for bi, (n, titulo, bajada) in enumerate(bloques):
                if not _aviso(tracker, titulo, bajada, w, h):
                    print("[cabeza] cancelada.")
                    return False
                for i, (nx, ny) in enumerate(_grid(n)):
                    cx, cy = int(nx * w), int(ny * h)
                    got, t0 = 0, time.time()
                    settle = FIRST_SETTLE_S if i == 0 else SETTLE_S
                    while got < SAMPLES_PER_POINT:
                        frame = next(frames)
                        elapsed = time.time() - t0
                        midiendo = elapsed >= settle

                        img = ui.canvas(w, h)
                        radio = 26 if midiendo else int(20 + 10 * abs(
                            np.sin(elapsed * 3.0)))
                        color = ui.GOOD if midiendo else ui.ACCENT
                        cv2.circle(img, (cx, cy), radio + 10, (30, 30, 34), -1)
                        cv2.circle(img, (cx, cy), radio, color, -1)
                        cv2.circle(img, (cx, cy), 5, (12, 12, 14), -1)
                        ui.text_center(img, f"{titulo}  -  {i + 1}/{n}",
                                       w // 2, h - 56, 24, ui.TEXT_DIM)
                        cv2.imshow(WINDOW, img)
                        if (cv2.waitKey(1) & 0xFF) == 27:
                            print("[cabeza] cancelada.")
                            return False
                        if not midiendo:
                            continue

                        f, blink = tracker.extract(frame)
                        if f is None or blink:
                            if elapsed > settle + 6.0:
                                break
                            continue
                        X.append(f)
                        y.append((cx, cy))
                        poses.append(tuple(f[-6:]))
                        got += 1
    finally:
        cv2.destroyWindow(WINDOW)

    if len(X) < total * 4:
        print(f"[cabeza] solo {len(X)} muestras útiles: muy pocas.")
        return False

    Xa, ya = np.asarray(X, np.float32), np.asarray(y, np.float32)
    tracker.model.add(Xa, ya)
    err = tracker.model.fit()
    tracker.trained = True

    # La referencia sale de las muestras del PRIMER bloque, que es la postura
    # natural. Usar la mediana de todo daría un punto intermedio entre las tres
    # posturas, donde la persona no se sienta nunca.
    _fijar_referencia(tracker, poses[:9 * SAMPLES_PER_POINT])

    dx, dy, dd = tracker.model.pose_spread()
    print(f"[cabeza] {len(Xa)} muestras nuevas, {len(tracker.model.X)} en total")
    print(f"[cabeza] variación de postura cubierta: cx={dx:.3f} cy={dy:.3f} "
          f"d={dd:.3f}")
    if max(dx, dy) < 0.02:
        print("[cabeza] AVISO: casi no te moviste entre bloques, así que el "
              "modelo no va a tolerar que te muevas después. El arreglo "
              "depende de esa variación.")
    print(f"[cabeza] error de AJUSTE {err:.0f} px — es sobre los datos de "
          f"entrenamiento, optimista. El real sale de `accuracy`.")
    return True


def _fijar_referencia(tracker, poses) -> None:
    """Pose de referencia desde las features de cabeza del bloque natural."""
    from .tracker import FacePose

    if not poses:
        return
    m = np.median(np.asarray(poses, dtype=np.float64), axis=0)
    cx, cy, d = float(m[0]), float(m[1]), float(m[2])
    if not (0.05 < cx < 0.95 and d > 0.01):
        return
    tracker.reference_pose = FacePose(cx=cx, cy=cy, eye_span=d,
                                      roll_deg=0.0, yaw_ratio=0.0,
                                      pitch_ratio=0.0)
