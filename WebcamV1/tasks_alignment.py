"""El alineador facial de GazeFollower, portado a la API Tasks de MediaPipe.

Por qué existe. GazeFollower usa `mp.solutions.face_mesh`, la API legacy, y
**ninguna versión de MediaPipe con rueda para Python 3.12 la tiene**: se probó
0.10.35, 0.10.33, 0.10.21, 0.10.20, 0.10.18 y 0.10.14, y en todas el paquete
expone solo `Image`, `ImageFormat` y `tasks`. La versión que su requirements.txt
fija, 0.10.10, no tiene rueda para 3.12 en absoluto.

Eso deja dos caminos: quedarse en Python 3.11 con una MediaPipe de 2024, o
reimplementar la pieza. `face_alignment` es uno de los cinco componentes
intercambiables de GazeFollower, así que reimplementarla es la opción barata, y
además paga dos veces:

* Vuelve el proyecto a Python 3.12 y a la MediaPipe actual, la misma que ya usa
  el módulo de la RealSense - un solo alineador para los dos caminos de cámara.
* Si la licencia CC BY-NC-SA de GazeFollower obliga a cambiar el estimador por
  uno permisivo, esta pieza sobrevive.

La geometría de abajo es una réplica deliberada de la suya, no una mejora. El
estimador MGazeNet fue entrenado con recortes producidos por *esa* geometría, así
que cualquier "mejora" en el encuadre del ojo le entrega al modelo algo distinto
de lo que vio en entrenamiento. Las constantes raras -padding de 20 unidades,
alto igual a 0.75 del ancho, repartido 0.6 arriba y 0.4 abajo- se copian tal
cual por esa razón, y no porque tengan sentido por sí mismas. Lo mismo vale para
el orden de canales: ver `_landmarks`.

**Hasta dónde llega la equivalencia, medido.** Contra el original corriendo en
Python 3.11 con mediapipe 0.10.10, sobre la misma imagen:

    caja facial      2 px de diferencia
    cajas oculares   2-5 px
    mirada cruda     1.77 cm

Los 2 px son ruido de redondeo. Los 1.77 cm **no**: sobran, y no se pueden
parchear, porque el `FaceMesh` legacy y el `FaceLandmarker` de Tasks son modelos
distintos. Con los canales convertidos "bien" la diferencia era 2.87 cm, así que
replicar su tubería ayuda pero no cierra la brecha.

Dos advertencias que salen de eso:

* La medición es sobre una **cara sintética de elipses**, muy lejos de la
  distribución de entrenamiento de ambos modelos. El desacuerdo ahí no predice
  el desacuerdo sobre una cara real, en ninguna dirección.
* Lo que **sí** se puede afirmar: hay que calibrar con la misma tubería con la
  que se va a inferir. La calibración SVR aprende un mapeo desde los rasgos que
  produce el alineador, así que absorbe una diferencia sistemática de puntos -
  pero solo si es la misma en calibración y en uso. Calibrar con el original y
  correr con este port, o al revés, es el error a no cometer.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

# Los contornos de párpado que GazeFollower usa para el área de apertura ocular.
# Copiados textualmente: el área en píxeles cuadrados de este polígono es lo que
# su filtro y su registro esperan como "openness", así que cambiar el contorno
# cambia la escala del número.
LIP_VERTICES = [61, 91, 14, 178, 402, 324, 95]
LEFT_VERTICES = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153,
                 145, 144, 163, 7, 33]
RIGHT_VERTICES = [362, 388, 384, 385, 386, 387, 388, 466, 263, 249, 380, 373,
                  374, 380, 381, 382, 362]

# Constantes del encuadre ocular, replicadas. Ver la nota del docstring.
X_PADDING_UNITS = 20.0
EYE_BOX_ASPECT = 0.75
Y_ABOVE = 0.6
Y_BELOW = 0.4


def _polygon_area(vertices: np.ndarray) -> float:
    """Área por la fórmula del cordón. Igual que la suya, incluido el orden."""
    x, y = vertices[:, 0], vertices[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


class TasksFaceAlignment:
    """Reemplazo de `MediaPipeFaceAlignment` sobre `mediapipe.tasks`.

    Se pasa al constructor: `GazeFollower(face_alignment=TasksFaceAlignment())`.

    Modo VIDEO y no IMAGE, igual que en el módulo de la RealSense y por la misma
    razón: arrastra estado de seguimiento entre cuadros, lo que baja el temblor
    de los puntos. Y el temblor en el borde del párpado y en la comisura es
    temblor en el recorte que recibe la CNN, que es temblor en la mirada.
    """

    def __init__(self, model_path: str | Path | None = None,
                 min_confidence: float = 0.1):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        import mediapipe as mp

        self._mp = mp
        path = Path(model_path) if model_path else _default_model()
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            # Umbrales bajos, como los suyos (0.1). Un umbral alto pierde la
            # cara justo cuando la persona se mueve, que es cuando más importa
            # no perder el hilo.
            min_face_detection_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts = 0

    # ------------------------------------------------------------------ API

    def detect(self, timestamp, image):
        """Misma firma y mismo tipo de retorno que `MediaPipeFaceAlignment`."""
        from gazefollower.misc import FaceInfo

        info = FaceInfo()
        info.timestamp = timestamp
        h, w = image.shape[:2]
        info.img_w, info.img_h = w, h

        pts = self._landmarks(image, timestamp)
        if pts is None:
            info.status = False
            info.can_gaze_estimation = False
            return info
        info.status = True

        # Píxeles y enteros, como los suyos: z se escala por el ANCHO de la
        # imagen, no por el alto. Es asimétrico y es a propósito - MediaPipe
        # entrega z en las mismas unidades que x.
        mesh = np.column_stack([
            np.round(pts[:, 0] * w),
            np.round(pts[:, 1] * h),
            np.round(pts[:, 2] * w),
        ]).astype(np.int16)

        # --- descarta caras cortadas por el borde inferior -------------------
        if float(np.mean(mesh[:, 1][LIP_VERTICES])) >= h:
            info.can_gaze_estimation = False
            return info

        info.face_rect = _face_rect(mesh, w, h)

        eyes = _eye_rects(mesh, w, h)
        if eyes is None:
            # Un ojo toca el borde: el recorte quedaría incompleto y la CNN
            # recibiría medio ojo. Mejor declararlo inestimable.
            info.can_gaze_estimation = False
            return info
        info.left_rect, info.right_rect = eyes

        info.left_eye_openness = _polygon_area(mesh[:, :2][LEFT_VERTICES])
        info.right_eye_openness = _polygon_area(mesh[:, :2][RIGHT_VERTICES])
        info.face_landmarks = mesh
        info.can_gaze_estimation = True
        return info

    def release(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass

    # ------------------------------------------------------------ internals

    def landmarks_normalised(self, image, timestamp) -> np.ndarray | None:
        """Los 478 puntos en coordenadas normalizadas, o None.

        Lo mismo que `_landmarks` pero público, porque la guía de colocación lo
        necesita: `detect()` devuelve cajas y áreas, y tira los puntos crudos,
        que es justo lo que hace falta para medir dónde está la cabeza.

        Reusa la misma detección y el mismo orden de canales que la estimación
        de mirada. Si midiera la pose con puntos de otra tubería, la guía
        llevaría a la persona a una posición distinta de aquella en la que el
        estimador fue calibrado - y esa guía es precisamente lo que existe para
        evitar eso.
        """
        return self._landmarks(image, timestamp)

    def _landmarks(self, image, timestamp) -> np.ndarray | None:
        """Los puntos, alimentando **BGR sin convertir**, igual que el original.

        Esto parece un bug y es deliberado. `MediaPipeFaceAlignment` hace
        `face_mesh.process(image)` con la imagen en BGR -su propio docstring dice
        "expected in BGR format"- mientras la API legacy de MediaPipe espera RGB.
        O sea que el original le entrega los canales rojo y azul invertidos al
        detector.

        Convertir a RGB "bien" produce puntos distintos, y medido sobre la misma
        imagen la mirada resultante se corre **2.87 cm**; alimentando BGR como
        ellos, la diferencia baja a 1.77 cm y las cajas pasan de diferir 17 px a
        2 px. MGazeNet se construyó sobre esa tubería, así que lo que importa no
        es la corrección de los canales sino la consistencia con el estimador que
        viene después.
        """
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                  data=np.ascontiguousarray(image))
        # VIDEO exige marcas de tiempo estrictamente crecientes, y el llamador
        # puede pasar la misma dos veces si va muy rápido.
        ts = int(timestamp)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts

        result = self._landmarker.detect_for_video(mp_image, ts)
        if not result.face_landmarks:
            return None
        lm = result.face_landmarks[0]
        if len(lm) < 478:
            raise RuntimeError(
                f"MediaPipe devolvió {len(lm)} puntos y MGazeNet espera 478. "
                f"Falta el refinamiento del iris."
            )
        return np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float64)


def _face_rect(mesh: np.ndarray, w: int, h: int) -> list[int]:
    """Caja de la cara, cuadrada con el mismo `delta` que ellos usan.

    El recuadro sale del bounding box de los 478 puntos y después se estrecha
    horizontalmente y se estira verticalmente por `(ancho - alto) / 4`. El
    efecto es acercarlo a un cuadrado, que es la forma que la CNN espera.
    """
    min_x, max_x = int(max(0, mesh[:, 0].min())), int(min(w, mesh[:, 0].max()))
    min_y, max_y = int(max(0, mesh[:, 1].min())), int(min(h, mesh[:, 1].max()))

    face_w = math.fabs(min_x - max_x)
    face_h = math.fabs(min_y - max_y)
    delta = (face_w - face_h) / 4.0

    x0, y0 = min_x + delta, min_y - delta
    x1, y1 = max_x - delta, max_y + delta
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(float(w - 1), x1), min(float(h - 1), y1)
    return [int(x0), int(y0), int(x1) - int(x0), int(y1) - int(y0)]


def _eye_rects(mesh: np.ndarray, w: int, h: int):
    """Las dos cajas oculares, o None si alguna toca el borde.

    Nota sobre los nombres, que en el original están espejados: `left_rect` se
    construye con los puntos 33 y 133, que son la comisura externa e interna del
    ojo **derecho del sujeto** - el que aparece a la izquierda de la imagen. Se
    replica el nombre tal cual, porque es el que el estimador espera; invertirlo
    para que "left" signifique el ojo izquierdo espejaría la mirada horizontal.
    """
    x = mesh[:, 0].astype(np.float64)
    y = mesh[:, 1].astype(np.float64)

    # La distancia entre comisuras internas dividida en 100 unidades. Todo el
    # padding se expresa en esas unidades, así que el encuadre escala con el
    # tamaño de la cara en la imagen - y por eso funciona a distintas distancias.
    scale = math.fabs(x[362] - x[133]) / 100.0
    pad = X_PADDING_UNITS * scale

    boxes = []
    for outer, inner in ((33, 133), (362, 263)):
        xs = [x[outer] - pad, x[inner] + pad]
        box_h = math.fabs(xs[0] - xs[1]) * EYE_BOX_ASPECT
        cy = (y[outer] + y[inner]) / 2.0
        ys = [cy - box_h * Y_ABOVE, cy + box_h * Y_BELOW]

        x0, y0, x1, y1 = int(xs[0]), int(ys[0]), int(xs[1]), int(ys[1])
        if x0 <= 0 or y0 <= 0 or x1 >= w or y1 >= h:
            return None
        boxes.append([x0, y0, x1 - x0, y1 - y0])
    return boxes[0], boxes[1]


def _default_model() -> Path:
    """Reusa el `face_landmarker.task` que el módulo de la RealSense ya bajó."""
    cache = Path.home() / ".cache" / "aac_realsense" / "face_landmarker.task"
    if cache.exists():
        return cache
    raise FileNotFoundError(
        f"no está el modelo de MediaPipe en {cache}. Lo baja "
        f"`RealsenseV1/main.py smoke`, o pasá la ruta con model_path."
    )
