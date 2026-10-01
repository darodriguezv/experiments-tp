"""Estimador que SÍ ve dónde está la cabeza.

## El problema, que es de diseño y no de ajuste

`eyetrax.GazeEstimator.extract_features` normaliza la cabeza hasta hacerla
desaparecer. Mirado en su código: centra todos los puntos en el punto medio de
los ojos, los rota a un marco local construido con la línea de los ojos y la
coronilla, y los divide por la distancia interocular. De los 486 valores que
produce, los únicos que conservan algo de la postura son los tres ángulos
yaw/pitch/roll que agrega al final.

O sea que **la traslación y la distancia de la cabeza se eliminan a propósito**.

Para un clasificador de expresiones eso es correcto: quita una variable que
molesta. Para estimar dónde mira alguien en una pantalla es fatal, porque el
punto de pantalla depende de dónde está el ojo en el espacio. La misma
geometría ocular a veinte centímetros a la izquierda apunta a otro lado, y el
modelo no tiene con qué distinguir los dos casos: le llega exactamente el mismo
vector.

Por eso calibrar en una posición y correrse unos centímetros degrada tanto. No
es ruido ni falta de suavizado: es que la información necesaria fue descartada
antes de llegar al regresor.

## Lo que hace este módulo

Devuelve las mismas features que eyetrax **más seis términos de cabeza**:

    cx, cy        dónde está el punto medio de los ojos en la imagen
    d             distancia interocular (proxy de qué tan cerca está)
    1/d, cx/d, cy/d   los mismos, escalados por distancia

Los tres primeros son literalmente lo que eyetrax dividió fuera. Los tres
últimos están porque el desplazamiento en pantalla que produce mover la cabeza
no es independiente de la distancia: moverse cinco centímetros pesa mucho más
sentado cerca que lejos. Un modelo lineal sin esos cocientes solo puede aplicar
una corrección aditiva, que es la aproximación equivocada.

**Nada de esto sirve si la calibración ocurre en una sola postura.** Si durante
la recolección `cx`, `cy` y `d` no varían, son constantes y el regresor no
aprende nada de ellas - y encima la regularización las manda a cero. Por eso
`calibrate_ui` pide moverse entre bloques de puntos: la variación en los datos
es la mitad del arreglo, y la otra mitad es que exista la columna.

## Por qué se replica la normalización en vez de llamarla

`extract_features` no devuelve el centro ni la distancia, y tampoco guarda los
puntos crudos, así que obtenerlos exigiría una segunda pasada de MediaPipe: 33
ms más por cuadro, de 26 fps a 15. La matemática de abajo es una **réplica
deliberada** de la suya -mismo anclaje, mismos ejes, misma escala- para que las
483 primeras columnas sean idénticas y la comparación contra EyeTrax siga
siendo honesta. Si cambiaran su normalización, esto queda desactualizado en
silencio; el chequeo `las features base coinciden con eyetrax` existe para que
no pase.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from eyetrax.constants import (LEFT_EYE_INDICES, MUTUAL_INDICES,
                               RIGHT_EYE_INDICES)

HEAD_TERMS = 6
"""cx, cy, d, 1/d, cx/d, cy/d. Ver el docstring del módulo."""

SUBSET = LEFT_EYE_INDICES + RIGHT_EYE_INDICES + MUTUAL_INDICES
BASE_DIM = len(SUBSET) * 3 + 3
FULL_DIM = BASE_DIM + HEAD_TERMS


def features_from_landmarks(points: np.ndarray) -> np.ndarray:
    """486 features de eyetrax + 6 de cabeza, desde los puntos crudos.

    Las primeras 486 replican `eyetrax.GazeEstimator.extract_features` exacto.
    """
    left_corner = points[33]
    right_corner = points[263]
    top_of_head = points[10]

    eye_center = (left_corner + right_corner) / 2.0
    shifted = points - eye_center

    x_axis = right_corner - left_corner
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-9)
    y_approx = top_of_head - eye_center
    y_approx = y_approx / (np.linalg.norm(y_approx) + 1e-9)
    y_axis = y_approx - np.dot(y_approx, x_axis) * x_axis
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-9)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-9)
    R = np.column_stack((x_axis, y_axis, z_axis))

    rotated = (R.T @ shifted.T).T
    left_rot = R.T @ (left_corner - eye_center)
    right_rot = R.T @ (right_corner - eye_center)
    inter_eye = float(np.linalg.norm(right_rot - left_rot))
    if inter_eye > 1e-7:
        rotated = rotated / inter_eye

    base = rotated[SUBSET].flatten()
    yaw = np.arctan2(R[1, 0], R[0, 0])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    roll = np.arctan2(R[2, 1], R[2, 2])
    base = np.concatenate([base, [yaw, pitch, roll]])

    # --- lo que eyetrax tira, devuelto como features explícitas -------------
    cx, cy = float(eye_center[0]), float(eye_center[1])
    d = max(inter_eye, 1e-6)
    head = np.array([cx, cy, d, 1.0 / d, cx / d, cy / d], dtype=np.float64)
    return np.concatenate([base, head]).astype(np.float32)


class HeadAwareExtractor:
    """Una pasada de MediaPipe por cuadro, features completas y parpadeo."""

    def __init__(self, estimator):
        # Se reusa el landmarker que el estimador de eyetrax ya construyó: son
        # ~1 s de carga y 33 ms por cuadro, y tener dos corriendo a la vez
        # duplicaría ambos costos para obtener exactamente los mismos puntos.
        self._landmarker = estimator._face_landmarker
        self._mp = estimator._mp
        self._last_ts = 0
        self._ear = []

    def __call__(self, frame):
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts = int(time.time() * 1000)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts

        result = self._landmarker.detect_for_video(image, ts)
        if not result.face_landmarks:
            return None, False
        lm = result.face_landmarks[0]
        points = np.array([(p.x, p.y, p.z) for p in lm], dtype=np.float32)

        blink = self._blink(points)
        return features_from_landmarks(points), blink

    def _blink(self, p) -> bool:
        """Mismo criterio que eyetrax: EAR contra su propia media móvil."""
        def ear(outer, inner, top, bottom):
            w = np.linalg.norm(p[outer][:2] - p[inner][:2])
            h = np.linalg.norm(p[top][:2] - p[bottom][:2])
            return h / (w + 1e-9)

        value = (ear(33, 133, 159, 145) + ear(263, 362, 386, 374)) / 2.0
        self._ear.append(value)
        if len(self._ear) > 50:
            self._ear.pop(0)
        umbral = float(np.mean(self._ear)) * 0.8 if len(self._ear) >= 15 else 0.2
        return value < umbral


class HeadAwareModel:
    """Ridge sobre las features extendidas, con las muestras guardadas.

    Se usa un ridge propio y no el de eyetrax porque su `predict` espera 486
    columnas y acá son 492. Guardar las **muestras** y no solo los coeficientes
    es lo que permite que cada calibración se sume a las anteriores: dos
    ajustes no se pueden mezclar, dos conjuntos de muestras sí - y acumular
    posturas es precisamente de lo que depende este arreglo.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = None
        self.X = np.zeros((0, FULL_DIM), dtype=np.float32)
        self.y = np.zeros((0, 2), dtype=np.float32)
        self.session = np.zeros((0,), dtype=np.int32)

    @property
    def trained(self) -> bool:
        return self.model is not None

    def add(self, X, y) -> None:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        nueva = int(self.session.max()) + 1 if len(self.session) else 0
        self.X = np.vstack([self.X, X])
        self.y = np.vstack([self.y, y])
        self.session = np.concatenate(
            [self.session, np.full(len(y), nueva, dtype=np.int32)])

    def fit(self) -> float:
        """Ajusta y devuelve el error de entrenamiento en píxeles."""
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        # Estandarizar NO es cosmético acá: las features de eyetrax están
        # normalizadas a escala ~1 y los términos de cabeza valen entre 0.05 y
        # 30. Sin escalar, la penalización del ridge los aplastaría primero -
        # justo las columnas que este módulo existe para agregar.
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        self.model.fit(self.X, self.y)
        pred = self.model.predict(self.X)
        return float(np.mean(np.hypot(*(pred - self.y).T)))

    def predict(self, x) -> tuple[float, float]:
        p = self.model.predict(np.asarray(x, dtype=np.float32).reshape(1, -1))[0]
        return float(p[0]), float(p[1])

    # ------------------------------------------------------------ persistencia

    def save(self, path: Path, meta: dict, max_sessions: int = 6) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        vivas = np.unique(self.session)[-max_sessions:]
        keep = np.isin(self.session, vivas)
        np.savez_compressed(path, X=self.X[keep], y=self.y[keep],
                            session=self.session[keep])
        path.with_suffix(".meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, alpha: float = 1.0) -> "HeadAwareModel":
        m = cls(alpha=alpha)
        d = np.load(Path(path), allow_pickle=False)
        if d["X"].shape[1] != FULL_DIM:
            raise ValueError(
                f"las muestras tienen {d['X'].shape[1]} columnas y el extractor "
                f"produce {FULL_DIM}: recalibrá desde cero")
        m.X, m.y, m.session = d["X"], d["y"], d["session"]
        m.fit()
        return m

    def pose_spread(self) -> tuple[float, float, float]:
        """Cuánto varían cx, cy y d en lo acumulado.

        Es el diagnóstico que dice si el arreglo puede funcionar: con una sola
        postura estas tres son casi cero, las columnas de cabeza no aportan
        nada, y el modelo vuelve a ser el de eyetrax con seis ceros al final.
        """
        if len(self.X) < 2:
            return 0.0, 0.0, 0.0
        cx, cy, d = self.X[:, BASE_DIM], self.X[:, BASE_DIM + 1], self.X[:, BASE_DIM + 2]
        return (float(cx.max() - cx.min()), float(cy.max() - cy.min()),
                float(d.max() - d.min()))


# ---------------------------------------------------------------- el tracker

MODELS_SUBDIR = "headaware"


class HeadAwareTracker:
    """La interfaz de `GazeTracker`, con el modelo consciente de la cabeza.

    Hereda por composición y no por subclase: se construye un `GazeTracker`
    real por dentro y se le piden la pantalla, el suavizado, la guía de
    colocación y el landmarker. Lo único que cambia es qué features se extraen
    y qué modelo las consume - que es exactamente el alcance del arreglo.
    """

    def __init__(self, settings=None):
        from .tracker import GazeTracker

        self.inner = GazeTracker(settings)
        self.settings = self.inner.settings
        self.screen = self.inner.screen
        self.screen_w, self.screen_h = self.inner.screen_w, self.inner.screen_h
        self.smoother = self.inner.smoother
        self.reference_pose = None
        self.extract = HeadAwareExtractor(self.inner.estimator)
        self.model = HeadAwareModel()
        self.trained = False
        self.last_pose = None

    name = "eyetrax+cabeza/ridge"

    # --- lo que se delega sin cambios -------------------------------------
    def open_window(self, window):
        return self.inner.open_window(window)

    def face_pose(self, frame):
        return self.inner.face_pose(frame)

    def reset_smoother(self):
        self.inner.reset_smoother()
        self.smoother = self.inner.smoother

    def close(self):
        self.inner.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # --- lo que cambia ------------------------------------------------------
    def process(self, frame):
        from .tracker import GazeSample

        features, blink = self.extract(frame)
        if features is None:
            return GazeSample(ok=False, face=False, blink=False)
        if blink:
            return GazeSample(ok=False, face=True, blink=True)
        if not self.trained:
            return GazeSample(ok=False, face=True, blink=False)

        raw_x, raw_y = self.model.predict(features)
        sx, sy = self.smoother.step(int(raw_x), int(raw_y))
        return GazeSample(
            ok=True,
            x=int(np.clip(sx, 0, self.screen_w - 1)),
            y=int(np.clip(sy, 0, self.screen_h - 1)),
            raw_x=int(raw_x), raw_y=int(raw_y), blink=False, face=True)

    def calibrate(self, method: str = "9point") -> None:
        from . import headaware_ui

        self.trained = headaware_ui.run(self, method)

    def _path(self) -> Path:
        base = Path(self.settings.model_path).parent / MODELS_SUBDIR
        return base / f"{self.screen.width}x{self.screen.height}.npz"

    def save(self, path=None) -> Path:
        p = Path(path) if path else self._path()
        meta = {
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "estimator": self.name,
            "monitor_size": [self.screen_w, self.screen_h],
            "monitor_name": self.screen.name,
            "camera_index": self.settings.camera_index,
            "muestras": int(len(self.model.X)),
            "sesiones": int(len(np.unique(self.model.session))),
        }
        if self.reference_pose is not None:
            meta["reference_pose"] = self.reference_pose.as_dict()
        self.model.save(p, meta)
        return p

    def load(self, path=None) -> Path:
        from .tracker import FacePose

        p = Path(path) if path else self._path()
        if not p.exists():
            raise FileNotFoundError(
                f"No hay calibración consciente de cabeza en {p}. "
                f"Corré `run.cmd calibrate --head-aware` primero.")
        self.model = HeadAwareModel.load(p)
        self.trained = True
        meta_path = p.with_suffix(".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if "reference_pose" in meta:
                self.reference_pose = FacePose.from_dict(meta["reference_pose"])
            if meta.get("monitor_size") != [self.screen_w, self.screen_h]:
                print(f"[warn] calibrado en {meta.get('monitor_size')} y "
                      f"corriendo en [{self.screen_w}, {self.screen_h}]")
        dx, dy, dd = self.model.pose_spread()
        print(f"[cabeza] {len(self.model.X)} muestras de "
              f"{len(np.unique(self.model.session))} sesión(es); "
              f"variación de postura cx={dx:.3f} cy={dy:.3f} d={dd:.3f}")
        if max(dx, dy) < 0.02:
            print("[warn] la calibración tiene muy poca variación de postura: "
                  "el modelo no va a tolerar que te muevas. Recalibrá moviéndote "
                  "entre bloques.")
        return p


def make(settings):
    """Fábrica para `aac_eyetrax.*.run(make_tracker=...)`."""
    return HeadAwareTracker(settings)
