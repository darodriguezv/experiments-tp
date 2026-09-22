"""Thin wrapper over eyetrax.GazeEstimator with smoothing, clamping and dwell.

The AAC layer only ever talks to this module, so swapping the underlying
gaze library later means rewriting one file.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from eyetrax import GazeEstimator
from eyetrax.calibration import (
    run_5_point_calibration,
    run_9_point_calibration,
    run_lissajous_calibration,
)
from eyetrax.filters import KalmanEMASmoother, KalmanSmoother, KDESmoother, NoSmoother

from . import display, video
from .config import FEATURE_DIM, Settings

_CALIBRATORS = {
    "9point": run_9_point_calibration,
    "5point": run_5_point_calibration,
    "lissajous": run_lissajous_calibration,
}

# eyetrax's calibrators call get_screen_size() for their canvas size and draw
# into a window named "Calibration". Both are monitor-blind, so we override the
# size and pre-place the window before handing control over.
CALIBRATION_WINDOW = "Calibration"

_PATCH_TARGETS = (
    "eyetrax.calibration.nine_point",
    "eyetrax.calibration.five_point",
    "eyetrax.calibration.lissajous",
    "eyetrax.calibration.dense_grid",
    "eyetrax.calibration.adaptive",
)


def _force_screen_size(screen: display.Screen) -> None:
    """Point eyetrax's `get_screen_size` at the monitor we actually use.

    Each calibration module does `from ...screen import get_screen_size`, which
    binds the name at import time, so patching the origin module is not enough -
    every importer needs rebinding too.
    """
    import importlib

    import eyetrax.utils.screen as screen_mod

    def sized() -> tuple[int, int]:
        return screen.width, screen.height

    screen_mod.get_screen_size = sized
    for mod_name in _PATCH_TARGETS:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        if hasattr(mod, "get_screen_size"):
            mod.get_screen_size = sized


@dataclass
class FacePose:
    """Where the head is, in units that survive a change of camera resolution.

    Measured in image space on purpose. The obvious alternative - reusing the
    rotation matrix eyetrax builds for its features - does not work for a guide:
    that matrix puts the eye line on its x axis, so an in-plane tilt rotates the
    whole basis and vanishes from the extracted "roll", which comes out a
    constant 180 degrees for *any* level or tilted head. These quantities are
    each measured directly instead, so each one means what its name says.
    """

    cx: float
    cy: float
    """Midpoint between the outer eye corners, normalised to [0, 1] of the frame."""

    eye_span: float
    """Outer-corner-to-outer-corner eye distance as a fraction of frame width.
    A proxy for how close the person is sitting."""

    roll_deg: float
    """Tilt of the eye line in the image plane. 0 = level. Signed, real degrees."""

    yaw_ratio: float
    """How far the nose sits from the midpoint of the eye line, along that line,
    as a fraction of half the eye span. 0 = facing the camera, +/-1 = nose over
    an eye corner. Projected along the eye line, so head tilt does not leak in."""

    pitch_ratio: float
    """Nose offset perpendicular to the eye line, over the eye span. The baseline
    depends on the person's face and where the camera sits, so this is only ever
    compared against a recorded reference, never against an absolute zero."""

    def as_dict(self) -> dict:
        return {
            "cx": round(self.cx, 4), "cy": round(self.cy, 4),
            "eye_span": round(self.eye_span, 4),
            "roll_deg": round(self.roll_deg, 2),
            "yaw_ratio": round(self.yaw_ratio, 4),
            "pitch_ratio": round(self.pitch_ratio, 4),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FacePose":
        return cls(
            cx=float(d["cx"]), cy=float(d["cy"]), eye_span=float(d["eye_span"]),
            roll_deg=float(d.get("roll_deg", 0.0)),
            yaw_ratio=float(d.get("yaw_ratio", 0.0)),
            pitch_ratio=float(d.get("pitch_ratio", 0.0)),
        )


DEFAULT_POSE = FacePose(
    cx=0.5, cy=0.42, eye_span=0.17, roll_deg=0.0, yaw_ratio=0.0, pitch_ratio=0.0
)
"""Target used before any calibration exists: centred, eyes a little above the
middle of the frame, and close enough that the iris gets useful pixels."""


def pose_from_landmarks(pts, frame_w: int, frame_h: int) -> FacePose:
    """Build a FacePose from MediaPipe's normalised landmark array.

    Pure function of the landmarks so it can be tested against synthetic faces
    without a camera. Landmarks 33/263 are the outer eye corners and 4 is the
    nose tip; MediaPipe's y grows downward, which is handled here rather than
    left for callers to trip over.
    """
    left = np.asarray(pts[33][:2], dtype=np.float64)
    right = np.asarray(pts[263][:2], dtype=np.float64)
    nose = np.asarray(pts[4][:2], dtype=np.float64)

    # Work in pixels so the aspect ratio does not distort angles.
    scale = np.array([float(frame_w), float(frame_h)])
    left_px, right_px, nose_px = left * scale, right * scale, nose * scale

    eye_vec = right_px - left_px
    span_px = float(np.hypot(*eye_vec))
    centre = (left + right) / 2.0

    if span_px < 1e-6:
        return FacePose(float(centre[0]), float(centre[1]), 0.0, 0.0, 0.0, 0.0)

    # Tilt of the eye line, straight from the image. Level head -> 0 degrees.
    roll = float(np.degrees(np.arctan2(eye_vec[1], eye_vec[0])))

    # Project the nose onto the eye line (yaw) and its perpendicular (pitch), so
    # neither measure is polluted by the other or by tilt.
    u = eye_vec / span_px
    perp = np.array([-u[1], u[0]])
    centre_px = (left_px + right_px) / 2.0
    rel = nose_px - centre_px
    yaw_ratio = float(np.dot(rel, u) / (span_px / 2.0))
    pitch_ratio = float(np.dot(rel, perp) / span_px)

    return FacePose(
        cx=float(centre[0]),
        cy=float(centre[1]),
        eye_span=span_px / max(frame_w, 1),
        roll_deg=roll,
        yaw_ratio=yaw_ratio,
        pitch_ratio=pitch_ratio,
    )


@dataclass
class GazeSample:
    """One frame's worth of gaze state."""

    ok: bool
    """True when a face was found and the eyes were open."""

    x: int = 0
    y: int = 0
    """Smoothed, screen-clamped gaze point."""

    raw_x: int = 0
    raw_y: int = 0
    """Model output before smoothing - useful to visualise jitter."""

    blink: bool = False
    face: bool = False


class GazeTracker:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.settings.validate()

        self.screen = display.resolve(self.settings.monitor)
        self.screen_w, self.screen_h = self.screen.size
        _force_screen_size(self.screen)
        # Calibration must run at the same resolution as everything else, and
        # eyetrax opens its own capture, so redirect that too.
        video.patch_eyetrax_camera(*self.settings.capture_size)

        self.estimator = GazeEstimator(model_name=self.settings.model_name)
        self.smoother = self._make_smoother()
        self.trained = False
        self.reference_pose: FacePose | None = None
        """Head position recorded at calibration time, if known."""

    @property
    def name(self) -> str:
        """Qué estimador es, para que la pantalla no mienta.

        El recuadro de estado mostraba `modelo=ridge`, que sale de `Settings` y
        describe el regresor de EyeTrax. WebcamV1 hereda esa configuración pero
        no usa ese regresor, así que su pantalla decía lo mismo - y con dos
        módulos que comparten arnés, pantalla y tablero, eso alcanza para estar
        midiendo uno mientras uno cree estar midiendo el otro.
        """
        return f"eyetrax/{self.settings.model_name}"

    # ------------------------------------------------------------- head pose

    def face_pose(self, frame) -> FacePose | None:
        """Measure head position and orientation from one frame.

        Reaches into the estimator's MediaPipe landmarker because
        `extract_features` throws the raw landmarks away and we need absolute
        position, which the normalised feature vector no longer carries. Only
        the positioning screen uses this, and it never runs at the same time as
        gaze tracking, so there is no double-detection cost.
        """
        landmarker = getattr(self.estimator, "_face_landmarker", None)
        mp = getattr(self.estimator, "_mp", None)
        if landmarker is None or mp is None:
            return None

        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode demands strictly increasing timestamps, shared with
        # extract_features' own counter.
        ts = int(time.time() * 1000)
        last = getattr(self.estimator, "_mp_last_ts_ms", 0)
        if ts <= last:
            ts = last + 1
        self.estimator._mp_last_ts_ms = ts

        result = landmarker.detect_for_video(mp_image, ts)
        if not result.face_landmarks:
            return None
        lm = result.face_landmarks[0]

        pts = np.array([(p.x, p.y, p.z) for p in lm], dtype=np.float32)
        h, w = frame.shape[:2]
        return pose_from_landmarks(pts, w, h)

    def open_window(self, window: str) -> None:
        """Open a fullscreen window on the selected monitor."""
        display.open_fullscreen(window, self.screen)
        if not display.placement_ok(window, self.screen):
            print(
                f"[warn] window '{window}' may not have landed on monitor "
                f"{self.screen.index} - gaze coordinates assume "
                f"{self.screen_w}x{self.screen_h} at "
                f"({self.screen.x},{self.screen.y})."
            )

    # ------------------------------------------------------------------ setup

    def _make_smoother(self):
        name = self.settings.smoother
        if name == "none":
            return NoSmoother()
        if name == "kalman":
            return KalmanSmoother()
        if name == "kalman_ema":
            return KalmanEMASmoother(ema_alpha=self.settings.ema_alpha)
        if name == "kde":
            return KDESmoother(self.screen_w, self.screen_h)
        raise ValueError(f"unknown smoother {name!r}")

    def reset_smoother(self) -> None:
        """Drop smoothing history - call this between screens/tasks."""
        self.smoother = self._make_smoother()

    def calibrate(self, method: str = "9point") -> None:
        if method not in _CALIBRATORS:
            raise ValueError(f"method must be one of {tuple(_CALIBRATORS)}")

        # Place eyetrax's own window on the chosen monitor before it starts
        # drawing; its internal namedWindow call then becomes a no-op.
        self.open_window(CALIBRATION_WINDOW)
        _CALIBRATORS[method](self.estimator, camera_index=self.settings.camera_index)
        # eyetrax's calibration routines return silently when the user aborts
        # with ESC, so probe the model instead of trusting the call.
        self.trained = self._model_is_fitted()

    def _model_is_fitted(self) -> bool:
        try:
            self.estimator.predict(np.zeros((1, FEATURE_DIM), dtype=np.float32))
        except Exception:
            return False
        return True

    # -------------------------------------------------------- persistence

    @staticmethod
    def _meta_path(model_path: Path) -> Path:
        return model_path.with_suffix(".meta.json")

    def _meta(self) -> dict:
        meta = {
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": self.settings.model_name,
            "camera_index": self.settings.camera_index,
            "capture_size": list(self.settings.capture_size),
            "monitor_index": self.screen.index,
            "monitor_name": self.screen.name,
            "monitor_size": [self.screen_w, self.screen_h],
            "monitor_origin": [self.screen.x, self.screen.y],
        }
        if self.reference_pose is not None:
            meta["reference_pose"] = self.reference_pose.as_dict()
        return meta

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path or self.settings.model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.estimator.save_model(path)
        # A gaze model is only valid for the screen it was calibrated on, and
        # nothing in the pickle records that - hence the sidecar.
        self._meta_path(path).write_text(
            json.dumps(self._meta(), indent=2), encoding="utf-8"
        )
        return path

    def load(self, path: str | Path | None = None) -> Path:
        path = Path(path or self.settings.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"No calibration model at {path}. Run `python main.py calibrate` first."
            )
        self.estimator.load_model(path)
        self.trained = True
        self._warn_on_meta_mismatch(path)
        return path

    def _warn_on_meta_mismatch(self, path: Path) -> None:
        meta_path = self._meta_path(path)
        if not meta_path.exists():
            return
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return

        # Recover the head position this model was calibrated at, so the
        # positioning guide can steer the user back to it.
        if "reference_pose" in meta:
            try:
                self.reference_pose = FacePose.from_dict(meta["reference_pose"])
            except (KeyError, TypeError, ValueError):
                self.reference_pose = None

        if meta.get("monitor_size") != [self.screen_w, self.screen_h] or meta.get(
            "monitor_name"
        ) != self.screen.name:
            print(
                f"[warn] this model was calibrated on monitor "
                f"{meta.get('monitor_index')} ({meta.get('monitor_name')}, "
                f"{meta.get('monitor_size')}) but you are running on monitor "
                f"{self.screen.index} ({self.screen.name}, "
                f"[{self.screen_w}, {self.screen_h}]).\n"
                f"       Gaze will be systematically off - recalibrate on this screen."
            )
        if meta.get("camera_index") != self.settings.camera_index:
            print(
                f"[warn] model was calibrated with camera "
                f"{meta.get('camera_index')}, now using "
                f"{self.settings.camera_index}."
            )

    # ------------------------------------------------------------- per frame

    def process(self, frame) -> GazeSample:
        features, blink = self.estimator.extract_features(frame)
        if features is None:
            return GazeSample(ok=False, face=False, blink=False)
        if blink:
            return GazeSample(ok=False, face=True, blink=True)

        raw_x, raw_y = self.estimator.predict(np.array([features]))[0]
        raw_x, raw_y = int(raw_x), int(raw_y)
        sx, sy = self.smoother.step(raw_x, raw_y)

        return GazeSample(
            ok=True,
            x=int(np.clip(sx, 0, self.screen_w - 1)),
            y=int(np.clip(sy, 0, self.screen_h - 1)),
            raw_x=raw_x,
            raw_y=raw_y,
            blink=False,
            face=True,
        )

    def close(self) -> None:
        self.estimator.close()

    def __enter__(self) -> "GazeTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class MedianWindow:
    """Mediana móvil de puntos, sobre una ventana de **tiempo**.

    En segundos y no en cuadros porque los dos estimadores del proyecto corren
    a velocidades distintas -EyeTrax a 26 fps, GazeFollower a 14- y una ventana
    contada en cuadros significaría medio segundo en uno y un cuarto en el
    otro.

    Mediana y no promedio: el ruido de mirada tiene colas pesadas. Una sola
    muestra disparada corre el promedio y no corre la mediana, y esas muestras
    disparadas son justamente lo que hace saltar el cursor.
    """

    def __init__(self, window_s: float = 0.32) -> None:
        self.window_s = window_s
        self._points: deque = deque()

    def clear(self) -> None:
        self._points.clear()

    def push(self, x: int, y: int, now: float | None = None) -> tuple[int, int]:
        now = time.time() if now is None else now
        self._points.append((now, x, y))
        cutoff = now - self.window_s
        while len(self._points) > 1 and self._points[0][0] < cutoff:
            self._points.popleft()
        xs = sorted(p[1] for p in self._points)
        ys = sorted(p[2] for p in self._points)
        mid = len(xs) // 2
        if len(xs) % 2:
            return xs[mid], ys[mid]
        return (xs[mid - 1] + xs[mid]) // 2, (ys[mid - 1] + ys[mid]) // 2


class DwellSelector:
    """Fires when the gaze rests on the same target long enough.

    Short gaze losses (blinks, a dropped frame) are tolerated for
    ``blink_grace_ms`` so a natural blink does not restart the dwell.

    El problema que resuelve `step()`
    ---------------------------------
    La versión por objetivo -`update(target)`- reinicia la permanencia en
    cuanto el objetivo cambia, aunque sea **un solo cuadro**. La mirada no se
    queda quieta: microsacadas, temblor del estimador y un cursor a 14 fps
    hacen que el punto cruce el borde de la celda varias veces por segundo. El
    resultado es que la barra de progreso se reinicia sola y seleccionar se
    vuelve imposible por más quieto que uno mire.

    `step()` lo arregla con dos cosas que se complementan:

    * **Se decide sobre la mediana de una ventana corta**, no sobre el punto
      del cuadro. La mediana -y no el promedio- porque el ruido de mirada tiene
      colas pesadas: una sola muestra disparada corre el promedio y no corre la
      mediana.
    * **La celda sostenida se pega**: mientras la mediana siga dentro de esa
      celda *agrandada por un margen*, sigue siendo el objetivo. Salir exige
      irse claramente, no rozar el borde.

    Lo que **no** se hace es suavizar más el cursor. Alcanzaría para estabilizar
    el punto, pero el suavizado cuesta retardo, y el retardo se paga en cada
    selección. Acá el cursor se puede seguir dibujando con el punto vivo -que es
    lo que le dice a la persona hacia dónde va- mientras la selección se decide
    sobre la mediana. Estabilidad donde hace falta, sin lentitud donde molesta.
    """

    def __init__(self, settings: Settings):
        self.dwell = settings.dwell_ms / 1000.0
        self.refractory = settings.refractory_ms / 1000.0
        self.grace = settings.blink_grace_ms / 1000.0
        self.window_s = settings.fixation_window_ms / 1000.0
        self.stick = settings.stick_margin_ratio

        self.target: object | None = None
        self._started_at = 0.0
        self._lost_at: float | None = None
        self._blocked_until = 0.0
        self.window = MedianWindow(self.window_s)

    def reset(self) -> None:
        self.target = None
        self._lost_at = None

    def block(self, seconds: float | None = None) -> None:
        """Suppress selection for a while (used after a page switch)."""
        self._blocked_until = time.time() + (
            self.refractory if seconds is None else seconds
        )
        self.reset()
        # La ventana también se vacía: después de cambiar de página, los puntos
        # de la página anterior describen una pantalla que ya no está.
        self.window.clear()

    # ------------------------------------------------- fijación y pegado

    def feed(self, x: int, y: int, now: float) -> tuple[int, int]:
        """Agrega una muestra y devuelve el punto estable de la ventana."""
        return self.window.push(x, y, now)

    def _still_held(self, point: tuple[int, int], bounds) -> bool:
        """¿La mediana sigue dentro de la celda sostenida, con margen?"""
        if self.target is None or bounds is None:
            return False
        box = bounds(self.target)
        if box is None:
            return False
        x1, y1, x2, y2 = box
        mx = (x2 - x1) * self.stick
        my = (y2 - y1) * self.stick
        return (x1 - mx) <= point[0] < (x2 + mx) and \
               (y1 - my) <= point[1] < (y2 + my)

    def step(self, sample, hit_test, bounds=None, now=None):
        """Un cuadro completo. Devuelve (progreso, disparado, punto_estable).

        `hit_test(x, y)` da el objetivo bajo un punto; `bounds(objetivo)` da su
        caja, y es lo que habilita el pegado. Sin `bounds` se comporta como
        antes, solo que decidiendo sobre la mediana.
        """
        now = time.time() if now is None else now
        if sample is None or not getattr(sample, "ok", False):
            progress, fired = self.update(None, now)
            return progress, fired, None

        stable = self.feed(sample.x, sample.y, now)
        if self._still_held(stable, bounds):
            target = self.target
        else:
            target = hit_test(*stable)
        progress, fired = self.update(target, now)
        return progress, fired, stable

    def update(self, target: object | None, now: float | None = None):
        """Feed the currently-gazed target.

        Returns ``(progress, fired)`` where ``progress`` is 0..1 dwell
        completion for the current target and ``fired`` is the selected
        target, or ``None``.
        """
        now = time.time() if now is None else now

        if now < self._blocked_until:
            return 0.0, None

        if target is None:
            # Tolerate a brief loss without losing dwell progress.
            if self.target is not None:
                if self._lost_at is None:
                    self._lost_at = now
                if now - self._lost_at <= self.grace:
                    return self._progress(now), None
                self.reset()
            return 0.0, None

        self._lost_at = None

        if target != self.target:
            self.target = target
            self._started_at = now
            return 0.0, None

        progress = self._progress(now)
        if progress >= 1.0:
            fired = self.target
            self.reset()
            self._blocked_until = now + self.refractory
            return 1.0, fired
        return progress, None

    def _progress(self, now: float) -> float:
        if self.target is None:
            return 0.0
        return min(1.0, (now - self._started_at) / self.dwell)
