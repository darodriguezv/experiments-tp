"""The only module the AAC layer talks to.

Deliberately the same shape as the EyeTrax harness's `tracker.py` - `GazeSample`,
`GazeTracker.process`, `DwellSelector` - so the board, the accuracy harness and
the live view are the same programs against a different gaze engine. That is
what makes the comparison in this repo a comparison rather than two unrelated
projects, and it is what the EyeTrax README promised when it said swapping the
gaze library should mean rewriting one file.

`GazeSample` gains three fields the webcam version could not produce, all of
them coming from the same source - metric depth:

* `confidence`, so a caller can decline to act on a bad frame.
* `distance_cm`, a real measurement rather than a proxy for how close the user
  is sitting.
* `head_yaw/pitch/roll`, real angles in degrees, which the placement guide needs
  and which the webcam path got demonstrably wrong.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import (display, estimator, face3d, geometry, landmarks, smoothing,
               source)
from .config import CameraMount, Settings
from .frames import Frame


@dataclass
class GazeSample:
    """One frame's worth of gaze state."""

    ok: bool
    """True when a face was found, the eyes were open and a gaze point resulted."""

    x: int = 0
    y: int = 0
    """Smoothed, screen-clamped gaze point."""

    raw_x: int = 0
    raw_y: int = 0
    """Before smoothing - useful for seeing how much jitter is being absorbed."""

    blink: bool = False
    face: bool = False

    confidence: float = 0.0
    distance_cm: float = 0.0
    iris_px: float = 0.0
    """Mean iris radius in image pixels. The hard ceiling on angular precision -
    the pupil centre is quantised by it - so it belongs in the logs next to the
    error it explains."""

    head_yaw: float = 0.0
    head_pitch: float = 0.0
    head_roll: float = 0.0
    method: str = ""
    """Which head-pose path ran, "kabsch" or "pnp"."""

    on_screen: bool = True
    """False when the gaze ray landed outside the display before clamping. A
    genuine signal - the user looked away - and distinct from a bad frame, which
    is what `confidence` reports."""

    reason: str = ""


class GazeTracker:
    """Owns the landmarker, the estimator and the smoother."""

    def __init__(self, settings: Settings | None = None,
                 landmarker: landmarks.Landmarker | None = None):
        self.settings = settings or Settings()
        self.settings.validate()

        self.screen = display.resolve(self.settings.monitor)
        self.screen_w, self.screen_h = self.screen.size

        # Injected for tests; built lazily otherwise so importing this module
        # never costs the second or so that MediaPipe takes to initialise.
        self._landmarker = landmarker
        self._owns_landmarker = landmarker is None

        self.model: estimator.GazeModel | None = None
        self.estimator = None
        self.smoother = smoothing.make(self.settings.smoother,
                                       self.settings.ema_alpha,
                                       float(self.settings.fps))
        self.trained = False
        self.last_head: face3d.Head | None = None

    # ------------------------------------------------------------- components

    @property
    def landmarker(self) -> landmarks.Landmarker:
        if self._landmarker is None:
            self._landmarker = landmarks.MediaPipeLandmarker()
        return self._landmarker

    def default_screen_plane(self, accel: np.ndarray | None = None
                             ) -> geometry.ScreenPlane:
        """The screen pose implied by the mount, before any refinement.

        Takes the physical screen size from EDID when available and the camera's
        tilt from the accelerometer when available, so the starting geometry is
        measured rather than assumed wherever the hardware permits.
        """
        mount = CameraMount.from_screen(
            self.screen,
            offset_x_mm=self.settings.mount.offset_x_mm,
            offset_y_mm=self.settings.mount.offset_y_mm,
            offset_z_mm=self.settings.mount.offset_z_mm,
            pitch_deg=self.settings.mount.pitch_deg,
            yaw_deg=self.settings.mount.yaw_deg,
            roll_deg=self.settings.mount.roll_deg,
        ).with_gravity(accel)
        self.settings.mount = mount
        return mount.screen_plane(self.screen_w, self.screen_h)

    def open_source(self, *, synthetic: bool = False) -> source.FrameSource:
        return source.open_source(self.settings, synthetic=synthetic)

    def open_window(self, window: str) -> None:
        display.open_fullscreen(window, self.screen)
        if not display.placement_ok(window, self.screen):
            print(
                f"[aviso] la ventana '{window}' puede no haber quedado en la "
                f"pantalla {self.screen.index}; las coordenadas de mirada "
                f"asumen {self.screen_w}x{self.screen_h} en "
                f"({self.screen.x},{self.screen.y})."
            )

    def reset_smoother(self) -> None:
        self.smoother = smoothing.make(self.settings.smoother,
                                       self.settings.ema_alpha,
                                       float(self.settings.fps))

    # ------------------------------------------------------------ persistence

    def load(self, path: str | Path | None = None) -> Path:
        path = Path(path or self.settings.model_path)
        self.model = estimator.GazeModel.load(path)
        self._warn_on_meta_mismatch(path)
        self._build_estimator()
        self.trained = True
        return path

    def attach(self, model: estimator.GazeModel) -> None:
        """Use a model already in memory, skipping the round trip through disk."""
        self.model = model
        self._build_estimator()
        self.trained = True

    def _build_estimator(self) -> None:
        if self.settings.estimator == "hybrid":
            regressor = estimator.HybridRegressor.from_dict(
                self.model.meta.get("hybrid", {}))
            if regressor.coeffs is None:
                raise ValueError(
                    "el modelo guardado no tiene un regresor híbrido "
                    "entrenado. Calibre con --estimator hybrid."
                )
            self.estimator = estimator.HybridEstimator(
                self.model, regressor, self.settings)
        else:
            self.estimator = estimator.GeometricEstimator(self.model, self.settings)

    def _warn_on_meta_mismatch(self, path: Path) -> None:
        """A calibration is only valid for the screen and camera that made it.

        With a geometric model the consequence is sharper than with a
        regression: the saved screen plane is a *physical* pose in camera
        coordinates, so loading it against a different monitor or a moved camera
        does not degrade gracefully - it points the rays at where the old screen
        used to be.
        """
        meta = self.model.meta if self.model else {}
        saved_screen = meta.get("monitor", {})
        if (saved_screen.get("name") != self.screen.name
                or saved_screen.get("size") != [self.screen_w, self.screen_h]):
            print(
                f"[aviso] este modelo se calibró en la pantalla "
                f"{saved_screen.get('index')} ({saved_screen.get('name')}, "
                f"{saved_screen.get('size')}) y ahora corre en la "
                f"{self.screen.index} ({self.screen.name}, "
                f"[{self.screen_w}, {self.screen_h}]).\n"
                f"        El plano de la pantalla guardado es una pose física: "
                f"apunta a donde estaba la pantalla anterior. Recalibre."
            )

        saved_serial = meta.get("camera", {}).get("serial")
        if saved_serial and self.settings.serial and saved_serial != self.settings.serial:
            print(f"[aviso] se calibró con la cámara {saved_serial} y ahora se "
                  f"usa la {self.settings.serial}.")

        if meta.get("mount", {}).get("measured_by", "").startswith("defaults"):
            print("[aviso] la geometría del montaje nunca se midió: se usaron "
                  "los valores por defecto. Mida los desplazamientos reales "
                  "(ver README) para ganar precisión sin más esfuerzo.")

    # ------------------------------------------------------------- per frame

    def process(self, frame: Frame) -> GazeSample:
        """One frame in, one gaze sample out."""
        if self.estimator is None:
            raise RuntimeError("llame a load() o attach() antes de process()")

        obs = self.landmarker.detect(frame.color)
        if obs is None:
            return GazeSample(ok=False, face=False, reason="sin cara")
        if obs.blink:
            # Reported as a distinct state rather than as "no face", so dwell can
            # keep its progress through a blink instead of restarting. For a user
            # with reduced eyelid control, blinks are frequent and long.
            return GazeSample(ok=False, face=True, blink=True, reason="parpadeo")

        est = self.estimator.predict(frame, obs)
        if not est.ok:
            head = est.head
            return GazeSample(ok=False, face=True, reason=est.reason,
                              distance_cm=(head.distance_m * 100.0) if head else 0.0)

        self.last_head = est.head
        sx, sy = self.smoother.step(est.x, est.y, est.confidence)
        yaw, pitch, roll = est.head.euler_deg()
        iris_px = float(np.mean([e.iris_radius_px for e in obs.eyes.values()])
                        ) if obs.eyes else 0.0

        on_screen = (0.0 <= est.x < self.screen_w and 0.0 <= est.y < self.screen_h)
        return GazeSample(
            ok=True,
            x=int(np.clip(sx, 0, self.screen_w - 1)),
            y=int(np.clip(sy, 0, self.screen_h - 1)),
            raw_x=int(np.clip(est.x, -9999, 9999)),
            raw_y=int(np.clip(est.y, -9999, 9999)),
            face=True,
            confidence=est.confidence,
            distance_cm=est.head.distance_m * 100.0,
            iris_px=iris_px,
            head_yaw=yaw, head_pitch=pitch, head_roll=roll,
            method=est.head.method,
            on_screen=on_screen,
        )

    def close(self) -> None:
        if self._owns_landmarker and self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> "GazeTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class DwellSelector:
    """Fires when the gaze rests on the same target long enough.

    Ported from the EyeTrax harness, where the blink-grace behaviour was already
    right, plus one addition the webcam version could not make: low-confidence
    frames neither advance nor reset the dwell. They are treated exactly like a
    blink - held, not counted.

    That matters because the two mistakes are not symmetric. Advancing on bad
    frames causes mis-selections, and a mis-selection costs a delete plus a
    retry. Resetting on bad frames costs the user the whole dwell and reads as
    the system ignoring them. Holding does neither.
    """

    def __init__(self, settings: Settings, min_confidence: float = 0.25):
        self.dwell = settings.dwell_ms / 1000.0
        self.refractory = settings.refractory_ms / 1000.0
        self.grace = settings.blink_grace_ms / 1000.0
        self.min_confidence = float(min_confidence)

        self.target: object | None = None
        self._started_at = 0.0
        self._lost_at: float | None = None
        self._blocked_until = 0.0

    def reset(self) -> None:
        self.target = None
        self._lost_at = None

    def block(self, seconds: float | None = None) -> None:
        """Suppress selection for a while - used after a page switch."""
        self._blocked_until = time.time() + (
            self.refractory if seconds is None else seconds)
        self.reset()

    def update(self, target: object | None, confidence: float = 1.0,
               now: float | None = None) -> tuple[float, object | None]:
        """Feed the currently-gazed target. Returns (progress, fired target)."""
        now = time.time() if now is None else now

        if now < self._blocked_until:
            return 0.0, None

        # A frame we do not trust is a frame we hold on, not one we act on.
        uncertain = target is not None and confidence < self.min_confidence
        if target is None or uncertain:
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
