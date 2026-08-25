"""Turning a frame into a screen coordinate, two different ways.

`GeometricEstimator` is the real one: metric head pose, an eye model, and a ray
intersected with the physical plane of the display. It is head-pose invariant by
construction, which is the property the webcam pipeline lacked.

`HybridRegressor` is the comparison arm. It is a regression like EyeTrax's, but
over *metric* features - 3D eye position, gaze angles, head pose - instead of
486 raw landmark coordinates. Keeping it here is not hedging: an experiments
repo needs the baseline measured on the same rig by the same accuracy harness,
or "the geometric model is better" is an assertion rather than a result. It also
isolates the two claims being made, since a metric feature set and a geometric
model are separate improvements and they can be credited separately.

Both end with a residual correction. A sphere-and-ray model does not represent
corneal refraction, the cornea's asphericity, or the systematic offset between
where MediaPipe puts an iris landmark and where the pupil actually is. Those
leave a smooth, repeatable error field across the screen, which is exactly what
a low-order polynomial removes and exactly what a per-frame filter cannot.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import eyemodel, face3d, geometry, landmarks
from .config import CameraMount, Settings
from .frames import Frame, Intrinsics


@dataclass
class GazeEstimate:
    """One frame's result, before smoothing."""

    ok: bool
    x: float = 0.0
    y: float = 0.0
    """Screen pixels, unclamped and unsmoothed. Off-screen values are real
    information - the person is not looking at the board - so they survive to
    the caller."""

    confidence: float = 0.0
    """0..1. The webcam path had no such thing: a regression always returns a
    point, however far off, so a bad frame was indistinguishable from a good
    one. Here a low value has causes that can be named - depth lost, head fit
    poor, iris too small, gaze too eccentric - and dwell selection can hold
    instead of committing."""

    head: face3d.Head | None = None
    rays: dict = field(default_factory=dict)
    reason: str = ""
    """Why `ok` is False, for the status line and the logs."""


class ResidualCorrector:
    """A low-order polynomial over the screen, fitted to leftover error.

    Fitted on per-target **medians**, not on every sample: targets differ in how
    many usable frames they contributed, and least squares over raw samples
    would quietly weight the easy centre of the screen above the corners, which
    are the points that need the correction most.

    Ridge-regularised, and the regularisation is not decoration - a quadratic in
    two variables has six coefficients per axis and a nine-point calibration
    gives nine observations. Without a penalty the fit would interpolate the
    calibration points and invent structure between them.
    """

    def __init__(self, order: int = 2, ridge: float = 1e-3):
        self.order = int(order)
        self.ridge = float(ridge)
        self.coeffs: np.ndarray | None = None

    # ---------------------------------------------------------------- design

    def _features(self, uv: np.ndarray) -> np.ndarray:
        """Polynomial basis in screen coordinates normalised to [-1, 1].

        Normalising first is what makes a single ridge value sensible across
        both axes and any screen resolution; on raw pixels the quadratic terms
        would be six orders of magnitude larger than the constant.
        """
        uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
        u, v = uv[:, 0], uv[:, 1]
        cols = [np.ones_like(u)]
        for total in range(1, self.order + 1):
            for i in range(total + 1):
                cols.append((u ** (total - i)) * (v ** i))
        return np.column_stack(cols)

    @property
    def n_features(self) -> int:
        return (self.order + 1) * (self.order + 2) // 2

    # ------------------------------------------------------------------- fit

    def fit(self, predicted_uv: np.ndarray, target_uv: np.ndarray) -> float:
        """Fit the correction. Returns the RMS residual it leaves, in [-1, 1] units.

        Both arguments are normalised screen coordinates. The correction is
        keyed on the *predicted* position, not the target, because that is what
        is available at inference time.
        """
        pred = np.atleast_2d(np.asarray(predicted_uv, dtype=np.float64))
        targ = np.atleast_2d(np.asarray(target_uv, dtype=np.float64))
        # Drop the polynomial order until there is more data than there are
        # coefficients. Fitting six coefficients to five points is not a
        # correction, it is memorisation, and it would make the screen edges
        # worse in exchange for making the calibration report look good. Order 0
        # is a plain bias, which is the honest part of this correction anyway.
        while self.order > 0 and len(pred) < self.n_features + 2:
            self.order -= 1
        X = self._features(pred)
        Y = targ - pred

        # Do not penalise the constant: a uniform offset is exactly the sort of
        # error this is for, and shrinking it would leave a known bias in place.
        penalty = np.eye(X.shape[1]) * self.ridge
        penalty[0, 0] = 0.0
        self.coeffs = np.linalg.solve(X.T @ X + penalty, X.T @ Y)

        resid = (X @ self.coeffs) - Y
        return float(np.sqrt(np.mean(np.sum(resid ** 2, axis=1))))

    def apply(self, uv: np.ndarray) -> np.ndarray:
        """Correct normalised screen coordinates.

        Always returns shape (n, 2), whatever the input shape - one predictable
        contract is worth more than the convenience of collapsing a single row.
        Identity when unfitted, so an uncalibrated corrector is safe to call.
        """
        uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
        if self.coeffs is None:
            return uv
        return uv + self._features(uv) @ self.coeffs

    def as_dict(self) -> dict:
        return {"order": self.order, "ridge": self.ridge,
                "coeffs": None if self.coeffs is None else self.coeffs.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "ResidualCorrector":
        c = cls(order=int(d.get("order", 2)), ridge=float(d.get("ridge", 1e-3)))
        if d.get("coeffs") is not None:
            c.coeffs = np.asarray(d["coeffs"], dtype=np.float64)
        return c


@dataclass
class GazeModel:
    """The complete calibrated artifact, saved as JSON.

    JSON rather than a pickle on purpose. The webcam harness stored a pickled
    sklearn estimator, which ties a saved calibration to the exact library
    versions that produced it and cannot be read by anything else. This model is
    eleven physical parameters, a screen pose, a skull template and a few
    polynomial coefficients - all of it meaningful numbers that belong in a
    thesis appendix and should still load in two years.
    """

    template: face3d.HeadTemplate
    eyes: eyemodel.EyeGeometry
    screen: geometry.ScreenPlane
    corrector: ResidualCorrector = field(default_factory=ResidualCorrector)
    mount: CameraMount = field(default_factory=CameraMount)
    intrinsics: Intrinsics | None = None
    meta: dict = field(default_factory=dict)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "aac_realsense.gaze_model/1",
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "template": self.template.as_dict(),
            "eyes": self.eyes.as_dict(),
            "screen": self.screen.as_dict(),
            "corrector": self.corrector.as_dict(),
            "mount": self.mount.as_dict(),
            "intrinsics": None if self.intrinsics is None else self.intrinsics.as_dict(),
            "meta": self.meta,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "GazeModel":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"no hay calibración en {path}. Ejecute "
                f"`python main.py calibrate` primero."
            )
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            template=face3d.HeadTemplate.from_dict(d["template"]),
            eyes=eyemodel.EyeGeometry.from_dict(d["eyes"]),
            screen=geometry.ScreenPlane.from_dict(d["screen"]),
            corrector=ResidualCorrector.from_dict(d.get("corrector", {})),
            mount=CameraMount.from_dict(d.get("mount", {})),
            intrinsics=(Intrinsics.from_dict(d["intrinsics"])
                        if d.get("intrinsics") else None),
            meta=d.get("meta", {}),
        )


class GeometricEstimator:
    """Metric gaze: head pose -> eye model -> screen plane intersection."""

    def __init__(self, model: GazeModel, settings: Settings | None = None):
        self.model = model
        self.settings = settings or Settings()
        self.head_tracker = face3d.HeadTracker(model.template)

    # ------------------------------------------------------------------ core

    def predict(self, frame: Frame, obs: landmarks.FaceObservation) -> GazeEstimate:
        head = self.head_tracker.track(frame, obs)
        if head is None:
            return GazeEstimate(ok=False, reason="no se pudo estimar la pose de la cabeza")

        rays = eyemodel.gaze_rays(head, obs, self.model.eyes, frame.intrinsics)
        screen = self.model.screen

        hits: dict[str, tuple[float, float]] = {}
        weights: dict[str, float] = {}
        for key, ray in rays.items():
            hit = screen.intersect(ray.origin, ray.direction)
            if hit is None:
                continue
            hits[key] = hit
            weights[key] = eyemodel.eye_weight(ray, obs.eyes[key])

        point = eyemodel.combine_on_screen(hits, weights)
        if point is None:
            return GazeEstimate(ok=False, head=head, rays=rays,
                                reason="la mirada no cruza el plano de la pantalla")

        x, y = self._corrected(point, screen)
        return GazeEstimate(
            ok=True, x=x, y=y, head=head, rays=rays,
            confidence=confidence_of(head, weights),
        )

    def _corrected(self, point: tuple[float, float],
                   screen: geometry.ScreenPlane) -> tuple[float, float]:
        """Apply the residual correction, in normalised screen coordinates."""
        u = point[0] / screen.width_px * 2.0 - 1.0
        v = point[1] / screen.height_px * 2.0 - 1.0
        cu, cv = np.atleast_2d(self.model.corrector.apply(np.array([[u, v]])))[0]
        return ((cu + 1.0) / 2.0 * screen.width_px,
                (cv + 1.0) / 2.0 * screen.height_px)


def confidence_of(head: face3d.Head, weights: dict[str, float]) -> float:
    """Combine the independent things that can go wrong into one 0..1 value.

    Module-level rather than a method because both estimators need it and it
    depends on nothing but its arguments - the geometric and hybrid arms must
    score frames identically or the comparison between them is confounded by
    their confidence, not just their predictions.

    Each term is a physical quantity with a known failure threshold, which is
    what makes a low value actionable rather than a vibe: it can be attributed,
    and `positioning` can tell the user which one to fix.
    """
    # A rigid fit residual over ~5 mm means the template no longer matches the
    # observation - wrong person, a hand on the face, or bad depth.
    fit_term = float(np.clip(1.0 - head.residual_m / 0.005, 0.0, 1.0))
    # PnP is a real fallback, not an equal: without depth it has no independent
    # constraint on distance, so it is more sensitive to landmark noise.
    method_term = 1.0 if head.method == "kabsch" else 0.75
    eye_term = max(weights.values()) if weights else 0.0
    # Both eyes agreeing is strong evidence; a single eye is workable.
    both_term = 1.0 if len([w for w in weights.values() if w > 0]) == 2 else 0.8
    return float(fit_term * method_term * eye_term * both_term)


class HybridRegressor:
    """Ridge regression over metric features. The comparison arm.

    The feature vector is deliberately tiny - 16 numbers, all of them physical -
    against EyeTrax's 486 raw landmark coordinates. That is the point of the
    comparison: if most of the webcam pipeline's edge collapse came from
    regressing on non-metric features rather than from regression itself, this
    arm will show it by doing markedly better than EyeTrax while still falling
    short of the geometric model.

    Solved in closed form with numpy, so there is no sklearn dependency and no
    pickle in the saved model.
    """

    N_FEATURES = 16

    def __init__(self, ridge: float = 1e-2):
        self.ridge = float(ridge)
        self.coeffs: np.ndarray | None = None

    @staticmethod
    def features(head: face3d.Head, rays: dict[str, eyemodel.GazeRay]) -> np.ndarray:
        """Metric, low-dimensional, and interpretable.

        Everything here is in metres or radians and means something physically,
        so the fitted weights can be sanity-checked - unlike a weight on
        "landmark 214's normalised z".
        """
        yaw, pitch, roll = head.euler_deg()
        feats = [head.translation[0], head.translation[1], head.translation[2],
                 np.radians(yaw), np.radians(pitch), np.radians(roll)]
        for key in landmarks.EYE_KEYS:
            ray = rays.get(key)
            if ray is None:
                feats += [0.0, 0.0, 0.0, 0.0, 0.0]
                continue
            # Gaze angles in the *head* frame, so head rotation is not counted
            # twice - once here and once in the Euler terms above.
            d_head = head.basis.T @ ray.direction
            g_yaw, g_pitch = geometry.direction_to_yaw_pitch(d_head)
            feats += [g_yaw, g_pitch, ray.origin[0], ray.origin[1], ray.origin[2]]
        return np.asarray(feats, dtype=np.float64)

    def _design(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return np.column_stack([np.ones(len(X)), X])

    def fit(self, X: np.ndarray, Y: np.ndarray) -> float:
        A = self._design(X)
        penalty = np.eye(A.shape[1]) * self.ridge
        penalty[0, 0] = 0.0
        self.coeffs = np.linalg.solve(A.T @ A + penalty, A.T @ np.asarray(Y, float))
        resid = A @ self.coeffs - Y
        return float(np.sqrt(np.mean(np.sum(resid ** 2, axis=1))))

    def predict_one(self, x: np.ndarray) -> tuple[float, float]:
        if self.coeffs is None:
            raise RuntimeError("HybridRegressor sin entrenar")
        out = self._design(x[None, :]) @ self.coeffs
        return float(out[0, 0]), float(out[0, 1])

    def as_dict(self) -> dict:
        return {"ridge": self.ridge,
                "coeffs": None if self.coeffs is None else self.coeffs.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "HybridRegressor":
        r = cls(ridge=float(d.get("ridge", 1e-2)))
        if d.get("coeffs") is not None:
            r.coeffs = np.asarray(d["coeffs"], dtype=np.float64)
        return r


class HybridEstimator:
    """Wraps `HybridRegressor` in the `GeometricEstimator` interface.

    Shares the head tracker and eye model with the geometric path, so the two
    arms differ *only* in how gaze becomes a screen coordinate - by physical
    intersection or by learned mapping. Any other difference would confound the
    comparison.
    """

    def __init__(self, model: GazeModel, regressor: HybridRegressor,
                 settings: Settings | None = None):
        self.model = model
        self.regressor = regressor
        self.settings = settings or Settings()
        self.head_tracker = face3d.HeadTracker(model.template)

    def predict(self, frame: Frame, obs: landmarks.FaceObservation) -> GazeEstimate:
        head = self.head_tracker.track(frame, obs)
        if head is None:
            return GazeEstimate(ok=False, reason="no se pudo estimar la pose de la cabeza")

        rays = eyemodel.gaze_rays(head, obs, self.model.eyes, frame.intrinsics)
        if not rays:
            return GazeEstimate(ok=False, head=head, reason="no se detectó el iris")

        x, y = self.regressor.predict_one(HybridRegressor.features(head, rays))
        weights = {k: eyemodel.eye_weight(r, obs.eyes[k]) for k, r in rays.items()}
        return GazeEstimate(
            ok=True, x=x, y=y, head=head, rays=rays,
            confidence=confidence_of(head, weights),
        )
