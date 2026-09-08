"""The calibration fit: eleven eye parameters plus the screen pose, jointly.

What is being solved. Each calibration sample records a head pose, an iris ray
per eye and the screen pixel the person was looking at. The forward model turns
the first two into a predicted pixel; the fit adjusts the parameters so
predictions match targets. Bounded Levenberg-Marquardt over 11-17 parameters and
a few hundred observations - small, fast, and well-posed provided the
degeneracies below are respected.

**The degeneracy that matters.** Pushing the eyeball centre 5 mm towards the
screen and pulling the screen 5 mm towards the eye produce nearly the same
predictions from a single head position. So does trading eyeball radius against
centre depth. Three things keep the parameters physically meaningful rather than
merely convenient:

1. *Anatomical bounds.* The radius stays in 10.5-13.5 mm and the centres within
   a centimetre of where the template puts them.
2. *A prior on the mount.* The screen pose may move a few millimetres and a
   couple of degrees from the measured value, not arbitrarily.
3. *Head movement during calibration.* This is the real fix, and the reason
   `calibrate --passes 2` exists. The eyeball centre is fixed in *head*
   coordinates while the screen is fixed in *camera* coordinates, so the moment
   the head occupies two genuinely different positions the two stop being
   interchangeable. One posture leaves them confounded and the fit will happily
   report a small residual while having distributed the error wrongly - which
   then shows up as drift the first time the user shifts in their chair.

That last point is worth stating plainly because it inverts the webcam
intuition: there, moving your head between calibration and use is what breaks
everything. Here, moving your head *during* calibration is what makes it work.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import estimator, eyemodel, geometry, landmarks
from .frames import Intrinsics

# Slack allowed on the hand-measured mount. A tape measure against a monitor
# bezel is good to a few millimetres; the IMU pins pitch and roll to a fraction
# of a degree, so the angular slack mostly covers yaw and screen non-verticality.
MOUNT_SHIFT_M = 0.030
MOUNT_ROTATE_DEG = 5.0


@dataclass
class Sample:
    """One recorded frame during calibration.

    Iris observations are stored as **unit rays** rather than pixels, so the
    intrinsics and the distortion model are applied once, here, and the
    optimiser never sees them. That also makes a stored calibration set
    replayable against a changed distortion model.
    """

    target_px: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    ray_dirs: dict[str, np.ndarray]
    weights: dict[str, float]
    target_index: int = 0
    pass_index: int = 0
    head_distance_m: float = 0.0

    @classmethod
    def build(cls, target_px, head, obs: landmarks.FaceObservation,
              intrin: Intrinsics, target_index: int = 0,
              pass_index: int = 0) -> "Sample":
        rays, weights = {}, {}
        for key in landmarks.EYE_KEYS:
            eye = obs.eyes.get(key)
            if eye is None:
                continue
            # Mean of the four ring points plus the centre landmark, matching
            # what `eyemodel.gaze_ray` does at inference time.
            iris_px = np.mean(np.vstack([eye.iris_ring,
                                         eye.iris_centre[None, :]]), axis=0)
            rays[key] = intrin.ray(iris_px)
            weights[key] = min(1.0, eye.iris_radius_px / 12.0)
        return cls(
            target_px=np.asarray(target_px, dtype=np.float64),
            rotation=np.asarray(head.rotation, dtype=np.float64),
            translation=np.asarray(head.translation, dtype=np.float64),
            ray_dirs=rays, weights=weights,
            target_index=int(target_index), pass_index=int(pass_index),
            head_distance_m=float(head.distance_m),
        )


@dataclass
class FitReport:
    """What the fit achieved, in the units a thesis needs."""

    samples: int
    targets: int
    passes: int
    rms_px: float
    p95_px: float
    rms_deg: float
    geometric_rms_px: float
    """Error before the residual correction. The gap between this and `rms_px`
    says how much of the accuracy is physics and how much is a fitted
    polynomial - which is exactly the honesty a model-based claim requires."""

    residual_rms_px: float
    head_spread_mm: float
    """How far the head actually moved across the calibration set. Small values
    mean the eye/screen degeneracy was never broken, so the fitted parameters
    are only valid near the posture that was calibrated - the very failure this
    approach exists to avoid. Reported so it cannot go unnoticed."""

    radius_mm: float
    kappa_deg: dict[str, tuple[float, float]]
    mount_shift_mm: float
    mount_rotate_deg: float
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> str:
        lines = [
            f"  muestras          {self.samples} en {self.targets} objetivos, "
            f"{self.passes} pasada(s)",
            f"  error RMS         {self.rms_px:6.1f} px   ({self.rms_deg:.2f}°)",
            f"  error p95         {self.p95_px:6.1f} px",
            f"  solo geometría    {self.geometric_rms_px:6.1f} px   "
            f"(antes de la corrección polinómica)",
            f"  radio corneal     {self.radius_mm:5.2f} mm",
            f"  kappa derecho     {self.kappa_deg['right'][0]:+5.2f}° / "
            f"{self.kappa_deg['right'][1]:+5.2f}°",
            f"  kappa izquierdo   {self.kappa_deg['left'][0]:+5.2f}° / "
            f"{self.kappa_deg['left'][1]:+5.2f}°",
            f"  ajuste del montaje {self.mount_shift_mm:5.1f} mm / "
            f"{self.mount_rotate_deg:.2f}°",
            f"  movimiento de cabeza {self.head_spread_mm:5.1f} mm",
        ]
        if self.warnings:
            lines.append("  avisos:")
            lines += [f"    - {w}" for w in self.warnings]
        return "\n".join(lines)


# --------------------------------------------------------------- screen pose


def _perturb_screen(base: geometry.ScreenPlane, params: np.ndarray,
                    ) -> geometry.ScreenPlane:
    """Apply a 6-vector (3 mm of shift, 3 of rotation vector) to a screen pose.

    Parameterised as a delta from the measured pose rather than absolutely, so
    the bounds express "how wrong can my tape measure be" - a question with an
    answer - instead of "where might a screen be", which has none.
    """
    shift = params[:3]
    rotvec = params[3:6]
    angle = float(np.linalg.norm(rotvec))
    R = (np.eye(3) if angle < 1e-12
         else geometry.rodrigues(rotvec / angle, angle))
    return geometry.ScreenPlane(
        origin=base.origin + shift,
        right=R @ base.right,
        down=R @ base.down,
        width_m=base.width_m, height_m=base.height_m,
        width_px=base.width_px, height_px=base.height_px,
    )


# ---------------------------------------------------------------------- solve


def _stack(samples: list[Sample], eye: str):
    """Gather one eye's usable samples into contiguous arrays."""
    idx = [i for i, s in enumerate(samples)
           if eye in s.ray_dirs and s.weights.get(eye, 0.0) > 0.0]
    if not idx:
        return None
    return (
        np.array(idx),
        np.stack([samples[i].rotation for i in idx]),
        np.stack([samples[i].translation for i in idx]),
        np.stack([samples[i].ray_dirs[eye] for i in idx]),
        np.array([samples[i].target_px for i in idx], dtype=np.float64),
        np.array([samples[i].weights[eye] for i in idx], dtype=np.float64),
    )


def fit(samples: list[Sample], init_eyes: eyemodel.EyeGeometry,
        init_screen: geometry.ScreenPlane, *,
        refine_mount: bool = True,
        residual_correction: bool = True,
        residual_order: int = 2,
        mount_shift_m: float = MOUNT_SHIFT_M,
        mount_rotate_deg: float = MOUNT_ROTATE_DEG,
        verbose: bool = True) -> tuple[eyemodel.EyeGeometry,
                                       geometry.ScreenPlane,
                                       estimator.ResidualCorrector,
                                       FitReport]:
    """Solve for the eye parameters and, optionally, the screen pose."""
    from scipy.optimize import least_squares

    if len(samples) < 20:
        raise ValueError(
            f"solo {len(samples)} muestras de calibración; hacen falta al "
            f"menos 20. Revise la detección de la cara y la iluminación."
        )

    per_eye = {eye: _stack(samples, eye) for eye in landmarks.EYE_KEYS}
    per_eye = {k: v for k, v in per_eye.items() if v is not None}
    if not per_eye:
        raise ValueError("ninguna muestra tiene un iris utilizable")

    n_eye_params = 11
    x0 = init_eyes.to_vector()
    lo, hi = eyemodel.EyeGeometry.bounds(init_eyes)

    if refine_mount:
        x0 = np.concatenate([x0, np.zeros(6)])
        mount_lo = np.array([-mount_shift_m] * 3
                            + [-math.radians(mount_rotate_deg)] * 3)
        lo = np.concatenate([lo, mount_lo])
        hi = np.concatenate([hi, -mount_lo])

    def unpack(x):
        eyes = eyemodel.EyeGeometry.from_vector(x[:n_eye_params])
        screen = (_perturb_screen(init_screen, x[n_eye_params:])
                  if refine_mount else init_screen)
        return eyes, screen

    # Precompute the 3D point each target sits at, per eye's sample set. Fixed
    # for the eye parameters but not for the screen pose, so it is recomputed
    # inside `residuals` when the mount is being refined.
    def target_points(screen):
        # Vectorised rather than calling `px_to_point` per sample: with the
        # mount being refined this runs on every one of a few hundred optimiser
        # evaluations, and the per-sample version dominated the whole solve.
        out = {}
        for eye, (_, _, _, _, targets, _) in per_eye.items():
            u = targets[:, 0] / screen.width_px * screen.width_m
            v = targets[:, 1] / screen.height_px * screen.height_m
            out[eye] = (screen.origin
                        + u[:, None] * screen.right
                        + v[:, None] * screen.down)
        return out

    fixed_points = None if refine_mount else target_points(init_screen)

    def residuals(x):
        """Residual in **radians of gaze angle**, not in screen pixels.

        This is the difference between a solver that converges and one that sits
        still, and the reason is worth recording. In pixels the residual is
        unbounded: displace the eyeball centre by about its own radius and the
        camera ray grazes the corneal sphere tangentially, the recovered axis
        comes out nearly parallel to the screen, and the intersection flies off
        to tens of thousands of pixels. Feed those to `soft_l1` and the loss
        saturates, the gradient vanishes, and the optimiser never leaves its
        starting point - which is exactly what happened here before the change.

        Angle has none of that. It is bounded by pi whatever the parameters do,
        it varies smoothly through the tangent case, and it is the quantity the
        model is actually uncertain about - pixels are just angle multiplied by a
        distance that itself depends on the parameters being fitted.
        """
        eyes, screen = unpack(x)
        points = fixed_points if fixed_points is not None else target_points(screen)
        out = []
        for eye, (_, R, t, dirs, targets, w) in per_eye.items():
            params = eyes.eye(eye)
            centre_cam, visual = eyemodel.forward_batch_rays(
                R, t, dirs, params.centre_head,
                params.kappa_yaw, params.kappa_pitch, eyes.radius_m)
            desired = geometry.normalize(points[eye] - centre_cam)
            # Difference of two unit vectors: equals the angle in radians to
            # first order, stays smooth and bounded far outside that regime, and
            # needs no arccos (whose derivative blows up at zero error).
            out.append(((visual - desired) * w[:, None]).reshape(-1))
        return np.concatenate(out)

    # f_scale in radians: one degree. A handful of samples always land badly - a
    # blink caught mid-close, a landmark that jumped - and plain least squares
    # would let those few dominate a fit this small.
    result = least_squares(residuals, x0, bounds=(lo, hi), method="trf",
                           x_scale="jac", loss="soft_l1",
                           f_scale=math.radians(1.0), max_nfev=400, verbose=0)

    eyes, screen = unpack(result.x)

    # ------------------------------------------------- measure what we achieved
    predicted, targets_px, indices = [], [], []
    for eye, (idx, R, t, dirs, targets, w) in per_eye.items():
        params = eyes.eye(eye)
        pred = eyemodel.forward_batch(
            R, t, dirs, params.centre_head,
            params.kappa_yaw, params.kappa_pitch, eyes.radius_m, screen)
        good = np.isfinite(pred).all(axis=1)
        predicted.append(pred[good])
        targets_px.append(targets[good])
        indices.append(idx[good])

    predicted = np.vstack(predicted)
    targets_px = np.vstack(targets_px)
    indices = np.concatenate(indices)

    geo_err = np.linalg.norm(predicted - targets_px, axis=1)
    geometric_rms = float(np.sqrt(np.mean(geo_err ** 2)))

    # ----------------------------------------------------- residual correction
    corrector = estimator.ResidualCorrector(order=residual_order)
    residual_rms = 0.0
    if residual_correction:
        # Fit on per-target medians. Targets contribute unequal sample counts -
        # a target where the face was briefly lost has fewer - and raw samples
        # would weight the easy screen centre above the corners that need this
        # correction most.
        by_target: dict[int, list[int]] = {}
        for row, sample_i in enumerate(indices):
            by_target.setdefault(samples[sample_i].target_index, []).append(row)

        pred_uv, targ_uv = [], []
        for rows in by_target.values():
            rows = np.array(rows)
            p = np.median(predicted[rows], axis=0)
            g = np.median(targets_px[rows], axis=0)
            pred_uv.append([p[0] / screen.width_px * 2.0 - 1.0,
                            p[1] / screen.height_px * 2.0 - 1.0])
            targ_uv.append([g[0] / screen.width_px * 2.0 - 1.0,
                            g[1] / screen.height_px * 2.0 - 1.0])
        residual_rms = corrector.fit(np.array(pred_uv), np.array(targ_uv))

    # Final error, with the correction applied to every sample.
    uv = np.column_stack([predicted[:, 0] / screen.width_px * 2.0 - 1.0,
                          predicted[:, 1] / screen.height_px * 2.0 - 1.0])
    corrected = corrector.apply(uv)
    final = np.column_stack([(corrected[:, 0] + 1.0) / 2.0 * screen.width_px,
                             (corrected[:, 1] + 1.0) / 2.0 * screen.height_px])
    err = np.linalg.norm(final - targets_px, axis=1)

    # Degrees at the real measured distance, not a hardcoded px-per-degree.
    mean_eye = np.mean([s.translation for s in samples], axis=0)
    px_per_deg = screen.px_per_degree(mean_eye)

    head_positions = np.array([s.translation for s in samples])
    head_spread_mm = float(np.linalg.norm(head_positions.std(axis=0)) * 1000.0)

    mount_shift_mm = (float(np.linalg.norm(result.x[n_eye_params:n_eye_params + 3]))
                      * 1000.0 if refine_mount else 0.0)
    mount_rotate_deg = (math.degrees(
        float(np.linalg.norm(result.x[n_eye_params + 3:n_eye_params + 6])))
        if refine_mount else 0.0)

    report = FitReport(
        samples=len(samples),
        targets=len({s.target_index for s in samples}),
        passes=len({s.pass_index for s in samples}),
        rms_px=float(np.sqrt(np.mean(err ** 2))),
        p95_px=float(np.percentile(err, 95)),
        rms_deg=float(np.sqrt(np.mean(err ** 2)) / max(px_per_deg, 1e-6)),
        geometric_rms_px=geometric_rms,
        residual_rms_px=float(residual_rms * screen.width_px / 2.0),
        head_spread_mm=head_spread_mm,
        radius_mm=eyes.radius_m * 1000.0,
        kappa_deg={
            "right": (math.degrees(eyes.right.kappa_yaw),
                      math.degrees(eyes.right.kappa_pitch)),
            "left": (math.degrees(eyes.left.kappa_yaw),
                     math.degrees(eyes.left.kappa_pitch)),
        },
        mount_shift_mm=mount_shift_mm,
        mount_rotate_deg=mount_rotate_deg,
        warnings=_warn(eyes, result, head_spread_mm, geometric_rms,
                       report_rms=float(np.sqrt(np.mean(err ** 2)))),
    )
    if verbose:
        print("\n=== ajuste de calibración ===")
        print(report.describe())
    return eyes, screen, corrector, report


def _warn(eyes: eyemodel.EyeGeometry, result, head_spread_mm: float,
          geometric_rms: float, report_rms: float) -> list[str]:
    """Flag the ways a numerically successful fit can still be untrustworthy."""
    warnings: list[str] = []

    if head_spread_mm < 15.0:
        warnings.append(
            f"la cabeza se movió muy poco durante la calibración "
            f"({head_spread_mm:.0f} mm). La posición del ojo y la de la "
            f"pantalla quedan confundidas entre sí, así que el ajuste vale "
            f"cerca de esa postura y no lejos de ella. Use "
            f"`calibrate --passes 2`, que pide cambiar de postura entre "
            f"pasadas: es lo único que separa esos dos parámetros."
        )

    v = result.x[:11]
    # A parameter pinned to its bound means the model wanted to go further,
    # which means something outside the model is being absorbed.
    at_radius_bound = (abs(v[10] - eyemodel.RADIUS_MIN_M) < 1e-5
                       or abs(v[10] - eyemodel.RADIUS_MAX_M) < 1e-5)
    if at_radius_bound:
        warnings.append(
            f"el radio corneal quedó en el límite ({v[10] * 1000:.1f} mm), lo "
            f"que suele significar que el modelo está absorbiendo un error de "
            f"otra parte: revise las medidas del montaje en CameraMount."
        )

    for eye, base in (("derecho", 3), ("izquierdo", 8)):
        if max(abs(v[base]), abs(v[base + 1])) > math.radians(
                eyemodel.MAX_KAPPA_DEG - 0.5):
            warnings.append(
                f"el kappa del ojo {eye} llegó al límite "
                f"({math.degrees(v[base]):+.1f}°). Es más grande de lo "
                f"anatómicamente esperable (4-5°); casi siempre es la pantalla "
                f"o el montaje mal medidos, no el ojo."
            )

    if geometric_rms > 3.0 * max(report_rms, 1.0):
        warnings.append(
            f"la corrección polinómica está haciendo casi todo el trabajo "
            f"({geometric_rms:.0f} px -> {report_rms:.0f} px). Eso es un ajuste "
            f"de curva, no un modelo geométrico, y no va a extrapolar a "
            f"posturas nuevas. Revise la geometría del montaje."
        )

    if not result.success:
        warnings.append(
            f"el optimizador terminó sin converger ({result.message}). "
            f"Los parámetros son los mejores encontrados, pero desconfíe."
        )
    return warnings
