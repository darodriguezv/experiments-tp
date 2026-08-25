"""From an iris pixel to a metric gaze ray.

The model per eye, which is the classical model-based formulation and is used
here because it is the one that generalises across head positions:

* The eyeball centre sits at a **fixed point in head coordinates**. Not in
  camera coordinates - that is the whole trick. Once the head pose is metric,
  the eyeball centre moves with the head for free, so the user may lean, turn
  or slump without invalidating anything.
* The pupil lies on a sphere of radius R about that centre. The camera ray
  through the observed iris pixel meets that sphere at the pupil, which turns a
  2D observation into a 3D one without ever reading depth at the eye - where
  depth is least reliable, the cornea being specular, the sclera textureless
  and the pupil a hole to an IR projector.
* Centre-to-pupil is the **optical** axis. Where the person is actually looking
  is the **visual** axis, offset by a per-eye angle (kappa) because the fovea is
  not at the back of the eye. Kappa is typically 4-5 degrees horizontally, which
  at 60 cm is around 200 px - so a model that ignores it is not usable, and one
  that folds it into a per-position correction is back to being head-dependent.

Parameter count is deliberately small: 3 for each eyeball centre, 2 for each
kappa, 1 shared radius - 11 in total. That is what makes a nine-point
calibration enough. An appearance regression over 486 landmark features has
hundreds of parameters and correspondingly no ability to extrapolate past the
poses it saw, which is exactly the edge collapse visible in this repo's EyeTrax
logs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import geometry, landmarks
from .face3d import Head, HeadTemplate
from .frames import Intrinsics

# Anatomical priors. These bound the optimiser rather than being fitted freely:
# eyeball radius trades off almost exactly against the centre's depth, so
# without bounds the pair wanders far from anatomy while fitting the calibration
# points just as well - and then extrapolates badly to everywhere else.
RADIUS_INIT_M = 0.0120
RADIUS_MIN_M = 0.0105
RADIUS_MAX_M = 0.0135

CENTRE_BEHIND_CORNERS_M = 0.0120
"""How far behind the plane of the eye corners to place the initial eyeball
centre. About 12 mm on an adult."""

KAPPA_INIT_DEG = 4.5
"""Horizontal kappa, mirrored between the eyes: each eye's fovea sits temporal
to its optical axis, so the two offsets point in opposite directions in head
coordinates. Initialising both with the same sign is a classic and expensive
mistake - calibration will partly absorb it and leave a residual that looks
like a mysterious horizontal bias."""

MAX_KAPPA_DEG = 12.0


@dataclass
class EyeParams:
    """One eye's fitted geometry."""

    centre_head: np.ndarray
    """Eyeball centre, in head-local metres."""

    kappa_yaw: float = 0.0
    kappa_pitch: float = 0.0
    """Optical-to-visual axis offset, in radians, in head coordinates."""

    def as_dict(self) -> dict:
        # Rounded for readability - these files are meant to be opened and read,
        # and appear in a thesis appendix. Kappa keeps six decimal *degrees*
        # rather than four: four decimals is 1.7e-6 radians, and at 60 cm that
        # is already a visible fraction of a pixel on screen. Micrometres for
        # the centre are well past anything the camera can resolve.
        return {
            "centre_head_m": [round(float(v), 7) for v in self.centre_head],
            "kappa_yaw_deg": round(math.degrees(self.kappa_yaw), 6),
            "kappa_pitch_deg": round(math.degrees(self.kappa_pitch), 6),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EyeParams":
        return cls(
            centre_head=np.asarray(d["centre_head_m"], dtype=np.float64),
            kappa_yaw=math.radians(float(d.get("kappa_yaw_deg", 0.0))),
            kappa_pitch=math.radians(float(d.get("kappa_pitch_deg", 0.0))),
        )


@dataclass
class EyeGeometry:
    """Both eyes plus the shared corneal radius."""

    right: EyeParams
    left: EyeParams
    radius_m: float = RADIUS_INIT_M

    def eye(self, key: str) -> EyeParams:
        return self.right if key == "right" else self.left

    # ------------------------------------------------------ optimiser interop

    def to_vector(self) -> np.ndarray:
        """Flatten to the 11-vector the calibration optimiser works on."""
        return np.concatenate([
            self.right.centre_head, [self.right.kappa_yaw, self.right.kappa_pitch],
            self.left.centre_head, [self.left.kappa_yaw, self.left.kappa_pitch],
            [self.radius_m],
        ])

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "EyeGeometry":
        v = np.asarray(v, dtype=np.float64)
        return cls(
            right=EyeParams(v[0:3].copy(), float(v[3]), float(v[4])),
            left=EyeParams(v[5:8].copy(), float(v[8]), float(v[9])),
            radius_m=float(v[10]),
        )

    @staticmethod
    def bounds(init: "EyeGeometry", centre_slack_m: float = 0.010,
               ) -> tuple[np.ndarray, np.ndarray]:
        """Box bounds around an initial guess.

        The centres are allowed to move by a centimetre - enough to cover real
        anatomical variation and a mediocre landmark-to-eyeball relationship,
        far too little to let them run off and absorb an error that actually
        belongs to the screen plane. That separation is what keeps the fitted
        parameters physically meaningful instead of merely convenient.
        """
        v = init.to_vector()
        lo, hi = v.copy(), v.copy()
        kappa_max = math.radians(MAX_KAPPA_DEG)
        for base in (0, 5):
            lo[base:base + 3] = v[base:base + 3] - centre_slack_m
            hi[base:base + 3] = v[base:base + 3] + centre_slack_m
            lo[base + 3:base + 5] = -kappa_max
            hi[base + 3:base + 5] = kappa_max
        lo[10], hi[10] = RADIUS_MIN_M, RADIUS_MAX_M
        return lo, hi

    # -------------------------------------------------------- serialisation

    def as_dict(self) -> dict:
        return {"right": self.right.as_dict(), "left": self.left.as_dict(),
                "radius_m": round(self.radius_m, 6)}

    @classmethod
    def from_dict(cls, d: dict) -> "EyeGeometry":
        return cls(right=EyeParams.from_dict(d["right"]),
                   left=EyeParams.from_dict(d["left"]),
                   radius_m=float(d.get("radius_m", RADIUS_INIT_M)))

    @classmethod
    def initial(cls, template: HeadTemplate) -> "EyeGeometry":
        """Anatomical starting point, derived from this person's own template.

        Each eyeball centre starts at the midpoint of that eye's corners,
        pushed 12 mm into the head along the head's +z. Deriving it from the
        measured template rather than from population averages means the
        starting point already reflects this person's interpupillary distance
        and orbit spacing, which is most of what varies.
        """
        def centre_for(inner: int, outer: int, sign: float) -> EyeParams:
            mid = (template.point_of(inner) + template.point_of(outer)) / 2.0
            centre = mid + np.array([0.0, 0.0, CENTRE_BEHIND_CORNERS_M])
            return EyeParams(centre, sign * math.radians(KAPPA_INIT_DEG), 0.0)

        # Each fovea is temporal to its own optical axis, so the two initial
        # kappas are mirrored in the head frame (+x is the subject's left).
        return cls(
            right=centre_for(landmarks.RIGHT_EYE_INNER, landmarks.RIGHT_EYE_OUTER, +1.0),
            left=centre_for(landmarks.LEFT_EYE_INNER, landmarks.LEFT_EYE_OUTER, -1.0),
            radius_m=RADIUS_INIT_M,
        )


@dataclass
class GazeRay:
    """One eye's gaze, in camera coordinates."""

    origin: np.ndarray
    """The eyeball centre. Rays start here rather than at the pupil because the
    centre is the stable quantity - it is fitted, while the pupil is observed
    and therefore noisy."""

    direction: np.ndarray
    """Unit visual axis: where the person is looking."""

    optical: np.ndarray
    """Unit optical axis, before kappa. Kept for diagnostics: the angle between
    the two should stay near the fitted kappa, and drift means the eye model is
    being asked to explain something it does not represent."""

    pupil: np.ndarray
    eye: str
    eccentricity_deg: float
    """How far the eye is rotated from looking straight ahead in the head frame.
    Beyond roughly 25 degrees the sphere model degrades - the cornea is not
    spherical out there - so this gates confidence rather than being ignored."""


def gaze_ray(head: Head, eye_key: str, eye_obs: landmarks.Eye,
             geom: EyeGeometry, intrin: Intrinsics) -> GazeRay:
    """Build one eye's gaze ray from its iris observation.

    The iris centre comes from the mean of MediaPipe's four-point ring rather
    than from the single centre landmark: averaging four independent estimates
    cuts the noise, and iris jitter is gaze jitter.
    """
    params = geom.eye(eye_key)
    centre_cam = head.to_camera(params.centre_head)

    iris_px = np.mean(np.vstack([eye_obs.iris_ring,
                                 eye_obs.iris_centre[None, :]]), axis=0)
    ray_dir = intrin.ray(iris_px)

    # Tangent fallback rather than a miss: at wide gaze angles a slightly wrong
    # radius makes the ray graze past the sphere, and dropping those frames
    # would blind the board exactly at its edge tiles.
    pupil = geometry.ray_sphere_tangent(np.zeros(3), ray_dir, centre_cam,
                                        geom.radius_m)

    optical = geometry.normalize(pupil - centre_cam)
    visual = geometry.apply_kappa(optical, params.kappa_yaw, params.kappa_pitch,
                                 head.basis)

    # Eccentricity is measured in the head frame, so head rotation does not
    # count as eye rotation.
    forward_head = np.array([0.0, 0.0, -1.0])   # out of the face
    ecc = geometry.angle_between(head.basis.T @ optical, forward_head)

    return GazeRay(origin=centre_cam, direction=visual, optical=optical,
                   pupil=pupil, eye=eye_key, eccentricity_deg=ecc)


def gaze_rays(head: Head, obs: landmarks.FaceObservation, geom: EyeGeometry,
              intrin: Intrinsics) -> dict[str, GazeRay]:
    return {key: gaze_ray(head, key, obs.eyes[key], geom, intrin)
            for key in landmarks.EYE_KEYS if key in obs.eyes}


def combine_on_screen(hits: dict[str, tuple[float, float]],
                      weights: dict[str, float] | None = None,
                      ) -> tuple[float, float] | None:
    """Merge the two eyes' screen intersections into one point.

    Combined on the *screen*, not by averaging the two directions. At a 60 cm
    working distance the eyes converge by several degrees, so a mean direction
    is a direction neither eye has and it points between them rather than at
    the target. Two rays that each hit the target, averaged where they land,
    hit the target.
    """
    usable = {k: v for k, v in hits.items() if v is not None}
    if not usable:
        return None
    if weights is None:
        weights = {k: 1.0 for k in usable}

    total = sum(max(weights.get(k, 0.0), 0.0) for k in usable)
    if total <= 0.0:
        pts = np.array(list(usable.values()), dtype=np.float64)
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())

    x = sum(usable[k][0] * max(weights.get(k, 0.0), 0.0) for k in usable) / total
    y = sum(usable[k][1] * max(weights.get(k, 0.0), 0.0) for k in usable) / total
    return float(x), float(y)


def forward_batch_rays(rotations: np.ndarray, translations: np.ndarray,
                       ray_dirs: np.ndarray, centre_head: np.ndarray,
                       kappa_yaw: float, kappa_pitch: float, radius_m: float,
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised eye model: many frames of one eye -> (eye centres, visual axes).

    Mathematically identical to `gaze_ray`, done for a whole calibration set at
    once - the optimiser evaluates this hundreds of times and the per-sample
    Python version is far too slow.

    Having two implementations of one model is a genuine hazard: if this drifts
    from the scalar path, calibration would faithfully optimise a model that
    inference does not use, and nothing would look wrong from outside.
    `smoke.py` asserts the two agree to 1e-9, and that assertion is what makes
    keeping this safe.

    Shapes: rotations (n, 3, 3), translations (n, 3), ray_dirs (n, 3) unit
    vectors from the camera origin through the iris pixel.
    """
    centre_cam = np.einsum("nij,j->ni", rotations, centre_head) + translations

    # --- ray/sphere, with the same tangent fallback as the scalar path -------
    d = ray_dirs / np.linalg.norm(ray_dirs, axis=1, keepdims=True)
    oc = -centre_cam                                     # ray starts at origin
    b = 2.0 * np.einsum("ni,ni->n", d, oc)
    c = np.einsum("ni,ni->n", oc, oc) - radius_m ** 2
    disc = b * b - 4.0 * c

    hit = np.sqrt(np.maximum(disc, 0.0))
    t_near = (-b - hit) / 2.0
    t_far = (-b + hit) / 2.0
    t_hit = np.where(t_near > geometry.EPS, t_near, t_far)

    pupil = t_hit[:, None] * d
    # Misses: project the closest approach onto the sphere, as ray_sphere_tangent
    # does. Vectorised branchlessly so the optimiser sees a continuous function.
    t_close = np.maximum(np.einsum("ni,ni->n", centre_cam, d), geometry.EPS)
    closest = t_close[:, None] * d
    tangent = centre_cam + radius_m * geometry.normalize(closest - centre_cam)
    missed = (disc < 0.0) | (t_hit <= geometry.EPS)
    pupil = np.where(missed[:, None], tangent, pupil)

    optical = geometry.normalize(pupil - centre_cam)

    # --- kappa, applied in the head frame -----------------------------------
    optical_head = np.einsum("nji,nj->ni", rotations, optical)   # R.T @ optical
    pitch = np.arcsin(np.clip(optical_head[:, 1], -1.0, 1.0)) + kappa_pitch
    yaw = np.arctan2(optical_head[:, 0], optical_head[:, 2]) + kappa_yaw
    visual_head = np.stack([np.sin(yaw) * np.cos(pitch),
                            np.sin(pitch),
                            np.cos(yaw) * np.cos(pitch)], axis=1)
    visual = np.einsum("nij,nj->ni", rotations, visual_head)
    return centre_cam, visual


def forward_batch(rotations: np.ndarray, translations: np.ndarray,
                  ray_dirs: np.ndarray, centre_head: np.ndarray,
                  kappa_yaw: float, kappa_pitch: float, radius_m: float,
                  screen: geometry.ScreenPlane) -> np.ndarray:
    """`forward_batch_rays` plus the screen intersection. Returns (n, 2) pixels.

    Rows whose ray never meets the plane come back as NaN. Used for reporting
    and for fitting the residual corrector - **not** for the optimiser's
    residual, which works in angle instead. See `fitting.residuals` for why.
    """
    centre_cam, visual = forward_batch_rays(
        rotations, translations, ray_dirs, centre_head, kappa_yaw, kappa_pitch,
        radius_m)

    normal = screen.normal
    denom = visual @ normal
    numer = (screen.origin - centre_cam) @ normal
    with np.errstate(divide="ignore", invalid="ignore"):
        t_plane = numer / denom
    bad = ~np.isfinite(t_plane) | (np.abs(denom) < 1e-9) | (t_plane <= 0.0)

    land = centre_cam + t_plane[:, None] * visual
    rel = land - screen.origin
    px = np.stack([(rel @ screen.right) / screen.width_m * screen.width_px,
                   (rel @ screen.down) / screen.height_m * screen.height_px], axis=1)
    px[bad] = np.nan
    return px


def eye_weight(ray: GazeRay, eye_obs: landmarks.Eye,
               min_iris_px: float = 4.0) -> float:
    """How much to trust one eye this frame.

    Two things degrade an eye independently, so both are folded in:

    * **Iris size in pixels.** The pupil centre is quantised by it, so a small
      iris is a hard precision limit no model can beat. This is also the term
      that will fall off when the user turns away, or when one eye is further
      from the camera than the other.
    * **Eccentricity.** The sphere model is good near the centre of gaze and
      degrades past roughly 25 degrees, where the cornea stops being spherical.

    Down-weighting rather than rejecting matters for this population: with
    facial asymmetry or partial ptosis one eye may be permanently the weaker
    signal, and a hard cutoff would then throw away half the information on
    every single frame.
    """
    iris = eye_obs.iris_radius_px
    if iris < min_iris_px:
        return 0.0
    size_term = min(1.0, iris / 12.0)
    ecc_term = float(np.clip(1.0 - (ray.eccentricity_deg - 20.0) / 20.0, 0.15, 1.0))
    return float(size_term * ecc_term)
