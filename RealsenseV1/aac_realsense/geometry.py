"""Pure 3D geometry. No camera, no MediaPipe, no OpenCV - just numpy.

Everything the gaze model needs sits here so it can be tested against
closed-form answers on a machine with no RealSense attached. That matters more
than usual for this module: the whole accuracy argument for moving off a webcam
rests on this math being right, and a sign error in a cross product produces a
result that looks plausible and is wrong by ten degrees.

Conventions, fixed once and relied on everywhere:

* **Camera frame** is the RealSense colour optical frame: +x right, +y down,
  +z forward (out of the lens). This is what `rs2_deproject_pixel_to_point`
  returns, so no conversion happens anywhere.
* **Units are metres.** Depth arrives as uint16 and is scaled once, at the
  source, and never again.
* Rotations are 3x3 matrices mapping **head-local -> camera**, that is,
  ``p_cam = R @ p_head + t``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-12


# --------------------------------------------------------------------- basics


def normalize(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """Unit vector(s). A zero vector is returned unchanged rather than as NaN."""
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return np.where(n < EPS, v, v / np.maximum(n, EPS))


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in degrees between two vectors."""
    a, b = normalize(np.asarray(a, float)), normalize(np.asarray(b, float))
    return float(np.degrees(np.arccos(np.clip(float(np.dot(a, b)), -1.0, 1.0))))


def rodrigues(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotation matrix from an axis-angle pair."""
    k = normalize(np.asarray(axis, dtype=np.float64))
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle_rad) * K + (1.0 - np.cos(angle_rad)) * (K @ K)


def yaw_pitch_to_direction(yaw_rad: float, pitch_rad: float) -> np.ndarray:
    """Direction from spherical angles in the camera frame.

    yaw rotates about the (downward) y axis and pitch about x, chosen so that
    (0, 0) points straight along +z - a person looking directly into the lens.
    Note +y is *down*, so a positive pitch looks downward; this is the sign
    convention the whole codebase uses, precisely because it is the one the
    camera hands us and converting would only create places to get it wrong.
    """
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    return np.array([sy * cp, sp, cy * cp], dtype=np.float64)


def direction_to_yaw_pitch(d: np.ndarray) -> tuple[float, float]:
    """Inverse of `yaw_pitch_to_direction`."""
    d = normalize(np.asarray(d, dtype=np.float64))
    pitch = float(np.arcsin(np.clip(d[1], -1.0, 1.0)))
    yaw = float(np.arctan2(d[0], d[2]))
    return yaw, pitch


def apply_kappa(direction: np.ndarray, kappa_yaw: float, kappa_pitch: float,
                basis: np.ndarray) -> np.ndarray:
    """Rotate `direction` by a small offset expressed in a local `basis`.

    Models the angle between the eye's optical axis (what geometry gives us:
    eyeball centre -> pupil) and its visual axis (where the person is actually
    looking, through the fovea, which is not at the back of the eye). That
    offset is a property of the eye and is fixed in *head* coordinates, not in
    camera coordinates, which is why the basis has to be passed in - applying
    it in the camera frame would make it drift as the head turns, and head
    invariance is the entire reason for using a depth camera.

    `basis` holds the head's x/y/z axes as columns, so it maps head -> camera.
    """
    d_head = basis.T @ normalize(direction)
    yaw, pitch = direction_to_yaw_pitch(d_head)
    return basis @ yaw_pitch_to_direction(yaw + kappa_yaw, pitch + kappa_pitch)


# ------------------------------------------------------------- intersections


def ray_sphere(origin: np.ndarray, direction: np.ndarray,
               centre: np.ndarray, radius: float) -> np.ndarray | None:
    """Nearest intersection of a ray with a sphere, or None if it misses.

    Used to turn a 2D pupil observation into a 3D point: the pupil sits on the
    corneal sphere, so the camera ray through the pupil pixel meets that sphere
    at the pupil. The *nearest* root is the one facing the camera.

    Preferred over reading the depth map at the pupil, which is unreliable
    there for real reasons: the cornea is specular, the sclera is low-texture,
    the pupil is a black hole to an IR projector, and the whole eye spans only
    a few disparity pixels.
    """
    o = np.asarray(origin, dtype=np.float64)
    d = normalize(np.asarray(direction, dtype=np.float64))
    oc = o - np.asarray(centre, dtype=np.float64)

    b = 2.0 * float(np.dot(d, oc))
    c = float(np.dot(oc, oc)) - radius * radius
    disc = b * b - 4.0 * c
    if disc < 0.0:
        return None

    root = np.sqrt(disc)
    t1, t2 = (-b - root) / 2.0, (-b + root) / 2.0
    t = t1 if t1 > EPS else t2
    if t <= EPS:
        return None
    return o + t * d


def ray_sphere_tangent(origin: np.ndarray, direction: np.ndarray,
                       centre: np.ndarray, radius: float) -> np.ndarray:
    """Like `ray_sphere`, but never fails.

    A ray that misses is projected onto the closest point of the sphere
    instead. Misses happen for ordinary reasons - a slightly wrong radius, a
    noisy landmark at an extreme gaze angle - and returning None there would
    drop frames exactly where the gaze is most eccentric, which is exactly
    where an AAC board needs its edge tiles to keep working.
    """
    hit = ray_sphere(origin, direction, centre, radius)
    if hit is not None:
        return hit

    o = np.asarray(origin, dtype=np.float64)
    d = normalize(np.asarray(direction, dtype=np.float64))
    c = np.asarray(centre, dtype=np.float64)
    t = max(float(np.dot(c - o, d)), EPS)   # closest approach along the ray
    return c + radius * normalize((o + t * d) - c)


def ray_plane(origin: np.ndarray, direction: np.ndarray,
              point: np.ndarray, normal: np.ndarray) -> np.ndarray | None:
    """Intersection of a ray with a plane, or None if parallel / behind."""
    o = np.asarray(origin, dtype=np.float64)
    d = normalize(np.asarray(direction, dtype=np.float64))
    n = normalize(np.asarray(normal, dtype=np.float64))

    denom = float(np.dot(d, n))
    if abs(denom) < 1e-9:
        return None
    t = float(np.dot(np.asarray(point, dtype=np.float64) - o, n)) / denom
    if t <= 0.0:
        return None                      # plane is behind the eye
    return o + t * d


# ------------------------------------------------------------------- fitting


def kabsch(src: np.ndarray, dst: np.ndarray,
           weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Rigid transform (R, t) minimising ||R @ src + t - dst||.

    Rotation only, no scale. The head template is already metric because it was
    built from depth, so allowing scale here would let a bad depth frame
    quietly shrink the head instead of showing up as residual.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError("kabsch needs matching point sets with at least 3 points")

    w = (np.ones(len(src)) if weights is None
         else np.asarray(weights, dtype=np.float64))
    w = np.maximum(w, 0.0)
    total = float(w.sum())
    if total < EPS:
        raise ValueError("kabsch got zero total weight")
    w = w / total

    mu_s = (w[:, None] * src).sum(axis=0)
    mu_d = (w[:, None] * dst).sum(axis=0)
    H = (w[:, None] * (src - mu_s)).T @ (dst - mu_d)

    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0.0:
        # Reflection instead of rotation - flip the least significant axis.
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    return R, mu_d - R @ mu_s


def kabsch_robust(src: np.ndarray, dst: np.ndarray,
                  weights: np.ndarray | None = None,
                  iterations: int = 3,
                  reject_sigma: float = 2.5,
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kabsch with iterative outlier rejection. Returns (R, t, weights).

    Facial landmarks fail in bursts, not independently: a hand across the chin,
    a depth hole on one cheek, a hair strand. Plain least squares spreads that
    error over the whole fit and tilts the head by a degree or two, which is a
    hundred pixels on screen.
    """
    w = (np.ones(len(src)) if weights is None
         else np.asarray(weights, dtype=np.float64).copy())

    R, t = kabsch(src, dst, w)
    for _ in range(max(0, iterations - 1)):
        resid = np.linalg.norm((src @ R.T + t) - dst, axis=1)
        active = w > 0
        if active.sum() < 4:
            break
        # MAD-based scale, so the outliers being detected cannot inflate the
        # threshold that is supposed to catch them.
        med = float(np.median(resid[active]))
        mad = float(np.median(np.abs(resid[active] - med))) * 1.4826
        keep = resid <= med + reject_sigma * max(mad, 1e-4)
        if keep.sum() < 4 or keep.all():
            break
        w = np.where(keep, w, 0.0)
        R, t = kabsch(src, dst, w)
    return R, t, w


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares plane through a point cloud -> (centroid, unit normal)."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        raise ValueError("fit_plane needs at least 3 points")
    centroid = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - centroid)
    return centroid, normalize(Vt[-1])


def orthonormal_basis(x_hint: np.ndarray, z_hint: np.ndarray) -> np.ndarray:
    """Nearest orthonormal basis to two rough axis directions, as columns.

    `x_hint` is honoured exactly - it comes from the eye line, the most
    reliable direction on a face - while `z_hint` only decides where "forward"
    is.
    """
    x = normalize(np.asarray(x_hint, dtype=np.float64))
    z = np.asarray(z_hint, dtype=np.float64)
    z = normalize(z - float(np.dot(z, x)) * x)
    if np.linalg.norm(z) < 1e-6:
        # Degenerate hint: any direction perpendicular to x will do.
        tmp = np.array([0.0, 0.0, 1.0]) if abs(x[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        z = normalize(tmp - float(np.dot(tmp, x)) * x)
    return np.column_stack([x, np.cross(z, x), z])


# --------------------------------------------------------------- screen plane


@dataclass
class ScreenPlane:
    """The physical display, in camera coordinates.

    A gaze ray becomes a pixel by intersecting this plane. Held as an origin
    plus two metric axis vectors rather than as a pose plus intrinsics, because
    that is exactly what the intersection needs and it makes the pixel mapping
    one dot product per axis.

    * `origin` - top-left corner of the active area, in metres.
    * `right`  - unit vector along the top edge, towards increasing pixel x.
    * `down`   - unit vector along the left edge, towards increasing pixel y.
    * `width_m` / `height_m` - physical size of the active area.
    """

    origin: np.ndarray
    right: np.ndarray
    down: np.ndarray
    width_m: float
    height_m: float
    width_px: int
    height_px: int

    def __post_init__(self) -> None:
        self.origin = np.asarray(self.origin, dtype=np.float64).reshape(3)
        # Re-orthonormalise: the axes arrive either from hand-measured angles
        # or from an optimiser, and neither guarantees a clean frame.
        basis = orthonormal_basis(self.right, np.cross(self.right, self.down))
        self.right = basis[:, 0]
        self.down = normalize(np.cross(basis[:, 2], basis[:, 0]))
        self.width_m = float(self.width_m)
        self.height_m = float(self.height_m)

    @property
    def normal(self) -> np.ndarray:
        """Unit normal pointing away from the screen, towards the viewer.

        `right x down`, in that order. Since `right` and `down` are the screen
        frame's own x and y carried into camera coordinates by a rotation, and
        the cross product is preserved by rotations, this is the screen frame's
        +z - which is defined as pointing at the user (`offset_z_mm` is positive
        towards them). Reversing the operands gives a normal pointing into the
        back of the panel; the ray/plane intersection would not notice, since it
        divides by the same sign twice, but `px_per_degree` and every future
        caller would.
        """
        return normalize(np.cross(self.right, self.down))

    @property
    def centre(self) -> np.ndarray:
        return (self.origin
                + self.right * (self.width_m / 2.0)
                + self.down * (self.height_m / 2.0))

    def px_per_m(self) -> tuple[float, float]:
        return self.width_px / self.width_m, self.height_px / self.height_m

    def point_to_px(self, p: np.ndarray) -> tuple[float, float]:
        """Project a 3D point onto the plane and return pixel coordinates.

        Deliberately unclamped: an off-screen result is information the caller
        needs - it means the person is not looking at the board - and clamping
        here would silently pin every stray estimate onto an edge tile.
        """
        rel = np.asarray(p, dtype=np.float64) - self.origin
        u = float(np.dot(rel, self.right)) / self.width_m * self.width_px
        v = float(np.dot(rel, self.down)) / self.height_m * self.height_px
        return u, v

    def px_to_point(self, x: float, y: float) -> np.ndarray:
        """Inverse of `point_to_px`: a screen pixel as a 3D point."""
        return (self.origin
                + self.right * (float(x) / self.width_px * self.width_m)
                + self.down * (float(y) / self.height_px * self.height_m))

    def intersect(self, origin: np.ndarray,
                  direction: np.ndarray) -> tuple[float, float] | None:
        """Where a gaze ray lands, in screen pixels."""
        hit = ray_plane(origin, direction, self.origin, self.normal)
        return None if hit is None else self.point_to_px(hit)

    def px_per_degree(self, eye: np.ndarray) -> float:
        """Screen pixels subtended by one degree of gaze, at this eye position.

        The honest conversion between the two units this project cares about.
        The webcam harness had to hardcode 40 px/deg; with a metric camera the
        real number is available every frame, so error can finally be reported
        in degrees without a fudge factor.
        """
        d = abs(float(np.dot(np.asarray(eye, dtype=np.float64) - self.origin,
                             self.normal)))
        px_x, _ = self.px_per_m()
        return float(np.tan(np.radians(1.0)) * max(d, 1e-3) * px_x)

    def as_dict(self) -> dict:
        return {
            "origin_m": [round(v, 6) for v in self.origin.tolist()],
            "right": [round(v, 6) for v in self.right.tolist()],
            "down": [round(v, 6) for v in self.down.tolist()],
            "width_m": round(self.width_m, 6),
            "height_m": round(self.height_m, 6),
            "width_px": int(self.width_px),
            "height_px": int(self.height_px),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ScreenPlane":
        return cls(
            origin=np.asarray(d["origin_m"], dtype=np.float64),
            right=np.asarray(d["right"], dtype=np.float64),
            down=np.asarray(d["down"], dtype=np.float64),
            width_m=float(d["width_m"]),
            height_m=float(d["height_m"]),
            width_px=int(d["width_px"]),
            height_px=int(d["height_px"]),
        )
