"""The data a frame source hands over: intrinsics and a colour+depth pair.

Pure data and pure numpy, with no pyrealsense2 import anywhere, so the entire
geometric model can be exercised - and asserted against closed-form answers -
on a machine with no SDK and no camera. Given that the camera for this project
is not in hand yet, that property is doing real work rather than being tidy for
its own sake.

Depth is metres everywhere below this line. The uint16-to-metres conversion
happens once, in `source.py`, using the scale the device reports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Distortion models, as librealsense names them. Only the two that D400 colour
# and depth streams actually use are handled; anything else falls back to a
# plain pinhole, which is what an all-zero coefficient vector means anyway.
DIST_NONE = "none"
DIST_BROWN = "brown_conrady"
DIST_INVERSE_BROWN = "inverse_brown_conrady"


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics plus a distortion model, in the colour frame.

    Deprojection is reimplemented here in numpy rather than calling
    `rs2_deproject_pixel_to_point`, for two reasons: the geometry then runs
    without the SDK, and a whole array of landmarks converts in one call
    instead of sixty. `smoke` asserts this against the SDK's own
    implementation whenever a camera is attached, which is the only honest way
    to know the two agree.
    """

    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    model: str = DIST_NONE
    coeffs: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)

    @property
    def has_distortion(self) -> bool:
        return self.model != DIST_NONE and any(abs(c) > 1e-12 for c in self.coeffs)

    def normalized(self, px: np.ndarray) -> np.ndarray:
        """Pixels -> normalised image coordinates, distortion removed.

        Shape (..., 2) in, (..., 2) out. This is the ray direction through each
        pixel at unit depth, which is all the gaze model needs.
        """
        px = np.asarray(px, dtype=np.float64)
        x = (px[..., 0] - self.ppx) / self.fx
        y = (px[..., 1] - self.ppy) / self.fy

        if not self.has_distortion:
            return np.stack([x, y], axis=-1)

        # Both Brown-Conrady variants get the same iterative undistortion, in the
        # same division form and with the same ten iterations librealsense uses.
        #
        # That is not what the model names suggest, and it is not what older
        # librealsense did: "inverse" once meant the coefficients already mapped
        # distorted-to-undistorted, so one forward evaluation was the whole
        # undistortion. Current SDKs run the iterative solve for both. This was
        # settled by measurement rather than by reading the names - a single
        # forward evaluation disagreed with pyrealsense2 2.58.3 by 32 mm at 0.6 m,
        # in the opposite direction, because it expands where the real transform
        # contracts. `smoke.py` checks this against the SDK on every run, so if
        # it changes again we find out here instead of on the camera.
        c = list(self.coeffs) + [0.0] * max(0, 5 - len(self.coeffs))
        x0, y0 = x, y
        ux, uy = x.copy(), y.copy()
        for _ in range(10):
            r2 = ux * ux + uy * uy
            icd = 1.0 / (1.0 + ((c[4] * r2 + c[1]) * r2 + c[0]) * r2)
            dx = 2.0 * c[2] * ux * uy + c[3] * (r2 + 2.0 * ux * ux)
            dy = 2.0 * c[3] * ux * uy + c[2] * (r2 + 2.0 * uy * uy)
            ux = (x0 - dx) * icd
            uy = (y0 - dy) * icd
        return np.stack([ux, uy], axis=-1)

    def ray(self, px: np.ndarray) -> np.ndarray:
        """Unit direction(s) in the camera frame through pixel(s) `px`."""
        n = self.normalized(px)
        v = np.concatenate([n, np.ones(n.shape[:-1] + (1,))], axis=-1)
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    def deproject(self, px: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
        """Pixel(s) plus depth in metres -> 3D point(s) in the camera frame.

        `depth_m` is depth along +z, which is what a RealSense reports - not
        range along the ray. Hence scaling the normalised coordinates rather
        than the unit ray; confusing the two costs a few millimetres at the
        edge of the frame.
        """
        n = self.normalized(px)
        z = np.asarray(depth_m, dtype=np.float64)[..., None]
        return np.concatenate([n * z, z], axis=-1)

    def project(self, points: np.ndarray) -> np.ndarray:
        """3D camera-frame point(s) -> pixel(s). Distortion is not re-applied.

        Used to draw overlays and to build synthetic test observations, so the
        pinhole approximation is deliberate: it keeps `project` an exact
        inverse of `deproject` in the undistorted case the tests rely on.
        """
        p = np.asarray(points, dtype=np.float64)
        z = np.maximum(p[..., 2], 1e-6)
        return np.stack([p[..., 0] / z * self.fx + self.ppx,
                         p[..., 1] / z * self.fy + self.ppy], axis=-1)

    def scaled_to(self, width: int, height: int) -> "Intrinsics":
        """Same optics, different sampling grid."""
        sx, sy = width / self.width, height / self.height
        return Intrinsics(width, height, self.fx * sx, self.fy * sy,
                          self.ppx * sx, self.ppy * sy, self.model, self.coeffs)

    def hfov_deg(self) -> float:
        return float(2.0 * math.degrees(math.atan(self.width / (2.0 * self.fx))))

    def px_per_mm_at(self, distance_m: float) -> float:
        """Image pixels per millimetre of object, at `distance_m`.

        The number that actually sets the precision ceiling: an iris is about
        12 mm across, and how many pixels land on it is what the gaze estimate
        is made of. Printed by `preview` so the framing can be judged directly
        instead of by eye.
        """
        return float(self.fx / max(distance_m, 1e-3) / 1000.0)

    def as_dict(self) -> dict:
        return {
            "width": self.width, "height": self.height,
            "fx": self.fx, "fy": self.fy, "ppx": self.ppx, "ppy": self.ppy,
            "model": self.model, "coeffs": list(self.coeffs),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Intrinsics":
        return cls(int(d["width"]), int(d["height"]),
                   float(d["fx"]), float(d["fy"]),
                   float(d["ppx"]), float(d["ppy"]),
                   str(d.get("model", DIST_NONE)),
                   tuple(float(c) for c in d.get("coeffs", ())))

    @classmethod
    def from_rs(cls, rs_intrin) -> "Intrinsics":
        name = str(rs_intrin.model).split(".")[-1].lower()
        if "inverse_brown" in name:
            model = DIST_INVERSE_BROWN
        elif "brown" in name:
            model = DIST_BROWN
        else:
            model = DIST_NONE
        return cls(
            width=int(rs_intrin.width), height=int(rs_intrin.height),
            fx=float(rs_intrin.fx), fy=float(rs_intrin.fy),
            ppx=float(rs_intrin.ppx), ppy=float(rs_intrin.ppy),
            model=model, coeffs=tuple(float(c) for c in rs_intrin.coeffs),
        )

    @classmethod
    def synthetic(cls, width: int = 1280, height: int = 720,
                  hfov_deg: float = 69.4) -> "Intrinsics":
        """Plausible D435i colour intrinsics, for tests and for `--synthetic`.

        69.4 degrees is the D435i's nominal colour horizontal field of view.
        Square pixels and a centred principal point are close enough to real
        for anything that does not need the actual factory calibration.
        """
        fx = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
        return cls(width, height, fx, fx, width / 2.0, height / 2.0)


@dataclass
class Frame:
    """One synchronised colour + depth pair, ready for the gaze model."""

    color: np.ndarray
    """BGR uint8, (h, w, 3). BGR because the stream is requested as bgr8, so
    OpenCV needs no conversion and MediaPipe needs exactly one."""

    depth_m: np.ndarray
    """float32 (h, w), metres, aligned to `color`. Zero means "no depth here" -
    a real and frequent condition on hair, eyebrows and specular skin - and
    every consumer must treat it as missing rather than as "at the lens"."""

    intrinsics: Intrinsics
    """Of the colour stream, which after alignment is the frame both images
    live in."""

    timestamp: float = 0.0
    frame_number: int = 0

    accel: np.ndarray | None = None
    """Latest accelerometer sample, rotated into the colour frame. None on a
    D435 without the IMU, or before the first motion frame arrives."""

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.color.shape[:2]
        return w, h

    @property
    def has_depth(self) -> bool:
        return self.depth_m is not None and bool(np.any(self.depth_m > 0.0))

    def depth_fill_rate(self) -> float:
        """Fraction of pixels carrying a depth value. Diagnostic, not per-frame."""
        if self.depth_m is None or self.depth_m.size == 0:
            return 0.0
        return float(np.count_nonzero(self.depth_m) / self.depth_m.size)
