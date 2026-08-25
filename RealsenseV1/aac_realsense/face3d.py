"""Metric 3D head tracking: the thing a webcam cannot do.

The core idea, and the reason this rewrite exists:

A webcam gives landmarks with no scale, so an appearance model has to learn the
pixel-to-screen mapping for one particular head position and silently breaks
when the head leaves it. The EyeTrax runs in this repo show exactly that -
error near the screen centre around 60-90 px, but 405 px in the top-left corner
in the same session, which is bias, not noise, and no amount of smoothing
touches it.

Depth removes the ambiguity. Two stages:

1. **Build a personalised metric skull template** from depth, once, during
   calibration. Depth's job is scale.
2. **Track pose against that template** every frame - by Kabsch when depth is
   healthy, by `solvePnP` when it is not. Because the template is already
   metric, PnP recovers true metric translation too, so depth dropout on a
   given frame costs accuracy rather than causing failure. Depth on faces drops
   out constantly - hair, eyebrows, specular skin, the eye itself - so this
   fallback is the normal path, not an edge case.

The head frame is chosen to be the identity when the subject faces the camera
squarely: +x towards the subject's left (image right), +y down, +z into the back
of the head. Debuggability is worth more here than elegance, because a wrong
sign in this frame produces gaze that is plausibly wrong rather than obviously
broken.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import geometry, landmarks
from .frames import Frame, Intrinsics

MIN_RIGID_POINTS = 8
"""Fewest depth-valid rigid landmarks for a Kabsch fit to be trustworthy. Below
this the fit is dominated by whichever few points survived, which correlates
with head pose and so produces a *biased* estimate, not just a noisy one."""

FACE_DEPTH_BAND_M = 0.10
"""How far a landmark's depth may sit from the face's own depth before it is
rejected. Catches the real failure of aligned depth on a face: at the silhouette
- temples, jaw edge, the side of the nose - a landmark pixel straddles the
boundary and picks up the wall two metres behind, which as a 3D point drags the
whole rigid fit around.

100 mm, not the 60 mm this started at. A head is genuinely about 70 mm deep from
nose tip to temple, and more than that once it turns, so a 60 mm band rejected
real temple points - measurably, at 18 degrees of yaw. Those are the worst ones
to lose: the temples are the widest-spaced points in the set and therefore the
ones that pin down yaw, so rejecting them biases the pose exactly when the head
is turned. Background is normally 300 mm or more behind a seated user, so 100 mm
still rejects everything this is meant to reject."""


# ------------------------------------------------------------- depth sampling


def sample_depth(depth_m: np.ndarray, px: np.ndarray, radius: int = 2,
                 reference_m: float | None = None,
                 band_m: float = FACE_DEPTH_BAND_M) -> np.ndarray:
    """Robust depth at each pixel in `px`. Zero where no usable depth exists.

    Takes the median of the valid values in a small window rather than the
    single pixel, because aligned depth is speckled with holes and a lone
    pixel is as likely to be a hole as not. The median also survives a window
    that straddles a depth edge, which the mean does not.

    When `reference_m` is given, values further than `band_m` from it are
    discarded first - see FACE_DEPTH_BAND_M.
    """
    px = np.atleast_2d(np.asarray(px, dtype=np.float64))
    h, w = depth_m.shape[:2]
    out = np.zeros(len(px), dtype=np.float64)

    for i, (x, y) in enumerate(px):
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = max(0, xi - radius), min(w, xi + radius + 1)
        y0, y1 = max(0, yi - radius), min(h, yi + radius + 1)
        if x0 >= x1 or y0 >= y1:
            continue

        patch = depth_m[y0:y1, x0:x1].reshape(-1)
        valid = patch[patch > 0.0]
        if reference_m is not None and valid.size:
            valid = valid[np.abs(valid - reference_m) <= band_m]
        if valid.size:
            out[i] = float(np.median(valid))
    return out


def face_depth(depth_m: np.ndarray, obs: landmarks.FaceObservation) -> float:
    """One robust depth for the whole face, used as the rejection reference.

    Sampled over the nose ridge and eye corners: they sit in the middle of the
    face, face the camera almost head-on, and are surrounded by more face - so
    a window there cannot straddle the silhouette.
    """
    anchors = obs.landmarks_px[[168, 6, 197, 195, 5, 4, 33, 133, 362, 263]]
    got = sample_depth(depth_m, anchors, radius=3)
    valid = got[got > 0.0]
    return float(np.median(valid)) if valid.size else 0.0


def rigid_points(frame: Frame, obs: landmarks.FaceObservation,
                 indices: tuple[int, ...] = landmarks.RIGID_INDICES,
                 ) -> tuple[np.ndarray | None, np.ndarray]:
    """Deproject the rigid landmarks of one frame. (points_cam, valid_mask).

    The single place depth becomes 3D, so the rejection band and the patch
    radius are applied identically during calibration and at run time. Points
    without depth come back filled with the face's own depth and flagged
    invalid: keeping the array rectangular lets everything downstream index by
    landmark, while the mask is what actually decides who is trusted.
    """
    ref = face_depth(frame.depth_m, obs)
    if ref <= 0.0:
        return None, np.zeros(len(indices), dtype=bool)

    px = obs.landmarks_px[list(indices)]
    depths = sample_depth(frame.depth_m, px, radius=2, reference_m=ref)
    valid = depths > 0.0
    pts_cam = frame.intrinsics.deproject(px, np.where(valid, depths, ref))
    return pts_cam, valid


def fit_pose(template: "HeadTemplate", pts_cam: np.ndarray,
             valid: np.ndarray) -> "Head | None":
    """Rigid pose of `template` against observed 3D points, by robust Kabsch.

    Shared by the live tracker and by calibration, so a calibration sample's
    head pose is computed exactly the way the run-time pose will be. If these
    two ever diverged, the fitted eye parameters would be tuned against poses
    that never occur in use.
    """
    if valid.sum() < MIN_RIGID_POINTS:
        return None
    src = template.points[valid]
    dst = pts_cam[valid]
    try:
        R, t, w = geometry.kabsch_robust(src, dst, template.confidence[valid])
    except ValueError:
        return None

    used = w > 0
    if used.sum() < MIN_RIGID_POINTS:
        return None
    resid = np.linalg.norm((src[used] @ R.T + t) - dst[used], axis=1)
    return Head(rotation=R, translation=t, method="kabsch",
                residual_m=float(np.sqrt(np.mean(resid ** 2))),
                points_valid=int(used.sum()),
                distance_m=float(np.linalg.norm(t)))


# ------------------------------------------------------------------- template


@dataclass
class HeadTemplate:
    """A personalised, metric, rigid model of one person's skull.

    `points` are in head-local metres, indexed the same as
    `landmarks.RIGID_INDICES`. `confidence` records how often each landmark
    contributed, so points that were mostly missing during calibration get
    down-weighted forever after rather than being trusted equally.
    """

    indices: tuple[int, ...]
    points: np.ndarray
    confidence: np.ndarray
    samples: int = 0

    @property
    def eye_span_m(self) -> float:
        """Outer corner to outer corner, a decent proxy for head size."""
        i = list(self.indices)
        try:
            a = self.points[i.index(landmarks.RIGHT_EYE_OUTER)]
            b = self.points[i.index(landmarks.LEFT_EYE_OUTER)]
        except ValueError:
            return 0.0
        return float(np.linalg.norm(b - a))

    def index_of(self, landmark_index: int) -> int:
        return list(self.indices).index(landmark_index)

    def point_of(self, landmark_index: int) -> np.ndarray:
        return self.points[self.index_of(landmark_index)]

    def as_dict(self) -> dict:
        return {
            "indices": list(self.indices),
            "points_m": self.points.tolist(),
            "confidence": self.confidence.tolist(),
            "samples": int(self.samples),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HeadTemplate":
        return cls(
            indices=tuple(int(i) for i in d["indices"]),
            points=np.asarray(d["points_m"], dtype=np.float64),
            confidence=np.asarray(d["confidence"], dtype=np.float64),
            samples=int(d.get("samples", 0)),
        )


CHEEK_INDICES = (205, 425, 116, 345)
"""Cheekbone landmarks used to define the head's "down" axis.

Chosen because they *bracket* the eye corners in protrusion rather than sitting
in front of them: on a real face 205/425 are slightly behind the eye-corner
plane and 116/345 slightly in front, so their centroid sits at almost the same
depth as the eye corners. The vector between the two centroids is therefore
nearly pure "down", with a couple of degrees of tilt instead of tens.
"""


def bootstrap_basis(points_cam: np.ndarray, indices: tuple[int, ...],
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Define a head frame from one frame's 3D points -> (origin, basis).

    Used to give the template a consistent frame to be averaged in. Once the
    template exists, pose comes from fitting it, not from this construction -
    which is the point of having a template at all, since this uses a handful of
    landmarks and would be far too noisy to drive gaze directly.

    **Why "down" comes from the cheekbones and not from the nose.** The obvious
    construction is to take the head's backward axis as
    ``eye_centre - nose_tip``. That is wrong, and wrong by a lot: on a real face
    the nose tip sits roughly 35 mm below and 30 mm in front of the eye-corner
    centroid, so that vector is tilted about ``atan2(35, 30)`` - some 49 degrees
    - away from the actual face normal. Every angle read off the resulting frame
    inherits that offset. Measured on the real camera it reported a steady
    44.9 degrees of pitch for someone sitting square to the lens, which blocked
    the placement guide outright.

    It hid for a while because the synthetic fixture placed the nose tip level
    with the eyes, specifically so this construction would return the identity.
    A test fixture built for the convenience of the code under test is a fixture
    that cannot find this class of bug; `synth.py` now uses a realistic nose and
    `smoke` asserts that a frontal head reports angles near zero.

    Note that the gaze estimate itself was never affected: a constant rotation
    between the template frame and anatomy cancels between `TemplateBuilder` and
    `fit_pose`. What it broke was every *reported* head angle, and the placement
    gate that depends on one.
    """
    idx = list(indices)
    right_outer = points_cam[idx.index(landmarks.RIGHT_EYE_OUTER)]
    left_outer = points_cam[idx.index(landmarks.LEFT_EYE_OUTER)]
    right_inner = points_cam[idx.index(landmarks.RIGHT_EYE_INNER)]
    left_inner = points_cam[idx.index(landmarks.LEFT_EYE_INNER)]

    # +x along the eye line towards the subject's left, which is image right.
    # MediaPipe 33 is the subject's *right* eye and therefore appears on the
    # left of the image; mixing these up mirrors horizontal gaze.
    x_hint = left_outer - right_outer
    eye_centre = (right_outer + left_outer + right_inner + left_inner) / 4.0

    cheeks = np.array([points_cam[idx.index(i)] for i in CHEEK_INDICES
                       if i in idx])
    if len(cheeks) >= 2:
        y_hint = cheeks.mean(axis=0) - eye_centre        # down the face
        z_hint = np.cross(x_hint, y_hint)                # into the back of the head
    else:
        # No cheekbones available: fall back to the nose, biased but defined.
        # Only reachable if the template has lost those landmarks entirely.
        z_hint = eye_centre - points_cam[idx.index(4)]

    return eye_centre, geometry.orthonormal_basis(x_hint, z_hint)


class TemplateBuilder:
    """Accumulates depth-backed 3D observations into a `HeadTemplate`.

    Each frame is expressed in its own bootstrap head frame before averaging,
    so the person may move during template capture - which they will, since
    this runs while they are also following calibration dots.
    """

    def __init__(self, indices: tuple[int, ...] = landmarks.RIGID_INDICES):
        self.indices = indices
        self._frames: list[np.ndarray] = []   # (n, 3) head-local, NaN = missing

    @property
    def count(self) -> int:
        return len(self._frames)

    def add(self, frame: Frame, obs: landmarks.FaceObservation) -> bool:
        """Fold one frame in. False if it was unusable."""
        pts_cam, valid = rigid_points(frame, obs, self.indices)
        if pts_cam is None:
            return False
        return self.add_observation(pts_cam, valid)

    def add_observation(self, pts_cam: np.ndarray, valid: np.ndarray) -> bool:
        """Fold in a set of 3D rigid points already extracted from a frame.

        Separate from `add` so calibration can keep only these - 26 points and a
        mask, a few hundred bytes - instead of the frames they came from. Holding
        300 aligned depth maps to build the template afterwards would be over a
        gigabyte, and the template genuinely does want to be built from the whole
        calibration set rather than from a separate capture step.
        """
        if valid.sum() < MIN_RIGID_POINTS:
            return False

        # The bootstrap frame needs the eye corners and the nose specifically,
        # not merely enough points in total.
        idx = list(self.indices)
        needed = (landmarks.RIGHT_EYE_OUTER, landmarks.LEFT_EYE_OUTER,
                  landmarks.RIGHT_EYE_INNER, landmarks.LEFT_EYE_INNER)
        if not all(valid[idx.index(n)] for n in needed):
            return False
        # At least two cheekbones, since they define the "down" axis now.
        if sum(valid[idx.index(i)] for i in CHEEK_INDICES if i in idx) < 2:
            return False

        origin, basis = bootstrap_basis(pts_cam, self.indices)
        local = (pts_cam - origin) @ basis        # i.e. basis.T @ v, vectorised
        self._frames.append(np.where(valid[:, None], local, np.nan))
        return True

    def build(self) -> HeadTemplate:
        """Per-landmark median across frames, ignoring the missing ones."""
        if not self._frames:
            raise ValueError(
                "no se juntó ninguna observación válida con profundidad; "
                "sin plantilla métrica no hay modelo geométrico"
            )
        stack = np.stack(self._frames)               # (frames, n, 3)
        with np.errstate(invalid="ignore"):
            points = np.nanmedian(stack, axis=0)
        seen = np.sum(~np.isnan(stack[:, :, 0]), axis=0)
        confidence = seen / float(len(self._frames))

        # A landmark never seen has a NaN median. Drop it from the template
        # rather than substituting something: a made-up template point is a
        # permanent bias in every future pose fit.
        keep = ~np.isnan(points).any(axis=1)
        if keep.sum() < MIN_RIGID_POINTS:
            raise ValueError(
                f"solo {int(keep.sum())} puntos rígidos tuvieron profundidad "
                f"válida (hacen falta {MIN_RIGID_POINTS}). Mejore la "
                f"iluminación, acerque la cara o revise el emisor IR."
            )
        return HeadTemplate(
            indices=tuple(int(i) for i, k in zip(self.indices, keep) if k),
            points=points[keep],
            confidence=confidence[keep],
            samples=len(self._frames),
        )


# ------------------------------------------------------------------- tracking


@dataclass
class Head:
    """One frame's head pose, in metres, in the camera frame."""

    rotation: np.ndarray
    """3x3, head-local -> camera."""

    translation: np.ndarray
    """Head-frame origin in camera coordinates."""

    method: str
    """"kabsch" (depth) or "pnp" (2D + metric template). Logged because the two
    have different error characteristics and a session that silently ran on PnP
    throughout is worth knowing about."""

    residual_m: float
    """RMS fit residual. The honest per-frame quality signal that the EyeTrax
    path never had - a regression always returns a point, however wrong."""

    points_valid: int
    distance_m: float

    def to_head(self, p_cam: np.ndarray) -> np.ndarray:
        return self.rotation.T @ (np.asarray(p_cam, dtype=np.float64)
                                  - self.translation)

    def to_camera(self, p_head: np.ndarray) -> np.ndarray:
        return self.rotation @ np.asarray(p_head, dtype=np.float64) + self.translation

    @property
    def basis(self) -> np.ndarray:
        """Head axes as columns, i.e. the rotation. Named for readability at
        the call sites that pass it to `geometry.apply_kappa`."""
        return self.rotation

    def euler_deg(self) -> tuple[float, float, float]:
        """(yaw, pitch, roll) in degrees, for the placement guide and logs.

        Measured directly off the head axes rather than by decomposing the
        matrix in some Euler convention, so each number means what its name
        says. The EyeTrax harness got burned by exactly this: reading ZYX
        angles off a basis whose x axis *was* the eye line reported a constant
        180-degree roll for every head, and leaked real tilt into "yaw".
        """
        x, y, z = self.rotation[:, 0], self.rotation[:, 1], self.rotation[:, 2]
        yaw = float(np.degrees(np.arctan2(z[0], z[2])))
        pitch = float(np.degrees(np.arcsin(np.clip(-z[1], -1.0, 1.0))))
        roll = float(np.degrees(np.arctan2(x[1], y[1])))
        return yaw, pitch, roll


class HeadTracker:
    """Fits a `HeadTemplate` to each frame."""

    def __init__(self, template: HeadTemplate):
        self.template = template
        self._last: Head | None = None

    def track(self, frame: Frame, obs: landmarks.FaceObservation) -> Head | None:
        pts_cam, valid = rigid_points(frame, obs, self.template.indices)

        if pts_cam is not None:
            head = fit_pose(self.template, pts_cam, valid)
            if head is not None:
                self._last = head
                return head

        # Depth was missing or too sparse. PnP still works, because the template
        # is metric - this is the normal fallback, not a failure path.
        px = obs.landmarks_px[list(self.template.indices)]
        ref = float(np.median(pts_cam[:, 2])) if pts_cam is not None else 0.0
        head = self._pnp(px, frame.intrinsics, ref)
        if head is not None:
            self._last = head
        return head

    # ------------------------------------------------------------- strategies

    def _pnp(self, px: np.ndarray, intrin: Intrinsics,
             ref_depth: float) -> Head | None:
        """Pose from 2D landmarks plus the metric template.

        Works because the template carries true scale from the depth stream, so
        PnP recovers metric translation and not just a direction. On a webcam
        the same call would need an assumed head size, and a 5 % size error is
        a 5 % distance error, which lands squarely on the gaze estimate.

        Pixels are undistorted first and a zero distortion vector is passed in,
        rather than handing OpenCV the RealSense coefficients: the two libraries
        do not agree on coefficient order for every model, and `Intrinsics`
        already implements the exact model librealsense reports.
        """
        import cv2

        undistorted = intrin.normalized(px)
        image_pts = np.column_stack([
            undistorted[:, 0] * intrin.fx + intrin.ppx,
            undistorted[:, 1] * intrin.fy + intrin.ppy,
        ]).astype(np.float64)

        camera_matrix = np.array([[intrin.fx, 0.0, intrin.ppx],
                                  [0.0, intrin.fy, intrin.ppy],
                                  [0.0, 0.0, 1.0]], dtype=np.float64)

        guess_r = guess_t = None
        use_guess = False
        if self._last is not None:
            guess_r, _ = cv2.Rodrigues(self._last.rotation)
            guess_t = self._last.translation.reshape(3, 1).copy()
            use_guess = True

        ok, rvec, tvec = cv2.solvePnP(
            self.template.points.astype(np.float64), image_pts,
            camera_matrix, np.zeros(5),
            rvec=guess_r, tvec=guess_t, useExtrinsicGuess=use_guess,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None

        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3)

        # Reprojection residual, converted to metres at the head's distance so
        # it is comparable with the Kabsch residual and one threshold serves
        # both paths.
        projected = intrin.project(self.template.points @ R.T + t)
        px_resid = float(np.sqrt(np.mean(np.sum((projected - image_pts) ** 2, axis=1))))
        depth = float(t[2]) if t[2] > 0 else (ref_depth or 0.6)
        return Head(rotation=R, translation=t, method="pnp",
                    residual_m=px_resid * depth / max(intrin.fx, 1.0),
                    points_valid=len(self.template.points),
                    distance_m=float(np.linalg.norm(t)))
