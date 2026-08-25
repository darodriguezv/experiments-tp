"""Synthetic scenes with exact ground truth.

This module is the answer to a specific problem: the camera is not in hand, so
none of the geometry can be checked against a real measurement. What *can* be
done is stronger than a plausibility check - the forward problem has a closed
form, so a scene can be constructed from known parameters, run through the real
estimator, and compared against the exact answer.

Concretely: pick a head pose, an eye geometry and a screen pixel; compute where
the pupil must appear for the person to be looking at that pixel; hand the
resulting image coordinates to the actual pipeline; and require it to return the
pixel we started from. That closes the loop through `Intrinsics`, `face3d`,
`eyemodel`, `ScreenPlane` and `estimator` without a single mock, and it fails
loudly on a sign error that a real-camera test would only turn into
"hmm, accuracy is worse than expected".

The property worth testing hardest is head invariance, and it is testable here
in a way it is not on hardware: generate the *same* target from several
different head poses with the *same* eye parameters. A correct implementation
returns the same pixel from all of them, to floating point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import eyemodel, face3d, geometry, landmarks
from .config import CameraMount
from .frames import Frame, Intrinsics

# A synthetic head, in head-local metres. +x towards the subject's left, +y
# down, +z into the back of the head.
#
# The nose is placed where a nose actually is: below the eyes and protruding.
# An earlier version put the nose tip level with the eyes so that
# `bootstrap_basis` would return exactly the identity, which made the assertions
# tidier - and hid a 49-degree bias in that very function for as long as the
# fixture was the only thing testing it. A fixture shaped for the convenience of
# the code under test cannot find that class of bug.
SYNTH_FACE: dict[int, tuple[float, float, float]] = {
    # eye corners
    landmarks.RIGHT_EYE_OUTER: (-0.045, 0.000, 0.008),
    landmarks.RIGHT_EYE_INNER: (-0.015, 0.000, 0.014),
    landmarks.LEFT_EYE_INNER: (0.015, 0.000, 0.014),
    landmarks.LEFT_EYE_OUTER: (0.045, 0.000, 0.008),
    # nose ridge, descending and protruding, as a real one does
    168: (0.000, -0.008, 0.006),
    6: (0.000, -0.004, 0.000),
    197: (0.000, -0.002, -0.004),
    195: (0.000, 0.010, -0.014),
    5: (0.000, 0.024, -0.024),
    4: (0.000, 0.035, -0.030),
    1: (0.000, 0.044, -0.024),
    19: (0.000, 0.050, -0.018),
    # bony orbit
    226: (-0.038, -0.004, 0.012),
    446: (0.038, -0.004, 0.012),
    244: (-0.012, -0.002, 0.016),
    464: (0.012, -0.002, 0.016),
    # cheekbones
    116: (-0.052, 0.022, 0.016),
    345: (0.052, 0.022, 0.016),
    123: (-0.044, 0.030, 0.010),
    352: (0.044, 0.030, 0.010),
    205: (-0.038, 0.026, 0.004),
    425: (0.038, 0.026, 0.004),
    # temples
    227: (-0.062, -0.006, 0.038),
    447: (0.062, -0.006, 0.038),
    # sides of the nose
    129: (-0.016, 0.020, -0.016),
    358: (0.016, 0.020, -0.016),
}


def synth_template() -> face3d.HeadTemplate:
    """A `HeadTemplate` built straight from `SYNTH_FACE`, all points confident."""
    indices = tuple(i for i in landmarks.RIGID_INDICES if i in SYNTH_FACE)
    points = np.array([SYNTH_FACE[i] for i in indices], dtype=np.float64)
    return face3d.HeadTemplate(indices=indices, points=points,
                               confidence=np.ones(len(indices)), samples=1)


def synth_eyes(kappa_right_deg: float = 4.2, kappa_left_deg: float = -4.4,
               kappa_pitch_deg: float = 1.1,
               radius_m: float = 0.0121) -> eyemodel.EyeGeometry:
    """A ground-truth eye geometry, with the two kappas mirrored as in life."""
    return eyemodel.EyeGeometry(
        right=eyemodel.EyeParams(
            centre_head=np.array([-0.031, -0.001, 0.024]),
            kappa_yaw=math.radians(kappa_right_deg),
            kappa_pitch=math.radians(kappa_pitch_deg)),
        left=eyemodel.EyeParams(
            centre_head=np.array([0.031, -0.001, 0.024]),
            kappa_yaw=math.radians(kappa_left_deg),
            kappa_pitch=math.radians(kappa_pitch_deg)),
        radius_m=radius_m,
    )


def head_pose(distance_m: float = 0.58, offset: tuple[float, float] = (0.0, 0.0),
              yaw_deg: float = 0.0, pitch_deg: float = 0.0,
              roll_deg: float = 0.0) -> face3d.Head:
    """A head at a chosen place and orientation in the camera frame."""
    R = (geometry.rodrigues([0.0, 1.0, 0.0], math.radians(yaw_deg))
         @ geometry.rodrigues([1.0, 0.0, 0.0], math.radians(pitch_deg))
         @ geometry.rodrigues([0.0, 0.0, 1.0], math.radians(roll_deg)))
    t = np.array([offset[0], offset[1], distance_m], dtype=np.float64)
    return face3d.Head(rotation=R, translation=t, method="kabsch",
                       residual_m=0.0, points_valid=len(SYNTH_FACE),
                       distance_m=float(np.linalg.norm(t)))


def pupil_for_target(head: face3d.Head, eye_key: str,
                     geom: eyemodel.EyeGeometry,
                     screen: geometry.ScreenPlane,
                     target_px: tuple[float, float]) -> np.ndarray:
    """Where the pupil must be for this eye to be looking at `target_px`.

    The exact inverse of `eyemodel.gaze_ray`: take the visual axis the target
    demands, undo kappa to recover the optical axis, and step one corneal radius
    along it from the eyeball centre. Because it is an inverse rather than a
    reimplementation, a sign error in `apply_kappa` cannot cancel itself out
    here - it would break the round trip in exactly the way the test looks for.
    """
    params = geom.eye(eye_key)
    centre_cam = head.to_camera(params.centre_head)
    target_point = screen.px_to_point(*target_px)

    visual = geometry.normalize(target_point - centre_cam)
    # Undo kappa in the head frame, where it is defined.
    visual_head = head.basis.T @ visual
    yaw, pitch = geometry.direction_to_yaw_pitch(visual_head)
    optical_head = geometry.yaw_pitch_to_direction(yaw - params.kappa_yaw,
                                                   pitch - params.kappa_pitch)
    optical = head.basis @ optical_head
    return centre_cam + geom.radius_m * optical


@dataclass
class SyntheticScene:
    """A complete synthetic observation, plus the truth that generated it."""

    frame: Frame
    observation: landmarks.FaceObservation
    head: face3d.Head
    target_px: tuple[float, float]


def make_scene(head: face3d.Head, geom: eyemodel.EyeGeometry,
               screen: geometry.ScreenPlane, target_px: tuple[float, float],
               intrin: Intrinsics | None = None,
               iris_radius_px: float = 11.0,
               depth_noise_m: float = 0.0,
               rng: np.random.Generator | None = None) -> SyntheticScene:
    """Render one frame's worth of observations for a known configuration.

    `depth_noise_m` injects Gaussian noise into the depth map, which is how the
    tests probe robustness rather than only correctness: a D435 at 60 cm has
    about 1 mm of depth RMS, so passing 0.001 asks "does this still work with
    the noise the real sensor has?".
    """
    intrin = intrin or Intrinsics.synthetic()
    rng = rng or np.random.default_rng(0)

    # --- rigid landmarks, projected and given depth ---------------------------
    pts_px = np.zeros((landmarks.NUM_LANDMARKS, 2), dtype=np.float64)

    centres, zs = [], []
    for index, local in SYNTH_FACE.items():
        p_cam = head.to_camera(np.asarray(local, dtype=np.float64))
        uv = intrin.project(p_cam)
        pts_px[index] = uv
        z = float(p_cam[2])
        if depth_noise_m:
            z += float(rng.normal(0.0, depth_noise_m))
        centres.append(uv)
        zs.append(z)

    depth = _render_depth(np.array(centres), np.array(zs),
                          intrin.width, intrin.height)

    # --- irises, placed where they must be to look at the target -------------
    for key, centre_idx, ring in (
        ("right", landmarks.RIGHT_IRIS_CENTRE, landmarks.RIGHT_IRIS_RING),
        ("left", landmarks.LEFT_IRIS_CENTRE, landmarks.LEFT_IRIS_RING),
    ):
        pupil = pupil_for_target(head, key, geom, screen, target_px)
        uv = intrin.project(pupil)
        pts_px[centre_idx] = uv
        # A ring centred on the pupil, so the mean of ring-plus-centre - which
        # is what both the estimator and the fit actually use - lands exactly on
        # the pupil pixel.
        for k, ring_idx in enumerate(ring):
            angle = 2.0 * math.pi * k / len(ring)
            pts_px[ring_idx] = uv + iris_radius_px * np.array(
                [math.cos(angle), math.sin(angle)])

    # --- eyelids, so the blink test has something sane to measure ------------
    for top, bottom, inner, outer in (
        (landmarks.RIGHT_LID_TOP, landmarks.RIGHT_LID_BOTTOM,
         landmarks.RIGHT_EYE_INNER, landmarks.RIGHT_EYE_OUTER),
        (landmarks.LEFT_LID_TOP, landmarks.LEFT_LID_BOTTOM,
         landmarks.LEFT_EYE_INNER, landmarks.LEFT_EYE_OUTER),
    ):
        mid = (pts_px[inner] + pts_px[outer]) / 2.0
        span = float(np.linalg.norm(pts_px[outer] - pts_px[inner]))
        pts_px[top] = mid - np.array([0.0, span * 0.16])
        pts_px[bottom] = mid + np.array([0.0, span * 0.16])

    color = np.full((intrin.height, intrin.width, 3), 60, dtype=np.uint8)
    frame = Frame(color=color, depth_m=depth, intrinsics=intrin,
                  timestamp=0.0, frame_number=1,
                  accel=np.array([0.0, -9.81, 0.0]))

    obs = landmarks._observation(pts_px, None)
    return SyntheticScene(frame=frame, observation=obs, head=head,
                          target_px=target_px)


def _render_depth(centres: np.ndarray, zs: np.ndarray, width: int, height: int,
                  reach_px: float = 9.0) -> np.ndarray:
    """Depth map where each pixel takes the depth of its nearest landmark.

    The obvious implementation - stamp a square patch per landmark - is wrong,
    and wrong in a way that quietly corrupted this fixture until a test caught
    it. Landmarks on the nose ridge are 4-8 mm apart, which at 60 cm is 5-12 px,
    so their patches overlap and whichever is written last wins. Landmark 197
    was reporting landmark 195's depth: a systematic 26 mm error that flowed into
    the head template as 10 mm of distorted geometry and from there into a
    hopeless calibration fit.

    Nearest-landmark assignment has no such ordering dependence. It is also
    closer to what the sensor does, since a real depth map is a surface sample
    rather than a set of independent stamps - and it means the median window
    `sample_depth` takes around a landmark is dominated by that landmark's own
    depth, which is the behaviour the production code is entitled to assume.
    """
    depth = np.zeros((height, width), dtype=np.float32)
    if len(centres) == 0:
        return depth

    # Only touch the face's bounding box; the rest stays "no depth", as it would
    # be on a real frame where the background is out of the emitter's reach.
    pad = int(reach_px) + 2
    x0 = max(0, int(np.floor(centres[:, 0].min())) - pad)
    x1 = min(width, int(np.ceil(centres[:, 0].max())) + pad)
    y0 = max(0, int(np.floor(centres[:, 1].min())) - pad)
    y1 = min(height, int(np.ceil(centres[:, 1].max())) + pad)
    if x0 >= x1 or y0 >= y1:
        return depth

    ys, xs = np.mgrid[y0:y1, x0:x1]
    d2 = ((xs[..., None] - centres[:, 0]) ** 2
          + (ys[..., None] - centres[:, 1]) ** 2)
    nearest = np.argmin(d2, axis=2)
    within = np.take_along_axis(d2, nearest[..., None], axis=2)[..., 0] <= reach_px ** 2
    depth[y0:y1, x0:x1] = np.where(within, zs[nearest], 0.0).astype(np.float32)
    return depth


def default_screen(width_px: int = 1920, height_px: int = 1080,
                   ) -> geometry.ScreenPlane:
    """A plausible mount: camera on the bottom bezel of a 24-inch panel, tilted up."""
    return CameraMount(offset_x_mm=0.0, offset_y_mm=172.0, offset_z_mm=18.0,
                       pitch_deg=7.0, yaw_deg=0.0, roll_deg=0.0,
                       screen_width_mm=527.0, screen_height_mm=296.0,
                       ).screen_plane(width_px, height_px)


def scripted_landmarker(scenes: list[SyntheticScene]) -> landmarks.Landmarker:
    """A landmarker that replays a list of scenes, for end-to-end runs."""
    return landmarks.ScriptedLandmarker(
        [s.observation.landmarks_px for s in scenes])
