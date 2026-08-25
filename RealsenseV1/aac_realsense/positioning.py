"""Face placement guide - and a different job than the webcam version had.

In the EyeTrax harness this screen existed to put the user back into the exact
posture they calibrated in, because an appearance model is only valid near that
posture. It had to be strict, and it had to be shown again before every single
run.

A geometric model does not need that. The eyeball centre is fixed in head
coordinates and the head pose is measured metrically every frame, so leaning,
turning or slumping is handled by the model rather than forbidden by the
interface. That is the entire point of the change of camera, and it is worth
being concrete about what it means for someone with ALS: posture drifts over a
session, and it drifts *more* as the disease progresses. A system that demands a
reproducible posture is asking for something the user cannot supply.

So this screen now checks only what the **sensor** genuinely needs:

* **Depth range.** Under about 30 cm the D435's stereo pair has no overlap and
  returns nothing; past about 90 cm the iris is too few pixels to be worth
  much. Neither is a preference - both are hard sensor limits.
* **Iris pixels.** Measured directly, in pixels, because that is the physical
  quantity that sets the angular precision ceiling.
* **Depth coverage on the face.** If the emitter is off, the user is backlit by
  sunlight, or the face is beyond the emitter's reach, there is no metric scale
  and the geometric model cannot run at all.
* **Extreme head angle.** Past roughly 35 degrees of yaw one eye is occluded and
  the visible iris is a foreshortened ellipse. This is a real model limit, not
  tidiness.

Everything is reported in units the user or the operator can act on - centimetres
and degrees, not normalised landmark ratios - and the numbers come from the
depth stream rather than from a proxy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from . import face3d, landmarks, ui
from .config import Settings
from .frames import Frame

WINDOW = "CAA - colocación"

# Sensor limits, not preferences. See the module docstring.
MIN_DISTANCE_CM = 35.0
MAX_DISTANCE_CM = 85.0
IDEAL_DISTANCE_CM = 55.0
MIN_IRIS_PX = 7.0
GOOD_IRIS_PX = 11.0
MAX_YAW_DEG = 35.0
MAX_PITCH_DEG = 28.0
MIN_DEPTH_COVERAGE = 0.45

HOLD_SECONDS = 1.0
"""How long everything must stay in range before continuing. Short, because
this screen is a sensor check rather than a posture contract."""

MEDIAN_WINDOW = 7
"""Frames in the temporal median. Without it the warnings flicker line to line
and become unreadable, which is worse than being slightly stale."""


@dataclass
class Placement:
    """One frame's measurement of where the head is, in real units."""

    distance_cm: float
    iris_px: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    depth_coverage: float
    centre_offset: tuple[float, float]
    """Face centre relative to the frame centre, as a fraction of frame size."""

    def problems(self) -> list[tuple[str, str]]:
        """Everything currently wrong, worst first, as (message, severity).

        Ordered by how much it costs, so the top line is always the thing most
        worth acting on. Showing four simultaneous complaints teaches people to
        ignore the whole panel.
        """
        out: list[tuple[str, str]] = []

        if self.depth_coverage < MIN_DEPTH_COVERAGE:
            out.append((
                "NO HAY PROFUNDIDAD SOBRE LA CARA",
                "El emisor infrarrojo no llega o hay demasiada luz solar. "
                "Sin profundidad no hay escala métrica.",
            ))
        if self.distance_cm < MIN_DISTANCE_CM:
            out.append(("ALÉJESE UN POCO",
                        f"{self.distance_cm:.0f} cm; por debajo de "
                        f"{MIN_DISTANCE_CM:.0f} cm la cámara no mide profundidad."))
        elif self.distance_cm > MAX_DISTANCE_CM:
            out.append(("ACÉRQUESE A LA CÁMARA",
                        f"{self.distance_cm:.0f} cm; el iris queda con muy "
                        f"pocos píxeles."))
        if abs(self.yaw_deg) > MAX_YAW_DEG:
            out.append(("GIRE LA CABEZA HACIA LA CÁMARA",
                        f"{self.yaw_deg:+.0f}° de giro; un ojo queda tapado."))
        if abs(self.pitch_deg) > MAX_PITCH_DEG:
            out.append(("MIRE MÁS DE FRENTE",
                        f"{self.pitch_deg:+.0f}° de inclinación."))
        if self.iris_px < MIN_IRIS_PX:
            out.append(("EL IRIS SE VE MUY CHICO",
                        f"{self.iris_px:.1f} px de radio. Acérquese, suba la "
                        f"resolución de color o mejore la luz."))
        return out

    @property
    def ok(self) -> bool:
        return not self.problems()

    def quality(self) -> float:
        """0..1 summary, for the on-screen bar. Not used for any decision."""
        d = 1.0 - min(1.0, abs(self.distance_cm - IDEAL_DISTANCE_CM) / 30.0)
        i = min(1.0, self.iris_px / GOOD_IRIS_PX)
        a = 1.0 - min(1.0, max(abs(self.yaw_deg) / MAX_YAW_DEG,
                               abs(self.pitch_deg) / MAX_PITCH_DEG))
        return float(np.clip(d * 0.3 + i * 0.4 + a * 0.15
                             + min(1.0, self.depth_coverage / 0.8) * 0.15, 0.0, 1.0))

    def as_dict(self) -> dict:
        return {
            "distance_cm": round(self.distance_cm, 1),
            "iris_px": round(self.iris_px, 2),
            "yaw_deg": round(self.yaw_deg, 1),
            "pitch_deg": round(self.pitch_deg, 1),
            "roll_deg": round(self.roll_deg, 1),
            "depth_coverage": round(self.depth_coverage, 3),
        }


def measure(frame: Frame, obs: landmarks.FaceObservation,
            head: face3d.Head | None = None) -> Placement | None:
    """Measure placement from one frame.

    Works before a template exists - during the very first calibration there is
    nothing to fit - so head angles fall back to the bootstrap construction from
    the eye corners and nose. That is noisier than a template fit but it is only
    driving a coarse "are you roughly facing the camera" check.
    """
    ref = face3d.face_depth(frame.depth_m, obs)
    if ref <= 0.0:
        return None

    w, h = frame.size
    iris_px = float(np.mean([obs.eyes[k].iris_radius_px
                             for k in obs.eyes])) if obs.eyes else 0.0

    # Depth coverage over the rigid landmarks: the honest measure of whether the
    # geometric model has anything to work with on *this* face right now.
    rigid_px = obs.rigid_px()
    depths = face3d.sample_depth(frame.depth_m, rigid_px, radius=2,
                                 reference_m=ref)
    coverage = float(np.count_nonzero(depths) / max(len(depths), 1))

    if head is not None:
        yaw, pitch, roll = head.euler_deg()
        distance_cm = head.distance_m * 100.0
    else:
        yaw, pitch, roll = _angles_without_template(frame, obs, ref)
        distance_cm = ref * 100.0

    eye_mid = (obs.landmarks_px[landmarks.RIGHT_EYE_OUTER]
               + obs.landmarks_px[landmarks.LEFT_EYE_OUTER]) / 2.0
    centre_offset = (float(eye_mid[0] / w - 0.5), float(eye_mid[1] / h - 0.5))

    return Placement(distance_cm=distance_cm, iris_px=iris_px,
                     yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll,
                     depth_coverage=coverage, centre_offset=centre_offset)


def _angles_without_template(frame: Frame, obs: landmarks.FaceObservation,
                             ref: float) -> tuple[float, float, float]:
    """Coarse head angles for use before a template exists.

    Delegates the frame construction to `face3d.bootstrap_basis` rather than
    building a second one here. The duplicate is exactly how this went wrong
    once already: two constructions of "the head frame" drifted apart, and the
    copy in this file kept a nose-derived axis that reported a steady 45 degrees
    of pitch for a head sitting square to the camera. One definition, one bias,
    one place to fix it.

    Measured in 3D from depth rather than from image-space ratios. The webcam
    harness had to use ratios and documented the trap it fell into: reading
    Euler angles off a basis whose x axis *is* the eye line reports a constant
    180-degree roll for every head and leaks real tilt into "yaw". With metric
    points the angles are directly measurable and each one means what its name
    says.
    """
    pts_cam, valid = face3d.rigid_points(frame, obs)
    if pts_cam is None:
        return 0.0, 0.0, 0.0
    # Missing points already carry the face depth as a stand-in, which is close
    # enough for a coarse "are you facing the camera" check.
    _, basis = face3d.bootstrap_basis(pts_cam, landmarks.RIGID_INDICES)
    x, y, z = basis[:, 0], basis[:, 1], basis[:, 2]
    yaw = float(np.degrees(np.arctan2(z[0], z[2])))
    pitch = float(np.degrees(np.arcsin(np.clip(-z[1], -1.0, 1.0))))
    roll = float(np.degrees(np.arctan2(x[1], y[1])))
    return yaw, pitch, roll


class _Median:
    """Per-field temporal median over a short window."""

    def __init__(self, window: int = MEDIAN_WINDOW):
        self.window = window
        self._rows: list[list[float]] = []

    def push(self, p: Placement) -> Placement:
        row = [p.distance_cm, p.iris_px, p.yaw_deg, p.pitch_deg, p.roll_deg,
               p.depth_coverage, p.centre_offset[0], p.centre_offset[1]]
        self._rows.append(row)
        del self._rows[:-self.window]
        m = np.median(np.array(self._rows), axis=0)
        return Placement(distance_cm=m[0], iris_px=m[1], yaw_deg=m[2],
                         pitch_deg=m[3], roll_deg=m[4], depth_coverage=m[5],
                         centre_offset=(m[6], m[7]))

    def reset(self) -> None:
        self._rows.clear()


def _draw(screen_w: int, screen_h: int, frame: Frame,
          placement: Placement | None, held: float, title: str,
          subtitle: str) -> np.ndarray:
    img = ui.canvas(screen_w, screen_h)

    ui.text_center(img, title, screen_w // 2, int(screen_h * 0.07), 46,
                   ui.TEXT, bold=True)
    ui.text_center(img, subtitle, screen_w // 2, int(screen_h * 0.13), 26,
                   ui.TEXT_DIM)

    # Mirrored preview, on purpose: moving towards the guide then becomes
    # intuitive and nobody has to reason about left and right.
    preview_h = int(screen_h * 0.52)
    preview_w = int(preview_h * frame.color.shape[1] / frame.color.shape[0])
    preview = cv2.resize(cv2.flip(frame.color, 1), (preview_w, preview_h))

    px0 = (screen_w - preview_w) // 2
    py0 = int(screen_h * 0.18)
    img[py0:py0 + preview_h, px0:px0 + preview_w] = preview
    cv2.rectangle(img, (px0 - 2, py0 - 2), (px0 + preview_w, py0 + preview_h),
                  ui.TILE, 2)

    # Target oval: where a well-placed face lands.
    cx, cy = px0 + preview_w // 2, py0 + int(preview_h * 0.46)
    good = placement is not None and placement.ok
    cv2.ellipse(img, (cx, cy), (int(preview_w * 0.20), int(preview_h * 0.32)),
                0, 0, 360, ui.GOOD if good else ui.ACCENT, 3, cv2.LINE_AA)

    y = py0 + preview_h + 30
    if placement is None:
        ui.text_center(img, "NO SE DETECTA LA CARA", screen_w // 2, y, 40,
                       ui.WARN, bold=True)
        return img

    problems = placement.problems()
    if problems:
        headline, detail = problems[0]
        ui.text_center(img, headline, screen_w // 2, y, 44, ui.WARN, bold=True)
        ui.text_center(img, detail, screen_w // 2, y + 44, 24, ui.TEXT_DIM)
        if len(problems) > 1:
            ui.text_center(img, f"(y {len(problems) - 1} cosa(s) más)",
                           screen_w // 2, y + 76, 20, ui.TEXT_DIM)
    else:
        ui.text_center(img, "POSICIÓN CORRECTA - MANTÉNGASE ASÍ",
                       screen_w // 2, y, 44, ui.GOOD, bold=True)
        bar_w = int(screen_w * 0.3)
        bx = (screen_w - bar_w) // 2
        by = y + 46
        cv2.rectangle(img, (bx, by), (bx + bar_w, by + 16), ui.TILE, -1)
        cv2.rectangle(img, (bx, by),
                      (bx + int(bar_w * min(1.0, held / HOLD_SECONDS)), by + 16),
                      ui.GOOD, -1)

    # The numbers, always visible: an operator setting the device up for someone
    # else needs the measurements, not just the verdict.
    ui.text_center(
        img,
        f"distancia {placement.distance_cm:.0f} cm    "
        f"iris {placement.iris_px:.1f} px    "
        f"giro {placement.yaw_deg:+.0f}°  inclinación {placement.pitch_deg:+.0f}°  "
        f"rotación {placement.roll_deg:+.0f}°    "
        f"profundidad {placement.depth_coverage * 100:.0f}%",
        screen_w // 2, int(screen_h * 0.90), 24, ui.TEXT_DIM,
    )
    ui.text_center(img, "ESC para cancelar", screen_w // 2,
                   int(screen_h * 0.95), 22, ui.WARN)
    return img


def run(settings: Settings, tracker, src, *,
        title: str = "COLOQUE SU CARA EN LA GUÍA",
        subtitle: str = "Solo se verifica que el sensor pueda medirle bien",
        ) -> Placement | None:
    """Show the guide until placement is good and held. None if cancelled.

    Takes an already-open frame source rather than opening one, so the camera is
    started once per run instead of once per step - on a RealSense that matters,
    since start-up includes waiting out auto-exposure convergence.
    """
    tracker.open_window(WINDOW)
    median = _Median()
    good_since: float | None = None
    landmarker = tracker.landmarker
    head_tracker = (face3d.HeadTracker(tracker.model.template)
                    if getattr(tracker, "model", None) is not None else None)

    try:
        for frame in src.frames():
            obs = landmarker.detect(frame.color)
            placement = None
            if obs is not None and not obs.blink:
                head = head_tracker.track(frame, obs) if head_tracker else None
                raw = measure(frame, obs, head)
                if raw is not None:
                    placement = median.push(raw)

            now = time.time()
            if placement is not None and placement.ok:
                good_since = good_since or now
            else:
                good_since = None
            held = (now - good_since) if good_since else 0.0

            cv2.imshow(WINDOW, _draw(tracker.screen_w, tracker.screen_h, frame,
                                     placement, held, title, subtitle))
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                return None
            if held >= HOLD_SECONDS:
                return placement
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return None
    finally:
        cv2.destroyWindow(WINDOW)
        cv2.waitKey(1)
    return None
