"""Every tunable in one place, plus the camera-to-screen mounting geometry.

The mounting block is the part with no webcam equivalent, and it is what buys
the head-pose invariance. An appearance regression maps pixels straight to
screen coordinates and therefore has to relearn the whole relationship whenever
the head moves. A geometric model intersects a metric gaze ray with the
physical plane of the display, so the head may move freely - but only if the
code knows where that plane *is*, in camera coordinates.

Three DOF of that pose are easy to measure with a tape measure (the offsets),
one is easy enough (yaw, normally zero), and two are genuinely awkward by hand
(pitch and roll of the camera relative to vertical). Those last two are the
ones the D435i's IMU measures for us from the gravity vector, which is the
single best reason to prefer the "i" model. Whatever is left over is refined
numerically during calibration, so the hand-measured numbers only need to be
close, not exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from . import geometry

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"
RECORDINGS_DIR = ROOT / "recordings"

DEFAULT_MODEL_PATH = MODELS_DIR / "gaze_model.json"

SMOOTHERS = ("none", "ema", "kalman", "kalman_ema", "median")
ESTIMATORS = ("geometric", "hybrid")
CALIBRATIONS = ("9point", "13point", "16point", "5point")


def ensure_dirs() -> None:
    for d in (MODELS_DIR, LOGS_DIR, RECORDINGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------- mount geometry


@dataclass
class CameraMount:
    """Where the camera sits relative to the display.

    All offsets are from the **centre of the active screen area** to the centre
    of the camera's colour lens, measured in the screen's own axes:

    * ``offset_x_mm`` - positive towards the screen's right edge.
    * ``offset_y_mm`` - positive **downwards**, so a camera clipped to the
      bottom bezel has a positive value and one on the top bezel a negative
      one. Down-positive matches both the screen pixel axis and the camera's
      own +y, which removes a whole class of sign mistake.
    * ``offset_z_mm`` - positive towards the user, i.e. how far the lens stands
      out in front of the glass.

    Angles describe how the camera is aimed relative to the screen's normal:

    * ``pitch_deg`` - positive aims the lens **upwards**. A camera on the
      bottom bezel is usually tilted up a little to frame a face.
    * ``yaw_deg`` - positive aims it towards the screen's right edge.
    * ``roll_deg`` - rotation about the lens axis.

    Pitch and roll are overwritten by `with_gravity` when an IMU reading is
    available, because a tape measure cannot read an angle but an accelerometer
    can.
    """

    offset_x_mm: float = 0.0
    offset_y_mm: float = 175.0
    offset_z_mm: float = 20.0

    pitch_deg: float = 8.0
    yaw_deg: float = 0.0
    roll_deg: float = 0.0

    screen_width_mm: float = 527.0
    screen_height_mm: float = 296.0
    """Physical size of the *active area*. screeninfo reports this on most
    machines (see `from_screen`); the defaults are a 24-inch 16:9 panel."""

    measured_by: str = "defaults"
    """Provenance, so a run can say whether the geometry was measured, sensed
    or guessed. A guessed mount still calibrates, just from further away."""

    # ------------------------------------------------------------------ build

    @classmethod
    def from_screen(cls, screen, **overrides) -> "CameraMount":
        """Seed the physical screen size from a `display.Screen`.

        screeninfo exposes EDID millimetres, which beats deriving size from a
        nominal diagonal: the diagonal is rounded to whole inches by every
        vendor, and a 1 % size error is a 1 % gaze error everywhere on screen.
        """
        w_mm, h_mm = getattr(screen, "width_mm", None), getattr(screen, "height_mm", None)
        if not w_mm or not h_mm:
            # Fall back to the defaults' aspect-corrected size rather than
            # inventing a diagonal, and say so in `measured_by`.
            return cls(measured_by="defaults (EDID size unavailable)", **overrides)
        return cls(screen_width_mm=float(w_mm), screen_height_mm=float(h_mm),
                   measured_by="EDID screen size", **overrides)

    def with_gravity(self, accel_cam: np.ndarray | None) -> "CameraMount":
        """Replace pitch and roll with values read off the accelerometer.

        `accel_cam` is a raw accelerometer sample expressed in the **colour**
        camera frame. Note what an accelerometer at rest actually reports: not
        gravity, but the reaction force opposing it, which points *up* in the
        world. Since the camera's +y axis points down, a level D435i reads
        ``(0, -9.8, 0)``. World-down in camera axes is therefore ``-accel``,
        and getting that sign backwards silently tilts the screen plane by
        twice the mount angle - which is why it is spelled out here.

        Assumes the screen is vertical, which holds for a monitor on a stand or
        an arm and does not hold for a laptop lid; with the screen vertical its
        own down axis *is* world-down, so one accelerometer sample fixes two of
        the three rotational DOF. Yaw about the vertical leaves gravity
        unchanged and so stays hand-measured.

        Returns self unchanged when there is no IMU or the sample looks wrong,
        so a D435 without the "i" quietly degrades to the measured numbers.
        """
        if accel_cam is None:
            return self
        a = np.asarray(accel_cam, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(a))
        # Reject anything not plausibly 1 g: the camera was being moved, or the
        # stream handed us a zero-filled frame during start-up.
        if not 8.0 <= norm <= 11.5:
            return self

        down = -a / norm                 # world "down", in camera axes

        # With R_screen<-camera = Ry(yaw) @ Rx(pitch) @ Rz(roll), world-down
        # maps into the camera frame as
        #     (sin(roll)cos(pitch),  cos(roll)cos(pitch),  -sin(pitch))
        # which inverts directly. Level and square gives down = (0, 1, 0) -> 0, 0.
        pitch = math.degrees(math.atan2(-down[2], math.hypot(down[0], down[1])))
        roll = math.degrees(math.atan2(down[0], down[1]))
        return replace(self, pitch_deg=pitch, roll_deg=roll,
                       measured_by=f"{self.measured_by} + IMU gravity")

    # -------------------------------------------------------------- geometry

    def rotation_screen_from_camera(self) -> np.ndarray:
        """R such that ``v_screen = R @ v_camera``.

        Identity means the camera looks straight out of the screen with its
        axes aligned to the pixel axes. Applied yaw-then-pitch-then-roll; at
        the few degrees these angles ever reach, the order is immaterial, but
        it is fixed here so the calibration optimiser and the forward model can
        never disagree about it.
        """
        p = math.radians(self.pitch_deg)
        y = math.radians(self.yaw_deg)
        r = math.radians(self.roll_deg)

        # Rotating the camera's +z towards screen -y (upwards) for pitch > 0.
        rx = np.array([[1.0, 0.0, 0.0],
                       [0.0, math.cos(p), -math.sin(p)],
                       [0.0, math.sin(p), math.cos(p)]])
        ry = np.array([[math.cos(y), 0.0, math.sin(y)],
                       [0.0, 1.0, 0.0],
                       [-math.sin(y), 0.0, math.cos(y)]])
        rz = np.array([[math.cos(r), -math.sin(r), 0.0],
                       [math.sin(r), math.cos(r), 0.0],
                       [0.0, 0.0, 1.0]])
        return ry @ rx @ rz

    def screen_plane(self, width_px: int, height_px: int) -> geometry.ScreenPlane:
        """Build the `ScreenPlane` this mount implies, in camera coordinates."""
        w_m = self.screen_width_mm / 1000.0
        h_m = self.screen_height_mm / 1000.0

        # Camera position expressed in the screen's frame, whose origin is the
        # top-left corner of the active area.
        cam_in_screen = np.array([
            w_m / 2.0 + self.offset_x_mm / 1000.0,
            h_m / 2.0 + self.offset_y_mm / 1000.0,
            self.offset_z_mm / 1000.0,
        ])

        r_sc = self.rotation_screen_from_camera()
        r_cs = r_sc.T                       # screen -> camera

        return geometry.ScreenPlane(
            origin=r_cs @ (-cam_in_screen),  # top-left corner, in camera coords
            right=r_cs @ np.array([1.0, 0.0, 0.0]),
            down=r_cs @ np.array([0.0, 1.0, 0.0]),
            width_m=w_m, height_m=h_m,
            width_px=int(width_px), height_px=int(height_px),
        )

    def describe(self) -> str:
        return (
            f"cámara en ({self.offset_x_mm:+.0f}, {self.offset_y_mm:+.0f}, "
            f"{self.offset_z_mm:+.0f}) mm del centro de la pantalla, "
            f"inclinación {self.pitch_deg:+.1f}° / giro {self.yaw_deg:+.1f}° / "
            f"rotación {self.roll_deg:+.1f}°; pantalla "
            f"{self.screen_width_mm:.0f}x{self.screen_height_mm:.0f} mm "
            f"[{self.measured_by}]"
        )

    def as_dict(self) -> dict:
        return {
            "offset_x_mm": self.offset_x_mm,
            "offset_y_mm": self.offset_y_mm,
            "offset_z_mm": self.offset_z_mm,
            "pitch_deg": self.pitch_deg,
            "yaw_deg": self.yaw_deg,
            "roll_deg": self.roll_deg,
            "screen_width_mm": self.screen_width_mm,
            "screen_height_mm": self.screen_height_mm,
            "measured_by": self.measured_by,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CameraMount":
        known = {k: d[k] for k in d if k in cls.__dataclass_fields__}
        return cls(**known)


# ------------------------------------------------------------------- settings


@dataclass
class Settings:
    """Everything tunable, so experiments stay reproducible."""

    # --- device / streams ---
    serial: str | None = None
    """Which RealSense to use, by serial number. None takes the first one."""

    color_size: tuple[int, int] = (1280, 720)
    """Colour stream resolution. Iris pixels are where angular precision comes
    from, so this is the first thing to raise if precision disappoints. 1920x1080
    is available at 30 fps and gives ~40 % more pixels across the iris."""

    depth_size: tuple[int, int] = (848, 480)
    """848x480 is the D435's sweet spot: it is a native sensor mode, so no
    scaling happens on-chip, and it keeps the full depth field of view. 1280x720
    trades frame rate and near-range for a modest disparity gain that the face
    does not benefit from, because a face has plenty of texture already."""

    fps: int = 30

    align_depth_to_color: bool = True
    """Reproject depth into the colour camera's frame so a landmark pixel
    indexes the matching depth pixel. Costs a few ms and removes the entire
    class of bug where the two sensors' 55 mm baseline is forgotten."""

    laser_power: float = 150.0
    """Emitter power, 0-360. The dot pattern is what makes depth work on skin,
    which is otherwise almost textureless. Only lower it if a user reports
    discomfort - it is invisible 850 nm light, but eye-safety margin matters
    when the device sits 60 cm from someone's face for hours."""

    visual_preset: str = "medium density"
    """Depth sensor visual preset, matched by name against what the camera
    reports. "high accuracy", "medium density", "default", "high density" and
    "hand" are the usual D400 options; "none" skips setting it.

    Started as "high accuracy" on the reasoning that a face at 60 cm has ample
    fill rate so confidence is what matters. Measured on the real D435 that was
    wrong: High Accuracy gave about 62 % depth coverage over the rigid landmarks
    against roughly 96 % for the lower-confidence presets.

    Coverage wins here, and not by a small margin, because of *which* points go
    missing. High Accuracy discards low-confidence pixels, and the
    low-confidence ones on a face are the peripheral ones - temples, cheek
    edges - since they sit at the silhouette. Those are exactly the
    widest-spaced points in the rigid set and therefore the ones that pin down
    head rotation. Losing them does not add noise, it adds *bias*, and the
    robust Kabsch fit can absorb noise but cannot invent a point that is not
    there. The same reasoning already forced the depth rejection band from 60 mm
    to 100 mm.

    Confirm with `accuracy`, which is the real arbiter."""

    use_imu: bool = True
    """Read the accelerometer to measure the camera's tilt. Silently skipped on
    a D435 without the IMU."""

    depth_filters: bool = True
    """Run the spatial/temporal filter chain on depth."""

    # --- gaze model ---
    estimator: str = "geometric"
    model_path: Path = DEFAULT_MODEL_PATH
    mount: CameraMount = field(default_factory=CameraMount)

    eyeball_radius_mm: float = 12.0
    """Initial guess for the corneal sphere radius. Refined per user during
    calibration inside a tight anatomical bound; the population spread is
    roughly 11-13 mm and letting it wander outside that band only lets it
    absorb error that belongs elsewhere."""

    refine_mount: bool = True
    """Let calibration adjust the mount pose. On by default because a hand
    measurement is good to maybe 5 mm and 1 degree, and 1 degree is 40 px."""

    residual_correction: bool = True
    """Fit a low-order polynomial to whatever error the geometric model leaves.
    This is what absorbs corneal refraction and landmark bias - the parts a
    sphere-and-ray model does not represent."""

    # --- smoothing ---
    smoother: str = "kalman_ema"
    ema_alpha: float = 0.25

    # --- AAC dwell selection ---
    dwell_ms: int = 800
    refractory_ms: int = 600
    blink_grace_ms: int = 250

    # --- session ---
    monitor: int | str = "primary"
    check_position: bool = True
    show_camera: bool = True
    show_raw_gaze: bool = False
    speak: bool = True
    log_session: bool = True

    playback: Path | None = None
    """Play a recorded .bag instead of opening a camera. This is how the
    pipeline is exercised - and regression-tested - without hardware."""

    record: Path | None = None
    """Record the session to a .bag. Worth doing on every real run: a recorded
    session can be replayed against a changed model, which turns "is the new
    estimator better?" into an offline question instead of another appointment
    with a participant."""

    @property
    def color_wh(self) -> tuple[int, int]:
        return self.color_size

    def validate(self) -> None:
        from . import display

        display.resolve(self.monitor)     # raises with a monitor listing if bogus
        if self.estimator not in ESTIMATORS:
            raise ValueError(f"estimator debe ser uno de {ESTIMATORS}")
        if self.smoother not in SMOOTHERS:
            raise ValueError(f"smoother debe ser uno de {SMOOTHERS}")
        if not 0.0 <= self.ema_alpha <= 1.0:
            raise ValueError("ema_alpha debe estar en [0.0, 1.0]")
        if self.dwell_ms <= 0:
            raise ValueError("dwell_ms debe ser > 0")
        if not isinstance(self.visual_preset, str):
            raise ValueError("visual_preset debe ser un nombre de preset")
        if not 0.0 <= self.laser_power <= 360.0:
            raise ValueError("laser_power debe estar en [0, 360]")
        if not 8.0 <= self.eyeball_radius_mm <= 16.0:
            raise ValueError("eyeball_radius_mm fuera del rango anatómico (8-16 mm)")
        if self.playback is not None and self.record is not None:
            # librealsense enforces this too, but its error arrives much later.
            raise ValueError("no se puede grabar y reproducir a la vez")
