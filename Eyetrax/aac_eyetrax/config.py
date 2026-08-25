"""Central configuration for the EyeTrax AAC test harness."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from eyetrax.constants import (
    LEFT_EYE_INDICES,
    MUTUAL_INDICES,
    RIGHT_EYE_INDICES,
)

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"

DEFAULT_MODEL_PATH = MODELS_DIR / "gaze_model.pkl"

# Smoothers exposed by this harness. "kde" needs the screen size, so it is
# constructed separately in tracker.py.
SMOOTHERS = ("none", "kalman", "kalman_ema", "kde")

# Regression backends shipped with eyetrax 0.4.
GAZE_MODELS = ("ridge", "elastic_net", "linear_svr", "svr", "tiny_mlp")

CALIBRATIONS = ("9point", "5point", "lissajous")

# Length of one eyetrax feature vector: every tracked landmark as (x, y, z)
# in the head-normalised frame, plus yaw/pitch/roll. Derived rather than
# hardcoded so an eyetrax update that changes the landmark set is caught.
FEATURE_DIM = (
    len(LEFT_EYE_INDICES + RIGHT_EYE_INDICES + MUTUAL_INDICES) * 3 + 3
)


@dataclass
class Settings:
    """Everything tunable in one place, so experiments stay reproducible."""

    camera_index: int = 0

    capture_width: int = 1280
    capture_height: int = 720
    """Requested camera resolution. Gaze precision scales with pixels on the
    iris, and eyetrax leaves this at the driver default (usually 640x480).
    Set both to 0 to keep whatever the driver hands over."""

    monitor: int | str = "primary"
    """Which screen to run on: an index from `main.py monitors`, or
    "primary" / "secondary". eyetrax itself always uses get_monitors()[0],
    which is not necessarily the primary one."""

    # --- gaze regression ---
    model_name: str = "ridge"
    model_path: Path = DEFAULT_MODEL_PATH

    # --- smoothing ---
    smoother: str = "kalman_ema"
    ema_alpha: float = 0.25

    # --- AAC dwell selection ---
    dwell_ms: int = 800
    """How long the gaze must rest on a tile before it fires."""

    refractory_ms: int = 600
    """Dead time after a selection, so one long look cannot double-fire."""

    blink_grace_ms: int = 250
    """Losing the face (blink) for less than this does not reset the dwell."""

    # --- UI ---
    check_position: bool = True
    """Show the face-placement guide before running. A gaze model only holds
    for roughly the head position it was calibrated at, so this is on by
    default."""

    show_camera: bool = True
    show_raw_gaze: bool = False
    speak: bool = True
    log_session: bool = True

    @property
    def capture_size(self) -> tuple[int, int] | tuple[None, None]:
        if self.capture_width and self.capture_height:
            return self.capture_width, self.capture_height
        return None, None

    def validate(self) -> None:
        from .display import resolve

        resolve(self.monitor)  # raises with a monitor listing if bogus
        if self.model_name not in GAZE_MODELS:
            raise ValueError(f"model_name must be one of {GAZE_MODELS}")
        if self.smoother not in SMOOTHERS:
            raise ValueError(f"smoother must be one of {SMOOTHERS}")
        if not 0.0 <= self.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in [0.0, 1.0]")
        if self.dwell_ms <= 0:
            raise ValueError("dwell_ms must be > 0")
        if bool(self.capture_width) != bool(self.capture_height):
            raise ValueError(
                "set both capture_width and capture_height, or neither"
            )


def ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
