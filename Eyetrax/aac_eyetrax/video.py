"""Camera opening at an explicit resolution.

eyetrax calls `cv2.VideoCapture(index)` and never sets a capture size, so you
get the driver default - 640x480 on most built-in webcams. Gaze precision comes
almost entirely from how many pixels land on the iris, so running 720p or 1080p
instead is the cheapest accuracy win available.

Two Windows quirks this handles:

- The MSMF backend throws `_step >= minstep` if you change the frame size on a
  capture that has already delivered frames, so the size must be set on a fresh
  capture before the first read.
- Cameras silently clamp to the nearest supported mode, so the achieved size is
  always read back rather than assumed.
"""

from __future__ import annotations

from contextlib import contextmanager

import cv2


def open_camera(
    index: int = 0,
    width: int | None = None,
    height: int | None = None,
    *,
    warn: bool = True,
) -> cv2.VideoCapture:
    """Open a camera, requesting `width` x `height` if given.

    Keeps eyetrax's fallback behaviour: if camera 0 will not open, try 1.
    """
    for candidate in ([index] if index != 0 else [0, 1]):
        cap = cv2.VideoCapture(candidate)
        if not cap.isOpened():
            cap.release()
            continue

        if width and height:
            # Must happen before the first read on the MSMF backend.
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
            got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if warn and (got_w, got_h) != (int(width), int(height)):
                print(
                    f"[camera] requested {width}x{height} but the camera gave "
                    f"{got_w}x{got_h} (nearest supported mode)"
                )
        return cap

    raise RuntimeError(
        f"cannot open camera {index}"
        + (" (fallback to camera 1 also failed)" if index == 0 else "")
    )


def capture_size(cap: cv2.VideoCapture) -> tuple[int, int]:
    return (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )


@contextmanager
def camera(index: int = 0, width: int | None = None, height: int | None = None):
    cap = open_camera(index, width, height)
    try:
        yield cap
    finally:
        cap.release()


def iter_frames(cap: cv2.VideoCapture):
    """Infinite generator of frames, skipping read failures."""
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        yield frame


# eyetrax's calibrators call open_camera themselves, and each module bound the
# name at import time, so every importer needs rebinding (same pattern as
# get_screen_size in tracker.py).
_PATCH_TARGETS = (
    "eyetrax.calibration.nine_point",
    "eyetrax.calibration.five_point",
    "eyetrax.calibration.lissajous",
    "eyetrax.calibration.dense_grid",
    "eyetrax.calibration.adaptive",
    "eyetrax.filters.kalman",
)


def patch_eyetrax_camera(width: int | None, height: int | None) -> None:
    """Make eyetrax's own capture calls use our resolution."""
    import importlib

    import eyetrax.utils.video as video_mod

    def sized(index: int = 0):
        return open_camera(index, width, height)

    video_mod.open_camera = sized
    for mod_name in _PATCH_TARGETS:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        if hasattr(mod, "open_camera"):
            mod.open_camera = sized
