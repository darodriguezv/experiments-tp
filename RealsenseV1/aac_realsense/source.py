"""Where frames come from: a live camera, a recorded .bag, or a synthetic stand-in.

Everything downstream consumes `frames.Frame` and never pyrealsense2, for two
reasons that both matter to this project specifically:

1. **The camera is not always present.** A recording replays through the exact
   same code path as a live device, so "did this change improve the estimator?"
   becomes an offline question instead of another session with a participant.
   For someone with ALS, whose good hours are limited and whose condition
   progresses, that is not a convenience - it is what makes a repeatable study
   protocol possible at all.
2. **The geometry has to be testable.** `SyntheticSource` plus an injected
   landmarker lets the full pipeline run headless, with ground truth known
   exactly, on a machine with no SDK installed.

Notes on the device settings chosen here, since several are not obvious:

* Depth is **aligned to colour**, so a landmark pixel indexes the matching
  depth pixel. Skipping this leaves the sensors' ~55 mm baseline unaccounted
  for, which lands the "depth of the iris" on the cheek behind it.
* The **IMU runs on its own pipeline**. Combining motion with depth+colour in
  one pipeline is a long-standing source of `wait_for_frames` timeouts and of
  IMU streams that stop after a few minutes, and the IMU here is only ever read
  as an occasional tilt reference - it needs no synchronisation with video.
* Auto-exposure is left **on**, but its convergence is waited out at start-up.
  Reading frames during convergence yields dark, noisy, motion-blurred iris
  pixels, and those frames would otherwise land in the calibration set.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from . import rsdevice
from .config import Settings
from .frames import Frame, Intrinsics

WARMUP_FRAMES = 30
"""Frames dropped after start-up while auto-exposure and auto-white-balance
converge. At 30 fps this is one second, which measurements on D400 modules put
comfortably past the knee of the convergence curve."""


class FrameSource:
    """Common interface: a context manager that yields `Frame`s."""

    intrinsics: Intrinsics
    label: str = "source"

    def read(self) -> Frame | None:
        """Next frame, or None when the source is exhausted."""
        raise NotImplementedError

    max_consecutive_misses = 0
    """How many `read()` failures in a row before the stream is declared over.

    Zero for a recording or the synthetic source, where a miss genuinely is the
    end of the data. A live camera overrides it, because for a camera a miss is
    almost always transient - a USB hiccup, a frameset that arrived without its
    colour half - and treating one as end-of-stream ends the whole session.
    """

    def frames(self):
        """Iterate until exhausted, skipping dropped frames.

        The skipping is the point, and it was missing: an earlier version
        returned on the first `None`, so a single dropped frame silently ended
        calibration, the board, or a preview - which is exactly what it looks
        like when nothing appears on screen at all and no error is printed.
        """
        misses = 0
        warned = False
        while True:
            frame = self.read()
            if frame is not None:
                misses = 0
                yield frame
                continue

            misses += 1
            if misses > self.max_consecutive_misses:
                if self.max_consecutive_misses:
                    print(f"[fuente] {misses} cuadros perdidos seguidos: se "
                          f"da la fuente por terminada.")
                return
            if not warned:
                # Once per run, not per frame: a stream that is dropping the
                # occasional frame is worth knowing about, and worth not
                # drowning the console over.
                print("[fuente] se perdió un cuadro; continuando.")
                warned = True

    def close(self) -> None:
        pass

    def describe(self) -> str:
        return self.label

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------- live/bag


class _RsPipeline(FrameSource):
    """Shared machinery for the live camera and .bag playback.

    Both are a librealsense pipeline with an aligner; they differ only in how
    the config is seeded and in whether device options can be set. Keeping them
    one class is what guarantees a replayed session goes through identical
    processing to the live one - if they diverged, offline comparisons would
    stop meaning anything.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.rs = rsdevice.import_rs()
        rs = self.rs

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self._configure()

        self.profile = self.pipeline.start(self.config)
        self.device = self.profile.get_device()

        self.align = rs.align(rs.stream.color) if settings.align_depth_to_color else None
        self.depth_scale = self._depth_scale()
        self._filters = self._build_filters() if settings.depth_filters else []

        self.intrinsics = self._color_intrinsics()
        self.imu = None
        self._frame_number = 0

    # ------------------------------------------------------------- subclass API

    def _configure(self) -> None:
        raise NotImplementedError

    # ---------------------------------------------------------------- internals

    def _depth_scale(self) -> float:
        try:
            return float(self.device.first_depth_sensor().get_depth_scale())
        except Exception:
            # 0.001 is the D400 default; a wrong scale would silently rescale
            # the entire head, so this fallback is announced rather than quiet.
            print("[fuente] no se pudo leer la escala de profundidad; se asume 0.001 m")
            return 0.001

    def _color_intrinsics(self) -> Intrinsics:
        rs = self.rs
        stream = self.profile.get_stream(rs.stream.color)
        return Intrinsics.from_rs(stream.as_video_stream_profile().get_intrinsics())

    def _build_filters(self) -> list:
        """Depth post-processing, in librealsense's recommended order.

        Deliberately conservative for a face at 60 cm:

        * No decimation. It is the standard first step, but it throws away
          exactly the lateral resolution the eye region has least of.
        * Spatial and temporal filtering run **in disparity space**, which is
          where the noise is actually uniform; filtering metres directly
          over-smooths far pixels and under-smooths near ones.
        * No hole filling. It invents plausible depth, and an invented depth on
          the eye socket is worse than a known gap - the gap is detectable and
          falls back to the rigid template, whereas a fabricated value quietly
          biases the fit.

        The chain runs **after** alignment, which is the prevailing pyrealsense2
        pattern and follows Intel's recommended filter order. Flagging it
        honestly: this is the one part of the pipeline that has not been
        exercised against hardware, and the disparity transform is operating on
        a resampled grid rather than the native one. If depth looks wrong on the
        real camera, `--no-depth-filters` isolates this chain in one step, and
        dropping just the two `disparity_transform` stages is the next thing to
        try.
        """
        rs = self.rs
        threshold = rs.threshold_filter()
        # Nothing useful for gaze lives outside this band: closer than 15 cm is
        # inside the D435's minimum range, past 1.5 m is the wall behind.
        threshold.set_option(rs.option.min_distance, 0.15)
        threshold.set_option(rs.option.max_distance, 1.5)

        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, 2)
        spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        spatial.set_option(rs.option.filter_smooth_delta, 20)

        temporal = rs.temporal_filter()
        temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
        temporal.set_option(rs.option.filter_smooth_delta, 20)

        return [threshold, rs.disparity_transform(True), spatial, temporal,
                rs.disparity_transform(False)]

    def _apply_filters(self, depth_frame):
        for f in self._filters:
            depth_frame = f.process(depth_frame)
        return depth_frame

    def _to_frame(self, frames) -> Frame | None:
        if self.align is not None:
            frames = self.align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame:
            return None

        color = np.asanyarray(color_frame.get_data())
        if depth_frame:
            depth_frame = self._apply_filters(depth_frame)
            raw = np.asanyarray(depth_frame.get_data())
            depth_m = raw.astype(np.float32) * np.float32(self.depth_scale)
        else:
            depth_m = np.zeros(color.shape[:2], dtype=np.float32)

        self._frame_number += 1
        return Frame(
            color=np.ascontiguousarray(color),
            depth_m=depth_m,
            intrinsics=self.intrinsics,
            timestamp=float(color_frame.get_timestamp()) / 1000.0,
            frame_number=self._frame_number,
            accel=self.imu.accel_in_color_frame() if self.imu else None,
        )

    def read(self) -> Frame | None:
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
        except RuntimeError as exc:
            # Playback raises this at end of file; live raises it on timeout,
            # which after 5 s means the camera really is gone.
            if "didn't arrive" in str(exc) or "Frame did not arrive" in str(exc):
                return None
            raise
        return self._to_frame(frames)

    def close(self) -> None:
        if self.imu is not None:
            self.imu.close()
            self.imu = None
        try:
            self.pipeline.stop()
        except Exception:
            pass


class RealSenseSource(_RsPipeline):
    """A live camera, configured for close-range face and iris work."""

    max_consecutive_misses = 90
    """Three seconds at 30 fps. Long enough to ride out a USB hiccup or a
    frameset that arrived without its colour half; short enough that an
    unplugged camera stops the run rather than hanging."""

    def __init__(self, settings: Settings, info: rsdevice.DeviceInfo | None = None):
        self.info = info or rsdevice.find(settings.serial)
        super().__init__(settings)
        self.label = f"{self.info.model} {self.info.serial}"

        self._apply_device_options()
        if self.settings.use_imu and self.info.has_imu:
            self.imu = ImuReader(self.info.serial, self.profile, self.rs)
        self._warmup()

    def _configure(self) -> None:
        rs, s = self.rs, self.settings
        self.config.enable_device(self.info.serial)

        cw, ch = self._pick_color_mode()
        self.config.enable_stream(rs.stream.color, cw, ch, rs.format.bgr8, s.fps)
        dw, dh = s.depth_size
        self.config.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, s.fps)

        if s.record is not None:
            Path(s.record).parent.mkdir(parents=True, exist_ok=True)
            self.config.enable_record_to_file(str(s.record))
            print(f"[fuente] grabando la sesión -> {s.record}")

    def _pick_color_mode(self) -> tuple[int, int]:
        """Honour the requested colour mode, or fall back with an explanation.

        Asking for an unsupported profile makes `pipeline.start` throw a
        message that names no alternative, which on USB 2 is a genuinely
        confusing failure. Checking first lets the fallback be deliberate and
        say why it happened.
        """
        w, h = self.settings.color_size
        if self.info.supports("color", w, h, self.settings.fps):
            return w, h

        best = self.info.best_color(self.settings.fps)
        if best is None:
            raise rsdevice.RealSenseUnavailable(
                f"la cámara no ofrece ningún modo de color a {self.settings.fps} fps"
            )
        print(
            f"[fuente] {w}x{h}@{self.settings.fps} no está disponible"
            f"{' (USB 2 limita el ancho de banda)' if not self.info.usb3 else ''}; "
            f"se usa {best[0]}x{best[1]}. Menos píxeles sobre el iris = menos "
            f"precisión angular."
        )
        return best

    def _apply_device_options(self) -> None:
        """Set depth options for a face at arm's length.

        Order matters: the visual preset overwrites individual depth options,
        so it goes first and the laser power after.
        """
        rs, s = self.rs, self.settings
        try:
            depth_sensor = self.device.first_depth_sensor()
        except Exception:
            return

        if s.visual_preset and s.visual_preset.lower() != "none":
            self._set_visual_preset(depth_sensor, s.visual_preset.lower())

        for option, value in ((rs.option.emitter_enabled, 1.0),
                              (rs.option.laser_power, s.laser_power)):
            try:
                if depth_sensor.supports(option):
                    depth_sensor.set_option(option, value)
            except Exception as exc:
                print(f"[fuente] no se pudo fijar {option}: {exc}")

        # Timestamps from the device clock rather than the host's, so a
        # recording's timing survives replay on a different machine.
        for sensor in self.device.query_sensors():
            try:
                if sensor.supports(rs.option.global_time_enabled):
                    sensor.set_option(rs.option.global_time_enabled, 1.0)
            except Exception:
                pass

    def _set_visual_preset(self, sensor, wanted: str) -> None:
        """Select a visual preset by name rather than by magic index.

        The numeric values of `rs2_rs400_visual_preset` have been renumbered
        across SDK versions, so the robust way is to read back each value's
        description and match the name.
        """
        rs = self.rs
        try:
            if not sensor.supports(rs.option.visual_preset):
                return
            rng = sensor.get_option_range(rs.option.visual_preset)
            for i in range(int(rng.min), int(rng.max) + 1):
                name = str(sensor.get_option_value_description(
                    rs.option.visual_preset, i))
                if wanted in name.lower():
                    sensor.set_option(rs.option.visual_preset, float(i))
                    print(f"[fuente] preset de profundidad: {name}")
                    return
            print(f"[fuente] la cámara no ofrece un preset {wanted!r}")
        except Exception as exc:
            print(f"[fuente] no se pudo fijar el preset de profundidad: {exc}")

    def _warmup(self) -> None:
        """Drop frames until auto-exposure settles.

        Without this the first second of every run is dark and noisy. That
        would not matter much for a preview, but calibration starts
        immediately after opening the camera, and those frames would go
        straight into the model as mislocated landmarks.
        """
        for _ in range(WARMUP_FRAMES):
            try:
                self.pipeline.wait_for_frames(timeout_ms=2000)
            except RuntimeError:
                break
        self._frame_number = 0


class BagSource(_RsPipeline):
    """Playback of a recorded session.

    Runs with `set_real_time(False)`, so every recorded frame is delivered even
    if processing is slower than the recording. Real-time playback drops frames
    to keep the clock, which would make an offline accuracy number depend on
    how busy the machine was - exactly the kind of irreproducibility this whole
    replay path exists to remove.
    """

    def __init__(self, settings: Settings, path: Path | None = None):
        self.path = Path(path or settings.playback)
        if not self.path.exists():
            raise FileNotFoundError(f"no existe la grabación {self.path}")
        super().__init__(settings)
        self.label = f"grabación {self.path.name}"

        try:
            playback = self.device.as_playback()
            playback.set_real_time(False)
        except Exception:
            print("[fuente] no se pudo desactivar el modo tiempo real de la "
                  "reproducción; puede perder cuadros")

    def _configure(self) -> None:
        # repeat_playback=False so `read` reports exhaustion instead of looping
        # forever, which is what an offline accuracy run needs.
        self.config.enable_device_from_file(str(self.path), repeat_playback=False)


# --------------------------------------------------------------------------- IMU


class ImuReader:
    """Accelerometer on its own pipeline, sampled in a background thread.

    Separate from the video pipeline on purpose: motion streams combined with
    depth+colour in a single pipeline are a documented source of
    `wait_for_frames` timeouts and of IMU streams that die after a few minutes.
    Nothing here needs synchronising with video - the reading is used to measure
    the camera's tilt, which changes when someone knocks the mount, not per
    frame - so a slow independent poll is strictly better.

    The samples are rotated into the **colour** frame, because that is where
    aligned depth is deprojected and therefore where the screen plane lives.
    The raw accelerometer shares axes with the *depth* sensor, and on a D435i
    the depth-to-colour rotation is near identity but the code should not
    assume that of every D400.
    """

    def __init__(self, serial: str, video_profile, rs, settle_s: float = 0.5):
        self.rs = rs
        self._lock = threading.Lock()
        self._accel: np.ndarray | None = None
        self._stop = threading.Event()
        self._rotation = self._depth_to_color_rotation(video_profile, rs)

        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.accel)

        # Starting both pipelines back to back is the reported trigger for the
        # timeouts above; a short pause reliably avoids it.
        time.sleep(settle_s)
        self.pipeline.start(cfg)

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @staticmethod
    def _depth_to_color_rotation(video_profile, rs) -> np.ndarray:
        """Rotation mapping depth-frame vectors into the colour frame."""
        try:
            depth = video_profile.get_stream(rs.stream.depth)
            color = video_profile.get_stream(rs.stream.color)
            ext = depth.get_extrinsics_to(color)
            # librealsense stores the rotation column-major.
            return np.asarray(ext.rotation, dtype=np.float64).reshape(3, 3).T
        except Exception:
            return np.eye(3)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                continue
            for f in frames:
                motion = f.as_motion_frame()
                if not motion:
                    continue
                if motion.get_profile().stream_type() != self.rs.stream.accel:
                    continue
                d = motion.get_motion_data()
                with self._lock:
                    self._accel = np.array([d.x, d.y, d.z], dtype=np.float64)
            # The tilt of a mounted camera is not a per-frame quantity; polling
            # at 200 Hz would only burn CPU that MediaPipe needs.
            time.sleep(0.05)

    def accel_in_color_frame(self) -> np.ndarray | None:
        with self._lock:
            accel = None if self._accel is None else self._accel.copy()
        return None if accel is None else self._rotation @ accel

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        try:
            self.pipeline.stop()
        except Exception:
            pass


def read_gravity(settings: Settings, samples: int = 20,
                 timeout_s: float = 3.0) -> np.ndarray | None:
    """Open the IMU briefly and average a few samples. None if unavailable.

    Averaging matters: a single accelerometer sample on an uncalibrated D435i
    carries enough noise and bias to move the derived tilt by a few tenths of a
    degree, and a tenth of a degree is four pixels on screen.
    """
    try:
        rs = rsdevice.import_rs()
        info = rsdevice.find(settings.serial)
    except rsdevice.RealSenseUnavailable:
        return None
    if not info.has_imu:
        return None

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(info.serial)
    cfg.enable_stream(rs.stream.accel)
    got: list[np.ndarray] = []
    try:
        pipeline.start(cfg)
        deadline = time.time() + timeout_s
        while len(got) < samples and time.time() < deadline:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=500)
            except RuntimeError:
                continue
            for f in frames:
                motion = f.as_motion_frame()
                if motion and motion.get_profile().stream_type() == rs.stream.accel:
                    d = motion.get_motion_data()
                    got.append(np.array([d.x, d.y, d.z], dtype=np.float64))
    except Exception:
        return None
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass

    return np.mean(got, axis=0) if got else None


# ----------------------------------------------------------------- synthetic


class SyntheticSource(FrameSource):
    """A camera-shaped object with no camera behind it.

    Renders a schematic face over a flat background at a fixed distance. It is
    not trying to fool MediaPipe - it exists so the windowing, the dwell logic,
    the board and the accuracy loop can be exercised end to end with no
    hardware, paired with `landmarks.ScriptedLandmarker` supplying the
    landmarks that a real detector would have produced.

    `head_distance_m` and the returned depth map are consistent with each
    other and with `intrinsics`, so anything computed from them is
    geometrically valid rather than merely non-crashing.
    """

    def __init__(self, settings: Settings | None = None,
                 width: int = 1280, height: int = 720,
                 head_distance_m: float = 0.6,
                 accel: np.ndarray | None = None):
        self.intrinsics = Intrinsics.synthetic(width, height)
        self.head_distance_m = float(head_distance_m)
        self.label = f"sintética {width}x{height} @ {head_distance_m:.2f} m"
        self._n = 0
        self._t0 = time.time()
        # A level, square mount: reaction force up, camera +y down.
        self.accel = np.array([0.0, -9.81, 0.0]) if accel is None else np.asarray(accel)

    def read(self) -> Frame | None:
        w, h = self.intrinsics.width, self.intrinsics.height
        color = np.full((h, w, 3), 40, dtype=np.uint8)
        depth = np.zeros((h, w), dtype=np.float32)

        # An ellipse standing in for a head, with depth to match, so anything
        # that samples depth over the face region gets a sane answer.
        cx, cy = w // 2, int(h * 0.45)
        ax, ay = int(w * 0.13), int(h * 0.26)
        yy, xx = np.mgrid[0:h, 0:w]
        face = ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2 <= 1.0
        color[face] = (150, 170, 190)
        depth[face] = np.float32(self.head_distance_m)

        self._n += 1
        return Frame(color=color, depth_m=depth, intrinsics=self.intrinsics,
                     timestamp=time.time() - self._t0, frame_number=self._n,
                     accel=self.accel.copy())


# --------------------------------------------------------------------- factory


def open_source(settings: Settings, *, synthetic: bool = False) -> FrameSource:
    """Build the source the settings ask for.

    Precedence is playback, then synthetic, then a live camera, because that is
    the order of specificity: a named recording is an explicit request, and
    `--synthetic` is an explicit "do not touch hardware".
    """
    if settings.playback is not None:
        return BagSource(settings)
    if synthetic:
        return SyntheticSource(settings, *settings.color_size)
    return RealSenseSource(settings)
