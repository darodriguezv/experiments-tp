"""Face and iris landmarks, and the index sets the gaze model relies on.

MediaPipe returns 478 points: 468 on the face mesh plus 10 iris points added by
the attention-mesh refinement. The iris ones are what make webcam-class gaze
estimation possible at all, and the rigid subset below is what makes metric
head tracking possible.

Two design points worth stating, because both are load-bearing:

* **The detector is injectable.** `Landmarker` is an interface, and
  `ScriptedLandmarker` implements it by replaying known landmark arrays. That
  is how the geometry gets tested with exact ground truth and no camera - the
  synthetic image a `SyntheticSource` renders would not fool a real detector,
  and it does not have to.
* **The rigid set excludes everything that deforms.** Mouth, jaw, brows and
  eyelids all move independently of the skull, so fitting a rigid head pose to
  them means a smile shows up as a head rotation. Only the nose ridge, the eye
  corners, the cheekbones and the temples are used.
"""

from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
CACHE_DIR = Path.home() / ".cache" / "aac_realsense"

NUM_LANDMARKS = 478

# --- iris -------------------------------------------------------------------
# MediaPipe's "left" and "right" are the *subject's*, so the right eye appears
# on the left of the image. Getting this backwards mirrors the horizontal gaze,
# which calibration then partly hides - so it is spelled out rather than
# inferred.
RIGHT_IRIS_CENTRE = 468
RIGHT_IRIS_RING = (469, 470, 471, 472)
LEFT_IRIS_CENTRE = 473
LEFT_IRIS_RING = (474, 475, 476, 477)

# --- eye corners ------------------------------------------------------------
RIGHT_EYE_OUTER, RIGHT_EYE_INNER = 33, 133
LEFT_EYE_INNER, LEFT_EYE_OUTER = 362, 263

# --- eyelids, for the blink test -------------------------------------------
RIGHT_LID_TOP, RIGHT_LID_BOTTOM = 159, 145
LEFT_LID_TOP, LEFT_LID_BOTTOM = 386, 374

# --- rigid skull points, for metric head pose ------------------------------
RIGID_INDICES: tuple[int, ...] = (
    # nose ridge: the most rigid line on a face, and it faces the camera
    168, 6, 197, 195, 5, 4, 1, 19,
    # eye corners
    33, 133, 362, 263,
    # bony orbit, outside the moving eyelid
    226, 446, 244, 464,
    # cheekbones
    116, 345, 123, 352, 205, 425,
    # temples
    227, 447,
    # sides of the nose
    129, 358,
)

EYE_KEYS = ("right", "left")


@dataclass
class Eye:
    """One eye's 2D observation, in colour-image pixels."""

    iris_centre: np.ndarray
    iris_ring: np.ndarray
    inner_corner: np.ndarray
    outer_corner: np.ndarray
    lid_opening_px: float

    @property
    def iris_radius_px(self) -> float:
        """Radius from the ring, which is steadier than any single point.

        Doubles as a quality signal: an iris under about 8 px across cannot
        support a good gaze estimate no matter what the model does, since the
        pupil centre is then quantised to a sizeable fraction of a degree.
        """
        return float(np.mean(np.linalg.norm(self.iris_ring - self.iris_centre, axis=1)))

    @property
    def corner_span_px(self) -> float:
        return float(np.linalg.norm(self.outer_corner - self.inner_corner))


@dataclass
class FaceObservation:
    """Everything one frame of detection yields."""

    landmarks_px: np.ndarray
    """(478, 2) float64, in colour-image pixels. Kept in pixels rather than
    MediaPipe's normalised units because every consumer needs pixels, and
    converting once here removes a recurring source of aspect-ratio bugs."""

    eyes: dict[str, Eye]
    blink: bool
    ear: float
    """Eye aspect ratio, the blink statistic. Exposed so a caller can see how
    close to the threshold a frame sat."""

    def rigid_px(self) -> np.ndarray:
        return self.landmarks_px[list(RIGID_INDICES)]


def _ear(pts: np.ndarray) -> float:
    """Eye aspect ratio: lid opening over corner span, averaged over both eyes.

    Scale-free by construction, so it does not change when the user leans in -
    which a raw pixel opening would, making a fixed threshold useless.
    """
    ratios = []
    for top, bottom, inner, outer in (
        (RIGHT_LID_TOP, RIGHT_LID_BOTTOM, RIGHT_EYE_INNER, RIGHT_EYE_OUTER),
        (LEFT_LID_TOP, LEFT_LID_BOTTOM, LEFT_EYE_INNER, LEFT_EYE_OUTER),
    ):
        span = float(np.linalg.norm(pts[outer] - pts[inner]))
        if span < 1e-6:
            continue
        ratios.append(float(np.linalg.norm(pts[top] - pts[bottom])) / span)
    return float(np.mean(ratios)) if ratios else 0.0


class BlinkDetector:
    """Adaptive eye-aspect-ratio threshold.

    A fixed threshold cannot work across people: eye shape, eyelid droop and
    camera angle all shift the open-eye baseline, and for the population this
    device is for, reduced eyelid control is part of the condition. So the
    baseline is learned from a rolling window of recent frames and a blink is
    a *relative* collapse below it.

    Ptosis and incomplete lid closure are both expected here, which is why the
    fallback when the baseline is not yet established is "not blinking": a
    false blink stalls the dwell and is felt immediately, whereas a missed one
    costs a single frame of gaze.
    """

    def __init__(self, window: int = 50, ratio: float = 0.72):
        self.window = window
        self.ratio = ratio
        self._history: list[float] = []

    def update(self, ear: float) -> bool:
        if ear <= 0.0:
            return False
        baseline = (float(np.mean(self._history))
                    if len(self._history) >= 10 else None)
        # Only open-eye frames enter the baseline, or a long blink would drag
        # the threshold down onto itself and blinks would stop registering.
        is_blink = baseline is not None and ear < self.ratio * baseline
        if not is_blink:
            self._history.append(ear)
            if len(self._history) > self.window:
                self._history.pop(0)
        return is_blink

    def reset(self) -> None:
        self._history.clear()


class Landmarker:
    """Interface: an image in, a `FaceObservation` or None out."""

    def detect(self, bgr: np.ndarray) -> FaceObservation | None:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self) -> "Landmarker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _observation(pts_px: np.ndarray, blink_detector: BlinkDetector | None
                 ) -> FaceObservation:
    """Assemble a FaceObservation from a full (478, 2) pixel array."""
    ear = _ear(pts_px)
    blink = blink_detector.update(ear) if blink_detector else False
    eyes = {
        "right": Eye(
            iris_centre=pts_px[RIGHT_IRIS_CENTRE],
            iris_ring=pts_px[list(RIGHT_IRIS_RING)],
            inner_corner=pts_px[RIGHT_EYE_INNER],
            outer_corner=pts_px[RIGHT_EYE_OUTER],
            lid_opening_px=float(np.linalg.norm(
                pts_px[RIGHT_LID_TOP] - pts_px[RIGHT_LID_BOTTOM])),
        ),
        "left": Eye(
            iris_centre=pts_px[LEFT_IRIS_CENTRE],
            iris_ring=pts_px[list(LEFT_IRIS_RING)],
            inner_corner=pts_px[LEFT_EYE_INNER],
            outer_corner=pts_px[LEFT_EYE_OUTER],
            lid_opening_px=float(np.linalg.norm(
                pts_px[LEFT_LID_TOP] - pts_px[LEFT_LID_BOTTOM])),
        ),
    }
    return FaceObservation(landmarks_px=pts_px, eyes=eyes, blink=blink, ear=ear)


class MediaPipeLandmarker(Landmarker):
    """MediaPipe FaceLandmarker in VIDEO mode.

    VIDEO rather than IMAGE mode because it carries tracking state between
    frames, which cuts landmark jitter substantially - and jitter on the iris
    centre is jitter on the gaze point. It requires strictly increasing
    timestamps, which is handled here.

    The Tasks API is tried first and the legacy `solutions.face_mesh` is the
    fallback; between them they cover every mediapipe release likely to be
    installed, and both yield the same 478 points.
    """

    def __init__(self, model_path: Path | None = None):
        self.blink = BlinkDetector()
        self._last_ts_ms = 0
        self._legacy = None
        self._task = None

        try:
            self._init_tasks(model_path)
        except Exception as exc:
            print(f"[puntos] la API Tasks no está disponible ({exc}); "
                  f"se prueba solutions.face_mesh")
            self._init_legacy()

    # ------------------------------------------------------------ back ends

    def _init_tasks(self, model_path: Path | None) -> None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        path = Path(model_path) if model_path else download_model()
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            # The refinement that produces the iris points. Without it there
            # are 468 landmarks and no iris, and no gaze estimate is possible.
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._task = vision.FaceLandmarker.create_from_options(options)

        import mediapipe as mp
        self._mp = mp

    def _init_legacy(self) -> None:
        import mediapipe as mp

        self._legacy = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,      # the iris points
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    # ------------------------------------------------------------- detection

    def detect(self, bgr: np.ndarray) -> FaceObservation | None:
        import cv2

        h, w = bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        if self._task is not None:
            mp_image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB, data=rgb)
            ts = max(int(time.time() * 1000), self._last_ts_ms + 1)
            self._last_ts_ms = ts
            result = self._task.detect_for_video(mp_image, ts)
            if not result.face_landmarks:
                return None
            lm = result.face_landmarks[0]
        else:
            result = self._legacy.process(rgb)
            if not result.multi_face_landmarks:
                return None
            lm = result.multi_face_landmarks[0].landmark

        if len(lm) < NUM_LANDMARKS:
            # 468 means the iris refinement is off. Fail loudly: silently
            # continuing would produce a gaze estimate built on the eyelid.
            raise RuntimeError(
                f"MediaPipe devolvió {len(lm)} puntos y hacen falta "
                f"{NUM_LANDMARKS}. Falta el refinamiento del iris."
            )

        pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float64)
        return _observation(pts, self.blink)

    def close(self) -> None:
        for backend in (self._task, self._legacy):
            try:
                if backend is not None:
                    backend.close()
            except Exception:
                pass


class ScriptedLandmarker(Landmarker):
    """Replays landmark arrays supplied by the caller. The test double.

    Lets the whole chain - head pose, eye model, screen intersection, dwell,
    board - run with ground truth known exactly and no camera. `smoke` uses it
    to assert that a synthetic eye aimed at a known screen pixel comes back as
    that pixel.
    """

    def __init__(self, sequence, blink_detection: bool = False):
        self._sequence = list(sequence)
        self._i = 0
        self.blink = BlinkDetector() if blink_detection else None

    def detect(self, bgr: np.ndarray) -> FaceObservation | None:
        if not self._sequence:
            return None
        pts = self._sequence[self._i % len(self._sequence)]
        self._i += 1
        if pts is None:
            return None
        return _observation(np.asarray(pts, dtype=np.float64), self.blink)


def download_model(path: Path | None = None) -> Path:
    """Fetch `face_landmarker.task` once and cache it (~3 MB)."""
    path = Path(path) if path else CACHE_DIR / "face_landmarker.task"
    if path.exists() and path.stat().st_size > 100_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[puntos] descargando el modelo de MediaPipe -> {path}")
    tmp = path.with_suffix(".part")
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.replace(path)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"no se pudo descargar {MODEL_URL}: {exc}\n"
            f"Descárguelo a mano y páselo con --landmark-model."
        ) from exc
    return path
