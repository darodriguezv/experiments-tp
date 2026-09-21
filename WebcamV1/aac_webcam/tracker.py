"""GazeFollower detrás de la interfaz de `aac_eyetrax.tracker.GazeTracker`.

Todo el arnés del proyecto -la guía de colocación, la medición de precisión, el
tablero por permanencia, el flujo completo- habla con un objeto que expone unas
diez cosas: `process`, `calibrate`, `save`, `load`, `face_pose`, `open_window`,
`reset_smoother`, y los atributos de pantalla. Este archivo provee exactamente
eso, con otro estimador adentro.

**Por qué importa que sea el mismo arnés.** La pregunta que la tesis tiene que
contestar es si GazeFollower mejora los 5.2° que dio EyeTrax. Si cada uno se
midiera con su propia rutina de precisión, su propia cuadrícula y su propio
suavizado, la diferencia sería inatribuible: no se sabría cuánto es el modelo y
cuánto el instrumento. Acá cambia una sola pieza y el resto es idéntico, línea
por línea, porque es literalmente el mismo módulo importado.

## La cadena, por cuadro

    cuadro BGR
      -> TasksFaceAlignment      recortes de cara y ojos + puntos faciales
      -> MGazeNet (MNN)          embedding, y sus dos primeros valores son
                                 la mirada CRUDA en cm respecto de la cámara
      -> SVRCalibration          embedding -> cm, aprendido de esta persona
      -> cm2px                   cm -> píxel, con la geometría física medida
      -> suavizado de Eyetrax    el mismo filtro que el otro estimador

Detalle que sorprende: una vez calibrada, la SVR predice **solo desde el
embedding** e ignora la mirada cruda. Esta la usa únicamente como respaldo
mientras no haya calibración, y por eso sin calibrar el cursor es inservible
aunque el modelo esté cargado.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT, ROOT.parent / "Eyetrax"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import gf_compat  # noqa: E402

gf_compat.patch()

from aac_eyetrax import display  # noqa: E402
from aac_eyetrax.tracker import FacePose, GazeSample, pose_from_landmarks  # noqa: E402
from tasks_alignment import TasksFaceAlignment  # noqa: E402

from .config import Settings  # noqa: E402

BLINK_RATIO = 0.35
"""Fracción de la apertura habitual por debajo de la cual se considera
parpadeo. Es una heurística **nuestra**, no de GazeFollower: su `FaceInfo` da
el área del párpado en píxeles cuadrados, que depende de la cara y de la
distancia, así que un umbral absoluto no sirve. Se compara contra una
referencia móvil de la propia persona.

Importa más de lo que parece: sin detección de parpadeo, el tablero resetea la
permanencia cada vez que la persona pestañea, y seleccionar una celda se vuelve
una prueba de aguante."""


class WebcamTracker:
    """Misma interfaz que `GazeTracker`, con GazeFollower adentro."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.settings.validate()

        self.screen = display.resolve(self.settings.monitor)
        self.screen_w, self.screen_h = self.screen.size

        gf_compat.init_logging(str(ROOT / "logs" / "gazefollower.log"))

        from gazefollower.calibration import SVRCalibration
        from gazefollower.gaze_estimator import MGazeNetGazeEstimator

        self.alignment = TasksFaceAlignment()
        self.estimator = MGazeNetGazeEstimator()
        self._retune_mnn()
        self._svr_cls = SVRCalibration
        self.calibration = self._new_calibration()

        self.smoother = self._make_smoother()
        self.trained = bool(getattr(self.calibration, "has_calibrated", False))
        self.reference_pose: FacePose | None = None

        self._last_ts = 0
        self._openness_ref = 0.0

    # ------------------------------------------------------------------ setup

    def _retune_mnn(self) -> None:
        """Rearma el modulo de MNN con otro reparto de hilos.

        GazeFollower lo fija en 4 al construirse y no lo expone. Se rehace acá
        en vez de parchear el paquete: es el mismo archivo de pesos y la misma
        API, solo cambia el `runtime_manager`. Si algo falla, se deja el suyo -
        un reparto subóptimo es molesto, quedarse sin estimador no.
        """
        import os

        n = self.settings.mnn_threads or max(2, (os.cpu_count() or 4) // 2)
        try:
            import MNN

            rt = MNN.nn.create_runtime_manager(
                ({"precision": "low", "backend": 0, "numThread": int(n)},))
            self.estimator.gaze_module = MNN.nn.load_module_from_file(
                str(self.estimator.model_path),
                ["face", "left", "right", "rect"], ["output_0"],
                runtime_manager=rt)
            self.mnn_threads = int(n)
        except Exception as exc:
            print(f"[warn] no se pudo reconfigurar MNN ({exc}); "
                  f"queda el reparto por defecto de GazeFollower")
            self.mnn_threads = 4

    def _new_calibration(self):
        """Construye la SVR, arreglando su recarga desde disco.

        **GazeFollower no puede releer su propia calibración.** Su constructor
        hace `self.svr_x.load(ruta)`, pero `load` en OpenCV es un método
        *estático*: devuelve un modelo nuevo y deja la instancia intacta.
        Verificado acá: tras `instance.load(p)` el modelo queda `isTrained()
        False` con `varCount 0`, mientras que `cv2.ml.SVM.load(p)` sí devuelve
        uno entrenado con sus 6 variables.

        Lo que lo hace peligroso y no solo molesto: acto seguido marcan
        `has_calibrated = True`. O sea que la segunda sesión y todas las
        siguientes arrancan **diciendo que están calibradas** y prediciendo con
        un modelo vacío. No hay excepción, no hay aviso, y la mirada
        simplemente no tiene sentido - que es indistinguible de "este
        estimador no sirve".

        Se corrige acá, sobre su objeto, en vez de parchear el paquete: es
        menos código y queda visible en el repositorio.
        """
        d = Path(self.settings.calibration_dir)
        d.mkdir(parents=True, exist_ok=True)
        cal = self._svr_cls(model_save_path=str(d))

        x_path, y_path = d / "svr_x.xml", d / "svr_y.xml"
        if x_path.exists() and y_path.exists():
            cal.svr_x = cv2.ml.SVM.load(str(x_path))
            cal.svr_y = cv2.ml.SVM.load(str(y_path))
            cal.has_calibrated = bool(cal.svr_x.isTrained()
                                      and cal.svr_y.isTrained())
            if not cal.has_calibrated:
                print(f"[warn] {d} tiene archivos de calibración que no "
                      f"contienen un modelo entrenado. Recalibrá.")
        return cal

    def _make_smoother(self):
        from eyetrax.filters import (KalmanEMASmoother, KalmanSmoother,
                                     KDESmoother, NoSmoother)

        name = self.settings.smoother
        if name == "none":
            return NoSmoother()
        if name == "kalman":
            return KalmanSmoother()
        if name == "kalman_ema":
            return KalmanEMASmoother(ema_alpha=self.settings.ema_alpha)
        if name == "kde":
            return KDESmoother(self.screen_w, self.screen_h)
        raise ValueError(f"suavizado desconocido {name!r}")

    def reset_smoother(self) -> None:
        self.smoother = self._make_smoother()

    def open_window(self, window: str) -> None:
        display.open_fullscreen(window, self.screen)
        if not display.placement_ok(window, self.screen):
            print(f"[warn] la ventana '{window}' puede no haber caído en el "
                  f"monitor {self.screen.index}; las coordenadas asumen "
                  f"{self.screen_w}x{self.screen_h}.")

    # ------------------------------------------------------------- por cuadro

    def _timestamp(self) -> int:
        """Estrictamente creciente: el modo VIDEO de MediaPipe lo exige."""
        ts = int(time.time() * 1000)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts
        return ts

    def cm_to_px(self, cm_x: float, cm_y: float) -> tuple[float, float]:
        """cm respecto de la cámara -> píxel de pantalla.

        Réplica de `gazefollower.misc.cm2px`, escrita acá para que la geometría
        del proyecto sea visible en un archivo del proyecto y no a tres saltos
        de import. La `y` se invierte porque el sistema en cm crece hacia
        arriba y el de píxeles hacia abajo.
        """
        w_cm, h_cm = self.settings.screen_cm
        cam_x, cam_y = self.settings.camera_cm
        px_x = (cm_x + cam_x) * self.screen_w / w_cm
        px_y = (-cm_y + cam_y) * self.screen_h / h_cm
        return px_x, px_y

    def _face_info(self, frame):
        return self.alignment.detect(self._timestamp(), frame)

    def _is_blink(self, face_info) -> bool:
        openness = min(float(face_info.left_eye_openness),
                       float(face_info.right_eye_openness))
        if openness > self._openness_ref:
            # Sube rápido y baja lento: la referencia tiene que ser "el ojo
            # bien abierto de esta persona", no un promedio que el propio
            # parpadeo arrastra hacia abajo.
            self._openness_ref = openness
        else:
            self._openness_ref *= 0.999
        if self._openness_ref <= 0.0:
            return False
        return openness < BLINK_RATIO * self._openness_ref

    def process(self, frame) -> GazeSample:
        face_info = self._face_info(frame)
        if not getattr(face_info, "status", False):
            return GazeSample(ok=False, face=False, blink=False)
        if not getattr(face_info, "can_gaze_estimation", False):
            # Cara encontrada pero un ojo toca el borde: el recorte estaría
            # incompleto y la CNN recibiría medio ojo.
            return GazeSample(ok=False, face=True, blink=False)
        if self._is_blink(face_info):
            return GazeSample(ok=False, face=True, blink=True)

        gaze_info = self.estimator.detect(frame, face_info)
        if not getattr(gaze_info, "status", False):
            return GazeSample(ok=False, face=True, blink=False)

        ok, (cm_x, cm_y) = self.calibration.predict(
            gaze_info.features, gaze_info.raw_gaze_coordinates)
        raw_x, raw_y = self.cm_to_px(float(cm_x), float(cm_y))
        sx, sy = self.smoother.step(int(raw_x), int(raw_y))
        return GazeSample(
            ok=True,
            x=int(np.clip(sx, 0, self.screen_w - 1)),
            y=int(np.clip(sy, 0, self.screen_h - 1)),
            raw_x=int(raw_x), raw_y=int(raw_y),
            blink=False, face=True,
        )

    def features(self, frame):
        """Embedding y mirada cruda en cm, o None. Lo usa la calibración."""
        face_info = self._face_info(frame)
        if not getattr(face_info, "can_gaze_estimation", False):
            return None
        gaze_info = self.estimator.detect(frame, face_info)
        if not getattr(gaze_info, "status", False):
            return None
        return np.asarray(gaze_info.features, dtype=np.float32)

    # --------------------------------------------------------------- cabeza

    def face_pose(self, frame) -> FacePose | None:
        """Posición de la cabeza, con la misma matemática que Eyetrax.

        `pose_from_landmarks` se importa en vez de reescribirse: la guía de
        colocación compara contra una pose de referencia, y si los dos módulos
        midieran el "yaw" de forma distinta la guía diría cosas distintas para
        la misma cabeza.
        """
        pts = self.alignment.landmarks_normalised(frame, self._timestamp())
        if pts is None:
            return None
        h, w = frame.shape[:2]
        return pose_from_landmarks(pts, w, h)

    # ---------------------------------------------------------- calibración

    def calibrate(self, method: str = "9point") -> None:
        from . import calibration_ui

        points = {"9point": 9, "5point": 5, "13point": 13}.get(method)
        if points is None:
            raise ValueError("method debe ser 9point, 5point o 13point")
        self.trained = calibration_ui.run(self, points)

    # -------------------------------------------------------- persistencia

    def _meta_path(self) -> Path:
        return Path(self.settings.calibration_dir) / "meta.json"

    def save(self, path: str | Path | None = None) -> Path:
        d = Path(path or self.settings.calibration_dir)
        d.mkdir(parents=True, exist_ok=True)
        if not self.calibration.save_model():
            raise RuntimeError(
                "la SVR no está entrenada: no hay nada que guardar")
        meta = {
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "estimator": "gazefollower/MGazeNet",
            "alignment": "TasksFaceAlignment (port a la API Tasks)",
            "camera_index": self.settings.camera_index,
            "capture_size": list(self.settings.capture_size),
            "monitor_index": self.screen.index,
            "monitor_name": self.screen.name,
            "monitor_size": [self.screen_w, self.screen_h],
            "screen_cm": list(self.settings.screen_cm),
            "camera_cm": list(self.settings.camera_cm),
        }
        if self.reference_pose is not None:
            meta["reference_pose"] = self.reference_pose.as_dict()
        self._meta_path().write_text(json.dumps(meta, indent=2),
                                     encoding="utf-8")
        return d

    def load(self, path: str | Path | None = None) -> Path:
        d = Path(path or self.settings.calibration_dir)
        if not (d / "svr_x.xml").exists() or not (d / "svr_y.xml").exists():
            raise FileNotFoundError(
                f"No hay calibración en {d}. Corré `run.cmd calibrate` primero."
            )
        self.settings.calibration_dir = d
        self.calibration = self._new_calibration()
        self.trained = bool(self.calibration.has_calibrated)
        self._warn_on_meta_mismatch()
        return d

    def _warn_on_meta_mismatch(self) -> None:
        p = self._meta_path()
        if not p.exists():
            return
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if "reference_pose" in meta:
            try:
                self.reference_pose = FacePose.from_dict(meta["reference_pose"])
            except (KeyError, TypeError, ValueError):
                self.reference_pose = None
        if meta.get("monitor_size") != [self.screen_w, self.screen_h]:
            print(f"[warn] calibrado en un monitor de {meta.get('monitor_size')} "
                  f"y ahora corriendo en [{self.screen_w}, {self.screen_h}]. "
                  f"El error va a ser sistemático: recalibrá.")
        if meta.get("screen_cm") and list(meta["screen_cm"]) != list(
                self.settings.screen_cm):
            print(f"[warn] calibrado con la pantalla declarada como "
                  f"{meta['screen_cm']} cm y ahora {list(self.settings.screen_cm)}. "
                  f"La conversión cm->px cambió bajo los pies del modelo.")

    # ------------------------------------------------------------------ vida

    def close(self) -> None:
        for piece in (self.alignment, self.estimator, self.calibration):
            try:
                piece.release()
            except Exception:
                pass

    def __enter__(self) -> "WebcamTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def make(settings) -> WebcamTracker:
    """Fábrica para pasarle a `aac_eyetrax.*.run(make_tracker=...)`."""
    return WebcamTracker(settings)
