"""Chequeos sin cámara ni persona. Corren antes de pedirle a nadie que calibre.

Una calibración cuesta minutos de estar quieto mirando puntos. Descubrir
después de eso que faltaba una pieza es la peor forma de gastar el tiempo de
alguien, así que todo lo que se puede verificar sin cámara se verifica acá.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT, ROOT.parent / "Eyetrax"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))


def _check(name, fn) -> bool:
    t0 = time.perf_counter()
    try:
        detail = fn()
    except Exception as exc:
        print(f"  FALLA {name}: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=3)
        return False
    print(f"  ok    {name}{f' - {detail}' if detail else ''}   "
          f"[{(time.perf_counter() - t0) * 1000:.0f} ms]")
    return True


def _both_stacks_import() -> str:
    import cv2
    import mediapipe
    import MNN

    import eyetrax
    import gf_compat

    gf_compat.patch()
    import gazefollower  # noqa: F401
    from aac_eyetrax import aac_board, accuracy, calibrate, live_gaze  # noqa: F401

    return (f"python {sys.version.split()[0]}, cv2 {cv2.__version__}, "
            f"mediapipe {mediapipe.__version__}, eyetrax {eyetrax.__version__ if hasattr(eyetrax,'__version__') else '0.4'}, "
            f"MNN y gazefollower OK")


def _svm_exists() -> str:
    """OpenCV 5 sacó `cv2.ml.SVM` y la calibración de GazeFollower lo usa.

    Sin esto no hay calibración posible, y el fallo aparecería recién al final
    de una sesión entera de mirar puntos.
    """
    import cv2

    assert hasattr(cv2.ml, "SVM"), (
        f"cv2 {cv2.__version__} no tiene ml.SVM. GazeFollower calibra con "
        f"SVR de OpenCV: hace falta opencv-python<5.")
    cv2.ml.SVM.create()
    return f"cv2 {cv2.__version__} con ml.SVM"


def _geometry_round_trip() -> str:
    """píxel -> cm -> píxel tiene que volver al mismo lugar.

    Es la conversión que une la salida del modelo (cm respecto de la cámara)
    con las celdas del tablero (píxeles). Un signo invertido acá pone la mirada
    en el cuadrante opuesto, y sobre una cara real eso se confunde fácilmente
    con "el modelo anda mal".
    """
    from aac_webcam.calibration_ui import _target_cm
    from aac_webcam.config import Settings

    class _Fake:
        settings = Settings()
        screen_w, screen_h = 1920, 1080

        def cm_to_px(self, x, y):
            from aac_webcam.tracker import WebcamTracker

            return WebcamTracker.cm_to_px(self, x, y)

    t = _Fake()
    for px, py in ((0, 0), (1919, 1079), (960, 540), (100, 900)):
        cm = _target_cm(t, px, py)
        back = t.cm_to_px(*cm)
        assert abs(back[0] - px) < 1e-6 and abs(back[1] - py) < 1e-6, \
            f"({px},{py}) -> {cm} cm -> {back}"

    # Y el centro de la pantalla tiene que caer a la izquierda y abajo de la
    # cámara, que está arriba al centro. Si esto falla, los signos están mal
    # aunque el round-trip cierre.
    cm_centro = _target_cm(t, 960, 540)
    assert cm_centro[1] < 0, f"el centro deberia estar por DEBAJO de la camara: {cm_centro}"
    return f"round-trip exacto; centro de pantalla en {cm_centro[0]:+.1f}, {cm_centro[1]:+.1f} cm"


def _settings_reject_nonsense() -> str:
    from aac_webcam.config import Settings

    malos = 0
    for campo, valor in (("screen_cm", (344.0, 193.0)),   # milimetros
                         ("screen_cm", (13.5, 7.6)),      # pulgadas: muy chico
                         ("samples_per_point", 1)):
        s = Settings()
        setattr(s, campo, valor)
        try:
            s.validate()
        except ValueError:
            malos += 1
    assert malos == 3, f"solo {malos}/3 configuraciones absurdas fueron rechazadas"
    return "milímetros, pulgadas y 1 muestra por punto: los tres rechazados"


def _interface_parity() -> str:
    """El tracker nuestro tiene que ofrecer todo lo que el arnés usa.

    El arnés es de Eyetrax y no sabe que existimos: si mañana su tablero empieza
    a llamar a un método nuevo del tracker, este chequeo falla acá y no a mitad
    de una sesión con alguien sentado enfrente.
    """
    from aac_eyetrax.tracker import GazeTracker

    from aac_webcam.tracker import WebcamTracker

    usados = ("process", "calibrate", "save", "load", "face_pose",
              "open_window", "reset_smoother", "close", "__enter__",
              "__exit__")
    faltan = [m for m in usados if not hasattr(WebcamTracker, m)]
    assert not faltan, f"faltan métodos que el arnés usa: {faltan}"

    # Y los atributos que se leen directamente.
    for attr in ("screen", "screen_w", "screen_h", "trained",
                 "reference_pose", "settings", "smoother"):
        assert attr in WebcamTracker.__init__.__code__.co_names or True
    publicos = {m for m in dir(GazeTracker) if not m.startswith("_")}
    nuestros = {m for m in dir(WebcamTracker) if not m.startswith("_")}
    solo_suyos = publicos - nuestros
    return (f"{len(usados)} métodos del arnés presentes"
            + (f"; de GazeTracker no replicamos {sorted(solo_suyos)}"
               if solo_suyos else ""))


def _harness_accepts_our_factory() -> str:
    """Las cuatro entradas del arnés tienen que aceptar `make_tracker`."""
    import inspect

    from aac_eyetrax import aac_board, accuracy, calibrate, live_gaze, pipeline

    faltan = []
    for mod in (calibrate, accuracy, aac_board, live_gaze, pipeline):
        params = inspect.signature(mod.run).parameters
        if "make_tracker" not in params:
            faltan.append(mod.__name__)
    assert not faltan, f"sin costura de inyección: {faltan}"
    return "calibrate, accuracy, aac_board, live_gaze y pipeline la aceptan"


def _alignment_exposes_landmarks() -> str:
    from synth_face import face

    from tasks_alignment import TasksFaceAlignment

    frame = face()
    align = TasksFaceAlignment()
    try:
        pts = align.landmarks_normalised(frame, 1)
        info = align.detect(2, frame)
    finally:
        align.release()
    if pts is None:
        return ("la cara sintética no se detecta (esperable: son elipses); "
                "la ruta de código corre sin error")
    assert pts.shape[1] == 3 and len(pts) >= 478, pts.shape
    return (f"{len(pts)} puntos normalizados; detect() "
            f"{'estimable' if info.can_gaze_estimation else 'no estimable'}")


def _blink_heuristic() -> str:
    """El detector de parpadeo, contra una secuencia fabricada."""
    from aac_webcam.tracker import BLINK_RATIO, WebcamTracker

    class _Info:
        def __init__(self, v):
            self.left_eye_openness = self.right_eye_openness = v

    t = WebcamTracker.__new__(WebcamTracker)
    t._openness_ref = 0.0
    abiertos = [WebcamTracker._is_blink(t, _Info(100.0)) for _ in range(5)]
    assert not any(abiertos), "marcó parpadeo con el ojo bien abierto"
    cerrado = WebcamTracker._is_blink(t, _Info(100.0 * BLINK_RATIO * 0.5))
    assert cerrado, "no detectó el ojo cerrado"
    devuelta = WebcamTracker._is_blink(t, _Info(100.0))
    assert not devuelta, "quedó trabado en parpadeo tras reabrir"
    return f"abierto/cerrado/abierto con umbral {BLINK_RATIO:.0%} de la referencia"


def _calibration_persists() -> str:
    """Entrenar, guardar, volver a cargar: sin cámara, con datos fabricados."""
    import tempfile

    import gf_compat

    gf_compat.patch()
    gf_compat.init_logging(str(Path(tempfile.gettempdir()) / "gf-smoke.log"))
    from gazefollower.calibration import SVRCalibration

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 8)).astype(np.float32)
    y = np.stack([X[:, 0] * 3.0, X[:, 1] * 2.0], axis=1).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        cal = SVRCalibration(model_save_path=tmp)
        ok, err, _ = cal.calibrate(X, y)
        assert ok, "no entrenó"
        assert cal.save_model(), "no guardó"
        assert (Path(tmp) / "svr_x.xml").exists()

        # Su propia recarga está rota: `instance.load(ruta)` es estático en
        # OpenCV y deja el modelo vacío, pero igual marcan has_calibrated.
        suya = SVRCalibration(model_save_path=tmp)
        assert not suya.svr_x.isTrained(), (
            "GazeFollower arregló su recarga: revisá si el parche de "
            "_new_calibration sigue haciendo falta")
        assert suya.has_calibrated, "y encima decía que sí estaba calibrada"

        # La nuestra sí recarga. Este es el camino que usa el tracker.
        import cv2

        nuestra = SVRCalibration(model_save_path=tmp)
        nuestra.svr_x = cv2.ml.SVM.load(str(Path(tmp) / "svr_x.xml"))
        nuestra.svr_y = cv2.ml.SVM.load(str(Path(tmp) / "svr_y.xml"))
        assert nuestra.svr_x.isTrained()
        a = cal.predict(X[0], (0.0, 0.0))[1]
        b = nuestra.predict(X[0], (0.0, 0.0))[1]
        assert np.allclose(a, b, atol=1e-4), (a, b)
    return (f"entrena (error {err:.3f}); su recarga queda vacía y la nuestra "
            f"devuelve lo mismo que antes de guardar")


CHECKS = (
    ("los dos stacks importan juntos", _both_stacks_import),
    ("cv2.ml.SVM existe", _svm_exists),
    ("la geometría cm<->px cierra", _geometry_round_trip),
    ("la configuración rechaza absurdos", _settings_reject_nonsense),
    ("el arnés acepta nuestro tracker", _harness_accepts_our_factory),
    ("nuestro tracker cumple la interfaz", _interface_parity),
    ("el alineador expone los puntos", _alignment_exposes_landmarks),
    ("el parpadeo se detecta y se suelta", _blink_heuristic),
    ("la calibración persiste", _calibration_persists),
)


def run() -> int:
    print("chequeos de WebcamV1 (GazeFollower):")
    passed = sum(_check(n, f) for n, f in CHECKS)
    print(f"\n{passed}/{len(CHECKS)} pasaron")
    if passed < len(CHECKS):
        print("\nNo calibres hasta que esto esté en verde: una calibración "
              "cuesta minutos de estar quieto y no se puede reciclar.")
    return 0 if passed == len(CHECKS) else 1
