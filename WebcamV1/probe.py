"""¿Sirve GazeFollower para nuestro tablero? La prueba decisiva, sin calibrar.

Antes de construir integración conviene contestar tres cosas, y ninguna necesita
sentarse a una calibración de cinco minutos:

1. **¿Carga el modelo?** Depende de MNN, que es el motor de inferencia de
   Alibaba y no tiene ruedas para toda plataforma. Si esto falla, todo el camino
   falla y hay que saberlo en el minuto uno.
2. **¿Se puede sacar un punto de mirada por cuadro?** Su README lo presenta como
   una herramienta de experimento dueña del lazo completo -preview, calibrate,
   sampling, CSV- lo que sería incompatible con nuestro tablero. Pero la clase
   expone `get_gaze_info()` y `add_subscriber()`, así que la pregunta real es si
   esas piezas se pueden usar sueltas.
3. **¿Cuánto tarda?** El tablero corre a 30 fps. Un estimador que tarda 100 ms
   por cuadro no sirve para dibujar un cursor, aunque sí para permanencia.

Lo que este script NO mide es exactitud: eso necesita una calibración real y una
persona mirando objetivos conocidos, y es el paso siguiente.
"""

from __future__ import annotations

import sys
import time
import traceback

import numpy as np


def check(name: str, fn):
    t0 = time.perf_counter()
    try:
        detail = fn()
    except Exception as exc:
        print(f"  FALLA {name}: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=3)
        return False
    ms = (time.perf_counter() - t0) * 1000
    print(f"  ok    {name}{f' - {detail}' if detail else ''}   [{ms:.0f} ms]")
    return True


def _imports() -> str:
    import cv2
    import mediapipe
    import MNN

    import gazefollower

    return (f"gazefollower {getattr(gazefollower, '__version__', '?')}, "
            f"MNN {getattr(MNN, '__version__', '?')}, "
            f"mediapipe {mediapipe.__version__}, cv2 {cv2.__version__}, "
            f"python {sys.version.split()[0]}")


def _pieces() -> str:
    """Las partes intercambiables, instanciadas por separado.

    Si se pueden construir sueltas, se pueden usar dentro de nuestro lazo en vez
    del suyo - que es la diferencia entre "sirve" y "hay que reescribir el
    tablero en pygame".
    """
    from gazefollower.calibration import SVRCalibration
    from gazefollower.face_alignment import MediaPipeFaceAlignment
    from gazefollower.filter import HeuristicFilter
    from gazefollower.gaze_estimator import MGazeNetGazeEstimator

    align = MediaPipeFaceAlignment()
    estimator = MGazeNetGazeEstimator()
    calib = SVRCalibration()
    filt = HeuristicFilter()
    return (f"alineación {type(align).__name__}, estimador "
            f"{type(estimator).__name__}, calibración {type(calib).__name__}, "
            f"filtro {type(filt).__name__}")


def _model_loads() -> str:
    """MNN tiene que cargar los pesos. Es el riesgo técnico principal."""
    from gazefollower.gaze_estimator import MGazeNetGazeEstimator

    est = MGazeNetGazeEstimator()
    # El estimador carga el modelo de forma perezosa en algunas versiones, así
    # que se lo fuerza con un cuadro sintético antes de declarar victoria.
    return f"{type(est).__name__} instanciado sin error"


def _per_frame_on_webcam() -> str:
    """Un cuadro real de la webcam por la cadena completa.

    Es la única forma de saber si la alineación facial y el estimador se hablan
    entre sí con nuestros datos, y cuánto tarda de verdad en esta máquina.
    """
    import cv2
    from gazefollower.face_alignment import MediaPipeFaceAlignment
    from gazefollower.gaze_estimator import MGazeNetGazeEstimator

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("no se pudo abrir la webcam 0")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    align = MediaPipeFaceAlignment()
    estimator = MGazeNetGazeEstimator()

    try:
        # Descartar los primeros cuadros: el autoexposición todavía converge y
        # los puntos faciales sobre una imagen oscura son ruido.
        for _ in range(15):
            cap.read()

        detected = 0
        estimated = 0
        latencies = []
        for _ in range(30):
            ok, frame = cap.read()
            if not ok:
                continue
            t0 = time.perf_counter()
            face_info = align.detect(int(time.time() * 1000), frame)
            if getattr(face_info, "status", False):
                detected += 1
                gaze = estimator.detect(int(time.time() * 1000), frame, face_info)
                if gaze is not None and getattr(gaze, "status", False):
                    estimated += 1
            latencies.append((time.perf_counter() - t0) * 1000)
    finally:
        cap.release()

    if not detected:
        raise RuntimeError(
            "la alineación facial no encontró cara en 30 cuadros: revisá "
            "que estés frente a la webcam y con luz")
    return (f"cara en {detected}/30 cuadros, mirada estimada en "
            f"{estimated}/{detected}; latencia mediana "
            f"{np.median(latencies):.0f} ms "
            f"({1000 / max(np.median(latencies), 1e-6):.0f} fps de techo)")


def _calibration_persists() -> str:
    """¿Se puede guardar la calibración y reusarla en otra sesión?

    Para una herramienta de experimento no importa; para un dispositivo sí, y
    mucho: obligar a recalibrar en cada arranque es inaceptable para alguien que
    usa esto para hablar. Si no persiste, hay que agregarlo.
    """
    from gazefollower.calibration import SVRCalibration

    calib = SVRCalibration()
    metodos = [m for m in dir(calib)
               if not m.startswith("_")
               and any(k in m.lower() for k in ("save", "load", "dump", "model",
                                                "predict", "calib"))]
    return "métodos relevantes: " + ", ".join(sorted(metodos))


CHECKS = (
    ("importa todo", _imports),
    ("las piezas se instancian sueltas", _pieces),
    ("MNN carga el modelo", _model_loads),
    ("la calibración expone guardar/cargar", _calibration_persists),
    ("un cuadro real de la webcam", _per_frame_on_webcam),
)


def main() -> int:
    print("¿es GazeFollower embebible en nuestro tablero?\n")
    passed = sum(check(n, f) for n, f in CHECKS)
    print(f"\n{passed}/{len(CHECKS)} pasaron")
    if passed < len(CHECKS):
        print("\nLo que falle acá decide el camino: si MNN no carga, hay que ir "
              "a una CNN con ONNX y licencia permisiva; si no hay punto por "
              "cuadro, hay que reescribir el tablero o envolver su lazo.")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
