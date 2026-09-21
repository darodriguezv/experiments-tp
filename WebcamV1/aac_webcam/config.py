"""Configuración de WebcamV1: lo de Eyetrax más la geometría física.

Este módulo **extiende** la configuración de Eyetrax en vez de copiarla, porque
el arnés -calibración, medición de precisión, tablero- es literalmente el mismo
código. Dos estimadores medidos con instrumentos distintos no se pueden
comparar, y comparar es todo el punto de tener WebcamV1 al lado de Eyetrax.

## Lo que hay que medir con una regla

MGazeNet no predice píxeles: predice **centímetros respecto de la cámara**, con
x hacia la derecha e y hacia arriba. Para llevar eso a una celda del tablero
hacen falta dos medidas físicas, y no hay forma de adivinarlas desde el
software:

* `screen_cm`: el ancho y alto del **área visible** de la pantalla, en cm.
* `camera_cm`: dónde está la cámara, en cm, desde la **esquina superior
  izquierda** de esa área. La `y` es negativa si la cámara está por encima del
  borde superior, que es el caso de casi todos los portátiles.

Si están mal, el error es sistemático. La calibración SVR absorbe parte -se
entrena en el mismo espacio de cm- pero no lo arregla: el modelo entrega cm
verdaderos y las etiquetas estarían en cm escalados, así que el ajuste empeora
en vez de fallar limpio. Peor todavía: el resultado sigue pareciendo razonable.

Por eso `main.py geometry` imprime lo que está asumiendo y pide confirmarlo
antes de calibrar.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_EYETRAX = ROOT.parent / "Eyetrax"
if str(_EYETRAX) not in sys.path:
    sys.path.insert(0, str(_EYETRAX))

try:
    from aac_eyetrax.config import Settings as EyetraxSettings
except ImportError as exc:  # pragma: no cover - depende del árbol de archivos
    raise ImportError(
        f"no se pudo importar la configuración de Eyetrax desde {_EYETRAX}: "
        f"{exc}\nWebcamV1 reusa su arnés a propósito; ver el docstring."
    ) from None

MODELS_DIR = ROOT / "models"
CALIBRATION_DIR = MODELS_DIR / "gazefollower"

# Medidas del portátil sobre el que se desarrolló esto, ya usadas para el
# presupuesto de error del tablero (344 x 193 mm a 1920x1080). Son un punto de
# partida verificable, NO una constante universal: en otra máquina hay que
# medir de nuevo, y `main.py geometry` existe para obligar a mirarlas.
DEFAULT_SCREEN_CM = (34.4, 19.3)
DEFAULT_CAMERA_CM = (17.2, -0.7)


@dataclass
class Settings(EyetraxSettings):
    """La configuración de Eyetrax más lo que GazeFollower necesita."""

    screen_cm: tuple[float, float] = DEFAULT_SCREEN_CM
    camera_cm: tuple[float, float] = DEFAULT_CAMERA_CM

    calibration_dir: Path = field(default_factory=lambda: CALIBRATION_DIR)
    """Dónde viven svr_x.xml y svr_y.xml. GazeFollower guarda por directorio y
    no por archivo, así que esto es una carpeta y no una ruta de modelo."""

    mnn_threads: int = 0
    """Hilos para la inferencia de MGazeNet. 0 = la mitad de los nucleos.

    GazeFollower fija 4. Medido acá sobre 16 nucleos logicos, el forward pasa
    de 20 ms con 4 hilos a 17 ms con 12 - o sea que el reparto importa poco por
    si solo. Lo que si importa es que MediaPipe y MNN no se peleen por los
    mismos nucleos: la alineacion corre justo antes del estimador en cada
    cuadro, y medir los dos juntos daba mucho peor que medir cada uno aparte.
    """

    calibration_points: int = 9
    samples_per_point: int = 18
    """Cuadros que se promedian por punto. A ~30 fps son unos 0.6 s de mirada
    quieta, después del tiempo de asentamiento."""

    # El suavizado de Eyetrax se reusa tal cual: son los mismos filtros y la
    # misma pantalla, así que una diferencia entre los dos estimadores no puede
    # atribuirse al filtro.

    def validate(self) -> None:
        super().validate()
        w, h = self.screen_cm
        if w <= 0 or h <= 0:
            raise ValueError("screen_cm tiene que ser positivo, en centímetros")
        if w < 15 or w > 200:
            # 15 cm de ancho es mas chico que cualquier portatil, y los dos
            # errores tipicos caen fuera por lados opuestos: milimetros (344)
            # por arriba, y la diagonal en pulgadas (13.5) por abajo.
            raise ValueError(
                f"screen_cm ancho = {w} no parece el ancho de una pantalla en "
                f"centimetros. ¿Pusiste milimetros, o la diagonal en pulgadas?")
        if self.samples_per_point < 3:
            raise ValueError("samples_per_point < 3 no promedia nada útil")

    def describe_geometry(self) -> str:
        from aac_eyetrax import display

        screen = display.resolve(self.monitor)
        w_cm, h_cm = self.screen_cm
        cx, cy = self.camera_cm
        px_per_cm_x = screen.width / w_cm
        px_per_cm_y = screen.height / h_cm
        return "\n".join([
            f"  pantalla   {screen.width}x{screen.height} px  =  "
            f"{w_cm:.1f} x {h_cm:.1f} cm",
            f"  densidad   {px_per_cm_x:.1f} x {px_per_cm_y:.1f} px/cm",
            f"  cámara     x={cx:.1f} cm desde el borde izquierdo, "
            f"y={cy:.1f} cm " + ("por ENCIMA del borde superior" if cy < 0
                                 else "por debajo del borde superior"),
            "",
            "  Si alguno de estos números no coincide con la regla, el error "
            "va a ser",
            "  sistemático y la calibración no lo va a arreglar. Se corrigen "
            "con",
            "  --screen-cm y --camera-cm.",
        ])
