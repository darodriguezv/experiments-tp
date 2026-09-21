"""Configuración de WebcamV1: lo de Eyetrax más la geometría física.

Este módulo **extiende** la configuración de Eyetrax en vez de copiarla, porque
el arnés -calibración, medición de precisión, tablero- es literalmente el mismo
código. Dos estimadores medidos con instrumentos distintos no se pueden
comparar, y comparar es todo el punto de tener WebcamV1 al lado de Eyetrax.

## La geometría física, y las dos pantallas

MGazeNet no predice píxeles: predice **centímetros respecto de la cámara**, con
x hacia la derecha e y hacia arriba. Para llevar eso a una celda hacen falta
dos medidas:

* `screen_cm`: ancho y alto del área visible, en cm. **Se deriva sola**: el
  sistema reporta los milímetros de cada monitor y `screeninfo` los expone. No
  hay que medir nada, y como sale del monitor elegido, cambiar de pantalla la
  recalcula sin tocar configuración.
* `camera_cm`: dónde está la cámara desde la esquina superior izquierda de
  **esa** área. Por defecto se asume el borde superior centrado, que es donde
  está en cualquier portátil.

## El caso de dos pantallas, que es el que importa acá

`camera_cm` es relativo a **la pantalla que se está mirando**, no al equipo. Si
el tablero corre en la pantalla del portátil, el default es correcto: la cámara
está justo arriba, centrada. Si el tablero corre en un monitor externo y la
webcam sigue siendo la del portátil, **la cámara ya no está sobre esa
pantalla** - puede estar 30 cm a un lado y 20 cm más abajo - y el default es
sencillamente falso.

Y no se puede detectar por software: nada le dice al sistema operativo dónde
está físicamente una webcam respecto de un monitor. Por eso la pantalla se
elige explícitamente y la suposición se imprime antes de calibrar, en vez de
quedar escondida en un valor por defecto.

Si están mal, el error es sistemático. La calibración SVR absorbe parte -se
entrena en el mismo espacio de cm- pero no lo arregla: el modelo entrega cm
verdaderos y las etiquetas estarían en cm corridos, así que el ajuste empeora
en vez de fallar limpio. Y el resultado sigue pareciendo razonable.
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

FALLBACK_SCREEN_CM = (34.4, 19.3)
"""Solo para cuando el sistema no reporta los milímetros de un monitor, cosa
que pasa con algunos adaptadores y máquinas virtuales. Son las del portátil de
desarrollo; en otra pantalla van a estar mal, y por eso se avisa cuando se usa."""

CAMERA_ABOVE_CM = 0.7
"""Cuánto sobresale la cámara por encima del borde visible, en un portátil
típico. Entra como `y` negativa."""


@dataclass
class Settings(EyetraxSettings):
    """La configuración de Eyetrax más lo que GazeFollower necesita."""

    screen_cm: tuple[float, float] | None = None
    """None = derivar del monitor elegido. Se fija a mano solo si el sistema
    miente sobre el tamaño, que pasa con algunos adaptadores."""

    camera_cm: tuple[float, float] | None = None
    """None = borde superior, centrado en el monitor elegido. **Hay que fijarlo
    a mano si el tablero corre en un monitor externo y la webcam es la del
    portátil**: ahí la cámara no está sobre esa pantalla."""

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

    # ------------------------------------------------- geometría derivada

    def calibration_dir_for(self, screen) -> Path:
        """Un directorio de calibración **por pantalla**.

        Con dos monitores esto no es prolijidad, es corrección. Un modelo de
        mirada vale para la pantalla en la que se calibró: mapea el embedding a
        centímetros respecto de la cámara, y esos centímetros se vuelven
        píxeles con el tamaño y la posición de *esa* pantalla. Reusarlo en la
        otra da un error sistemático grande.

        Con un solo directorio, cambiar de monitor cargaba en silencio la
        calibración de la otra pantalla. Separarlos hace que el caso malo sea
        «no hay calibración para este monitor, calibrá» en vez de «anda pero
        mal», que es mucho más difícil de diagnosticar.
        """
        if self.calibration_dir != CALIBRATION_DIR:
            return Path(self.calibration_dir)      # el usuario la fijó a mano
        slug = "".join(c if c.isalnum() else "_" for c in screen.name).strip("_")
        return CALIBRATION_DIR / f"{slug}_{screen.width}x{screen.height}"

    def geometry_for(self, screen) -> tuple[tuple[float, float],
                                            tuple[float, float], list[str]]:
        """(screen_cm, camera_cm, avisos) para una pantalla concreta.

        Con dos monitores la misma configuración significa cosas distintas
        según cuál se elija, así que todo lo que dependa de la pantalla se
        resuelve acá y en un solo lugar. Repartir esa decisión por el código
        sería pedir que las dos copias se desincronicen.
        """
        avisos: list[str] = []

        if self.screen_cm is not None:
            screen_cm = tuple(self.screen_cm)
        elif screen.width_mm and screen.height_mm:
            screen_cm = (screen.width_mm / 10.0, screen.height_mm / 10.0)
        else:
            screen_cm = FALLBACK_SCREEN_CM
            avisos.append(
                f"el sistema no reporta el tamaño físico de {screen.name}; se "
                f"usan {FALLBACK_SCREEN_CM[0]}x{FALLBACK_SCREEN_CM[1]} cm, que "
                f"son de OTRA pantalla. Medí y pasá --screen-cm.")

        if self.camera_cm is not None:
            camera_cm = tuple(self.camera_cm)
        else:
            camera_cm = (screen_cm[0] / 2.0, -CAMERA_ABOVE_CM)
            avisos.append("se asume la webcam en el borde superior de ESTA "
                          "pantalla, centrada.")
            diag = screen.diagonal_in or 0.0
            if diag > 18.0:
                # Heurística, no certeza. Nada le dice al sistema operativo
                # dónde está físicamente una webcam respecto de un monitor, así
                # que lo único honesto es marcar el caso probable en vez de
                # dejar el supuesto escondido en un valor por defecto.
                avisos.append(
                    f'{screen.name} mide {diag:.0f}" y probablemente sea un '
                    f"monitor externo. Si la webcam es la del portátil, NO "
                    f"está sobre esta pantalla: medí dónde queda respecto de "
                    f"su esquina superior izquierda y pasá --camera-cm.")
        return screen_cm, camera_cm, avisos

    def validate(self) -> None:
        super().validate()
        if self.samples_per_point < 3:
            raise ValueError("samples_per_point < 3 no promedia nada útil")
        if self.screen_cm is None:
            return         # se deriva del monitor: no hay nada más que validar
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

    def describe_geometry(self) -> str:
        from aac_eyetrax import display

        screen = display.resolve(self.monitor)
        (w_cm, h_cm), (cx, cy), avisos = self.geometry_for(screen)
        origen = "medido a mano" if self.screen_cm else "según el sistema"
        lines = [
            f"  pantalla   [{screen.index}] {screen.name}  "
            f"{'PRINCIPAL' if screen.is_primary else 'secundaria'}",
            f"             {screen.width}x{screen.height} px  =  "
            f"{w_cm:.1f} x {h_cm:.1f} cm  ({origen})",
            f"  densidad   {screen.width / w_cm:.1f} x "
            f"{screen.height / h_cm:.1f} px/cm",
            f"  cámara     x={cx:.1f} cm desde el borde izquierdo, "
            f"y={cy:.1f} cm " + ("por ENCIMA del borde superior" if cy < 0
                                 else "por debajo del borde superior"),
        ]
        if avisos:
            lines.append("")
            lines += [f"  ! {a}" for a in avisos]
        return "\n".join(lines)
