"""Importar GazeFollower en Python 3.12, donde su alineador no puede existir.

El problema es un detalle de su código, no nuestro. En `GazeFollower.py`:

    class GazeFollower:
        def __init__(self, ...,
                     face_alignment: FaceAlignment = MediaPipeFaceAlignment(),
                     ...)

`MediaPipeFaceAlignment()` es el valor por defecto de un argumento, así que se
construye **cuando se define la clase**, es decir al importar el paquete. Y su
constructor toca `mp.solutions.face_mesh`, que no existe en ninguna MediaPipe
con rueda para Python 3.12. Resultado: `import gazefollower` falla entero, aunque
uno vaya a pasarle su propio alineador y nunca use ese default.

(De paso: un objeto mutable como valor por defecto se comparte entre todas las
instancias de la clase. Acá significa que dos `GazeFollower` compartirían el
mismo detector de MediaPipe. No es nuestro problema mientras pasemos el nuestro,
pero conviene saberlo antes de confiar en el default.)

La solución es un stub de la API legacy, suficiente para que ese constructor no
explote al importar. El objeto que crea queda inservible, y da igual: se
descarta en el momento en que se pasa `face_alignment=TasksFaceAlignment()`.

Llamar `patch()` **antes** de cualquier `import gazefollower`.
"""

from __future__ import annotations

import sys
import types


class _DeadFaceMesh:
    """Acepta cualquier construcción y falla si alguien la usa de verdad.

    Deliberadamente no es un no-op silencioso. Si algún camino termina usando el
    alineador legacy en vez del nuestro, tiene que romperse con un mensaje que
    diga qué pasó - un stub que devuelve `None` produciría "no se detecta cara"
    para siempre y nadie sabría por qué.
    """

    def __init__(self, *args, **kwargs):
        pass

    def process(self, *args, **kwargs):
        raise RuntimeError(
            "se usó el alineador legacy de GazeFollower, que en Python 3.12 es "
            "un stub vacío. Pasá face_alignment=TasksFaceAlignment() al "
            "construir GazeFollower."
        )

    def close(self):
        pass


def patch() -> bool:
    """Inserta el stub si hace falta. Devuelve True si lo insertó.

    No hace nada cuando la API legacy existe de verdad -por ejemplo en un
    entorno con Python 3.11 y mediapipe 0.10.10- así que es seguro llamarla
    siempre.
    """
    import mediapipe as mp

    if hasattr(mp, "solutions"):
        return False

    face_mesh = types.ModuleType("mediapipe.solutions.face_mesh")
    face_mesh.FaceMesh = _DeadFaceMesh

    solutions = types.ModuleType("mediapipe.solutions")
    solutions.face_mesh = face_mesh

    # Tanto en sys.modules como como atributo: su código hace
    # `mp.solutions.face_mesh`, que resuelve por atributo, pero otro código
    # podría hacer `import mediapipe.solutions`.
    sys.modules["mediapipe.solutions"] = solutions
    sys.modules["mediapipe.solutions.face_mesh"] = face_mesh
    mp.solutions = solutions
    return True


def init_logging(log_file: str | None = None) -> None:
    """`Log.init()` antes de tocar la calibración, o revienta.

    GazeFollower usa un logger global con inicialización explícita, y varias de
    sus clases loguean en el camino normal - `SVRCalibration.predict()` escribe
    una línea cuando todavía no hay modelo entrenado, que es exactamente el
    estado en el que uno la usa la primera vez. Sin `Log.init()` eso levanta
    "Logger has not been initialized" desde dentro de un método que por lo demás
    funcionaba bien.

    Ojo con el argumento: `Log.init` espera la ruta de un **archivo**, no de un
    directorio, aunque el nombre no lo diga. Pasarle una carpeta da un
    `PermissionError` desde `logging.FileHandler`, que es un error bastante
    desconcertante para lo que en realidad pasó.

    No es opcional ni cosmético: es parte de construir el objeto.
    """
    from pathlib import Path

    from gazefollower.logger import Log

    path = (Path(log_file) if log_file
            else Path.home() / "GazeFollower" / "gazefollower.log")
    path.parent.mkdir(parents=True, exist_ok=True)
    Log.init(str(path))
