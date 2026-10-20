"""Qué se mide en cada tarea, y cómo se saca del CSV que el tablero escribe.

## Las variables dependientes

Se fijan acá, antes de ver ningún dato, porque elegirlas después de mirar los
resultados es como se construye un hallazgo que no existe.

* **Segundos por enunciado.** Desde que aparece la consigna hasta que la frase
  sale por el parlante. Es lo que la persona vive.
* **Permanencias por enunciado.** Cuántas selecciones costó. Es la unidad de
  esfuerzo del tablero y la única comparable entre condiciones: los segundos
  dependen del tiempo de permanencia configurado, las permanencias no.
* **Correcciones.** BORRAR y LIMPIAR. Separa "costó mucho" de "salió mal y hubo
  que rehacerlo", que son problemas distintos con soluciones distintas.
* **Acierto.** Si lo que se dijo comunica la intención pedida. Sin esto, la
  condición más rápida gana diciendo cualquier cosa rápido.

De ahí sale lo que de verdad interesa: **el ahorro** de cada condición contra
el deletreo, en permanencias y en segundos.

## El acierto no se puede automatizar del todo, y no se finge que sí

A la persona se le pide una intención, no una frase: "Tiene sed", no "escriba
'quiero agua'". Si se le dictara la frase, el estudio mediría transcripción y
la condición con modelo perdería todo sentido, porque no habría nada que
interpretar. El precio es que "quiero agua", "tengo sed" y "agua por favor" son
todas correctas y no se parecen entre sí en los tokens.

Entonces el F1 de tokens contra la intención de referencia **no alcanza**, y
usarlo solo sería inventar precisión. Lo que se hace:

* arriba de `UMBRAL_ALTO`, acierto automático;
* abajo de `UMBRAL_BAJO`, fallo automático;
* en el medio, `pendiente`: lo resuelve una persona y queda registrado quién y
  cuándo.

La zona gris es chica pero existe, y el informe la reporta. Un estudio que
devuelve 100 % de casos resueltos por una fórmula sobre paráfrasis libres está
escondiendo el juicio, no evitándolo.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

_SLM = Path(__file__).resolve().parents[2] / "slm"
if str(_SLM) not in sys.path:
    sys.path.insert(0, str(_SLM))

from aac_semantic.metrics import token_f1  # noqa: E402

UMBRAL_ALTO = 0.60
UMBRAL_BAJO = 0.25
"""Los dos bordes de la zona gris.

No son redondos por gusto: 0.60 es donde cae una paráfrasis que conserva el
verbo y el objeto ("me duele la espalda" contra "me duele mucho la espalda" da
0.89; "quiero agua" contra "por favor quiero agua" da 0.67). 0.25 es donde cae
compartir una sola palabra funcional, que no es comunicar lo mismo."""

CONTROLES = {"ESPACIO", "BORRAR", "LIMPIAR", "HABLAR", "ABC", "PALABRAS"}
CORRECCIONES = {"BORRAR", "LIMPIAR"}


@dataclass
class Enunciado:
    """Una tarea completada: lo que costó y si salió bien."""

    tarea_id: str
    dicho: str
    segundos: float
    permanencias: int
    correcciones: int
    navegacion: int
    """Cambios de página (ABC/PALABRAS). Se cuentan aparte porque son el costo
    de la ESTRUCTURA del tablero, no del mensaje: en la condición de deletreo
    son cero por definición, y mezclarlos la haría ver mejor de lo que es."""

    f1: float = 0.0
    veredicto: str = "pendiente"
    """`acierto`, `fallo` o `pendiente`."""

    resuelto_por: str = ""
    """Quién adjudicó un `pendiente`. Vacío si lo decidió el umbral."""

    @property
    def automatico(self) -> bool:
        return not self.resuelto_por


@dataclass
class Bloque:
    """Una condición completa de un participante."""

    participante: int
    condicion: str
    conjunto: int
    enunciados: list[Enunciado] = field(default_factory=list)
    incompleto: bool = False
    """El bloque se cortó antes de terminar las tareas (la persona abandonó o
    se cerró la ventana). Se marca en vez de descartarse en silencio: un bloque
    abandonado es un dato sobre la condición, no un archivo roto."""

    def resumen(self) -> dict:
        import numpy as np

        hechos = self.enunciados
        if not hechos:
            return {"n": 0}
        aciertos = [e for e in hechos if e.veredicto == "acierto"]
        return {
            "n": len(hechos),
            "pendientes": sum(1 for e in hechos if e.veredicto == "pendiente"),
            "tasa_acierto": len(aciertos) / len(hechos),
            # Las medianas van sobre los ACERTADOS: un enunciado que no
            # comunicó la intención puede haber sido rapidísimo, y promediarlo
            # con los buenos premia rendirse temprano.
            "segundos_p50": float(np.median([e.segundos for e in aciertos]))
                            if aciertos else float("nan"),
            "permanencias_p50": float(np.median(
                [e.permanencias for e in aciertos])) if aciertos else float("nan"),
            "correcciones_media": float(np.mean(
                [e.correcciones for e in hechos])),
            "navegacion_media": float(np.mean([e.navegacion for e in hechos])),
            "incompleto": self.incompleto,
        }


def leer_csv(path: str | Path) -> list[tuple[float, str]]:
    filas = []
    with Path(path).open(encoding="utf-8", newline="") as fh:
        for fila in csv.DictReader(fh):
            try:
                filas.append((float(fila["t_since_start_s"]), fila["label"]))
            except (KeyError, ValueError, TypeError):
                continue
    return filas


def extraer(path: str | Path, tareas: dict[str, str], *,
            participante: int = 0, condicion: str = "",
            conjunto: int = 0) -> Bloque:
    """Convierte un CSV de sesión en un `Bloque` medido.

    `tareas` mapea id -> intención de referencia, para puntuar el acierto.
    """
    filas = leer_csv(path)
    bloque = Bloque(participante, condicion, conjunto)

    inicio: float | None = None
    tarea_id = ""
    permanencias = correcciones = navegacion = 0
    cerrado = False

    for t, label in filas:
        if label.startswith("TAREA:"):
            inicio, tarea_id = t, label.split(":", 1)[1]
            permanencias = correcciones = navegacion = 0
        elif label == "BLOQUE:fin":
            cerrado = True
        elif label.startswith("DICHO:"):
            if inicio is None:
                # Un enunciado sin consigna previa: el tablero se usó suelto.
                # No se puede atribuir a ninguna tarea, así que no se cuenta.
                continue
            dicho = label.split(":", 1)[1]
            f1 = token_f1(dicho, tareas.get(tarea_id, ""))
            bloque.enunciados.append(Enunciado(
                tarea_id=tarea_id, dicho=dicho, segundos=t - inicio,
                permanencias=permanencias, correcciones=correcciones,
                navegacion=navegacion, f1=f1, veredicto=_veredicto(f1)))
            inicio = None
        else:
            permanencias += 1
            if label in CORRECCIONES:
                correcciones += 1
            elif label in ("ABC", "PALABRAS"):
                navegacion += 1

    bloque.incompleto = bool(bloque.enunciados) and not cerrado
    return bloque


def _veredicto(f1: float) -> str:
    if f1 >= UMBRAL_ALTO:
        return "acierto"
    if f1 < UMBRAL_BAJO:
        return "fallo"
    return "pendiente"


def ahorro(base: Bloque, otra: Bloque) -> dict:
    """Cuánto ahorra `otra` contra la línea de base, en permanencias y tiempo.

    Se compara sobre los enunciados ACERTADOS de cada bloque. Comparar todos
    premiaría a la condición donde se abandona antes, porque rendirse es
    rápido y barato.
    """
    a, b = base.resumen(), otra.resumen()
    if not a.get("n") or not b.get("n"):
        return {}
    import math

    def _delta(x, y):
        if math.isnan(x) or math.isnan(y) or x == 0:
            return float("nan")
        return (x - y) / x

    return {
        "permanencias_base": a["permanencias_p50"],
        "permanencias_otra": b["permanencias_p50"],
        "ahorro_permanencias": _delta(a["permanencias_p50"],
                                      b["permanencias_p50"]),
        "segundos_base": a["segundos_p50"],
        "segundos_otra": b["segundos_p50"],
        "ahorro_segundos": _delta(a["segundos_p50"], b["segundos_p50"]),
        "acierto_base": a["tasa_acierto"],
        "acierto_otra": b["tasa_acierto"],
    }
