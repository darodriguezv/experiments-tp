"""Correr un bloque: abrir el tablero con la condición puesta y guardar lo medido.

Un bloque es un participante haciendo UNA condición completa: seis tareas, de
corrido, en una sola apertura del tablero. De corrido a propósito -no una
apertura por tarea- porque cada apertura obliga a recalibrar, y entonces el
tiempo de la primera tarea de cada bloque incluiría una calibración entera.

## La calibración va en cada bloque, igual

Entre bloques la persona se levanta, toma agua, cambia de postura. Volver a
calibrar no es una molestia del protocolo: es la condición para que las tres
mediciones sean comparables. Un bloque medido con la calibración de hace veinte
minutos arrastra un error sistemático que ningún suavizado corrige, y ese error
se le atribuiría a la condición.

## Lo que el estado del estudio guarda, y lo que no

Guarda: número de participante, qué bloques hizo, a qué hora, y la ruta del CSV
del tablero. **No guarda** nombre, contacto ni nada que identifique a la
persona. El consentimiento se firma en papel y el papel se guarda aparte; acá
el participante es un número, y el número no sirve para encontrarlo.

Eso es deliberado y es lo que permite que esta carpeta exista sin más cuidado
que el habitual. Si alguna vez se agrega un campo con un nombre, deja de ser
cierto y hay que revisar la declaración al comité.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import diseno, protocolo

RAIZ = Path(__file__).resolve().parents[2]
DATOS = RAIZ / "estudio" / "datos"
ESTADO = DATOS / "participantes.json"
LOGS_TABLERO = RAIZ / "slm" / "logs"


@dataclass
class EstadoEstudio:
    """Quién hizo qué. Un número por persona, nunca un nombre."""

    participantes: dict = field(default_factory=dict)

    @classmethod
    def cargar(cls) -> "EstadoEstudio":
        if not ESTADO.exists():
            return cls()
        try:
            return cls(json.loads(ESTADO.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            return cls()

    def guardar(self) -> None:
        DATOS.mkdir(parents=True, exist_ok=True)
        tmp = ESTADO.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.participantes, ensure_ascii=False,
                                  indent=2), encoding="utf-8")
        tmp.replace(ESTADO)

    # ------------------------------------------------------------------ alta

    def alta(self, notas: str = "") -> int:
        """Da de alta al siguiente participante. Devuelve su número."""
        n = max((int(k) for k in self.participantes), default=0) + 1
        self.participantes[str(n)] = {
            "alta": time.time(),
            "notas": notas,
            "bloques": {},
        }
        self.guardar()
        return n

    def ficha(self, participante: int) -> dict:
        """Todo lo que hay que saber de un participante, listo para la UI."""
        p = self.participantes.get(str(participante))
        if p is None:
            return {}
        asig = diseno.asignar(participante)
        bloques = []
        for pos, cond, conj in asig.bloques():
            hecho = p["bloques"].get(cond)
            bloques.append({
                "posicion": pos,
                "condicion": cond,
                "nombre": diseno.NOMBRES[cond],
                "conjunto": conj,
                "instrucciones": protocolo.INSTRUCCIONES[cond],
                "con_guion": cond == "con_contexto",
                "hecho": bool(hecho),
                "csv": (hecho or {}).get("csv", ""),
                "enunciados": (hecho or {}).get("enunciados", 0),
            })
        # El siguiente bloque es el primero sin hacer, EN EL ORDEN ASIGNADO.
        # Dejar elegir el orden desde la pantalla rompería el contrabalanceo
        # sin que nadie se entere, que es la forma más barata de arruinar el
        # estudio entero.
        siguiente = next((b for b in bloques if not b["hecho"]), None)
        return {
            "participante": participante,
            "alta": p["alta"],
            "notas": p.get("notas", ""),
            "bloques": bloques,
            "siguiente": siguiente,
            "completo": siguiente is None,
        }

    def registrar(self, participante: int, condicion: str, csv: str,
                  enunciados: int) -> None:
        p = self.participantes.setdefault(
            str(participante), {"alta": time.time(), "notas": "",
                                "bloques": {}})
        p["bloques"][condicion] = {
            "csv": csv, "cuando": time.time(), "enunciados": enunciados}
        self.guardar()


def tareas_de(participante: int, condicion: str) -> list[protocolo.Tarea]:
    asig = diseno.asignar(participante)
    for _, cond, conj in asig.bloques():
        if cond == condicion:
            return list(protocolo.TAREAS[conj])
    raise ValueError(f"{condicion!r} no es una condición del participante")


def _csv_mas_nuevo(desde: float) -> str:
    """El CSV que el tablero acaba de escribir.

    Se busca por fecha de modificación posterior al arranque del bloque en vez
    de pedirle al tablero que informe su ruta: el tablero ya nombra sus
    archivos con la hora y agregarle un canal de vuelta solo para esto sería
    acoplarlo al estudio.
    """
    if not LOGS_TABLERO.exists():
        return ""
    nuevos = [p for p in LOGS_TABLERO.glob("aac-session-*.csv")
              if p.stat().st_mtime >= desde - 1]
    if not nuevos:
        return ""
    return str(max(nuevos, key=lambda p: p.stat().st_mtime))


def correr_bloque(participante: int, condicion: str, *, monitor: str = "0",
                  camera: int = 0, model: str | None = None,
                  en_proceso: bool = True) -> dict:
    """Abre el tablero con la condición puesta y devuelve lo medido."""
    from . import medidas

    cfg = protocolo.CONFIG_CONDICION[condicion]
    tareas = tareas_de(participante, condicion)
    t0 = time.time()

    if en_proceso:
        codigo = _abrir_tablero(cfg, tareas, monitor, camera, model)
    else:
        codigo = subprocess.call(
            [sys.executable, str(RAIZ / "estudio" / "main.py"), "bloque",
             "--participante", str(participante), "--condicion", condicion,
             "--monitor", str(monitor), "--camera", str(camera)]
            + (["--model", model] if model else []))

    csv = _csv_mas_nuevo(t0)
    bloque = None
    if csv:
        bloque = medidas.extraer(
            csv, {t.id: t.intencion for t in tareas},
            participante=participante, condicion=condicion,
            conjunto=diseno.asignar(participante).conjuntos[
                list(diseno.asignar(participante).secuencia).index(condicion)])
        EstadoEstudio.cargar().registrar(
            participante, condicion, csv, len(bloque.enunciados))

    return {
        "codigo": codigo,
        "csv": csv,
        "enunciados": len(bloque.enunciados) if bloque else 0,
        "incompleto": bool(bloque and bloque.incompleto),
        "resumen": bloque.resumen() if bloque else {},
    }


def _abrir_tablero(cfg: dict, tareas: list, monitor: str, camera: int,
                   model: str | None) -> int:
    """Llama al tablero por mirada con la condición ya aplicada."""
    sys.path.insert(0, str(RAIZ / "slm"))
    sys.path.insert(0, str(RAIZ / "Eyetrax"))
    from aac_eyetrax.config import Settings
    from aac_eyetrax.display import pick_screen
    from aac_semantic import gaze_board

    if str(monitor).lower() == "ask":
        pantalla = pick_screen("¿En qué pantalla va a estar el tablero?")
        if pantalla is None:
            return 130
        monitor = pantalla.index
    ajustes = Settings(monitor=int(monitor), camera_index=camera)
    ajustes.validate()
    return gaze_board.run(
        ajustes,
        model=model,
        guion=[(t.id, t.pista) for t in tareas],
        expandir=cfg["expandir"],
        contexto=protocolo.CASO.contexto_modelo if cfg["contexto"] else "",
        pagina_inicial=cfg["pagina_inicial"],
    )
