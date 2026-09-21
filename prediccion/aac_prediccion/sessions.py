"""Reconstruir enunciados desde los CSV que el tablero ya escribe.

`SessionLog` de RealsenseV1 guarda una fila por selección desde el primer día,
con la etiqueta de la celda y el instante. Eso alcanza para reconstruir qué
frases se dijeron, sin agregarle nada al tablero ni pedirle a nadie que anote
nada. El historial que haga falta ya se está juntando.

Dos decisiones que cambian lo que el modelo aprende:

* **Solo cuenta lo que llegó a HABLAR.** Un mensaje escrito y borrado es
  justamente lo que la persona decidió no decir. Aprenderlo empujaría al
  sistema hacia los errores del propio usuario y hacia los de la mirada: una
  celda seleccionada por error y corregida enseguida es ruido del estimador,
  no vocabulario.
* **BORRAR y LIMPIAR se aplican de verdad.** Reproducir la edición en vez de
  quedarse con la lista de celdas tocadas es la diferencia entre aprender la
  frase que se dijo y aprender el camino accidentado hasta ella.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

from .replay import Utterance

CONTROLS = {"ESPACIO", "BORRAR", "LIMPIAR", "HABLAR", "ABC", "PALABRAS"}


def _session_start(path: Path) -> float:
    """El instante absoluto del CSV: del nombre, con la fecha del archivo como
    respaldo. El log guarda segundos desde el arranque, y sin una base absoluta
    el decaimiento por recencia no tendría contra qué medirse."""
    stem = path.stem
    for marker in ("aac-session-", "session-"):
        if marker in stem:
            stamp = stem.split(marker, 1)[1]
            try:
                return time.mktime(time.strptime(stamp, "%Y%m%d-%H%M%S"))
            except ValueError:
                break
    return path.stat().st_mtime


def _flush(buffer: list[str], pending: list[str]) -> None:
    if pending:
        buffer.append("".join(pending).lower())
        pending.clear()


def parse_csv(path: Path) -> list[Utterance]:
    """Un CSV de sesión -> los enunciados que se dijeron en voz alta."""
    t0 = _session_start(path)
    out: list[Utterance] = []
    tokens: list[str] = []
    letters: list[str] = []

    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("event") != "select":
                continue
            label = (row.get("label") or "").strip()
            if not label:
                continue
            try:
                t = t0 + float(row.get("t_since_start_s") or 0.0)
            except ValueError:
                t = t0

            upper = label.upper()
            if upper == "HABLAR":
                _flush(tokens, letters)
                if tokens:
                    out.append(Utterance(list(tokens), t))
                tokens, letters = [], []
            elif upper == "LIMPIAR":
                tokens, letters = [], []
            elif upper == "BORRAR":
                if letters:
                    letters.pop()
                elif tokens:
                    tokens.pop()
            elif upper == "ESPACIO":
                _flush(tokens, letters)
            elif upper in CONTROLS:
                continue                      # cambio de página: no es texto
            elif len(label) == 1 and not label.isspace():
                letters.append(label)         # deletreo en curso
            else:
                _flush(tokens, letters)
                tokens.append(label.lower())
    return out


def load(path: Path) -> tuple[list[Utterance], int]:
    """Todos los enunciados de una carpeta de logs, en orden cronológico.

    Devuelve también cuántas sesiones no aportaron ninguno, que es una señal
    útil: muchas sesiones sin un solo HABLAR normalmente significa que se
    estaba calibrando o probando, no hablando.
    """
    files = ([path] if path.is_file()
             else sorted(path.glob("**/*.csv")) if path.exists() else [])
    out: list[Utterance] = []
    vacias = 0
    for file in files:
        try:
            got = parse_csv(file)
        except Exception:
            vacias += 1
            continue
        if not got:
            vacias += 1
        out.extend(got)
    out.sort(key=lambda u: u.t)
    return out, vacias
