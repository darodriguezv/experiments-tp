"""Medir el ahorro sin hacer trampa.

La regla que gobierna este archivo: **el modelo solo ve el pasado**. Cada
enunciado se predice con lo aprendido de los anteriores y recién después se
observa. Es obvio dicho así y es fácil de romper sin darse cuenta - entrenar
con todo el historial y después medir sobre el mismo historial da números
excelentes y completamente falsos, porque mide memoria, no predicción.

Lo que se cobra y por qué:

* **Permanencias**, que son el tiempo de la persona.
* **Celdas leídas**, que son su atención. Una franja de 5 casillas agrega 5
  celdas a *toda* selección, también a las que no usan ninguna. Un informe que
  muestre solo el ahorro de permanencias está escondiendo la mitad del trato.
* **Aciertos**, para separar «predice poco» de «predice mucho y mal».
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import board
from .predict import Predictor
from .store import UserModel


@dataclass
class Utterance:
    """Algo que la persona efectivamente dijo, con cuándo lo dijo."""

    tokens: list[str]
    t: float


@dataclass
class Result:
    baseline_dwells: int = 0
    predicted_dwells: int = 0
    steps: int = 0
    """Decisiones tomadas con predicción: una por permanencia gastada."""
    hits: int = 0
    slots_shown: int = 0
    utterances: int = 0
    cells_baseline: int = 0
    cells_predicted: int = 0
    per_utterance: list[tuple[int, int]] = field(default_factory=list)

    @property
    def savings(self) -> float:
        if not self.baseline_dwells:
            return 0.0
        return 1.0 - self.predicted_dwells / self.baseline_dwells

    @property
    def hit_rate(self) -> float:
        return self.hits / self.steps if self.steps else 0.0

    @property
    def cells_per_selection(self) -> float:
        return (self.cells_predicted / self.predicted_dwells
                if self.predicted_dwells else 0.0)

    @property
    def cells_per_selection_baseline(self) -> float:
        return (self.cells_baseline / self.baseline_dwells
                if self.baseline_dwells else 0.0)

    @property
    def dwells_per_utterance(self) -> tuple[float, float]:
        if not self.utterances:
            return 0.0, 0.0
        return (self.baseline_dwells / self.utterances,
                self.predicted_dwells / self.utterances)


def _with_prediction(tokens: list[str], page: str, predictor: Predictor,
                     now: float) -> tuple[int, str, int, int, int]:
    """Recorre un enunciado usando la franja de predicción cuando sirve.

    La política es voraz: de las casillas que coinciden con lo que viene, se
    toma la que consume más palabras. No es óptima en teoría - una elección
    peor ahora podría habilitar una mucho mejor después - pero es la única que
    una persona puede ejecutar: nadie mira cinco casillas calculando el camino
    de mínimo costo de una frase que todavía no terminó de pensar.
    """
    i = 0
    dwells = 0
    slots_shown = 0
    hits = 0
    cells = 0
    while i < len(tokens):
        proposals = predictor.propose(tokens[:i], page, now)
        best = None
        for prop in proposals:
            n = len(prop.tokens)
            if tokens[i:i + n] == prop.tokens and (best is None
                                                   or n > len(best.tokens)):
                best = prop
        cells += board.cells_visible(page, len(proposals))
        slots_shown += len(proposals)
        if best is not None:
            dwells += 1
            hits += 1
            i += len(best.tokens)
            continue
        # Sin casilla útil: un paso del camino normal. Ojo con el cambio de
        # página, que se cobra acá igual que en el camino sin predicción.
        cost, page = board.dwells_for([tokens[i]], page)
        dwells += cost
        # Las permanencias intermedias de deletreo también leen la pantalla.
        cells += board.cells_visible(page, len(proposals)) * (cost - 1)
        i += 1
    return dwells, page, slots_shown, hits, cells


def _baseline_cells(tokens: list[str], page: str) -> tuple[int, int, str]:
    dwells = 0
    cells = 0
    for token in tokens:
        cost, page = board.dwells_for([token], page)
        dwells += cost
        cells += board.cells_visible(page) * cost
    return dwells, cells, page


def replay(utterances: list[Utterance], model: UserModel | None = None,
           slots: int = 5, learn: bool = True) -> Result:
    """Reproduce una historia en orden y devuelve el ahorro.

    `learn=False` congela el modelo: sirve para separar cuánto aporta lo que ya
    sabía de cuánto aporta seguir aprendiendo durante la sesión.
    """
    model = model if model is not None else UserModel()
    predictor = Predictor(model, slots=slots)
    result = Result()

    page_base = board.DEFAULT_PAGE
    page_pred = board.DEFAULT_PAGE

    for utt in sorted(utterances, key=lambda u: u.t):
        if not utt.tokens:
            continue

        base, base_cells, page_base = _baseline_cells(utt.tokens, page_base)
        base += board.SPEAK_DWELLS
        base_cells += board.cells_visible(page_base)

        pred, page_pred, shown, hits, pred_cells = _with_prediction(
            utt.tokens, page_pred, predictor, utt.t)
        pred += board.SPEAK_DWELLS
        pred_cells += board.cells_visible(page_pred, slots)

        result.baseline_dwells += base
        result.predicted_dwells += pred
        result.cells_baseline += base_cells
        result.cells_predicted += pred_cells
        result.slots_shown += shown
        result.hits += hits
        result.steps += pred
        result.utterances += 1
        result.per_utterance.append((base, pred))

        # Y RECIÉN AHORA el modelo se entera. Mover esta línea arriba del
        # bloque anterior convierte toda la medición en una mentira.
        if learn:
            model.observe(utt.tokens, utt.t)

    return result
