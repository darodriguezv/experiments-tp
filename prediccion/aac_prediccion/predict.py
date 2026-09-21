"""Qué poner en las casillas de predicción.

## El hallazgo que define este módulo

La intuición de SwiftKey es «predecí la próxima palabra». **En este tablero eso
no ahorra nada**, y conviene verlo con números antes de escribir una línea de
código.

En un teclado, escribir «necesito» cuesta ocho pulsaciones, así que ofrecerla
entera ahorra siete. En este tablero NECESITO es una celda de la página `core`:
cuesta **una** permanencia. Ofrecerla en una casilla de predicción también
cuesta una. El ahorro es **cero** - y encima se pagó una celda más de lectura
en toda selección, acertada o no. Un port ingenuo de SwiftKey sería puro costo.

Donde sí hay ahorro es en otro lado:

| situación | sin predicción | con | ahorro |
| --- | --- | --- | --- |
| palabra de la página actual | 1 | 1 | **0** |
| palabra de la otra página | 2 (cambio + celda) | 1 | 1 |
| frase de 4 palabras ya dicha | 4 | 1 | **3** |
| palabra que el tablero no tiene («hermana») | 9 (cambio + 7 letras + espacio) | 1 | **8** |

O sea: **el valor está en las frases y en lo que hay que deletrear**, no en la
próxima palabra. Por eso el ranking de abajo no ordena por probabilidad sino
por *probabilidad x permanencias ahorradas*, y por eso una predicción correcta
con ahorro cero nunca ocupa una casilla.

Es el mismo principio que salió en el módulo de expansión: lo que se optimiza
tiene que ser lo que le sirve a la persona, no lo que el modelo hace bien.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import board
from .store import UserModel


@dataclass
class Proposal:
    """Una casilla candidata, con lo que cuesta y lo que ahorra."""

    tokens: list[str]
    p: float
    """Probabilidad estimada dentro de su fuente. No está calibrada entre
    fuentes distintas; es una heurística de ranking y se declara como tal."""
    saving: int
    """Permanencias que ahorra si acierta. Puede ser 0, y entonces se descarta."""
    source: str

    @property
    def score(self) -> float:
        return self.p * self.saving

    @property
    def label(self) -> str:
        return " ".join(self.tokens)


class Predictor:
    """Llena `slots` casillas fijas. Nunca reordena el tablero.

    Esto último no es un detalle de implementación. SwiftKey reordena sus
    sugerencias, pero **el teclado de abajo nunca se mueve**, y acá la razón es
    mucho más fuerte: quien maneja el tablero con la mirada aprende dónde está
    cada celda y va directo. Mover AGUA porque esta semana se pidió mucha
    destruye esa memoria espacial y obliga a barrer el tablero de nuevo - se
    pierde más en búsqueda visual de lo que cualquier predicción pueda ahorrar,
    y se pierde justo cuando la persona tiene más apuro.

    Por eso las predicciones viven en una franja de posición fija, y el
    vocabulario de `PAGES` se queda quieto para siempre.
    """

    def __init__(self, model: UserModel, slots: int = 5):
        self.model = model
        self.slots = slots

    def propose(self, tokens_so_far: list[str], page: str = board.DEFAULT_PAGE,
                now: float | None = None) -> list[Proposal]:
        prefix = tuple(tokens_so_far)
        out: list[Proposal] = []

        # --- frases enteras ---------------------------------------------------
        completions = self.model.phrase_completions(prefix, now,
                                                    limit=self.slots * 3)
        total = sum(w for _, w in completions)
        for tokens, w in completions:
            rest = tokens[len(prefix):]
            if not rest:
                continue
            cost, _ = board.dwells_for(rest, page)
            out.append(Proposal(rest, w / total if total else 0.0,
                                cost - 1, "frase"))

        # --- símbolo siguiente ------------------------------------------------
        nexts = self.model.next_symbols(prefix, now, limit=self.slots * 3)
        total = sum(w for _, w in nexts)
        for symbol, w in nexts:
            cost, _ = board.dwells_for([symbol], page)
            out.append(Proposal([symbol], w / total if total else 0.0,
                                cost - 1, "palabra"))

        # Ahorro cero = celda que solo cuesta leerla. Fuera, por probable que sea.
        out = [p for p in out if p.saving > 0]

        seen: set[tuple[str, ...]] = set()
        unique: list[Proposal] = []
        for prop in sorted(out, key=lambda p: -p.score):
            key = tuple(prop.tokens)
            if key in seen:
                continue
            seen.add(key)
            unique.append(prop)
        return unique[:self.slots]
