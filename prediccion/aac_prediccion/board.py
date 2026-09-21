"""El tablero visto como un costo en permanencias.

Este módulo no dibuja nada: traduce «decir esta frase» a «cuántas veces hay que
fijar la mirada», que es la única moneda que importa acá. A ~800 ms por
permanencia más el tiempo de encontrar la celda con los ojos, cada permanencia
ahorrada es tiempo real de una persona que no tiene otra forma de hablar.

El tablero se importa de RealsenseV1 en vez de copiarse, igual que hace el demo
del módulo de expansión. Una copia se desincroniza en silencio y entonces este
módulo estaría midiendo un tablero que ya no existe.
"""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

_SIBLING = Path(__file__).resolve().parents[2] / "RealsenseV1"
if str(_SIBLING) not in sys.path:
    sys.path.insert(0, str(_SIBLING))

try:
    from aac_realsense import board_layout
except ImportError as exc:  # pragma: no cover - depende del árbol de archivos
    raise ImportError(
        f"no se pudo importar el tablero desde {_SIBLING}: {exc}\n"
        f"Este módulo mide el tablero real, no una copia."
    ) from None


SPEAK_DWELLS = 1
"""HABLAR, al final de cada enunciado. Se cobra igual con predicción y sin
ella, así que no cambia el ahorro - pero omitirlo inflaría el porcentaje."""


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def _build_index() -> tuple[dict[str, str], set[str], dict[str, int]]:
    """Dónde vive cada símbolo, y cuántas celdas tiene que leer la vista."""
    word_page: dict[str, str] = {}
    letters: set[str] = set()
    cells: dict[str, int] = {}
    for page, rows in board_layout.PAGES.items():
        cells[page] = sum(len(row) for row in rows)
        for row in rows:
            for key in row:
                if key.kind == board_layout.WORD:
                    word_page.setdefault(key.label.lower(), page)
                elif key.kind == board_layout.LETTER:
                    letters.add(key.label.lower())
    return word_page, letters, cells


WORD_PAGE, LETTERS, PAGE_CELLS = _build_index()

VOCAB = tuple(sorted(WORD_PAGE))
"""Las palabras que el tablero ofrece en una sola permanencia."""

DEFAULT_PAGE = "core"


def page_of(token: str) -> str | None:
    """La página donde `token` es una celda, o None si hay que deletrearlo."""
    return WORD_PAGE.get(token.lower())


def spell_cost(token: str) -> int:
    """Permanencias para deletrear una palabra que el tablero no tiene.

    Una por letra más una por el ESPACIO final. Las tildes se cuentan sobre la
    letra base porque la página `abc` no las ofrece: quien escriba «está» va a
    poner ESTA. Eso es una limitación real del tablero, no una simplificación
    de la medición, y conviene que se vea acá en vez de esconderla.
    """
    plain = _strip_accents(token.lower())
    return len(plain) + 1


def dwells_for(tokens: list[str], start_page: str = DEFAULT_PAGE
               ) -> tuple[int, str]:
    """Costo de escribir `tokens` sin ninguna predicción.

    Devuelve las permanencias y en qué página queda el tablero, porque el costo
    del siguiente enunciado depende de dónde quedó el anterior.

    El cambio de página se cobra: es una permanencia sobre ABC o PALABRAS, y es
    exactamente el costo que la predicción puede ahorrar cuando la palabra que
    viene está en la otra página.
    """
    page = start_page
    total = 0
    for token in tokens:
        target = page_of(token)
        if target is not None:
            if target != page:
                total += 1
                page = target
            total += 1
        else:
            if page != "abc":
                total += 1
                page = "abc"
            total += spell_cost(token)
    return total, page


def cells_visible(page: str, slots: int = 0) -> int:
    """Celdas que la vista tiene que recorrer para elegir.

    Importa tanto como las permanencias y se olvida siempre. Cada casilla de
    predicción es una celda más que leer en **toda** selección, incluidas
    aquellas en las que la predicción falla. Es el mismo error que costó caro
    en el módulo de expansión: ahí se contaba un acierto aunque el usuario
    recibiera un solo candidato, acá sería contar el ahorro sin cobrar la
    lectura que lo hizo posible.
    """
    return PAGE_CELLS[page] + slots
