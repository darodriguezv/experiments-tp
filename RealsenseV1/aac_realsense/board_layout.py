"""Definición del tablero CAA y su geometría.

Las filas pueden tener distinta cantidad de celdas: cada fila se reparte a lo
ancho de forma independiente, así el vocabulario se edita a mano sin pelearse
con la cuadrícula.

Para adaptar el tablero a una persona concreta, edite `PAGES` y nada más.
"""

from __future__ import annotations

from dataclasses import dataclass

# Tipos de celda
WORD = "word"      # inserta la etiqueta con un espacio delante
LETTER = "letter"  # inserta la etiqueta tal cual
ACTION = "action"  # tecla de control, la maneja la aplicación


@dataclass(frozen=True)
class Key:
    label: str
    kind: str = WORD
    value: str | None = None
    """Contenido: el texto a insertar, o el nombre de la acción."""

    @property
    def payload(self) -> str:
        return self.value if self.value is not None else self.label


@dataclass
class Tile:
    key: Key
    x1: int
    y1: int
    x2: int
    y2: int
    tile_id: str
    """Identidad estable para el seguimiento de la permanencia (página+fila+col)."""

    def contains(self, x: int, y: int) -> bool:
        return self.x1 <= x < self.x2 and self.y1 <= y < self.y2


def _a(label: str, action: str) -> Key:
    return Key(label, ACTION, action)


def _l(ch: str) -> Key:
    return Key(ch, LETTER, ch)


CONTROLS_CORE = [
    _a("ABC", "page:abc"),
    _a("ESPACIO", "space"),
    _a("BORRAR", "delete"),
    _a("LIMPIAR", "clear"),
    _a("HABLAR", "speak"),
]

CONTROLS_ABC = [
    _a("PALABRAS", "page:core"),
    _a("ESPACIO", "space"),
    _a("BORRAR", "delete"),
    _a("LIMPIAR", "clear"),
    _a("HABLAR", "speak"),
]

PAGES: dict[str, list[list[Key]]] = {
    "core": [
        [Key("SÍ"), Key("NO"), Key("POR FAVOR"), Key("GRACIAS"), Key("HOLA")],
        [Key("QUIERO"), Key("NECESITO"), Key("ME SIENTO"), Key("AYUDA"),
         Key("BASTA")],
        [Key("AGUA"), Key("COMIDA"), Key("BAÑO"), Key("DOLOR"), Key("MEDICINA")],
        [Key("MÁS"), Key("MENOS"), Key("BIEN"), Key("MAL"), Key("TAL VEZ")],
        CONTROLS_CORE,
    ],
    "abc": [
        [_l(c) for c in "ABCDEF"],
        [_l(c) for c in "GHIJKL"],
        [_l(c) for c in "MNÑOPQ"],
        [_l(c) for c in "RSTUVW"],
        [_l("X"), _l("Y"), _l("Z"), _l("."), _l(","), _l("?")],
        CONTROLS_ABC,
    ],
}


def layout(
    page: str,
    screen_w: int,
    screen_h: int,
    *,
    message_h_ratio: float = 0.16,
    gap: int = 10,
) -> list[Tile]:
    """Convierte la definición de una página en celdas posicionadas."""
    rows = PAGES[page]
    top = int(screen_h * message_h_ratio)
    grid_h = screen_h - top

    tiles: list[Tile] = []
    row_h = grid_h / len(rows)
    for r, row in enumerate(rows):
        col_w = screen_w / len(row)
        for c, key in enumerate(row):
            x1 = int(c * col_w) + gap
            x2 = int((c + 1) * col_w) - gap
            y1 = top + int(r * row_h) + gap
            y2 = top + int((r + 1) * row_h) - gap
            tiles.append(Tile(key, x1, y1, x2, y2, f"{page}:{r}:{c}"))
    return tiles


def message_area(screen_w: int, screen_h: int, message_h_ratio: float = 0.16):
    return 0, 0, screen_w, int(screen_h * message_h_ratio)
