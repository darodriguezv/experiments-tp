"""El tablero con la franja de predicción, para probarlo a mano.

Lo que hay que mirar mientras se usa: el contador de arriba a la derecha. Va
sumando las permanencias que costó de verdad contra las que habría costado sin
la franja. Ese número **no** es el del `sim`: son dos usuarios distintos -acá el
usuario sos vos- y la única forma de saber si el mecanismo sirve en la práctica
es verlo con frases propias.

Aprende en vivo y guarda después de cada HABLAR, así que cerrar y volver a
abrir conserva lo aprendido. Es la diferencia entre probar una demo y probar el
módulo: la segunda vez que se dice algo tiene que costar menos.

El tablero de abajo **nunca se reordena**. Las casillas de arriba cambian; las
celdas del vocabulario se quedan donde están, por la razón que explica
`predict.Predictor`.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

from . import board, predict
from .store import UserModel

_SIBLING = Path(__file__).resolve().parents[2] / "RealsenseV1"
if str(_SIBLING) not in sys.path:
    sys.path.insert(0, str(_SIBLING))

from aac_realsense import board_layout, tts, ui  # noqa: E402

WINDOW = "CAA - predicción personalizada (demo)"
WIDTH, HEIGHT = 1280, 820
STRIP_TOP = 104
STRIP_H = 76
STATUS_H = 78


class State:
    def __init__(self, model: UserModel, path: Path, slots: int):
        self.model = model
        self.path = path
        self.slots = slots
        self.predictor = predict.Predictor(model, slots=slots)
        self.words: list[str] = []
        self.letters: list[str] = []
        self.page = board.DEFAULT_PAGE
        self.page_at_start = board.DEFAULT_PAGE
        self.dwells = 0
        """Permanencias gastadas en el enunciado en curso. Cada clic es una."""
        self.total_used = 0
        self.total_baseline = 0
        self.last: str = ""
        self.last_at = 0.0
        self.saved_now = 0

    @property
    def tokens(self) -> list[str]:
        out = list(self.words)
        if self.letters:
            out.append("".join(self.letters).lower())
        return out

    def proposals(self):
        # Se predice sobre lo ya confirmado: a mitad de deletrear una palabra
        # el prefijo todavía no es un token, y ofrecer sobre él sería adivinar
        # dos cosas a la vez. Es la limitación que el README anota como el
        # ahorro grande que queda sin tomar.
        if self.letters:
            return []
        return self.predictor.propose(self.words, self.page)


def _draw_message(img, state: State) -> None:
    text = " ".join(state.tokens).upper() or "TOQUE LAS CELDAS PARA ARMAR UNA FRASE"
    ui.rounded_rect(img, 12, 10, WIDTH - 13, 92, (30, 30, 34))
    size = ui.fit_text_size(text, WIDTH - 420, 54, bold=True, cap=44)
    ui.text_center(img, text, (WIDTH - 340) // 2, 51, size,
                   ui.TEXT_DIM if not state.tokens else ui.TEXT, bold=True)

    # El contador. Es lo que hay que mirar.
    base_now, _ = board.dwells_for(state.tokens, state.page_at_start)
    ui.text_at(img, "esta frase", WIDTH - 330, 20, 17, ui.TEXT_DIM)
    ui.text_at(img, f"{state.dwells} permanencias", WIDTH - 330, 40, 26,
               ui.TEXT, bold=True)
    ui.text_at(img, f"sin predecir: {base_now}", WIDTH - 330, 70, 18,
               ui.TEXT_DIM)

    if state.total_baseline:
        ahorro = 1.0 - state.total_used / state.total_baseline
        color = ui.GOOD if ahorro > 0 else ui.TEXT_DIM
        ui.text_at(img, "sesión", WIDTH - 150, 20, 17, ui.TEXT_DIM)
        ui.text_at(img, f"{ahorro:+.0%}", WIDTH - 150, 40, 26, color, bold=True)
        ui.text_at(img, f"{state.total_used} de {state.total_baseline}",
                   WIDTH - 150, 70, 18, ui.TEXT_DIM)


def _draw_strip(img, state: State) -> list:
    """La franja de predicción. Posición fija, siempre las mismas coordenadas."""
    proposals = state.proposals()
    zones = []
    gap = 10
    cell_w = (WIDTH - 24 - gap * (state.slots - 1)) // state.slots
    for i in range(state.slots):
        x1 = 12 + i * (cell_w + gap)
        x2 = x1 + cell_w
        if i < len(proposals):
            prop = proposals[i]
            ui.rounded_rect(img, x1, STRIP_TOP, x2, STRIP_TOP + STRIP_H,
                            (40, 58, 48))
            size = ui.fit_text_size(prop.label.upper(), cell_w - 18,
                                    STRIP_H - 34, bold=True, cap=26)
            ui.text_center(img, prop.label.upper(), (x1 + x2) // 2,
                           STRIP_TOP + STRIP_H // 2 - 6, size, ui.TEXT,
                           bold=True)
            # Lo que ahorra, escrito. Sin esto la franja es magia; con esto se
            # ve por qué una celda aparece y otra no.
            ui.text_center(img, f"-{prop.saving}", (x1 + x2) // 2,
                           STRIP_TOP + STRIP_H - 14, 16, ui.GOOD)
            zones.append((prop, (x1, STRIP_TOP, x2, STRIP_TOP + STRIP_H)))
        else:
            ui.rounded_rect(img, x1, STRIP_TOP, x2, STRIP_TOP + STRIP_H,
                            (28, 28, 32))
    if not proposals:
        msg = ("deletreando: la predicción espera al ESPACIO"
               if state.letters else
               "sin evidencia suficiente todavía - decí algunas frases")
        ui.text_center(img, msg, WIDTH // 2, STRIP_TOP + STRIP_H // 2, 20,
                       ui.TEXT_DIM)
    return zones


def _draw_board(img, state: State) -> list:
    tiles = board_layout.layout(state.page, WIDTH,
                                HEIGHT - STATUS_H - STRIP_TOP - STRIP_H,
                                message_h_ratio=0.0)
    offset = STRIP_TOP + STRIP_H + 12
    for tile in tiles:
        tile.y1 += offset
        tile.y2 += offset
        w, h = tile.x2 - tile.x1, tile.y2 - tile.y1
        if w <= 0 or h <= 0:
            continue
        base = ui.TILE
        if tile.key.kind == board_layout.ACTION:
            base = tuple(min(255, c + 16) for c in base)
        ui.rounded_rect(img, tile.x1, tile.y1, tile.x2, tile.y2, base)
        size = ui.fit_text_size(tile.key.label, w - 18, h - 18, bold=True)
        ui.text_center(img, tile.key.label, (tile.x1 + tile.x2) // 2,
                       (tile.y1 + tile.y2) // 2, size,
                       ui.ACCENT if tile.key.kind == board_layout.ACTION
                       else ui.TEXT, bold=True)
    return tiles


def _draw_status(img, state: State) -> None:
    lines = [
        f"aprendidas {state.model.utterances_seen} frases   ->  {state.path.name}"
        f"   |   {state.slots} casillas   |   ESC salir",
    ]
    if state.last and time.time() - state.last_at < 5.0:
        extra = (f"  ({state.saved_now} permanencias ahorradas)"
                 if state.saved_now else "")
        lines.append(f"dicho: {state.last!r}{extra}")
    ui.status_bar(img, lines, y=HEIGHT - 52, size=19, color=ui.TEXT_DIM)


def _speak(state: State) -> None:
    tokens = state.tokens
    if not tokens:
        return
    state.dwells += 1                       # la permanencia de HABLAR
    baseline, _ = board.dwells_for(tokens, state.page_at_start)
    baseline += board.SPEAK_DWELLS

    text = " ".join(tokens)
    tts.speak(text)

    state.total_used += state.dwells
    state.total_baseline += baseline
    state.saved_now = baseline - state.dwells
    state.last = text
    state.last_at = time.time()

    # Aprende y guarda. Solo lo que llegó a HABLAR, igual que `sessions.py`:
    # un borrador abandonado es lo que la persona decidió NO decir.
    state.model.observe(tokens)
    state.model.save(state.path)

    state.words.clear()
    state.letters.clear()
    state.dwells = 0
    state.page_at_start = state.page


def _on_tile(state: State, key) -> None:
    payload = key.payload
    state.dwells += 1
    if key.kind in (board_layout.WORD, board_layout.LETTER):
        if key.kind == board_layout.LETTER:
            state.letters.append(key.label)
        else:
            if state.letters:
                state.words.append("".join(state.letters).lower())
                state.letters.clear()
            state.words.append(key.label.lower())
        return
    if payload.startswith("page:"):
        state.page = payload.split(":", 1)[1]
        return
    if payload == "space":
        if state.letters:
            state.words.append("".join(state.letters).lower())
            state.letters.clear()
        return
    if payload == "delete":
        if state.letters:
            state.letters.pop()
        elif state.words:
            state.words.pop()
        return
    if payload == "clear":
        state.words.clear()
        state.letters.clear()
        # Las permanencias gastadas no se devuelven: se gastaron. Restarlas
        # haría que borrarse saliera gratis y el contador mentiría a favor.
        return
    if payload == "speak":
        _speak(state)


def run(model_path: str | None = None, slots: int = 5) -> int:
    path = Path(model_path or "mio.user.json")
    state = State(UserModel.load(path), path, slots)

    print(f"[demo] modelo: {path}  ({state.model.utterances_seen} frases "
          f"aprendidas)")
    print("[demo] clic para armar la frase, HABLAR para decirla y aprenderla.")
    print("[demo] la franja de arriba son las predicciones; el número verde "
          "es lo que ahorra cada una.")
    voice = tts.voice_name()
    print(f"[demo] voz: {voice or 'ninguna en español (se imprime el texto)'}")

    click: dict = {"at": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click["at"] = (x, y)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, WIDTH, HEIGHT)
    cv2.setMouseCallback(WINDOW, on_mouse)

    try:
        while True:
            img = ui.canvas(WIDTH, HEIGHT)
            _draw_message(img, state)
            zones = _draw_strip(img, state)
            tiles = _draw_board(img, state)
            _draw_status(img, state)

            cv2.imshow(WINDOW, img)
            if (cv2.waitKey(30) & 0xFF) == 27:
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break

            at = click["at"]
            click["at"] = None
            if at is None:
                continue
            x, y = at

            hit = False
            for prop, (x1, y1, x2, y2) in zones:
                if x1 <= x < x2 and y1 <= y < y2:
                    state.dwells += 1
                    if state.letters:
                        state.words.append("".join(state.letters).lower())
                        state.letters.clear()
                    state.words.extend(prop.tokens)
                    hit = True
                    break
            if hit:
                continue
            for tile in tiles:
                if tile.contains(x, y):
                    _on_tile(state, tile.key)
                    break
    finally:
        state.model.save(state.path)
        cv2.destroyAllWindows()
    print(f"[demo] guardado {state.path} con "
          f"{state.model.utterances_seen} frases")
    if state.total_baseline:
        print(f"[demo] sesión: {state.total_used} permanencias contra "
              f"{state.total_baseline} sin predicción "
              f"({1 - state.total_used / state.total_baseline:+.0%})")
    return 0
