"""El tablero, clickeable, con la expansión semántica enchufada.

Para qué existe: los números de `report` dicen si un modelo es fiel, pero no
dicen cómo se *siente* esperar dos segundos con la frase a medio armar, ni si
tres candidatos son suficientes cuando la frase es propia y no del corpus. Eso
hay que probarlo con el dedo.

Es el flujo real del producto menos la mirada: se arma la frase telegráfica
tocando celdas, se pide la expansión, y se elige uno de los candidatos. Donde el
sistema final pone permanencia de 800 ms, acá hay un clic — así que **la latencia
del modelo se siente más de lo que se va a sentir en uso real**, donde el usuario
ya venía gastando casi un segundo por palabra.

El tablero, el renderizado de texto y la voz se importan de
[RealsenseV1](../RealsenseV1) en vez de copiarse. No es solo evitar duplicación:
es que sea literalmente el mismo tablero, con el mismo vocabulario, así que lo
que se prueba acá es lo que va a haber.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from . import backends, guard, prompts

# El módulo de mirada es un hermano en el árbol de experimentos. Importarlo así
# mantiene un solo tablero y un solo renderizador de texto en el proyecto.
_SIBLING = Path(__file__).resolve().parents[2] / "RealsenseV1"
if str(_SIBLING) not in sys.path:
    sys.path.insert(0, str(_SIBLING))

try:
    from aac_realsense import board_layout, tts, ui
except ImportError as exc:  # pragma: no cover - depende del árbol de archivos
    raise ImportError(
        f"no se pudo importar el tablero desde {_SIBLING}: {exc}\n"
        f"El demo reutiliza board_layout, ui y tts de RealsenseV1 para que sea "
        f"el mismo tablero y no una copia que se desincroniza."
    ) from None

WINDOW = "CAA - expansión semántica (demo)"
WIDTH, HEIGHT = 1280, 800

# El 3B primero: es el que mejor mide (88 % de recall contra 70 % del 1.5B) y
# tarda 0.34 s, así que no hay razón para arrancar en uno peor. El 1.5B era el
# default y contestaba "POR FAVOR COMIDA" con un solo "comida".
MODELS = ["qwen2.5:3b", "qwen2.5:7b", "qwen2.5:1.5b", "llama3.2:3b",
          "phi3.5", "gemma2:2b", "qwen2.5:0.5b", "llama3.2:1b", "smollm2:1.7b"]
"""El 3b va primero porque es el que mide mejor, no por ser el mas chico: el 7b
empata en recall y DEVUELVE MENOS OPCIONES (2.39 contra 3.00), que es
justamente lo que se necesita. Esta en la lista para poder compararlos en vivo
con la tecla M."""

DEFAULT_VARIANT = "v6-glosario"
"""La única configuración que pasa todas las puertas de `metrics.usable`, junto
con qwen2.5:3b.

Se guarda por **nombre** y no por índice, y eso es el arreglo de un bug real: el
default era `variant_i = 3` con un comentario que decía "v4-literal". Cuando se
agregó v6 el índice 3 pasó a ser v3-json, así que el demo venía arrancando en
una variante sin la exigencia del literal ni el glosario - y el comentario
seguía afirmando lo contrario. Un índice como default es una referencia que se
rompe en silencio al insertar algo antes."""


@dataclass
class State:
    """Todo lo que la ventana necesita saber, en un lugar.

    Mutado desde el hilo del modelo y leído desde el hilo de dibujo, así que
    todo lo compartido pasa por `lock`. Sin eso la ventana puede dibujar una
    lista de candidatos a medio escribir - que no rompe nada, pero parpadea de
    una forma que hace imposible juzgar la latencia, que es justo lo que este
    demo mide.
    """

    words: list[str] = field(default_factory=list)
    page: str = "core"
    model_i: int = 0
    variant_name: str = DEFAULT_VARIANT

    candidates: list[str] = field(default_factory=list)
    thinking: bool = False
    latency_s: float = 0.0
    fmt: str = ""
    error: str = ""
    spoken: str = ""
    spoken_at: float = 0.0
    dropped: list = field(default_factory=list)
    fell_back: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def model(self) -> str:
        return MODELS[self.model_i % len(MODELS)]

    @property
    def variant(self) -> prompts.Prompt:
        return prompts.VARIANTS[self.variant_name]

    def cycle_variant(self, step: int = 1) -> None:
        names = list(prompts.VARIANTS)
        i = names.index(self.variant_name)
        self.variant_name = names[(i + step) % len(names)]

    @property
    def telegraphic(self) -> str:
        return " ".join(self.words)


def _expand(state: State) -> None:
    """Llama al modelo en un hilo aparte y deja el resultado en `state`.

    En un hilo porque un SLM tarda entre medio segundo y tres, y una ventana de
    OpenCV congelada no distingue "pensando" de "colgado". Mostrar el estado
    explícitamente es parte de lo que hay que evaluar: el usuario tiene que
    saber que el sistema lo escuchó.
    """
    telegraphic = state.telegraphic
    variant = state.variant
    # keep_alive="-1": el modelo se queda en RAM mientras el tablero este
    # abierto. Con el default de 5 minutos de Ollama, una pausa hace que la
    # frase siguiente pague la recarga -4.3 s el 3b, 6.9 s el 7b- y puede ser
    # la urgente.
    backend = backends.OllamaBackend(state.model, temperature=0.2,
                                     keep_alive="-1")

    completion = backend.complete(variant.system, variant.user(telegraphic),
                                  json_mode=variant.json_mode)
    candidates, fmt = prompts.parse_candidates(completion.text)

    # La salida de escape del diseño: el literal siempre disponible, aunque el
    # modelo no lo haya incluido. Si los candidatos generados están todos mal,
    # la persona tiene que poder decir exactamente lo que escribió - el tablero
    # pelado ya podía, y este módulo no debería quitarle eso.
    # El filtro determinista: descarta lo que perdió la negación o la orden de
    # detenerse, deduplica, y garantiza el literal. La polaridad pasa a 0 % por
    # construcción en vez de por tendencia del modelo.
    result = guard.guarded(telegraphic, candidates)
    candidates = result.candidates

    with state.lock:
        state.dropped = result.dropped
        state.fell_back = result.fell_back
        state.candidates = candidates
        state.latency_s = completion.latency_s
        state.fmt = fmt
        state.error = completion.error
        state.thinking = False


# ------------------------------------------------------------------- dibujo


def _draw_header(img, state: State) -> None:
    with state.lock:
        text = state.telegraphic.upper() or "TOQUE LAS CELDAS PARA ARMAR UNA FRASE"
        empty = not state.words
    ui.rounded_rect(img, 12, 10, WIDTH - 13, 96, (30, 30, 34))
    size = ui.fit_text_size(text, WIDTH - 80, 60, bold=True, cap=48)
    ui.text_center(img, text, WIDTH // 2, 53, size,
                   ui.TEXT_DIM if empty else ui.TEXT, bold=True)


def _draw_status(img, state: State) -> None:
    with state.lock:
        lines = [
            f"modelo  {state.model}   prompt  {state.variant.variant}"
            + (f"   {state.latency_s:.2f}s" if state.latency_s else "")
            + (f"   formato {state.fmt}" if state.fmt else ""),
            "clic: agregar palabra   |   M modelo   V prompt   "
            "BORRAR/LIMPIAR   |   ESC salir",
        ]
        if state.error:
            lines.append(f"ERROR: {state.error[:110]}")
        if state.dropped:
            reasons = "; ".join(f"{t!r} ({why})" for t, why in state.dropped[:2])
            lines.append(f"filtro descartó: {reasons}")
        if state.fell_back:
            lines.append("el filtro dejó menos de 2 opciones: solo el literal "
                         "es seguro acá")
        if state.spoken and time.time() - state.spoken_at < 4.0:
            lines.append(f"dicho: {state.spoken!r}")
    ui.status_bar(img, lines, y=HEIGHT - 74, size=19,
                  color=ui.WARN if state.error else ui.TEXT_DIM)


def _draw_board(img, state: State) -> list:
    tiles = board_layout.layout(state.page, WIDTH, HEIGHT - 80,
                                message_h_ratio=0.16)
    for tile in tiles:
        w, h = tile.x2 - tile.x1, tile.y2 - tile.y1
        if w <= 0 or h <= 0:
            continue
        base = ui.TILE
        if tile.key.kind == board_layout.ACTION:
            base = tuple(min(255, c + 16) for c in base)
        ui.rounded_rect(img, tile.x1, tile.y1, tile.x2, tile.y2, base)
        size = ui.fit_text_size(tile.key.label, w - 20, h - 20, bold=True)
        ui.text_center(img, tile.key.label, (tile.x1 + tile.x2) // 2,
                       (tile.y1 + tile.y2) // 2, size,
                       ui.ACCENT if tile.key.kind == board_layout.ACTION
                       else ui.TEXT, bold=True)
    return tiles


def _draw_candidates(img, state: State) -> list:
    """La pantalla de elección. Devuelve las zonas clickeables."""
    with state.lock:
        candidates = list(state.candidates)
        thinking = state.thinking
        latency = state.latency_s
        model = state.model
        telegraphic = state.telegraphic

    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (WIDTH, HEIGHT), (10, 10, 12), -1)
    cv2.addWeighted(overlay, 0.88, img, 0.12, 0, img)

    if thinking:
        ui.text_center(img, "PENSANDO...", WIDTH // 2, HEIGHT // 2 - 30, 64,
                       ui.ACCENT, bold=True)
        ui.text_center(img, f"{model} está proponiendo opciones",
                       WIDTH // 2, HEIGHT // 2 + 40, 28, ui.TEXT_DIM)
        return []

    ui.text_center(img, "¿CUÁL QUISO DECIR?", WIDTH // 2, 60, 44, ui.TEXT,
                   bold=True)
    ui.text_center(img, f"{model}   {latency:.2f}s   "
                        f"toque una opción para decirla en voz alta",
                   WIDTH // 2, 108, 24, ui.TEXT_DIM)

    zones = []
    if not candidates:
        ui.text_center(img, "el modelo no devolvió nada utilizable",
                       WIDTH // 2, HEIGHT // 2, 34, ui.WARN, bold=True)
    else:
        top, gap = 150, 14
        # La última opción es siempre el literal, y se pinta distinto: es la
        # salida de escape, no una sugerencia del modelo.
        height = min(110, (HEIGHT - top - 130 - gap * len(candidates))
                     // max(len(candidates), 1))
        for i, candidate in enumerate(candidates):
            y1 = top + i * (height + gap)
            y2 = y1 + height
            is_literal = candidate.strip().lower() == telegraphic.lower()
            ui.rounded_rect(img, 80, y1, WIDTH - 81, y2,
                            (38, 38, 44) if is_literal else ui.TILE)
            size = ui.fit_text_size(candidate, WIDTH - 260, height - 26,
                                    bold=not is_literal, cap=40)
            ui.text_center(img, candidate, WIDTH // 2, (y1 + y2) // 2, size,
                           ui.TEXT_DIM if is_literal else ui.TEXT,
                           bold=not is_literal)
            if is_literal:
                ui.text_at(img, "literal", 96, y1 + 8, 16, ui.TEXT_DIM)
            zones.append(("say", candidate, (80, y1, WIDTH - 81, y2)))

    y1 = HEIGHT - 110
    ui.rounded_rect(img, 80, y1, WIDTH - 81, y1 + 62, (46, 40, 40))
    ui.text_center(img, "VOLVER AL TABLERO", WIDTH // 2, y1 + 31, 30,
                   ui.TEXT, bold=True)
    zones.append(("back", "", (80, y1, WIDTH - 81, y1 + 62)))
    return zones


# ------------------------------------------------------------------- entrada


def _on_tile(state: State, label: str, kind: str, payload: str) -> str:
    """Aplica un toque en el tablero. Devuelve una acción para el lazo."""
    if kind == board_layout.WORD or kind == board_layout.LETTER:
        with state.lock:
            state.words.append(label)
        return ""
    if payload.startswith("page:"):
        with state.lock:
            state.page = payload.split(":", 1)[1]
        return ""
    if payload == "space":
        return ""
    if payload == "delete":
        with state.lock:
            if state.words:
                state.words.pop()
        return ""
    if payload == "clear":
        with state.lock:
            state.words.clear()
            state.candidates.clear()
            state.latency_s = 0.0
            state.error = ""
        return ""
    if payload == "speak":
        return "expand"
    return ""


def run(model: str | None = None, variant: str | None = None) -> int:
    """Abre la ventana. Devuelve un código de salida."""
    probe = backends.OllamaBackend(MODELS[0])
    if not probe.available():
        print(backends.ollama_report())
        return 1

    installed = set(probe.models())
    available = [m for m in MODELS
                 if any(i == m or i.split(":")[0] == m.split(":")[0]
                        for i in installed)]
    if not available:
        print("Ningún modelo instalado. Ver `main.py pull`.")
        return 1

    state = State()
    # Solo se ofrecen los que están: ciclar con M hasta uno ausente daría un
    # error de red que parece un fallo del demo.
    MODELS[:] = available
    if model:
        if model not in MODELS:
            print(f"{model} no está instalado. Disponibles: "
                  f"{', '.join(MODELS)}")
            return 1
        state.model_i = MODELS.index(model)
    if variant:
        if variant not in prompts.VARIANTS:
            print(f"variante desconocida {variant!r}")
            return 1
        state.variant_name = variant

    print(f"[demo] modelos disponibles: {', '.join(MODELS)}")
    print(f"[demo] arrancando con {state.model} / {state.variant.variant}")
    print("[demo] clic para armar la frase, HABLAR para expandir. "
          "M cambia de modelo, V de prompt, ESC sale.")
    voice = tts.voice_name()
    print(f"[demo] voz: {voice or 'ninguna en español (se imprime el texto)'}")

    click: dict = {"at": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click["at"] = (x, y)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, WIDTH, HEIGHT)
    cv2.setMouseCallback(WINDOW, on_mouse)

    showing_candidates = False
    try:
        while True:
            img = ui.canvas(WIDTH, HEIGHT)
            _draw_header(img, state)

            with state.lock:
                busy = state.thinking
                has_candidates = bool(state.candidates)
            if busy or has_candidates:
                showing_candidates = True

            if showing_candidates:
                zones = _draw_candidates(img, state)
                tiles = []
            else:
                tiles = _draw_board(img, state)
                zones = []
            _draw_status(img, state)

            cv2.imshow(WINDOW, img)
            key = cv2.waitKey(30) & 0xFF

            if key == 27:
                break
            if key == ord("m"):
                with state.lock:
                    state.model_i += 1
                    state.latency_s = 0.0
                print(f"[demo] modelo -> {state.model}")
            if key == ord("v"):
                with state.lock:
                    state.cycle_variant()
                print(f"[demo] prompt -> {state.variant.variant}")
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break

            at = click["at"]
            click["at"] = None
            if at is None:
                continue
            x, y = at

            if showing_candidates:
                for kind, payload, (x1, y1, x2, y2) in zones:
                    if x1 <= x < x2 and y1 <= y < y2:
                        if kind == "back":
                            with state.lock:
                                state.candidates.clear()
                            showing_candidates = False
                        else:
                            tts.speak(payload)
                            with state.lock:
                                state.spoken = payload
                                state.spoken_at = time.time()
                                state.words.clear()
                                state.candidates.clear()
                            showing_candidates = False
                        break
                continue

            for tile in tiles:
                if tile.contains(x, y):
                    action = _on_tile(state, tile.key.label, tile.key.kind,
                                      tile.key.payload)
                    if action == "expand":
                        with state.lock:
                            if not state.words:
                                break
                            state.thinking = True
                            state.candidates.clear()
                            state.error = ""
                        showing_candidates = True
                        threading.Thread(target=_expand, args=(state,),
                                         daemon=True).start()
                    break
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    return 0
