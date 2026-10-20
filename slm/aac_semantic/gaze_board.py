"""El tablero por mirada con la expansión del SLM enchufada.

Es el dispositivo completo hasta donde llega hoy: la mirada elige celdas, las
celdas arman un mensaje telegráfico, y el SLM propone frases entre las que la
persona elige **también con la mirada**. Hasta acá cada pieza se probaba por
separado -el tablero con la mirada, la expansión con el mouse- y eso escondía
la pregunta que importa: si el costo de elegir entre candidatos se paga con la
mirada, ¿sigue valiendo la pena?

## Por qué EyeTrax y no GazeFollower

Medido y decidido acá: EyeTrax corre a 26 fps contra 14, su licencia es MIT
contra CC BY-NC-SA, y sobre esta persona y esta webcam su punto es
visiblemente más estable. GazeFollower puede ganar en precisión bruta -su paper
lo dice, acá no se comprobó- pero un cursor que tiembla no se puede usar por
más exacto que sea su promedio.

Se importa `GazeTracker` y no se copia, así que todas las mejoras del arnés
-permanencia por fijación, pegado de celda, guía de colocación- llegan solas.

## La calibración es obligatoria

No hay bandera para saltearla, igual que en los otros módulos. Un modelo de
mirada vale para la postura con la que se calibró, y usar uno viejo falla en
silencio: el cursor se ve igual de firme estando corrido.

## Lo que se registra

Cada selección va a un CSV con el mismo formato que lee `prediccion/sessions.py`,
así que usar esto **alimenta** el módulo de predicción sin pedirle nada a nadie.
"""

from __future__ import annotations

import csv
import sys
import threading
import time
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parents[2]
for _sib in ("RealsenseV1", "Eyetrax"):
    _p = str(_ROOT / _sib)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from aac_eyetrax import calibrate as eyetrax_calibrate  # noqa: E402
from aac_eyetrax import positioning  # noqa: E402
from aac_eyetrax.config import Settings  # noqa: E402
from aac_eyetrax.tracker import DwellSelector, GazeTracker  # noqa: E402
from aac_eyetrax.video import camera, iter_frames  # noqa: E402
from aac_realsense import board_layout, tts, ui  # noqa: E402

from . import demo  # noqa: E402

WINDOW = "CAA - mirada + expansión semántica"
LOGS = _ROOT / "slm" / "logs"


class _Log:
    """CSV compatible con `prediccion/sessions.py`.

    Mismo formato a propósito: el módulo de predicción aprende de los
    enunciados que se dijeron en voz alta, y si este tablero escribiera un
    formato propio habría que mantener dos lectores para el mismo dato.
    """

    def __init__(self, enabled: bool):
        self.file = None
        self.t0 = time.time()
        if not enabled:
            return
        LOGS.mkdir(parents=True, exist_ok=True)
        path = LOGS / f"aac-session-{time.strftime('%Y%m%d-%H%M%S')}.csv"
        self.file = path.open("w", newline="", encoding="utf-8")
        self.w = csv.writer(self.file)
        self.w.writerow(["t_since_start_s", "event", "label",
                         "gaze_x", "gaze_y"])
        print(f"[tablero] registrando -> {path}")

    def selection(self, label: str, x: int, y: int) -> None:
        if self.file is None:
            return
        self.w.writerow([f"{time.time() - self.t0:.3f}", "select", label, x, y])
        self.file.flush()

    def close(self) -> None:
        if self.file is not None:
            self.file.close()


def _candidate_zones(state, w: int, h: int) -> list[tuple[str, str, tuple]]:
    """Las franjas clickeables de la pantalla de candidatos.

    Se calculan aparte del dibujo para que la permanencia y el render usen
    exactamente la misma geometría: si difirieran, la barra de progreso
    aparecería sobre una franja y se seleccionaría otra.
    """
    with state.lock:
        candidates = list(state.candidates)
    zones = []
    top, gap = int(h * 0.20), 16
    alto = min(140, (int(h * 0.62) - gap * max(len(candidates), 1))
               // max(len(candidates), 1))
    for i, texto in enumerate(candidates):
        y1 = top + i * (alto + gap)
        zones.append(("say", texto, (int(w * 0.07), y1,
                                     int(w * 0.93), y1 + alto)))
    y1 = int(h * 0.86)
    zones.append(("back", "", (int(w * 0.07), y1, int(w * 0.93), y1 + 84)))
    return zones


def _draw_candidates(img, state, zones, w: int, h: int, progress: float,
                     activa: str | None) -> None:
    with state.lock:
        pensando = state.thinking
        modelo = state.model
        latencia = state.latency_s
        telegrafico = state.telegraphic

    if pensando:
        ui.text_center(img, "PENSANDO...", w // 2, h // 2 - 30, 70, ui.ACCENT,
                       bold=True)
        ui.text_center(img, f"{modelo} está proponiendo opciones",
                       w // 2, h // 2 + 45, 30, ui.TEXT_DIM)
        return

    ui.text_center(img, "¿CUÁL QUISO DECIR?", w // 2, int(h * 0.09), 52,
                   ui.TEXT, bold=True)
    ui.text_center(img, f"{modelo}   {latencia:.2f}s   "
                        f"T={state.temperature}   mire una opción para decirla",
                   w // 2, int(h * 0.145), 26, ui.TEXT_DIM)

    # Un solo tamaño para las frases: son alternativas de lo MISMO, y mostrarlas
    # a tamaños distintos insinúa una jerarquía que no existe. La más larga
    # manda, que es la única que puede no entrar.
    frases = [(t, z) for k, t, z in zones if k != "back"]
    tam = ui.uniform_text_size(
        tuple(t for t, _ in frases),
        min(z[2] - z[0] for _, z in frases) - 220,
        min(z[3] - z[1] for _, z in frases) - 30,
        cap=44,
    ) if frases else 34

    for kind, texto, (x1, y1, x2, y2) in zones:
        es_literal = texto.strip().lower() == telegrafico.lower()
        clave = f"{kind}:{texto}"
        caliente = clave == activa
        if kind == "back":
            ui.tile(img, x1, y1, x2, y2,
                    ui.TILE_HOT if caliente else ui.TILE_ACTION, edge=ui.EDGE,
                    progress=progress if caliente else 0.0, accent=ui.ACCENT)
            ui.text_center(img, "VOLVER AL TABLERO", (x1 + x2) // 2,
                           (y1 + y2) // 2, 34, ui.TEXT_DIM, bold=True)
        else:
            ui.tile(img, x1, y1, x2, y2,
                    ui.TILE_HOT if caliente else
                    (ui.TILE_ACTION if es_literal else ui.TILE),
                    edge=ui.EDGE,
                    progress=progress if caliente else 0.0, accent=ui.ACCENT)
            ui.text_center(img, texto, (x1 + x2) // 2, (y1 + y2) // 2, tam,
                           ui.TEXT_DIM if es_literal else ui.TEXT,
                           bold=not es_literal)
            if es_literal:
                ui.text_at(img, "literal", x1 + 18, y1 + 10, 18, ui.TEXT_DIM)



class _Guion:
    """Las tareas que se le van pidiendo, cuando el tablero corre en un estudio.

    Vive acá y no en el módulo del estudio porque la frontera de una tarea es
    un evento del tablero -la persona tocó HABLAR- y medirla desde afuera
    obligaría a adivinar ese instante desde el reloj del conductor, que corre
    en otro proceso. El límite de la tarea lo marca quien lo ve.
    """

    def __init__(self, tareas: list[tuple[str, str]] | None):
        self.tareas = list(tareas or [])
        self.i = 0

    @property
    def activa(self) -> tuple[str, str] | None:
        return self.tareas[self.i] if self.i < len(self.tareas) else None

    @property
    def terminado(self) -> bool:
        return bool(self.tareas) and self.i >= len(self.tareas)

    def avanzar(self) -> None:
        self.i += 1

    def banda(self) -> str:
        """La línea que se le muestra. Vacía si no hay estudio corriendo."""
        act = self.activa
        if act is None:
            return ""
        return f"{self.i + 1}/{len(self.tareas)}   {act[1]}"


def run(settings: Settings, model: str | None = None,
        variant: str | None = None, temperature: float = 0.2,
        seed: int | None = 7, guion: list[tuple[str, str]] | None = None,
        expandir: bool = True, contexto: str = "",
        pagina_inicial: str = "core") -> int:
    """Abre el tablero. Calibra primero, siempre.

    `guion` convierte el tablero en un bloque de estudio: una lista de
    (id, pista) que se van pidiendo de a una y terminan la sesión al agotarse.
    `expandir=False` apaga el SLM y hace que HABLAR diga literalmente lo que se
    escribió, que es la condición de línea de base.
    """
    probe = backends_probe()
    if probe:
        print(probe)
        return 1

    state = demo.State(temperature=temperature, seed=seed, contexto=contexto)
    if model:
        if model not in demo.MODELS:
            print(f"{model} no está en la lista: {', '.join(demo.MODELS)}")
            return 1
        state.model_i = demo.MODELS.index(model)
    if variant:
        state.variant_name = variant

    print(f"[tablero] {state.model} / {state.variant.variant} / "
          f"T={state.temperature}"
          + ("" if state.seed is None else f" (semilla {state.seed} FIJA: la "
                                           f"temperatura no va a tener efecto)"))

    # El modelo se calienta MIENTRAS la persona calibra. Medido: la primera
    # llamada del día tarda 39 s -leer los pesos del disco- y 4.3 s cuando ya
    # están en la caché del sistema; las siguientes, 0.25 s. Sin esto el primer
    # HABLAR de cada sesión se cuelga, que es exactamente el momento en que
    # alguien concluye que no funciona. La calibración dura más que eso, así
    # que la espera sale gratis.
    threading.Thread(target=_warm_up, args=(state,), daemon=True).start()

    print("[tablero] la calibración es obligatoria y va primero.")
    if eyetrax_calibrate.run(settings) is None:
        print("[tablero] sin calibración no se sigue.")
        return 1

    with GazeTracker(settings) as tracker:
        tracker.load()
        w, h = tracker.screen_w, tracker.screen_h

        if settings.check_position and positioning.run(
                settings, tracker,
                title="VUELVA A LA POSICIÓN DE LA CALIBRACIÓN",
                subtitle="Alinee su cara con la guía antes de comenzar",
        ) is None:
            return 130

        page = pagina_inicial
        tiles = board_layout.layout(page, w, h)
        by_id = {t.tile_id: t for t in tiles}
        dwell = DwellSelector(settings)
        log = _Log(settings.log_session)
        mostrando = False
        script = _Guion(guion)
        if script.activa:
            log.selection(f"TAREA:{script.activa[0]}", 0, 0)

        tracker.open_window(WINDOW)
        try:
            with camera(settings.camera_index, *settings.capture_size) as cap:
                for frame in iter_frames(cap):
                    sample = tracker.process(frame)
                    img = ui.canvas(w, h)

                    with state.lock:
                        ocupado = state.thinking
                        hay = bool(state.candidates)
                    if ocupado or hay:
                        mostrando = True

                    if mostrando:
                        zones = _candidate_zones(state, w, h)

                        def _hit(x, y, _z=zones):
                            for kind, texto, (x1, y1, x2, y2) in _z:
                                if x1 <= x < x2 and y1 <= y < y2:
                                    return f"{kind}:{texto}"
                            return None

                        def _bounds(clave, _z=zones):
                            for kind, texto, caja in _z:
                                if f"{kind}:{texto}" == clave:
                                    return caja
                            return None

                        progress, fired, stable = dwell.step(
                            sample, _hit, _bounds)
                        _draw_candidates(img, state, zones, w, h, progress,
                                         dwell.target)
                        if fired is not None:
                            kind, _, texto = fired.partition(":")
                            dijo = kind == "say" and bool(texto)
                            if dijo:
                                tts.speak(texto)
                                log.selection(f"DICHO:{texto}",
                                              sample.x, sample.y)
                            with state.lock:
                                state.candidates.clear()
                                state.words.clear()
                            mostrando = False
                            dwell.block(0.8)
                            if dijo and _avanzar(script, log):
                                break
                    else:
                        progress, fired, stable = dwell.step(
                            sample,
                            lambda x, y: next(
                                (t.tile_id for t in tiles if t.contains(x, y)),
                                None),
                            lambda tid: (lambda t: None if t is None
                                         else (t.x1, t.y1, t.x2, t.y2))(
                                by_id.get(tid)))
                        _draw_board(img, state, tiles, dwell.target, progress,
                                    w, h, script.banda())
                        if fired is not None:
                            page, tiles, by_id, mostrar, dijo = _on_tile(
                                state, by_id[fired], page, w, h, log,
                                sample, dwell, usar_modelo=expandir)
                            if mostrar:
                                mostrando = True
                            if dijo and _avanzar(script, log):
                                break

                    if sample.ok:
                        if stable:
                            ui.gaze_cursor(img, sample.x, sample.y, ui.RAW, 7)
                            ui.gaze_cursor(img, stable[0], stable[1])
                        else:
                            ui.gaze_cursor(img, sample.x, sample.y)

                    cv2.imshow(WINDOW, img)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                    if cv2.getWindowProperty(WINDOW,
                                             cv2.WND_PROP_VISIBLE) < 1:
                        break
        finally:
            log.close()
            cv2.destroyAllWindows()
    return 0


def _tam_pagina(tiles) -> int:
    """Un solo tamaño de letra para toda la grilla.

    Ajustar cada etiqueta a su celda deja «SÍ» gigante al lado de «MEDICINA»
    diminuta: el ojo salta a la palabra más grande en vez de recorrer la
    grilla, y acá la grilla se recorre con la mirada. Ver `ui.uniform_text_size`.
    """
    if not tiles:
        return 28
    return ui.uniform_text_size(
        tuple(t.key.label for t in tiles),
        min(t.x2 - t.x1 for t in tiles) - 26,
        min(t.y2 - t.y1 for t in tiles) - 26,
    )


def _draw_board(img, state, tiles, activa, progress: float, w: int,
                h: int, tarea: str = "") -> None:
    with state.lock:
        texto = state.telegraphic.upper() or "MIRE LAS CELDAS PARA ARMAR UNA FRASE"
        vacio = not state.words
    ui.rounded_rect(img, 16, 12, w - 17, int(h * 0.145), ui.SURFACE, 20)
    ui.rounded_rect(img, 16, 12, 23, int(h * 0.145), ui.ACCENT, 3)

    if tarea:
        # Con guion, el panel se parte: arriba lo que hay que decir, abajo lo
        # que se lleva escrito. La consigna tiene que estar a la vista todo el
        # tiempo: si hay que acordarse de memoria, el estudio mide memoria.
        ui.text_center(img, tarea, w // 2, int(h * 0.042), 30, ui.ACCENT)
        y_msg, alto_msg = int(h * 0.105), int(h * 0.055)
    else:
        y_msg, alto_msg = int(h * 0.078), int(h * 0.085)
    size = ui.fit_text_size(texto, w - 110, alto_msg, bold=True, cap=54)
    ui.text_center(img, texto, w // 2, y_msg, size,
                   ui.TEXT_DIM if vacio else ui.TEXT, bold=True)

    for tile in tiles:
        caliente = tile.tile_id == activa
        accion = tile.key.kind == board_layout.ACTION
        emite = tile.key.payload == "speak"
        if caliente:
            base = ui.TILE_HOT
        elif emite:
            base = ui.TILE_COMMIT
        else:
            base = ui.TILE_ACTION if accion else ui.TILE
        ui.tile(img, tile.x1, tile.y1, tile.x2, tile.y2, base, edge=ui.EDGE,
                progress=progress if caliente else 0.0, accent=ui.ACCENT)
        ui.text_center(img, tile.key.label, (tile.x1 + tile.x2) // 2,
                       (tile.y1 + tile.y2) // 2, _tam_pagina(tiles),
                       ui.ACCENT if (emite and not caliente) else
                       (ui.TEXT if (caliente or not accion) else ui.TEXT_DIM),
                       bold=True)


def _avanzar(script: "_Guion", log) -> bool:
    """Pasa a la tarea siguiente. Devuelve True si el bloque terminó.

    Sin guion no hace nada y devuelve False, así el tablero suelto sigue
    funcionando igual: la persona dice una frase y sigue usándolo.
    """
    if not script.tareas:
        return False
    script.avanzar()
    if script.terminado:
        log.selection("BLOQUE:fin", 0, 0)
        return True
    log.selection(f"TAREA:{script.activa[0]}", 0, 0)
    return False


def _on_tile(state, tile, page: str, w: int, h: int, log, sample, dwell,
             usar_modelo: bool = True):
    """Aplica una selección.

    Devuelve (página, celdas, índice, mostrar_candidatos, dijo). `dijo` es
    verdadero cuando la frase ya salió por el parlante sin pasar por el modelo,
    que es la condición de línea de base del estudio.
    """
    key = tile.key
    log.selection(key.label, sample.x, sample.y)
    expandir = False
    dijo = False

    if key.kind in (board_layout.WORD, board_layout.LETTER):
        with state.lock:
            state.words.append(key.label)
    elif key.payload.startswith("page:"):
        page = key.payload.split(":", 1)[1]
        # La mirada que cambió de página no puede disparar la celda que quedó
        # debajo: es la selección accidental más frecuente del tablero.
        dwell.block(0.8)
    elif key.payload == "delete":
        with state.lock:
            if state.words:
                state.words.pop()
    elif key.payload == "clear":
        with state.lock:
            state.words.clear()
    elif key.payload == "speak":
        if not usar_modelo:
            # Línea de base: lo que se escribió es lo que se dice. Sin pantalla
            # de candidatos, porque elegir entre opciones es justamente la
            # función que esta condición existe para NO tener.
            with state.lock:
                texto = state.telegraphic.strip()
                state.words.clear()
            if texto:
                tts.speak(texto)
                log.selection(f"DICHO:{texto}", sample.x, sample.y)
                dijo = True
        else:
            with state.lock:
                hay = bool(state.words)
                if hay:
                    state.thinking = True
                    state.candidates.clear()
            if hay:
                threading.Thread(target=demo._expand, args=(state,),
                                 daemon=True).start()
                expandir = True

    tiles = board_layout.layout(page, w, h)
    return page, tiles, {t.tile_id: t for t in tiles}, expandir, dijo


def _warm_up(state) -> None:
    """Deja el modelo cargado en RAM antes del primer HABLAR.

    Manda una expansión de descarte -no toca `state`- solo para que Ollama
    levante los pesos. Se traga cualquier error a propósito: si el precalentado
    falla, la sesión tiene que seguir igual y el costo es volver a pagar la
    espera en el primer HABLAR, no quedarse sin tablero.
    """
    from . import backends

    try:
        t0 = time.time()
        backend = backends.OllamaBackend(state.model,
                                         temperature=state.temperature,
                                         seed=state.seed, keep_alive="-1")
        variant = state.variant
        backend.complete(variant.system, variant.user("agua"),
                         json_mode=variant.json_mode)
        print(f"[tablero] modelo {state.model} listo en RAM "
              f"({time.time() - t0:.0f}s, en paralelo con la calibración)")
    except Exception as exc:
        print(f"[tablero] no se pudo precalentar el modelo ({exc}); "
              f"el primer HABLAR va a tardar más")


def backends_probe() -> str:
    """Mensaje de error si Ollama no está, o cadena vacía si todo bien."""
    from . import backends

    probe = backends.OllamaBackend(demo.MODELS[0])
    if not probe.available():
        return backends.ollama_report()
    instalados = set(probe.models())
    disponibles = [m for m in demo.MODELS
                   if any(i == m or i.split(":")[0] == m.split(":")[0]
                          for i in instalados)]
    if not disponibles:
        return "Ningún modelo instalado. Ver `main.py pull`."
    demo.MODELS[:] = disponibles
    return ""
