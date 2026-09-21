"""Headless checks - no camera, no display. Run this first after installing.

Verifies the pieces that usually break on a fresh machine: the numpy<2 pin,
the MediaPipe FaceLandmarker download, screen detection, and that the
regression + dwell logic behave.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from .config import FEATURE_DIM


def _check(name: str, fn) -> bool:
    try:
        detail = fn()
    except Exception as exc:
        print(f"  FALLA {name}: {type(exc).__name__}: {exc}")
        return False
    print(f"  ok    {name}{f' - {detail}' if detail else ''}")
    return True


def _versions() -> str:
    import cv2
    import mediapipe
    import sklearn

    import eyetrax

    assert np.__version__.startswith("1."), (
        f"eyetrax pins numpy<2 but numpy {np.__version__} is installed"
    )
    return (
        f"eyetrax {eyetrax.__version__}, numpy {np.__version__}, "
        f"cv2 {cv2.__version__}, mediapipe {mediapipe.__version__}, "
        f"sklearn {sklearn.__version__}, python {sys.version.split()[0]}"
    )


def _screen() -> str:
    from eyetrax.utils.screen import get_screen_size

    w, h = get_screen_size()
    assert w > 0 and h > 0
    return f"eyetrax default (get_monitors()[0]) = {w}x{h}"


def _monitors() -> str:
    """Monitor selection must resolve by index and by primary/secondary."""
    from .display import list_screens, resolve

    screens = list_screens()
    assert screens, "no monitors detected"

    primary = resolve("primary")
    assert primary.is_primary or len(screens) == 1

    for i, s in enumerate(screens):
        assert resolve(i).name == s.name, f"index {i} resolved to the wrong screen"
        assert s.width > 0 and s.height > 0, f"monitor {i} has no size"

    for bad in (len(screens), "laptop", "ask"):
        try:
            resolve(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"resolve({bad!r}) should have been rejected")

    if len(screens) > 1:
        sec = resolve("secondary")
        assert not sec.is_primary, "'secondary' resolved to the primary monitor"
        note = f", secondary=[{sec.index}] {sec.width}x{sec.height}"
    else:
        note = " (single monitor)"
    return f"{len(screens)} monitor(s), primary=[{primary.index}]{note}"


def _landmarker() -> str:
    """Constructing GazeEstimator downloads face_landmarker.task on first run."""
    from eyetrax import GazeEstimator

    t0 = time.time()
    est = GazeEstimator()
    dt = time.time() - t0
    try:
        feats, blink = est.extract_features(np.zeros((480, 640, 3), np.uint8))
        assert feats is None and blink is False, "blank frame should yield no face"
    finally:
        est.close()
    return f"loaded in {dt:.1f}s, blank frame -> no face (correct)"


def _feature_dim() -> str:
    """No real face frame headlessly, so assert the shape eyetrax will produce."""
    assert FEATURE_DIM == 486, (
        f"eyetrax landmark set changed: feature dim is now {FEATURE_DIM}, not 486"
    )
    return f"{FEATURE_DIM} features per frame (161 landmarks x 3 + yaw/pitch/roll)"


def _regression() -> str:
    """Every model backend must fit synthetic data and predict screen coords."""
    from eyetrax.models import AVAILABLE_MODELS, create_model

    create_model("ridge")  # triggers auto-discovery
    rng = np.random.RandomState(0)
    X = rng.randn(60, FEATURE_DIM).astype(np.float32)
    W = rng.randn(FEATURE_DIM, 2) * 0.1
    y = X @ W * 100 + np.array([960.0, 540.0])

    trained = []
    for name in sorted(AVAILABLE_MODELS):
        model = create_model(name)
        model.train(X, y)
        pred = model.predict(X[:5])
        assert pred.shape == (5, 2), f"{name} returned {pred.shape}"
        assert np.isfinite(pred).all(), f"{name} returned non-finite values"
        trained.append(name)
    return f"trained + predicted with {', '.join(trained)}"


def _persistence() -> str:
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from eyetrax.models import BaseModel, create_model

    rng = np.random.RandomState(1)
    X = rng.randn(40, FEATURE_DIM).astype(np.float32)
    y = rng.uniform([0, 0], [1920, 1080], size=(40, 2))

    model = create_model("ridge")
    model.train(X, y)
    before = model.predict(X[:3])

    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "gaze_model.pkl"
        model.save(path)
        after = BaseModel.load(path).predict(X[:3])

    assert np.allclose(before, after), "predictions changed across save/load"
    return "save/load round-trips identically"


class _FakeTracker:
    """Stand-in so smoother construction can be tested without a camera/model."""

    def __init__(self, settings):
        self.settings = settings
        self.screen_w, self.screen_h = 1920, 1080


def _smoothers() -> str:
    from .config import SMOOTHERS, Settings
    from .tracker import GazeTracker

    names = []
    for name in SMOOTHERS:
        settings = Settings(smoother=name)
        settings.validate()
        # _make_smoother only reads settings + screen size, so it can run
        # against a stand-in instead of a real (camera-backed) tracker.
        smoother = GazeTracker._make_smoother(_FakeTracker(settings))
        out = smoother.step(500, 400)
        assert len(out) == 2, f"{name} returned {out!r}"
        names.append(name)
    return f"stepped {', '.join(names)}"


def _dwell() -> str:
    from .config import Settings
    from .tracker import DwellSelector

    sel = DwellSelector(Settings(dwell_ms=500, refractory_ms=300, blink_grace_ms=200))
    t = 1000.0

    assert sel.update("A", t)[1] is None, "fired instantly"
    assert sel.update("A", t + 0.2)[1] is None, "fired before dwell elapsed"
    assert 0.3 < sel.update("A", t + 0.25)[0] < 0.7, "progress not tracking"
    assert sel.update("A", t + 0.5)[1] == "A", "did not fire at dwell time"

    # Refractory: holding the same tile must not double-fire.
    assert sel.update("A", t + 0.6)[1] is None, "double-fired inside refractory"

    # Switching targets restarts the dwell.
    sel2 = DwellSelector(Settings(dwell_ms=500, blink_grace_ms=200))
    sel2.update("A", t)
    sel2.update("B", t + 0.4)
    assert sel2.update("B", t + 0.5)[1] is None, "target switch did not reset dwell"

    # A short blink (target None) inside the grace window keeps progress.
    sel3 = DwellSelector(Settings(dwell_ms=500, blink_grace_ms=200))
    sel3.update("A", t)
    sel3.update(None, t + 0.3)
    assert sel3.update("A", t + 0.5)[1] == "A", "blink grace lost dwell progress"

    # A long loss resets it.
    sel4 = DwellSelector(Settings(dwell_ms=500, blink_grace_ms=200))
    sel4.update("A", t)
    sel4.update(None, t + 0.1)
    sel4.update(None, t + 0.45)
    assert sel4.update("A", t + 0.5)[1] is None, "long loss did not reset dwell"
    return "fire, refractory, target switch, blink grace, long loss"


def _dwell_survives_jitter() -> str:
    """Una mirada temblorosa sobre una celda tiene que poder seleccionarla.

    Reproduce el fallo que hacía inusable el tablero: la mirada se queda sobre
    una celda pero el punto tiembla ±45 px alrededor de su borde, así que con
    `update(hit_test(punto))` el objetivo cambiaba cada pocos cuadros y la
    permanencia se reiniciaba sola. La barra de progreso latía sin llegar nunca.

    El temblor de acá no es exagerado: sobre una celda de 384x216 px, ±45 px es
    menos de un cuarto de su alto, y es lo que produce un estimador a 14 fps
    entre microsacadas.

    Se comprueban las dos mitades: que el camino viejo efectivamente falla -si
    algún día deja de fallar, este test está midiendo otra cosa- y que `step`
    con pegado sí dispara.
    """
    import random

    from .config import Settings
    from .tracker import DwellSelector, GazeSample

    # Dos celdas pegadas, como en el tablero.
    cajas = {"A": (0, 0, 384, 216), "B": (0, 216, 384, 432)}

    def hit(x, y):
        for k, (x1, y1, x2, y2) in cajas.items():
            if x1 <= x < x2 and y1 <= y < y2:
                return k
        return None

    def bounds(k):
        return cajas.get(k)

    def mirada(seed):
        """Mirando el borde inferior de A, con temblor que cruza a B."""
        rng = random.Random(seed)
        for i in range(40):
            yield (1000.0 + i * 0.07,                 # ~14 fps
                   192 + rng.randint(-20, 20),
                   200 + rng.randint(-45, 45))

    ajustes = Settings(dwell_ms=800, refractory_ms=600, blink_grace_ms=250)

    # --- camino viejo: por punto crudo, sin ventana ni pegado ---------------
    viejo = DwellSelector(ajustes)
    disparos_viejo = 0
    for t, x, y in mirada(7):
        if viejo.update(hit(x, y), t)[1] is not None:
            disparos_viejo += 1
    assert disparos_viejo == 0, (
        "el camino por punto crudo seleccionó pese al temblor: este test ya no "
        "reproduce el problema que motivó `step`")

    # --- camino nuevo: mediana + pegado ------------------------------------
    nuevo = DwellSelector(ajustes)
    disparos_nuevo, celda = 0, None
    for t, x, y in mirada(7):
        _, fired, _ = nuevo.step(GazeSample(ok=True, x=x, y=y, face=True),
                                 hit, bounds, now=t)
        if fired is not None:
            disparos_nuevo += 1
            celda = fired
    assert disparos_nuevo >= 1, "con mediana y pegado tampoco llegó a disparar"
    assert celda == "A", f"seleccionó {celda!r} en vez de la celda mirada"

    # Y el pegado no puede impedir un cambio real de celda: si la mirada se va
    # de verdad al centro de B, el objetivo tiene que seguirla.
    lejos = DwellSelector(ajustes)
    for i in range(12):
        lejos.step(GazeSample(ok=True, x=192, y=100, face=True), hit, bounds,
                   now=2000.0 + i * 0.07)
    assert lejos.target == "A"
    for i in range(12):
        lejos.step(GazeSample(ok=True, x=192, y=330, face=True), hit, bounds,
                   now=2001.0 + i * 0.07)
    assert lejos.target == "B", (
        f"el pegado no soltó la celda: sigue en {lejos.target!r}. Un margen "
        f"demasiado grande hace imposible cambiar de celda.")
    return ("con temblor de ±45 px: por punto crudo 0 selecciones, "
            "con mediana y pegado 1; y un cambio real de celda sí pasa")


def _board() -> str:
    from .aac_board import Message
    from .board_layout import PAGES, layout

    total = 0
    for page in PAGES:
        tiles = layout(page, 1920, 1080)
        total += len(tiles)
        ids = [t.tile_id for t in tiles]
        assert len(ids) == len(set(ids)), f"duplicate tile ids on page {page}"
        for t in tiles:
            assert t.x2 > t.x1 and t.y2 > t.y1, f"degenerate tile {t.tile_id}"
            assert 0 <= t.x1 and t.x2 <= 1920, f"tile {t.tile_id} off-screen in x"
            assert t.y2 <= 1080, f"tile {t.tile_id} off-screen in y"
            cx, cy = (t.x1 + t.x2) // 2, (t.y1 + t.y2) // 2
            hits = [o for o in tiles if o.contains(cx, cy)]
            assert hits == [t], f"tile {t.tile_id} centre overlaps {len(hits)} tiles"

    msg = Message()
    msg.add_word("I WANT")
    msg.add_word("WATER")
    assert msg.text == "i want water", msg.text
    msg.delete()
    assert msg.text == "i want", msg.text
    msg.add_raw("!")
    msg.clear()
    assert msg.text == ""
    return f"{total} tiles across {len(PAGES)} pages, message stack OK"


def _camera() -> str:
    """Informational - a missing camera fails the live demos, not this check."""
    import cv2

    cap = cv2.VideoCapture(0)
    try:
        if not cap.isOpened():
            return "camera 0 NOT available (live demos will not run)"
        ok, frame = cap.read()
        if not ok or frame is None:
            return "camera 0 opened but returned no frame"
        return f"camera 0 -> {frame.shape[1]}x{frame.shape[0]}"
    finally:
        cap.release()


def _accents() -> str:
    """El renderizador tiene que dibujar tildes y ñ, no cajitas ni nada vacío.

    Es la razón por la que el texto pasa por Pillow: cv2.putText con las fuentes
    Hershey no puede con "BAÑO" y lo dibujaría como "BANO" o peor.
    """
    from . import ui

    for word in ("BAÑO", "SÍ", "MÁS", "ACÉRCATE", "¿QUÉ?"):
        w, h = ui.text_size(word, 40, bold=True)
        assert w > 0 and h > 0, f"{word!r} no ocupa nada"
        _, alpha = ui._sprite(word, 40, True, (255, 255, 255))
        assert alpha.max() > 0, f"{word!r} se dibujó vacío"

    # "BAÑO" tiene que ser visiblemente distinto de "BANO": si la fuente no
    # tuviese la ñ, ambos saldrían iguales.
    a = ui._sprite("BAÑO", 40, True, (255, 255, 255))[1]
    b = ui._sprite("BANO", 40, True, (255, 255, 255))[1]
    assert a.shape != b.shape or not (a == b).all(), "la ñ no se está dibujando"
    return "tildes, ñ y signos de apertura se dibujan bien"


def _vocabulary() -> str:
    """El tablero tiene que estar en español."""
    from .board_layout import ACTION, PAGES, layout

    labels = {k.label for page in PAGES.values() for row in page for k in row}
    for expected in ("SÍ", "NO", "AGUA", "BAÑO", "DOLOR", "AYUDA", "HABLAR"):
        assert expected in labels, f"falta la celda {expected!r}"

    actions = {
        k.payload
        for page in PAGES.values() for row in page for k in row
        if k.kind == ACTION
    }
    for act in ("speak", "delete", "clear", "space", "page:abc", "page:core"):
        assert act in actions, f"falta la acción {act!r}"

    # Con acentos las etiquetas son más anchas: hay que verificar que igual entran.
    from . import ui

    for tile in layout("core", 1920, 1080):
        size = ui.fit_text_size(tile.key.label, tile.x2 - tile.x1 - 24,
                                tile.y2 - tile.y1 - 24, bold=True)
        tw, _ = ui.text_size(tile.key.label, size, bold=True)
        assert tw <= tile.x2 - tile.x1 - 24, f"{tile.key.label!r} no entra en la celda"
    return f"{len(labels)} etiquetas en español, todas entran en su celda"


def _synth_face(roll_deg=0.0, yaw=0.0, pitch=0.60, span=0.16, cx=0.5, cy=0.42,
                frame=(1280, 720)):
    """Landmarks sintéticos de una cara con geometría conocida.

    Los desplazamientos verticales se dividen por la relación de aspecto para
    que la forma sea un círculo real en píxeles y no una elipse.
    """
    w, h = frame
    aspect = w / h
    pts = np.zeros((478, 3), dtype=np.float32)
    half = span / 2.0
    a = np.radians(roll_deg)
    ca, sa = np.cos(a), np.sin(a)

    def place(dx, dy):
        return cx + (dx * ca - dy * sa), cy + (dx * sa + dy * ca) * aspect

    pts[33] = (*place(-half, 0.0), 0.0)
    pts[263] = (*place(half, 0.0), 0.0)
    pts[4] = (*place(yaw * half, pitch * span), 0.0)
    pts[10] = (*place(0.0, -1.3 * span), 0.0)
    return pts


def _pose_math() -> str:
    """Las medidas de pose tienen que medir lo que dice su nombre.

    Regresión concreta: la versión anterior reutilizaba la matriz de rotación de
    eyetrax, cuyo eje x es la línea de los ojos. Una inclinación en el plano de
    la imagen rota toda la base y desaparece del "roll" extraído, que daba
    exactamente 180 grados para CUALQUIER cabeza, derecha o inclinada. Con un
    objetivo de 0 grados eso disparaba "ENDERECE LA CABEZA" siempre.
    """
    from .tracker import pose_from_landmarks

    W, H = 1280, 720

    p = pose_from_landmarks(_synth_face(), W, H)
    assert abs(p.roll_deg) < 0.5, f"cabeza nivelada -> roll {p.roll_deg:+.1f}, esperaba 0"
    assert abs(p.yaw_ratio) < 0.02, f"de frente -> yaw {p.yaw_ratio:+.3f}"
    assert abs(p.eye_span - 0.16) < 0.005, f"eye_span {p.eye_span:.3f}"
    assert abs(p.cx - 0.5) < 0.005 and abs(p.cy - 0.42) < 0.005

    # El roll debe seguir la inclinación real, no saturarse.
    seen = []
    for d in (-25.0, -12.0, 0.0, 12.0, 25.0):
        got = pose_from_landmarks(_synth_face(roll_deg=d), W, H).roll_deg
        assert abs(got - d) < 1.0, f"inclinación {d:+.0f} -> roll {got:+.1f}"
        seen.append(round(got))
    assert len(set(seen)) == len(seen), f"el roll se satura: {seen}"

    # Una inclinación no debe filtrarse a yaw ni a pitch.
    for d in (-25.0, 25.0):
        q = pose_from_landmarks(_synth_face(roll_deg=d), W, H)
        assert abs(q.yaw_ratio) < 0.03, f"tilt {d:+.0f} contamina yaw ({q.yaw_ratio:+.3f})"
        assert abs(q.pitch_ratio - 0.60) < 0.03, f"tilt {d:+.0f} contamina pitch"

    # El yaw debe ser monótono y centrado en cero.
    for y in (-0.6, -0.3, 0.0, 0.3, 0.6):
        got = pose_from_landmarks(_synth_face(yaw=y), W, H).yaw_ratio
        assert abs(got - y) < 0.03, f"yaw {y:+.2f} -> {got:+.2f}"

    # Ojos superpuestos: no debe lanzar ni devolver NaN.
    degenerate = np.zeros((478, 3), dtype=np.float32)
    degenerate[33] = degenerate[263] = degenerate[4] = (0.5, 0.5, 0.0)
    d = pose_from_landmarks(degenerate, W, H)
    assert d.eye_span == 0.0 and d.roll_deg == 0.0

    return "roll sigue la inclinación real, sin fuga a yaw/pitch, caso degenerado seguro"


def _pose_hints() -> str:
    """Los avisos de colocación: sin referencia usan rangos; con referencia comparan."""
    from .positioning import (
        HYSTERESIS, SPAN_MAX, SPAN_MIN, PoseFilter, _hints,
    )
    from .tracker import DEFAULT_POSE, FacePose, pose_from_landmarks

    W, H = 1280, 720
    level = pose_from_landmarks(_synth_face(), W, H)

    # Una cabeza derecha y centrada no debe generar ningún aviso: este es el
    # síntoma que se reportó ("dice ENDERECE LA CABEZA aunque está recta").
    assert _hints(level, DEFAULT_POSE) == [], f"pose correcta -> {_hints(level, DEFAULT_POSE)}"

    tilted = pose_from_landmarks(_synth_face(roll_deg=25.0), W, H)
    assert any("ENDERECE" in x for x in _hints(tilted, DEFAULT_POSE)), \
        "no detecta la cabeza inclinada"
    turned = pose_from_landmarks(_synth_face(yaw=0.55), W, H)
    assert any("DE FRENTE" in x for x in _hints(turned, DEFAULT_POSE)), \
        "no detecta la cabeza girada"

    # Sin referencia, cualquier distancia dentro de la banda es aceptable.
    for span in (SPAN_MIN + 0.005, 0.17, SPAN_MAX - 0.005):
        p = pose_from_landmarks(_synth_face(span=span), W, H)
        assert _hints(p, DEFAULT_POSE) == [], f"span {span:.3f} rechazado sin referencia"
    far = pose_from_landmarks(_synth_face(span=SPAN_MIN - 0.02), W, H)
    assert any("ACÉRQUESE" in x for x in _hints(far, DEFAULT_POSE))
    near = pose_from_landmarks(_synth_face(span=SPAN_MAX + 0.03), W, H)
    assert any("ALÉJESE" in x for x in _hints(near, DEFAULT_POSE))

    # Con referencia se compara contra ella, no contra un absoluto.
    ref = pose_from_landmarks(_synth_face(span=0.13, roll_deg=6.0), W, H)
    same = pose_from_landmarks(_synth_face(span=0.13, roll_deg=6.0), W, H)
    assert _hints(same, ref, has_reference=True) == [], "no reconoce su propia referencia"
    moved = pose_from_landmarks(_synth_face(span=0.21, roll_deg=6.0), W, H)
    assert any("ALÉJESE" in x for x in _hints(moved, ref, has_reference=True))

    # El pitch solo se controla cuando hay referencia.
    odd_pitch = pose_from_landmarks(_synth_face(pitch=1.4), W, H)
    assert _hints(odd_pitch, DEFAULT_POSE) == [], "el pitch no debe gatillar sin referencia"
    ref_p = pose_from_landmarks(_synth_face(pitch=0.55), W, H)
    assert any("BARBILLA" in x for x in _hints(odd_pitch, ref_p, has_reference=True)), \
        "con referencia sí debe avisar del pitch"

    # La histéresis debe ensanchar la tolerancia, nunca estrecharla.
    edge = pose_from_landmarks(_synth_face(roll_deg=17.0), W, H)
    assert _hints(edge, DEFAULT_POSE, slack=1.0), "sin holgura debería avisar"
    assert not _hints(edge, DEFAULT_POSE, slack=HYSTERESIS), "la holgura no se aplica"

    # El filtro temporal debe descartar un cuadro disparatado.
    filt = PoseFilter()
    for _ in range(6):
        filt.update(level)
    glitch = FacePose(cx=0.9, cy=0.9, eye_span=0.5, roll_deg=70.0,
                      yaw_ratio=0.9, pitch_ratio=2.0)
    out = filt.update(glitch)
    assert abs(out.roll_deg) < 3.0, f"la mediana no filtró el glitch ({out.roll_deg:+.1f})"
    return "rangos sin referencia, comparación con referencia, histéresis y filtro OK"


def _pose_roundtrip() -> str:
    """FacePose debe ir y volver de JSON sin perder nada."""
    from .tracker import DEFAULT_POSE, FacePose

    back = FacePose.from_dict(DEFAULT_POSE.as_dict())
    for field in ("cx", "cy", "eye_span", "roll_deg", "yaw_ratio", "pitch_ratio"):
        assert abs(getattr(back, field) - getattr(DEFAULT_POSE, field)) < 1e-6, field
    # Un sidecar viejo sin estos campos no debe romper la carga.
    partial = FacePose.from_dict({"cx": 0.5, "cy": 0.4, "eye_span": 0.15})
    assert partial.roll_deg == 0.0 and partial.yaw_ratio == 0.0
    return "round-trip completo y tolerante con sidecars viejos"


def _tile_cache() -> str:
    """Las celdas se cachean ya compuestas; el tamaño debe coincidir siempre."""
    from .aac_board import _tile_bitmap
    from .board_layout import layout

    _tile_bitmap.cache_clear()
    tiles = layout("core", 1920, 1080)
    for tile in tiles:
        w, h = tile.x2 - tile.x1, tile.y2 - tile.y1
        bmp = _tile_bitmap(tile.key.label, tile.key.kind, w, h, False)
        assert bmp.shape == (h, w, 3), f"celda {tile.tile_id}: {bmp.shape} != {(h, w, 3)}"
    info = _tile_bitmap.cache_info()
    assert info.hits > 0 or info.currsize > 0
    return f"{info.currsize} celdas en caché, {info.hits} aciertos"


def _voice() -> str:
    """Informativo: si no hay voz en español, el tablero imprime el texto."""
    from . import tts

    if not tts.available():
        return "sin SAPI (fuera de Windows): las frases se imprimen"
    name = tts.voice_name()
    return f"voz en español: {name}" if name else \
        "NO hay voz en español instalada (se usará la predeterminada)"


def _ui_signatures() -> str:
    """Toda llamada a ui.* debe encajar con la firma real de la función.

    Existe por una falla concreta: al pasar el texto de cv2 a Pillow, status_bar
    cambió de `scale` a `size`, y una llamada quedó sin actualizar. Ese error solo
    aparecía al abrir el tablero, después de calibrar y medir - varios minutos de
    trabajo de la persona antes de reventar. Un chequeo estático lo detecta sin
    tener que abrir ninguna ventana.
    """
    import ast
    import inspect
    from pathlib import Path

    from . import ui

    pkg = Path(__file__).resolve().parent
    checked = 0
    problems: list[str] = []

    for mod_path in sorted(pkg.glob("*.py")):
        tree = ast.parse(mod_path.read_text(encoding="utf-8"), filename=str(mod_path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not (isinstance(fn, ast.Attribute)
                    and isinstance(fn.value, ast.Name)
                    and fn.value.id == "ui"):
                continue

            target = getattr(ui, fn.attr, None)
            if target is None:
                problems.append(f"{mod_path.name}:{node.lineno} ui.{fn.attr} no existe")
                continue
            if not callable(target):
                continue

            # Las llamadas con *args/**kwargs no se pueden validar estáticamente.
            if any(isinstance(a, ast.Starred) for a in node.args) or \
               any(k.arg is None for k in node.keywords):
                continue

            kwargs = {k.arg: None for k in node.keywords}
            try:
                inspect.signature(target).bind(*([None] * len(node.args)), **kwargs)
            except TypeError as exc:
                problems.append(
                    f"{mod_path.name}:{node.lineno} ui.{fn.attr}(...): {exc}"
                )
            checked += 1

    if problems:
        raise AssertionError("; ".join(problems))
    return f"{checked} llamadas a ui.* con firma correcta"


def _diagnosis() -> str:
    """El diagnóstico de precisión, sobre casos sintéticos.

    Aquí vivieron dos errores reales:

    * el "centro" se tomaba como `point_index == 0`, que en una grilla en
      serpentina es la esquina superior izquierda - uno de los peores puntos -,
      así que la comparación centro-vs-bordes quedaba invertida y no detectaba
      el límite del modelo lineal;
    * las estadísticas por punto usaban la media, así que un saccade al empezar
      la medición se reportaba como "temblor" y disparaba consejos de suavizado
      que no correspondían.
    """
    from .accuracy import PX_PER_DEG, _diagnose, _per_point, _summarise

    # Grilla 3x3 en orden serpentina: el índice 0 es la esquina, el 4 el centro.
    grid = [(230, 129), (960, 129), (1690, 129),
            (1690, 540), (960, 540), (230, 540),
            (230, 951), (960, 951), (1690, 951)]
    rng = np.random.RandomState(0)

    def rows(offsets, jitter=12.0, drift=None, n=45):
        """offsets: px de desvío por punto; drift: px extra al inicio."""
        out = []
        for i, (tx, ty) in enumerate(grid):
            ox, oy = offsets[i]
            for k in range(n):
                extra = 0.0
                if drift and drift[i] and k < n // 3:
                    extra = drift[i]
                gx = tx + ox + extra + rng.randn() * jitter
                gy = ty + oy + rng.randn() * jitter
                out.append(dict(point_index=i, target_x=tx, target_y=ty,
                                gaze_x=gx, gaze_y=gy, raw_x=gx, raw_y=gy,
                                error_px=float(np.hypot(gx - tx, gy - ty))))
        return out

    def tips_for(r):
        return _diagnose(_summarise(r), _per_point(r),
                         {"capture": (1280, 720), "brightness": 145.0,
                          "face_rate": 1.0})

    # 1. Todo sano: ningún consejo de problema.
    healthy = rows([(8, 6)] * 9)
    got = tips_for(healthy)
    assert len(got) == 1 and "No se ve nada mal" in got[0], got

    # 2. Saccade al inicio en el primer objetivo: debe detectarlo como tránsito
    #    y NO como temblor.
    drift = [0.0] * 9
    drift[0] = 600.0
    trans = rows([(8, 6)] * 9, drift=drift)
    got = tips_for(trans)
    assert any("en tránsito" in x or "todavía estaba llegando" in x for x in got), got
    assert not any("suavizado casi no ayuda" in x for x in got), \
        f"un saccade se reportó como temblor: {got}"
    pp = {int(q["point_index"]): q for q in _per_point(trans)}
    assert pp[0]["drift_px"] > 300, f"drift_px no lo captó: {pp[0]['drift_px']:.0f}"
    assert pp[0]["jitter_px"] < 4 * 12.0, \
        f"la mediana no protegió el temblor: {pp[0]['jitter_px']:.0f}"

    # 3. Error que crece hacia los bordes, con el centro bueno. El centro es el
    #    índice 4, no el 0: si se tomara el 0 como centro, no se detectaría.
    offs = []
    for (tx, ty) in grid:
        k = 0.30
        offs.append(((tx - 960) * k, (ty - 540) * k))
    edges = rows(offs)
    got = tips_for(edges)
    assert any("crece hacia los bordes" in x for x in got), got

    # 4. Entorno: baja resolución y poca luz se detectan.
    got = _diagnose(_summarise(healthy), _per_point(healthy),
                    {"capture": (640, 480), "brightness": 40.0, "face_rate": 0.5})
    assert any("640x480" in x for x in got), got
    assert any("oscura" in x for x in got), got
    assert any("% de los cuadros" in x for x in got), got

    # 5. El temblor real sí debe reportarse cuando de verdad es alto.
    noisy = rows([(8, 6)] * 9, jitter=140.0)
    got = tips_for(noisy)
    assert any("temblor es alto" in x for x in got), got
    return "sano, tránsito, bordes, entorno y temblor real distinguidos"


CHECKS = [
    ("versiones / numpy<2", _versions),
    ("tamaño de pantalla", _screen),
    ("selección de pantalla", _monitors),
    ("vector de características", _feature_dim),
    ("FaceLandmarker de mediapipe", _landmarker),
    ("modelos de regresión", _regression),
    ("guardado del modelo", _persistence),
    ("suavizadores", _smoothers),
    ("selección por permanencia", _dwell),
    ("la permanencia sobrevive al temblor", _dwell_survives_jitter),
    ("geometría del tablero", _board),
    ("firmas de ui.*", _ui_signatures),
    ("acentos y ñ", _accents),
    ("vocabulario en español", _vocabulary),
    ("medidas de pose", _pose_math),
    ("avisos de colocación", _pose_hints),
    ("pose en JSON", _pose_roundtrip),
    ("diagnóstico de precisión", _diagnosis),
    ("caché de celdas", _tile_cache),
    ("voz (informativo)", _voice),
    ("cámara (informativo)", _camera),
]


def run() -> int:
    print("Chequeos del sistema CAA (sin cámara ni pantalla)\n")
    results = [_check(name, fn) for name, fn in CHECKS]
    passed = sum(results)
    print(f"\n{passed}/{len(results)} chequeos pasaron")
    return 0 if passed == len(results) else 1
