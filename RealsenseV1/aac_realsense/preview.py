"""The three things to do before calibrating anything.

`preview` is what you run first with a camera you have never used. It needs no
calibration and it answers the questions that decide whether the rest can work
at all: is there depth on the face, how many pixels land on the iris, how far
away is the person, is the framing right. Every one of those is a hard limit
that no amount of model tuning can lift, so finding out early is worth a
dedicated command.

`mount` reads the accelerometer and prints the camera's real tilt. This replaces
the hardest hand measurement in the whole setup - you cannot put a protractor on
a camera clamped to a bezel, but the camera can measure its own attitude against
gravity to a fraction of a degree.

`record` writes a .bag. Worth doing on every real session: a recording replays
through the identical code path, so a changed estimator can be evaluated against
it offline instead of by booking another appointment with a participant. For a
study with people whose condition is progressing, being able to re-analyse
yesterday's session is not a nicety.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from . import landmarks, positioning, source, ui
from .config import RECORDINGS_DIR, CameraMount, Settings, ensure_dirs

WINDOW = "CAA - vista previa"


def _depth_colormap(depth_m: np.ndarray, near: float = 0.3,
                    far: float = 1.2) -> np.ndarray:
    """Colourise depth over the range a seated user occupies.

    Scaled to 0.3-1.2 m rather than to the frame's own min/max. Autoscaling
    looks better and is useless here: it changes meaning frame to frame, so you
    cannot tell whether the face got closer or the background changed. A fixed
    scale makes "the face is always this colour at this distance" a readable
    fact.
    """
    valid = depth_m > 0
    norm = np.zeros(depth_m.shape, dtype=np.uint8)
    norm[valid] = np.clip(
        (depth_m[valid] - near) / max(far - near, 1e-6) * 255.0, 0, 255
    ).astype(np.uint8)
    coloured = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    coloured[~valid] = (0, 0, 0)          # holes stay black, and look like holes
    return coloured


def run(settings: Settings) -> int:
    """Live colour + depth, with the numbers that decide feasibility."""
    with source.open_source(settings) as src:
        print(f"[vista] fuente: {src.describe()}")
        intrin = src.intrinsics
        print(f"[vista] color {intrin.width}x{intrin.height}  fx={intrin.fx:.1f}  "
              f"campo horizontal {intrin.hfov_deg():.1f}°")
        print(f"[vista] a 55 cm: {intrin.px_per_mm_at(0.55):.2f} px/mm, "
              f"iris de ~{intrin.px_per_mm_at(0.55) * 12:.0f} px de diámetro")
        print("[vista] ESC/Q salir | D alternar profundidad | L puntos faciales")

        landmarker = landmarks.MediaPipeLandmarker()
        show_depth = True
        show_landmarks = True
        frames, fps, t0 = 0, 0.0, time.time()

        try:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW, min(1600, intrin.width),
                             min(900, intrin.height))
            for frame in src.frames():
                obs = landmarker.detect(frame.color)
                canvas = frame.color.copy()

                if show_depth:
                    depth_vis = _depth_colormap(frame.depth_m)
                    canvas = cv2.addWeighted(canvas, 0.55, depth_vis, 0.45, 0)

                lines = [
                    f"{fps:4.1f} fps   profundidad "
                    f"{frame.depth_fill_rate() * 100:5.1f}% del cuadro",
                ]

                if obs is None:
                    # A banner, not a status line. Aiming a loose camera is the
                    # first thing anyone does with a new one, and the failure it
                    # produces - pointing at the ceiling - is invisible in a
                    # small line of text among five others. The depth fill rate
                    # is the tell: a ceiling or a far wall is past the 1.5 m
                    # threshold filter, so it comes back nearly empty.
                    h, w = canvas.shape[:2]
                    ui.banner(canvas, "NO SE DETECTA LA CARA")
                    hint = ("apuntá la cámara a tu cara"
                            if frame.depth_fill_rate() > 0.10 else
                            "casi no hay profundidad: apunta a algo lejano "
                            "(¿techo o pared?)")
                    ui.text_center(canvas, hint, w // 2, h // 2 + 70, 34,
                                   ui.TEXT_DIM, bold=True)
                    # Where a well-framed face should land, so aiming has a
                    # target rather than being trial and error.
                    cv2.rectangle(canvas, (int(w * 0.30), int(h * 0.15)),
                                  (int(w * 0.70), int(h * 0.85)),
                                  ui.ACCENT, 2)
                else:
                    if show_landmarks:
                        for p in obs.rigid_px():
                            cv2.circle(canvas, (int(p[0]), int(p[1])), 2,
                                       (90, 200, 255), -1, cv2.LINE_AA)
                        for key, eye in obs.eyes.items():
                            c = eye.iris_centre
                            cv2.circle(canvas, (int(c[0]), int(c[1])),
                                       max(2, int(eye.iris_radius_px)),
                                       (70, 255, 70), 1, cv2.LINE_AA)

                    place = positioning.measure(frame, obs)
                    if place is None:
                        lines.append("cara detectada pero SIN PROFUNDIDAD sobre "
                                     "ella - revise el emisor IR y la luz solar")
                    else:
                        lines.append(
                            f"distancia {place.distance_cm:5.1f} cm   "
                            f"iris {place.iris_px:4.1f} px de radio   "
                            f"profundidad sobre la cara "
                            f"{place.depth_coverage * 100:.0f}%")
                        lines.append(
                            f"cabeza  giro {place.yaw_deg:+5.1f}°  "
                            f"inclinación {place.pitch_deg:+5.1f}°  "
                            f"rotación {place.roll_deg:+5.1f}°   "
                            f"calidad {place.quality():.2f}")
                        problems = place.problems()
                        lines.append(problems[0][0] if problems
                                     else "ENCUADRE CORRECTO")

                if frame.accel is not None:
                    mount = CameraMount().with_gravity(frame.accel)
                    lines.append(
                        f"IMU: la cámara está inclinada "
                        f"{mount.pitch_deg:+5.1f}° y rotada "
                        f"{mount.roll_deg:+5.1f}° (asumiendo pantalla vertical)")
                else:
                    lines.append("sin IMU: la inclinación hay que medirla a mano")

                lines.append("ESC/Q salir   D profundidad   L puntos")
                ui.status_bar(canvas, lines, y=16, size=20)

                frames += 1
                if frames % 10 == 0:
                    now = time.time()
                    fps = 10.0 / max(now - t0, 1e-6)
                    t0 = now

                cv2.imshow(WINDOW, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("d"):
                    show_depth = not show_depth
                if key == ord("l"):
                    show_landmarks = not show_landmarks
        finally:
            landmarker.close()
            cv2.destroyAllWindows()
    return 0


def measure_mount(settings: Settings) -> int:
    """Report the mount geometry, using the accelerometer if there is one.

    Prints the command to paste, because the alternative - remembering which
    flag was which - is how a measured setup silently becomes an unmeasured one.

    On a D435 (no IMU) the tilt cannot be sensed, but the tape-measure half of
    the job is unchanged and is most of the value, so this still runs and still
    prints a usable command. Returning early with only an apology would have
    made the command useless on exactly the hardware that needs the guidance
    most.
    """
    from . import display

    print("[montaje] leyendo el acelerómetro; no mueva la cámara...")
    accel = source.read_gravity(settings, samples=40)
    sensed = accel is not None

    if sensed:
        norm = float(np.linalg.norm(accel))
        mount = CameraMount().with_gravity(accel)
        print(f"[montaje] aceleración medida "
              f"({accel[0]:+.3f}, {accel[1]:+.3f}, {accel[2]:+.3f}) m/s², "
              f"módulo {norm:.3f}")
        if abs(norm - 9.80665) > 0.35:
            print(f"[montaje] aviso: el módulo debería ser ~9.81 y es "
                  f"{norm:.2f}. El IMU de la D435i sale de fábrica sin "
                  f"calibrar; se puede calibrar con la herramienta "
                  f"rs-imu-calibration de Intel.")
        print(f"\n  inclinación (pitch)  {mount.pitch_deg:+6.2f}°   "
              f"(+ = el lente apunta hacia arriba)")
        print(f"  rotación    (roll)   {mount.roll_deg:+6.2f}°")
        print("\n  Esto supone que la pantalla es vertical. Cierto para un "
              "monitor en soporte o brazo;\n  NO cierto para la tapa de una "
              "notebook - en ese caso mida el ángulo de la tapa\n  y súmelo a "
              "mano.")
        print("  El giro (yaw) alrededor del eje vertical no cambia la "
              "gravedad, así que ese sí\n  hay que medirlo (normalmente es 0: "
              "la cámara mira de frente).")
        pitch_hint = f"{mount.pitch_deg:.1f}"
    else:
        print("[montaje] esta cámara no tiene acelerómetro (es una D435, no "
              "una D435i).")
        print("\n  La inclinación hay que medirla a mano. Dos formas, de menor "
              "a mayor esfuerzo:")
        print("    1. Si la cámara está apoyada plana sobre el escritorio o "
              "pegada al bisel\n"
              "       sin cuña, la inclinación es ~0°: use --pitch 0.")
        print("    2. Con una app de nivel/inclinómetro del teléfono apoyada "
              "sobre el cuerpo\n"
              "       de la cámara. Un grado son unos 38 px en la pantalla, "
              "así que vale\n"
              "       el minuto que cuesta.")
        print("\n  No se agobie con esto: la calibración refina la inclinación "
              "hasta 5°, así que\n  errarle por 2-3° es irrelevante. Lo que no "
              "puede es errarle por 15°.")
        pitch_hint = "0"

    screen = display.resolve(settings.monitor)
    edid = (f"{screen.width_mm}x{screen.height_mm} mm del EDID"
            if screen.width_mm else "el EDID no informa el tamaño físico")
    print(f"\n  pantalla elegida: [{screen.index}] {screen.name} "
          f"{screen.width}x{screen.height} px, {edid}")
    if screen.width_mm and screen.height_mm:
        print(f"  -> con la cámara centrada en el bisel inferior, "
              f"offset-y ≈ {screen.height_mm / 2:.0f} + (bisel hasta el lente) mm")

    print("\n  Falta medir con cinta, del centro de la pantalla al centro del "
          "lente:")
    print("    --offset-x  mm hacia la derecha (normalmente 0, cámara centrada)")
    print("    --offset-y  mm hacia abajo (cámara en el borde inferior -> "
          "positivo)")
    print("    --offset-z  mm hacia el usuario (cuánto sobresale el lente)")
    print("\n  Comando sugerido, completando los offsets:")
    print(f"    .\\run.cmd calibrate --monitor {screen.index} "
          f"--pitch {pitch_hint} --offset-y <mm> --offset-z <mm>")
    print("\n  No hace falta que sea exacto: la calibración refina el montaje "
          "hasta 30 mm y 5°.\n  Lo que importa es que el punto de partida no "
          "esté fuera de ese margen.")
    return 0


def record(settings: Settings, seconds: float) -> int:
    """Record colour + depth to a .bag for later offline analysis."""
    ensure_dirs()
    if settings.record is None:
        settings.record = RECORDINGS_DIR / f"session-{time.strftime('%Y%m%d-%H%M%S')}.bag"

    print(f"[grabación] {seconds:.0f} s -> {settings.record}")
    print("[grabación] los .bag de color+profundidad son grandes: cuente unos "
          "200-400 MB por minuto a 720p30. ESC corta antes.")

    with source.open_source(settings) as src:
        t0 = time.time()
        count = 0
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        try:
            for frame in src.frames():
                count += 1
                elapsed = time.time() - t0
                if elapsed >= seconds:
                    break

                canvas = frame.color.copy()
                ui.status_bar(canvas, [
                    f"GRABANDO   {elapsed:5.1f} / {seconds:.0f} s   "
                    f"{count} cuadros   "
                    f"profundidad {frame.depth_fill_rate() * 100:.0f}%",
                    "ESC para terminar antes",
                ], y=16, size=22)
                cv2.imshow(WINDOW, canvas)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break
        finally:
            cv2.destroyAllWindows()

    size_mb = settings.record.stat().st_size / 1e6 if settings.record.exists() else 0.0
    print(f"[grabación] {count} cuadros, {size_mb:.0f} MB -> {settings.record}")
    print(f"[grabación] reprodúzcala con: "
          f".\\run.cmd accuracy --playback {settings.record}")
    return 0
