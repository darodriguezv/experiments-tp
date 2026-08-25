"""Quantitative gaze check: show known targets, measure error, and diagnose it.

Two failures get confused with each other when people say "it's imprecise", and
they have opposite fixes:

* **Jitter (precision)** - the gaze point dances while you hold a fixation.
  A signal-quality problem: too few pixels on the iris, dim light, sitting too
  far back. Smoothing hides it at the cost of lag.
* **Offset (accuracy)** - the point is steady but lands in the wrong place.
  A calibration/model problem: stale calibration, head moved since, or a linear
  model that cannot reach the screen edges.

This module reports them separately, because "raise the resolution" and
"recalibrate with a nonlinear model" fix different things.
"""

from __future__ import annotations

import csv
import time

import cv2
import numpy as np

from eyetrax.calibration.common import compute_grid_points_from_shape

from . import positioning, ui
from .config import LOGS_DIR, Settings, ensure_dirs
from .tracker import GazeTracker
from .video import camera, capture_size, iter_frames

WINDOW = "CAA - prueba de precisión"

PX_PER_DEG = 40.0
"""Rough conversion on a 1080p screen at normal seating distance - enough to
compare against papers quoting degrees, not a calibrated figure."""


def _summarise(rows: list[dict]) -> dict:
    err = np.array([r["error_px"] for r in rows], dtype=np.float64)
    dx = np.array([r["gaze_x"] - r["target_x"] for r in rows], dtype=np.float64)
    dy = np.array([r["gaze_y"] - r["target_y"] for r in rows], dtype=np.float64)
    return {
        "samples": len(rows),
        "mean_px": float(err.mean()),
        "median_px": float(np.median(err)),
        "p95_px": float(np.percentile(err, 95)),
        "max_px": float(err.max()),
        "bias_x_px": float(dx.mean()),
        "bias_y_px": float(dy.mean()),
        "std_x_px": float(dx.std()),
        "std_y_px": float(dy.std()),
    }


def _per_point(rows: list[dict]) -> list[dict]:
    """Split precision from accuracy, one entry per target.

    Offset is where the fixation actually sat; jitter is how much it wandered
    around that place. Both use the **median**, not the mean: if the gaze is
    still travelling to the target when recording starts, a mean-based centre is
    dragged toward the old fixation and a mean-based spread explodes. That is
    not jitter, it is a saccade, and it should not be reported as noise.

    `drift_px` exposes exactly that: the median error over the first third of
    the window minus the last third. A large positive value means the recording
    started too early rather than that the gaze is unsteady.
    """
    out = []
    for idx in sorted({r["point_index"] for r in rows}):
        pts = [r for r in rows if r["point_index"] == idx]
        gx = np.array([p["gaze_x"] for p in pts], dtype=np.float64)
        gy = np.array([p["gaze_y"] for p in pts], dtype=np.float64)
        rx = np.array([p["raw_x"] for p in pts], dtype=np.float64)
        ry = np.array([p["raw_y"] for p in pts], dtype=np.float64)
        tx, ty = pts[0]["target_x"], pts[0]["target_y"]

        mx, my = float(np.median(gx)), float(np.median(gy))
        err = np.hypot(gx - tx, gy - ty)
        third = max(1, len(pts) // 3)

        out.append({
            "point_index": idx,
            "target_x": tx, "target_y": ty,
            "n": len(pts),
            "offset_px": float(np.hypot(mx - tx, my - ty)),
            "jitter_px": float(np.median(np.hypot(gx - mx, gy - my))),
            "raw_jitter_px": float(np.median(
                np.hypot(rx - np.median(rx), ry - np.median(ry))
            )),
            "drift_px": float(np.median(err[:third]) - np.median(err[-third:])),
        })
    return out


def _diagnose(stats: dict, points: list[dict], env: dict) -> list[str]:
    """Turn the numbers into a ranked list of things to actually change."""
    tips: list[str] = []
    jitter = float(np.mean([p["jitter_px"] for p in points]))
    raw_jitter = float(np.mean([p["raw_jitter_px"] for p in points]))
    offset = float(np.mean([p["offset_px"] for p in points]))

    # A recording window that starts before the eye has arrived reports the
    # saccade as error. Flagged first, because every other number is suspect
    # while it is happening.
    drifters = [p for p in points
                if p["drift_px"] > 2.0 * PX_PER_DEG
                and p["drift_px"] > 1.5 * max(p["jitter_px"], 1.0)]
    if drifters:
        worst_drift = max(drifters, key=lambda p: p["drift_px"])
        tips.append(
            f"En {len(drifters)} objetivo(s) la mirada todavía estaba llegando cuando "
            f"empezó la medición (objetivo {int(worst_drift['point_index'])}: "
            f"{worst_drift['drift_px']:.0f} px de diferencia entre el principio y el "
            "final). Suba --settle; esas muestras inflan la media, el p95 y el máximo."
        )

    cap_w, cap_h = env["capture"]
    if cap_w and cap_w <= 640:
        tips.append(
            f"La captura es de solo {cap_w}x{cap_h}. Vuelva a ejecutar con "
            "--capture 1280x720: la precisión proviene de los píxeles sobre el iris, "
            "y 720p tiene 4 veces más área."
        )
    if env["brightness"] < 60:
        tips.append(
            f"La imagen está oscura (media {env['brightness']:.0f}/255). Ponga luz "
            "de frente a la cara, no detrás: con poca luz los puntos del iris "
            "quedan ruidosos."
        )
    if env["face_rate"] < 0.85:
        tips.append(
            f"Solo se detectó cara en el {env['face_rate'] * 100:.0f}% de los cuadros. "
            "Corrija el encuadre y la luz antes de confiar en estos números."
        )

    if jitter > 3 * PX_PER_DEG:
        tips.append(
            f"El temblor es alto ({jitter:.0f} px = {jitter / PX_PER_DEG:.1f}°): es un "
            "problema de calidad de señal. Suba la resolución, agregue luz y "
            "acérquese: la cara debería ocupar casi toda la vista previa."
        )
    # Only worth mentioning when the jitter actually hurts - a smoother that
    # "barely helps" on an already-steady point is not a problem.
    if jitter > 1.5 * PX_PER_DEG and raw_jitter > 0 and jitter > 0.8 * raw_jitter:
        tips.append(
            f"El suavizado casi no ayuda ({raw_jitter:.0f} px crudo -> "
            f"{jitter:.0f} px suavizado). Pruebe --smoother kde, o "
            "--smoother kalman_ema --ema-alpha 0.6 para un punto más estable "
            "pero con más retardo."
        )

    # A linear model typically tracks the centre well and collapses at the edges.
    # Checked before the generic offset advice because it is the more specific
    # diagnosis, and requires a real absolute error so jitter cannot trigger it.
    edge_collapse = False
    if len(points) >= 5:
        # Geometrically nearest the screen centre - NOT point_index 0, which in a
        # serpentine grid is the top-left corner, i.e. one of the worst spots.
        cx = float(np.mean([p["target_x"] for p in points]))
        cy = float(np.mean([p["target_y"] for p in points]))
        centre = min(points, key=lambda p: (p["target_x"] - cx) ** 2
                     + (p["target_y"] - cy) ** 2)
        worst = sorted(points, key=lambda p: p["offset_px"])[-3:]
        worst_mean = float(np.mean([p["offset_px"] for p in worst]))
        if worst_mean > 1.5 * PX_PER_DEG and worst_mean > 2.0 * max(
            centre["offset_px"], 1.0
        ):
            edge_collapse = True
            tips.append(
                f"El error crece hacia los bordes (centro {centre['offset_px']:.0f} px "
                f"vs peor {worst_mean:.0f} px): el límite clásico del modelo "
                "lineal ridge. Pruebe --model svr o --model tiny_mlp con "
                "--method lissajous, o aleje las celdas de los bordes."
            )

    if not edge_collapse and offset > 2 * PX_PER_DEG and jitter < 1.5 * PX_PER_DEG:
        tips.append(
            f"El punto es estable ({jitter:.0f} px de temblor) pero cae a {offset:.0f} px "
            "del objetivo. Eso es calibración, no señal: recalibre y no cambie "
            "de postura después, y pruebe --method lissajous --model svr."
        )

    if abs(stats["bias_x_px"]) > PX_PER_DEG or abs(stats["bias_y_px"]) > PX_PER_DEG:
        tips.append(
            f"Hay un desvío constante (x {stats['bias_x_px']:+.0f}, "
            f"y {stats['bias_y_px']:+.0f} px), que casi siempre significa que la "
            "cabeza está en otra posición que durante la calibración. Recalibre "
            "en la postura real de uso."
        )

    if not tips:
        tips.append(
            "No se ve nada mal - esto es aproximadamente lo mejor que da una webcam. "
            "Dimensione el tablero con el p95 de arriba."
        )
    return tips


def run(
    settings: Settings,
    rows: int = 3,
    cols: int = 3,
    hold_s: float = 2.0,
    settle_s: float = 0.7,
    first_extra_settle_s: float = 1.3,
) -> dict | None:
    """Walk a rows x cols grid, sampling gaze error at each point.

    `settle_s` is discarded at the start of each point so the saccade and the
    smoother's lag are not counted as error.

    The **first** target gets `first_extra_settle_s` on top. Coming off the
    placement screen the eye has no idea where the first dot is, and measured
    runs showed it taking about 1.7 s to arrive - far longer than the 0.7 s that
    is plenty once the person is already following dots. Without this the first
    target contributes a saccade to the statistics and inflates the mean, the
    p95 and especially the maximum.
    """
    cap_size = (0, 0)
    frames_seen = faces_seen = 0
    brightness: list[float] = []

    with GazeTracker(settings) as tracker:
        tracker.load()
        w, h = tracker.screen_w, tracker.screen_h
        points = compute_grid_points_from_shape(
            rows, cols, w, h, margin_ratio=0.12, order="serpentine"
        )

        print(f"[precisión] pantalla {tracker.screen.index} "
              f"({tracker.screen.name}) {w}x{h}")
        print(f"[precisión] grilla {rows}x{cols}, {hold_s}s por punto, "
              f"{settle_s}s de asentado "
              f"({settle_s + first_extra_settle_s:.1f}s en el primero), "
              f"suavizado={settings.smoother}")
        print("[precisión] Siga el punto blanco con la mirada. ESC cancela.")

        # Volver a la posición de la calibración: sin esto el error es
        # sistemático y ningún suavizado lo corrige.
        if settings.check_position:
            if positioning.run(
                settings, tracker,
                title="VUELVA A LA POSICIÓN DE LA CALIBRACIÓN",
                subtitle="La medición solo es válida desde la posición calibrada",
            ) is None:
                print("[precisión] cancelado en la colocación.")
                return None

        samples: list[dict] = []
        tracker.open_window(WINDOW)

        with camera(settings.camera_index, *settings.capture_size) as cap:
            cap_size = capture_size(cap)
            frames = iter_frames(cap)
            aborted = False

            for idx, (tx, ty) in enumerate(points):
                t_start = time.time()
                tracker.reset_smoother()
                this_settle = settle_s + (first_extra_settle_s if idx == 0 else 0.0)

                while True:
                    elapsed = time.time() - t_start
                    if elapsed > this_settle + hold_s:
                        break

                    frame = next(frames)
                    sample = tracker.process(frame)
                    recording = elapsed >= this_settle

                    if recording:
                        frames_seen += 1
                        if sample.ok:
                            faces_seen += 1
                        if frames_seen % 10 == 0:
                            brightness.append(
                                float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
                            )

                    if recording and sample.ok:
                        err = float(np.hypot(sample.x - tx, sample.y - ty))
                        samples.append({
                            "point_index": idx,
                            "target_x": tx, "target_y": ty,
                            "gaze_x": sample.x, "gaze_y": sample.y,
                            "raw_x": sample.raw_x, "raw_y": sample.raw_y,
                            "error_px": err,
                        })

                    img = ui.canvas(w, h)
                    # Target: hollow while settling, filled while recording.
                    cv2.circle(img, (tx, ty), 26, ui.TEXT, 2, cv2.LINE_AA)
                    if recording:
                        cv2.circle(img, (tx, ty), 14, ui.TEXT, -1, cv2.LINE_AA)
                        frac = (elapsed - this_settle) / hold_s
                        cv2.ellipse(img, (tx, ty), (40, 40), 0, -90,
                                    -90 + 360 * frac, ui.ACCENT, 4, cv2.LINE_AA)

                    if sample.ok:
                        ui.gaze_cursor(img, sample.x, sample.y)
                        cv2.line(img, (tx, ty), (sample.x, sample.y),
                                 (60, 60, 70), 1, cv2.LINE_AA)
                    elif not sample.face:
                        ui.status_bar(img, ["NO SE DETECTA LA CARA"], color=ui.WARN)

                    ui.status_bar(img, [
                        f"punto {idx + 1}/{len(points)}   "
                        f"{'midiendo' if recording else 'asentando'}   "
                        f"muestras={len(samples)}   [ESC cancela]",
                    ], y=h - 40)

                    cv2.imshow(WINDOW, img)
                    if (cv2.waitKey(1) & 0xFF) == 27:
                        aborted = True
                        break

                if aborted:
                    break

        cv2.destroyAllWindows()

    if not samples:
        print("[precisión] no se juntaron muestras.")
        return None

    stats = _summarise(samples)
    per_point = _per_point(samples)
    env = {
        "capture": cap_size,
        "brightness": float(np.mean(brightness)) if brightness else 0.0,
        "face_rate": faces_seen / max(frames_seen, 1),
    }

    ensure_dirs()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = LOGS_DIR / f"accuracy-{stamp}.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)

    jitter = float(np.mean([p["jitter_px"] for p in per_point]))
    raw_jitter = float(np.mean([p["raw_jitter_px"] for p in per_point]))
    offset = float(np.mean([p["offset_px"] for p in per_point]))

    print("\n=== resumen de precisión ===")
    print(f"  model={settings.model_name}  smoother={settings.smoother}  "
          f"capture={cap_size[0]}x{cap_size[1]}  "
          f"brightness={env['brightness']:.0f}/255  "
          f"faces={env['face_rate'] * 100:.0f}%")
    print(f"  muestras     {stats['samples']}")
    for key in ("mean_px", "median_px", "p95_px", "max_px"):
        v = stats[key]
        print(f"  {key:<12} {v:7.1f} px  ({v / PX_PER_DEG:.1f}°)")
    print(f"  bias         x {stats['bias_x_px']:+7.1f} px   "
          f"y {stats['bias_y_px']:+7.1f} px")

    print("\n--- temblor vs desvío ---")
    print(f"  temblor (precisión) {jitter:7.1f} px  "
          f"({jitter / PX_PER_DEG:.1f}°)   crudo {raw_jitter:.0f} px")
    print(f"  desvío  (exactitud) {offset:7.1f} px  "
          f"({offset / PX_PER_DEG:.1f}°)")

    print("\n--- por objetivo (medianas, robustas a saccades) ---")
    print("   idx    objetivo      desvío     temblor    arrastre")
    for p in per_point:
        flag = "  <- mirada en tránsito" if p["drift_px"] > 2.0 * PX_PER_DEG else ""
        print(f"   {p['point_index']:>3}  ({p['target_x']:>4},{p['target_y']:>4})"
              f"  {p['offset_px']:>8.0f} px  {p['jitter_px']:>8.0f} px"
              f"  {p['drift_px']:>+8.0f} px{flag}")

    min_tile = 2 * stats["p95_px"]
    cols_fit = max(1, int(1920 // max(min_tile, 1)))
    print(f"\n  -> celda CAA mínima segura ~= {min_tile:.0f} px (2x p95), unas "
          f"{cols_fit} columna(s) en 1920 px")

    print("\n--- qué cambiar (lo más prometedor primero) ---")
    for i, tip in enumerate(_diagnose(stats, per_point, env), 1):
        print(f"  {i}. {tip}")

    print(f"\n  muestras crudas -> {path}")
    return stats
