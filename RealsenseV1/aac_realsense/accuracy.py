"""Quantitative gaze check: known targets, measured error, ranked diagnosis.

Keeps the EyeTrax harness's structure and its CSV column order deliberately, so
the two pipelines' numbers can be put side by side without a conversion script.
It adds three things that only a metric camera makes possible:

* **Real degrees.** The webcam version had to hardcode 40 px per degree. Here
  the eye's distance is measured every frame, so degrees are computed rather
  than assumed - which is what makes the numbers comparable with the literature.
* **A head-invariance test.** Error is regressed against how far the head sat
  from its average position. This is the measurement that decides whether the
  whole change of camera was worth it: an appearance model's error grows with
  head displacement, a correct geometric model's does not. It is reported as a
  slope in px per cm, so the claim is falsifiable rather than rhetorical.
* **A fallback audit.** How often the run used PnP instead of depth, and the
  depth coverage on the face. A session that silently ran without depth
  throughout is measuring something other than what it claims to.

The jitter-versus-offset split from the webcam harness is kept unchanged,
because that distinction was right and the two still have opposite fixes.
"""

from __future__ import annotations

import csv
import time

import cv2
import numpy as np

from . import calibrate, positioning, ui
from .config import LOGS_DIR, Settings, ensure_dirs
from .tracker import GazeTracker

WINDOW = "CAA - prueba de precisión"


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
    """Offset versus jitter, per target.

    Both use the **median**, not the mean. If the gaze is still travelling when
    recording starts, a mean-based centre is dragged towards the previous
    fixation and a mean-based spread explodes - that is a saccade, not noise,
    and reporting it as jitter would send someone off raising the resolution to
    fix a timing problem. `drift_px` exposes it directly: the median error over
    the first third of the window minus the last third.
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
                np.hypot(rx - np.median(rx), ry - np.median(ry)))),
            "drift_px": float(np.median(err[:third]) - np.median(err[-third:])),
        })
    return out


def _head_invariance(rows: list[dict]) -> dict:
    """Does error grow as the head leaves its average position?

    The single most informative number this harness produces, because it tests
    the premise rather than the tuning. Fitted as a least-squares line of error
    against head displacement, giving a slope in px/cm.

    An appearance-based model has a strongly positive slope by construction - it
    learned one head position. A correct geometric model should be flat. A
    *negative* or noisy slope with little displacement in the data means the test
    had nothing to measure, which is why the spread is reported alongside.
    """
    disp = np.array([r["head_disp_cm"] for r in rows], dtype=np.float64)
    err = np.array([r["error_px"] for r in rows], dtype=np.float64)
    spread = float(disp.max() - disp.min()) if len(disp) else 0.0

    if len(disp) < 20 or spread < 1.0:
        return {"slope_px_per_cm": 0.0, "spread_cm": spread, "valid": False,
                "near_px": 0.0, "far_px": 0.0}

    slope, intercept = np.polyfit(disp, err, 1)
    # Also reported as a direct comparison of the closest and furthest thirds,
    # which is harder to misread than a regression coefficient.
    order = np.argsort(disp)
    third = max(1, len(order) // 3)
    return {
        "slope_px_per_cm": float(slope),
        "spread_cm": spread,
        "valid": True,
        "near_px": float(np.median(err[order[:third]])),
        "far_px": float(np.median(err[order[-third:]])),
    }


def _diagnose(stats: dict, points: list[dict], env: dict,
              invariance: dict, px_per_deg: float) -> list[str]:
    """Rank what to actually change."""
    tips: list[str] = []
    jitter = float(np.mean([p["jitter_px"] for p in points]))
    raw_jitter = float(np.mean([p["raw_jitter_px"] for p in points]))
    offset = float(np.mean([p["offset_px"] for p in points]))

    # Flagged first: while this is happening every other number is suspect.
    drifters = [p for p in points
                if p["drift_px"] > 2.0 * px_per_deg
                and p["drift_px"] > 1.5 * max(p["jitter_px"], 1.0)]
    if drifters:
        worst = max(drifters, key=lambda p: p["drift_px"])
        tips.append(
            f"En {len(drifters)} objetivo(s) la mirada todavía estaba llegando "
            f"cuando empezó la medición (objetivo {int(worst['point_index'])}: "
            f"{worst['drift_px']:.0f} px entre el principio y el final). Suba "
            f"--settle; esas muestras inflan la media, el p95 y el máximo."
        )

    # --- the failures specific to this camera --------------------------------
    if env["depth_coverage"] < 0.5:
        tips.append(
            f"Solo el {env['depth_coverage'] * 100:.0f}% de los puntos de la "
            f"cara tuvo profundidad. Sin profundidad no hay escala métrica y "
            f"todo el modelo geométrico se apoya en el respaldo por PnP. "
            f"Causas: sol directo (satura el emisor infrarrojo de 850 nm), "
            f"emisor apagado, o la persona fuera del rango de 35-85 cm."
        )
    if env["pnp_rate"] > 0.3:
        tips.append(
            f"El {env['pnp_rate'] * 100:.0f}% de los cuadros resolvió la pose "
            f"por PnP en vez de por profundidad. Funciona - la plantilla es "
            f"métrica - pero es más sensible al ruido de los puntos faciales. "
            f"Arreglar la profundidad es la mejora más grande disponible."
        )
    if env["mean_confidence"] < 0.5:
        tips.append(
            f"La confianza media fue {env['mean_confidence']:.2f}. Mire el "
            f"desglose de arriba: normalmente es iris chico (acérquese o suba "
            f"a --color 1920x1080) o residuo de ajuste alto (la plantilla no "
            f"coincide: ¿es otra persona la que se calibró?)."
        )
    if env["iris_px"] and env["iris_px"] < 8.0:
        tips.append(
            f"El iris mide {env['iris_px']:.1f} px de radio. Ese es el techo "
            f"físico de la precisión angular, porque el centro de la pupila "
            f"queda cuantizado por él. Acérquese, o use "
            f"--color 1920x1080 (un 50% más de píxeles a lo ancho)."
        )

    # --- the premise ---------------------------------------------------------
    if invariance["valid"]:
        slope = invariance["slope_px_per_cm"]
        if slope > 8.0:
            tips.append(
                f"El error crece {slope:.1f} px por cada cm que la cabeza se "
                f"aleja de su posición media ({invariance['near_px']:.0f} px "
                f"cerca vs {invariance['far_px']:.0f} px lejos). Eso es "
                f"justamente lo que NO debería pasar con un modelo geométrico: "
                f"revise la geometría del montaje en CameraMount y recalibre "
                f"con --passes 2, que es lo que separa la posición del ojo de "
                f"la de la pantalla."
            )
    elif invariance["spread_cm"] < 1.0:
        tips.append(
            "La cabeza casi no se movió durante la medición "
            f"({invariance['spread_cm']:.1f} cm), así que esta prueba no dice "
            "nada sobre la invariancia a la postura - que es la ventaja "
            "principal de esta cámara. Repita moviéndose a propósito entre "
            "objetivos."
        )

    # --- the two classic failures, unchanged from the webcam harness ---------
    if jitter > 3 * px_per_deg:
        tips.append(
            f"El temblor es alto ({jitter:.0f} px = "
            f"{jitter / px_per_deg:.1f}°): es calidad de señal, no "
            f"calibración. Suba la resolución de color, agregue luz frontal y "
            f"acérquese."
        )
    if jitter > 1.5 * px_per_deg and raw_jitter > 0 and jitter > 0.8 * raw_jitter:
        tips.append(
            f"El suavizado casi no ayuda ({raw_jitter:.0f} px crudo -> "
            f"{jitter:.0f} px suavizado). Pruebe --smoother median, o "
            f"--smoother kalman_ema --ema-alpha 0.6."
        )

    edge_collapse = False
    if len(points) >= 5:
        cx = float(np.mean([p["target_x"] for p in points]))
        cy = float(np.mean([p["target_y"] for p in points]))
        centre = min(points, key=lambda p: (p["target_x"] - cx) ** 2
                     + (p["target_y"] - cy) ** 2)
        worst3 = sorted(points, key=lambda p: p["offset_px"])[-3:]
        worst_mean = float(np.mean([p["offset_px"] for p in worst3]))
        if worst_mean > 1.5 * px_per_deg and worst_mean > 2.0 * max(
                centre["offset_px"], 1.0):
            edge_collapse = True
            tips.append(
                f"El error crece hacia los bordes (centro "
                f"{centre['offset_px']:.0f} px vs peor {worst_mean:.0f} px). "
                f"En un modelo geométrico esto NO es el colapso lineal que "
                f"tenía ridge: suele ser el plano de la pantalla mal ubicado, "
                f"o el modelo de esfera perdiendo validez con la mirada muy "
                f"excéntrica. Revise el montaje y considere "
                f"--residual-order 3."
            )

    if not edge_collapse and offset > 2 * px_per_deg and jitter < 1.5 * px_per_deg:
        tips.append(
            f"El punto es estable ({jitter:.0f} px de temblor) pero cae a "
            f"{offset:.0f} px del objetivo. Con esta cámara eso apunta a la "
            f"geometría, no al modelo: mida de nuevo los desplazamientos del "
            f"montaje (ver README) y recalibre."
        )

    if abs(stats["bias_x_px"]) > px_per_deg or abs(stats["bias_y_px"]) > px_per_deg:
        tips.append(
            f"Hay un desvío constante (x {stats['bias_x_px']:+.0f}, "
            f"y {stats['bias_y_px']:+.0f} px). En el camino geométrico esto es "
            f"casi siempre un error de traslación del montaje: "
            f"{abs(stats['bias_y_px']) / max(px_per_deg, 1):.1f}° en y "
            f"equivale a unos "
            f"{abs(stats['bias_y_px']) / max(px_per_deg, 1) * 0.55 * 17.5:.0f} mm "
            f"de error en offset_y_mm a 55 cm."
        )

    if not tips:
        tips.append(
            "No se ve nada mal. Dimensione el tablero con el p95 de arriba y "
            "confirme la invariancia a la postura moviéndose a propósito."
        )
    return tips


def run(settings: Settings, rows: int = 3, cols: int = 3,
        hold_s: float = 2.0, settle_s: float = 0.7,
        first_extra_settle_s: float = 1.3) -> dict | None:
    """Walk a rows x cols grid, sampling gaze error at each point."""
    samples: list[dict] = []
    frames_seen = faces_seen = pnp_frames = 0
    depth_coverages: list[float] = []
    confidences: list[float] = []
    iris_radii: list[float] = []
    brightness: list[float] = []

    with GazeTracker(settings) as tracker:
        tracker.load()
        w, h = tracker.screen_w, tracker.screen_h
        targets = calibrate.grid_targets(rows, cols, w, h, margin=0.09)

        print(f"[precisión] pantalla {tracker.screen.index} "
              f"({tracker.screen.name}) {w}x{h}")
        print(f"[precisión] grilla {rows}x{cols}, {hold_s}s por punto, "
              f"{settle_s}s de asentado, suavizado={settings.smoother}, "
              f"estimador={settings.estimator}")
        print("[precisión] Siga el punto blanco con la mirada. ESC cancela.")
        print("[precisión] Si puede, muévase un poco entre objetivos: es lo "
              "que permite medir la invariancia a la postura.")

        with tracker.open_source() as src:
            if settings.check_position:
                if positioning.run(
                    settings, tracker, src,
                    title="COLÓQUESE PARA LA MEDICIÓN",
                    subtitle="No hace falta repetir la postura de la calibración",
                ) is None:
                    print("[precisión] cancelado en la colocación.")
                    return None

            tracker.open_window(WINDOW)
            frames = src.frames()
            aborted = False

            for idx, (tx, ty) in enumerate(targets):
                t_start = time.time()
                tracker.reset_smoother()
                this_settle = settle_s + (first_extra_settle_s if idx == 0 else 0.0)

                while True:
                    elapsed = time.time() - t_start
                    if elapsed > this_settle + hold_s:
                        break

                    frame = next(frames, None)
                    if frame is None:
                        aborted = True
                        break
                    sample = tracker.process(frame)
                    recording = elapsed >= this_settle

                    if recording:
                        frames_seen += 1
                        depth_coverages.append(frame.depth_fill_rate())
                        if frames_seen % 10 == 0:
                            brightness.append(float(cv2.cvtColor(
                                frame.color, cv2.COLOR_BGR2GRAY).mean()))
                        if sample.ok:
                            faces_seen += 1
                            confidences.append(sample.confidence)
                            iris_radii.append(sample.iris_px)
                            if sample.method == "pnp":
                                pnp_frames += 1

                    if recording and sample.ok:
                        err = float(np.hypot(sample.x - tx, sample.y - ty))
                        head = tracker.last_head
                        samples.append({
                            # First eight columns match the EyeTrax harness's
                            # CSV exactly, so the two are directly comparable.
                            "point_index": idx,
                            "target_x": tx, "target_y": ty,
                            "gaze_x": sample.x, "gaze_y": sample.y,
                            "raw_x": sample.raw_x, "raw_y": sample.raw_y,
                            "error_px": err,
                            # New, and only possible with a metric camera.
                            "confidence": round(sample.confidence, 4),
                            "distance_cm": round(sample.distance_cm, 1),
                            "head_yaw": round(sample.head_yaw, 2),
                            "head_pitch": round(sample.head_pitch, 2),
                            "head_roll": round(sample.head_roll, 2),
                            "method": sample.method,
                            "head_x_cm": round(head.translation[0] * 100, 2) if head else 0.0,
                            "head_y_cm": round(head.translation[1] * 100, 2) if head else 0.0,
                            "head_z_cm": round(head.translation[2] * 100, 2) if head else 0.0,
                            "fit_residual_mm": round(head.residual_m * 1000, 3) if head else 0.0,
                            "iris_px": round(sample.iris_px, 2),
                        })

                    img = ui.canvas(w, h)
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
                        f"punto {idx + 1}/{len(targets)}   "
                        f"{'midiendo' if recording else 'asentando'}   "
                        f"muestras={len(samples)}   "
                        f"conf={sample.confidence:.2f}  "
                        f"{sample.distance_cm:.0f} cm  {sample.method}   "
                        f"[ESC cancela]",
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

        # Head displacement from the run's own mean, for the invariance test.
        head_xyz = np.array([[s["head_x_cm"], s["head_y_cm"], s["head_z_cm"]]
                             for s in samples])
        centre = head_xyz.mean(axis=0)
        for s, p in zip(samples, np.linalg.norm(head_xyz - centre, axis=1)):
            s["head_disp_cm"] = round(float(p), 2)

        px_per_deg = tracker.model.screen.px_per_degree(
            np.array([centre[0], centre[1], centre[2]]) / 100.0)

    stats = _summarise(samples)
    per_point = _per_point(samples)
    invariance = _head_invariance(samples)
    env = {
        "brightness": float(np.mean(brightness)) if brightness else 0.0,
        "face_rate": faces_seen / max(frames_seen, 1),
        "depth_coverage": float(np.mean(depth_coverages)) if depth_coverages else 0.0,
        "pnp_rate": pnp_frames / max(faces_seen, 1),
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "iris_px": float(np.mean(iris_radii)) if iris_radii else 0.0,
    }

    ensure_dirs()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = LOGS_DIR / f"accuracy-{stamp}.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)

    _report(stats, per_point, env, invariance, px_per_deg, settings, path)
    return stats


def _report(stats: dict, per_point: list[dict], env: dict, invariance: dict,
            px_per_deg: float, settings: Settings, path) -> None:
    jitter = float(np.mean([p["jitter_px"] for p in per_point]))
    raw_jitter = float(np.mean([p["raw_jitter_px"] for p in per_point]))
    offset = float(np.mean([p["offset_px"] for p in per_point]))

    print("\n=== resumen de precisión ===")
    print(f"  estimador={settings.estimator}  suavizado={settings.smoother}  "
          f"brillo={env['brightness']:.0f}/255  "
          f"caras={env['face_rate'] * 100:.0f}%")
    print(f"  profundidad sobre la cara {env['depth_coverage'] * 100:.0f}%   "
          f"pose por PnP {env['pnp_rate'] * 100:.0f}%   "
          f"confianza media {env['mean_confidence']:.2f}")
    print(f"  escala medida: {px_per_deg:.1f} px por grado "
          f"(no un valor fijo: sale de la distancia real)")
    print(f"  muestras     {stats['samples']}")
    for key in ("mean_px", "median_px", "p95_px", "max_px"):
        v = stats[key]
        print(f"  {key:<12} {v:7.1f} px  ({v / px_per_deg:.2f}°)")
    print(f"  bias         x {stats['bias_x_px']:+7.1f} px   "
          f"y {stats['bias_y_px']:+7.1f} px")

    print("\n--- temblor vs desvío ---")
    print(f"  temblor (precisión) {jitter:7.1f} px  "
          f"({jitter / px_per_deg:.2f}°)   crudo {raw_jitter:.0f} px")
    print(f"  desvío  (exactitud) {offset:7.1f} px  "
          f"({offset / px_per_deg:.2f}°)")

    print("\n--- invariancia a la postura (la prueba de fondo) ---")
    if invariance["valid"]:
        print(f"  la cabeza se movió en un rango de "
              f"{invariance['spread_cm']:.1f} cm")
        print(f"  pendiente del error: {invariance['slope_px_per_cm']:+.2f} px/cm")
        print(f"  tercio más cerca de la media: {invariance['near_px']:.0f} px   "
              f"tercio más lejos: {invariance['far_px']:.0f} px")
        verdict = ("plano - el modelo es invariante a la postura, que es "
                   "exactamente lo buscado"
                   if abs(invariance["slope_px_per_cm"]) < 8.0
                   else "el error crece con el movimiento: la geometría no "
                        "está bien resuelta")
        print(f"  -> {verdict}")
    else:
        print(f"  no medible: la cabeza se movió solo "
              f"{invariance['spread_cm']:.1f} cm")

    print("\n--- por objetivo (medianas, robustas a saccades) ---")
    print("   idx    objetivo      desvío     temblor    arrastre")
    for p in per_point:
        flag = "  <- mirada en tránsito" if p["drift_px"] > 2.0 * px_per_deg else ""
        print(f"   {p['point_index']:>3}  ({p['target_x']:>4},{p['target_y']:>4})"
              f"  {p['offset_px']:>8.0f} px  {p['jitter_px']:>8.0f} px"
              f"  {p['drift_px']:>+8.0f} px{flag}")

    min_tile = 2 * stats["p95_px"]
    cols_fit = max(1, int(1920 // max(min_tile, 1)))
    print(f"\n  -> celda CAA mínima segura ~= {min_tile:.0f} px (2x p95), unas "
          f"{cols_fit} columna(s) en 1920 px")

    print("\n--- qué cambiar (lo más prometedor primero) ---")
    for i, tip in enumerate(
            _diagnose(stats, per_point, env, invariance, px_per_deg), 1):
        print(f"  {i}. {tip}")

    print(f"\n  muestras crudas -> {path}")
