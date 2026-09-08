"""Calibration: show targets, collect, fit, save.

The collection order is not the obvious one, and the reason is worth stating.
The head template has to be built from depth, but the head *pose* of each
calibration sample has to be measured against that template - so the template
must exist before the poses can be computed, while the best data for building it
is the calibration set itself.

Resolved by collecting once and processing twice. During the dot sequence each
frame contributes only what is small and sufficient: 26 rigid 3D points, a
validity mask, and one iris ray per eye. Then the template is built from all of
it, every frame's pose is fitted against that template, and the eye parameters
and screen pose are solved. Keeping the frames themselves would be over a
gigabyte of aligned depth for a normal session; keeping the extracted points is
a few hundred kilobytes and loses nothing the fit uses.

Two passes are offered and recommended. Between them the user is asked to shift
posture, which is the only thing that separates "the eye is here" from "the
screen is there" in the fit - see the note in `fitting.py`. It doubles as the
honest test of the whole premise: if the model is head-invariant, a second
posture should not degrade it.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import (estimator, eyemodel, face3d, fitting, geometry, landmarks,
               positioning, source, ui)
from .config import Settings, ensure_dirs
from .frames import Frame, Intrinsics
from .tracker import GazeTracker

WINDOW = "CAA - calibración"

SETTLE_S = 0.55
"""Time given to the saccade and to the eye settling before samples count. A
target's first frames are the eye still travelling; counting them teaches the
model that the user looks near the previous dot."""

HOLD_S = 1.1
"""Sampling time per target. At 30 fps that is around 33 frames, which after
blink and confidence rejection reliably leaves 20 or more."""


@dataclass
class RawSample:
    """The compact per-frame record. See the module docstring."""

    target_px: np.ndarray
    target_index: int
    pass_index: int
    points_cam: np.ndarray
    valid: np.ndarray
    iris_rays: dict[str, np.ndarray]
    iris_radius_px: dict[str, float]


SAMPLES_SUFFIX = ".samples.npz"
"""Where the raw collection is kept, next to the model.

Saving it is not an optimisation, it is the difference between a hypothesis
about the mount costing a keystroke and costing another session with the person
in the chair. The first real calibration on hardware ended with every mount
parameter pinned at its bound; the obvious next step was to refit with wider
bounds, and that was impossible because the samples had been discarded after
fitting. They are a few hundred kilobytes.
"""


def samples_path(model_path: Path) -> Path:
    return Path(model_path).with_suffix(SAMPLES_SUFFIX)


def save_samples(raw: list["RawSample"], intrin: Intrinsics, path: Path) -> Path:
    """Persist the collection so the fit can be redone offline, any number of
    times, against different bounds or a rebuilt template."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(landmarks.EYE_KEYS)
    np.savez_compressed(
        path,
        rigid_indices=np.asarray(landmarks.RIGID_INDICES, dtype=np.int32),
        target_px=np.stack([s.target_px for s in raw]),
        target_index=np.asarray([s.target_index for s in raw], dtype=np.int32),
        pass_index=np.asarray([s.pass_index for s in raw], dtype=np.int32),
        points_cam=np.stack([s.points_cam for s in raw]).astype(np.float32),
        valid=np.stack([s.valid for s in raw]),
        eye_keys=np.asarray(keys),
        # A ray of NaN marks "this eye was not observed in this frame", which
        # keeps the arrays rectangular without inventing an observation.
        iris_rays=np.stack([
            np.stack([s.iris_rays.get(k, np.full(3, np.nan)) for k in keys])
            for s in raw]).astype(np.float32),
        iris_radius_px=np.asarray(
            [[s.iris_radius_px.get(k, np.nan) for k in keys] for s in raw],
            dtype=np.float32),
        intrinsics=np.asarray(json.dumps(intrin.as_dict())),
    )
    return path


def load_samples(path: Path):
    """Inverse of `save_samples`. Returns (raw samples, intrinsics)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no hay muestras de calibración en {path}. Se guardan desde la "
            f"versión que agregó `refit`; una calibración anterior no las "
            f"tiene y hay que repetirla una vez."
        )
    d = np.load(path, allow_pickle=False)
    keys = [str(k) for k in d["eye_keys"]]
    intrin = Intrinsics.from_dict(json.loads(str(d["intrinsics"])))

    raw: list[RawSample] = []
    for i in range(len(d["target_index"])):
        rays, radii = {}, {}
        for j, key in enumerate(keys):
            ray = d["iris_rays"][i, j].astype(np.float64)
            if np.all(np.isfinite(ray)):
                rays[key] = ray
                radii[key] = float(d["iris_radius_px"][i, j])
        raw.append(RawSample(
            target_px=d["target_px"][i].astype(np.float64),
            target_index=int(d["target_index"][i]),
            pass_index=int(d["pass_index"][i]),
            points_cam=d["points_cam"][i].astype(np.float64),
            valid=d["valid"][i].astype(bool),
            iris_rays=rays, iris_radius_px=radii,
        ))
    return raw, intrin


def grid_targets(rows: int, cols: int, width: int, height: int,
                 margin: float = 0.09) -> list[tuple[int, int]]:
    """Targets on a grid, in serpentine order.

    Serpentine rather than raster so consecutive dots are adjacent: it halves
    the total saccade distance, which means less settling time wasted and less
    fatigue. Fatigue is a first-order concern here - calibration is something
    this user group may need to repeat often.

    The margin is 9 % rather than the 12 % the webcam harness used. Corner
    accuracy was that pipeline's weak point so it kept targets away from the
    edges; a geometric model has no such collapse, and sampling closer to the
    corners is what proves it.
    """
    xs = np.linspace(width * margin, width * (1.0 - margin), cols)
    ys = np.linspace(height * margin, height * (1.0 - margin), rows)
    out: list[tuple[int, int]] = []
    for r, y in enumerate(ys):
        row = xs if r % 2 == 0 else xs[::-1]
        out += [(int(round(x)), int(round(y))) for x in row]
    return out


PATTERNS = {
    "5point": (1, 5),
    "9point": (3, 3),
    "13point": (3, 5),
    "16point": (4, 4),
}


def _targets(method: str, width: int, height: int) -> list[tuple[int, int]]:
    if method == "5point":
        # A cross rather than a row: five collinear points cannot constrain the
        # vertical at all, which a (1, 5) grid would silently do.
        cx, cy = width // 2, height // 2
        mx, my = int(width * 0.11), int(height * 0.11)
        return [(cx, cy), (mx, my), (width - mx, my),
                (width - mx, height - my), (mx, height - my)]
    rows, cols = PATTERNS.get(method, (3, 3))
    return grid_targets(rows, cols, width, height)


def _draw_target(img, x: int, y: int, phase: str, progress: float) -> None:
    """A dot that says whether it is being measured, without needing words."""
    if phase == "settle":
        # Hollow and pulsing: "look here, not counted yet".
        pulse = 20 + int(8 * abs(np.sin(time.time() * 5.0)))
        cv2.circle(img, (x, y), pulse, ui.ACCENT, 2, cv2.LINE_AA)
        cv2.circle(img, (x, y), 4, ui.ACCENT, -1, cv2.LINE_AA)
        return

    cv2.circle(img, (x, y), 22, ui.TEXT, 2, cv2.LINE_AA)
    cv2.circle(img, (x, y), 9, ui.TEXT, -1, cv2.LINE_AA)
    cv2.ellipse(img, (x, y), (34, 34), 0, -90, -90 + 360 * progress,
                ui.GOOD, 4, cv2.LINE_AA)


def _collect_pass(tracker: GazeTracker, src: source.FrameSource,
                  targets: list[tuple[int, int]], pass_index: int,
                  ) -> tuple[list[RawSample], list[np.ndarray], bool]:
    """Walk the targets once. Returns (samples, accel readings, aborted)."""
    landmarker = tracker.landmarker
    samples: list[RawSample] = []
    accels: list[np.ndarray] = []
    frames = src.frames()

    for index, (tx, ty) in enumerate(targets):
        t0 = time.time()
        while True:
            elapsed = time.time() - t0
            if elapsed > SETTLE_S + HOLD_S:
                break

            frame = next(frames, None)
            if frame is None:
                return samples, accels, True
            if frame.accel is not None:
                accels.append(frame.accel)

            obs = landmarker.detect(frame.color)
            recording = elapsed >= SETTLE_S

            if recording and obs is not None and not obs.blink:
                sample = _extract(frame, obs, (tx, ty), index, pass_index)
                if sample is not None:
                    samples.append(sample)

            img = ui.canvas(tracker.screen_w, tracker.screen_h)
            _draw_target(img, tx, ty, "hold" if recording else "settle",
                         max(0.0, (elapsed - SETTLE_S) / HOLD_S))
            state = ("sin cara" if obs is None
                     else "parpadeo" if obs.blink else "midiendo")
            ui.status_bar(img, [
                f"pasada {pass_index + 1}   punto {index + 1}/{len(targets)}   "
                f"{state}   muestras={len(samples)}   [ESC cancela]",
            ], y=tracker.screen_h - 40)

            cv2.imshow(WINDOW, img)
            if (cv2.waitKey(1) & 0xFF) == 27:
                return samples, accels, True

    return samples, accels, False


def _extract(frame: Frame, obs: landmarks.FaceObservation,
             target: tuple[int, int], index: int,
             pass_index: int) -> RawSample | None:
    """Reduce one frame to the few hundred bytes the fit needs."""
    pts_cam, valid = face3d.rigid_points(frame, obs)
    if pts_cam is None or valid.sum() < face3d.MIN_RIGID_POINTS:
        return None

    rays, radii = {}, {}
    for key in landmarks.EYE_KEYS:
        eye = obs.eyes.get(key)
        if eye is None:
            continue
        iris_px = np.mean(np.vstack([eye.iris_ring,
                                     eye.iris_centre[None, :]]), axis=0)
        rays[key] = frame.intrinsics.ray(iris_px)
        radii[key] = eye.iris_radius_px
    if not rays:
        return None

    return RawSample(target_px=np.array(target, dtype=np.float64),
                     target_index=index, pass_index=pass_index,
                     points_cam=pts_cam, valid=valid,
                     iris_rays=rays, iris_radius_px=radii)


def _reposition_screen(tracker: GazeTracker) -> bool:
    """Ask for a genuinely different posture between passes. False on ESC.

    Phrased as a small, specific movement. "Move your head" produces either
    nothing or a wild change; "shift about a hand's width" produces the 5-15 cm
    that actually breaks the degeneracy without leaving the frame.
    """
    w, h = tracker.screen_w, tracker.screen_h
    img = ui.canvas(w, h)
    ui.text_center(img, "CAMBIE UN POCO DE POSTURA", w // 2, int(h * 0.30), 60,
                   ui.ACCENT, bold=True)
    for i, line in enumerate([
        "Muévase unos 10 cm: acérquese o aléjese,",
        "o corra la cabeza hacia un costado.",
        "",
        "No hace falta precisión: lo único importante es",
        "que esta pasada NO sea desde el mismo lugar que la anterior.",
        "",
        "Con dos posturas distintas el sistema puede separar",
        "dónde está su ojo de dónde está la pantalla.",
        "Con una sola, no puede.",
    ]):
        ui.text_center(img, line, w // 2, int(h * 0.45) + i * 38, 27, ui.TEXT_DIM)
    ui.text_center(img, "ENTER para seguir     ESC para cancelar", w // 2,
                   int(h * 0.88), 30, ui.ACCENT)

    while True:
        cv2.imshow(WINDOW, img)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 32):
            return True
        if key == 27:
            return False
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            return False


def run(settings: Settings, method: str = "9point", passes: int = 2,
        ) -> Path | None:
    """Run calibration end to end and save the model. None if cancelled."""
    ensure_dirs()
    print(f"[calibración] patrón={method} pasadas={passes} "
          f"estimador={settings.estimator}")

    with GazeTracker(settings) as tracker:
        targets = _targets(method, tracker.screen_w, tracker.screen_h)
        print(f"[calibración] pantalla {tracker.screen.index} "
              f"({tracker.screen.name}) {tracker.screen_w}x{tracker.screen_h}, "
              f"{len(targets)} objetivos")

        with tracker.open_source() as src:
            print(f"[calibración] fuente: {src.describe()}")
            intrin = src.intrinsics
            print(f"[calibración] color {intrin.width}x{intrin.height}, "
                  f"fx={intrin.fx:.1f}, campo horizontal "
                  f"{intrin.hfov_deg():.1f}°, "
                  f"{intrin.px_per_mm_at(0.55):.2f} px/mm a 55 cm "
                  f"-> iris de ~{intrin.px_per_mm_at(0.55) * 12:.0f} px")

            if settings.check_position:
                if positioning.run(settings, tracker, src) is None:
                    print("[calibración] cancelada en la colocación.")
                    return None

            tracker.open_window(WINDOW)
            raw: list[RawSample] = []
            accels: list[np.ndarray] = []

            for p in range(max(1, passes)):
                if p > 0 and not _reposition_screen(tracker):
                    print("[calibración] cancelada entre pasadas.")
                    return None

                got, acc, aborted = _collect_pass(tracker, src, targets, p)
                raw += got
                accels += acc
                print(f"[calibración] pasada {p + 1}: {len(got)} muestras")
                if aborted:
                    print("[calibración] cancelada - no se guardó nada.")
                    cv2.destroyAllWindows()
                    return None

            cv2.destroyAllWindows()

        # ----------------------------------------------------------- process
        if len(raw) < 20:
            print(f"\n[calibración] solo {len(raw)} muestras utilizables. "
                  f"Casi siempre es falta de profundidad sobre la cara: "
                  f"revise la luz (el sol directo tapa el emisor infrarrojo) "
                  f"y la distancia.")
            return None

        template = _build_template(raw)
        print(f"[calibración] plantilla craneal: {len(template.points)} puntos, "
              f"distancia entre ojos {template.eye_span_m * 1000:.0f} mm, "
              f"de {template.samples} cuadros")

        accel = np.mean(accels, axis=0) if accels else None
        screen0 = tracker.default_screen_plane(accel)
        print(f"[calibración] montaje: {settings.mount.describe()}")

        eyes0 = eyemodel.EyeGeometry.initial(template)
        samples = _to_fit_samples(raw, template)
        if len(samples) < 20:
            print(f"[calibración] solo {len(samples)} muestras conservaron una "
                  f"pose de cabeza utilizable.")
            return None

        eyes, screen, corrector, report = fitting.fit(
            samples, eyes0, screen0,
            refine_mount=settings.refine_mount,
            residual_correction=settings.residual_correction,
        )

        # The intrinsics are saved with the model because the calibration is
        # only valid for the optics that produced it: a different colour
        # resolution changes fx, and every iris ray with it.
        model = estimator.GazeModel(
            template=template, eyes=eyes, screen=screen, corrector=corrector,
            mount=settings.mount, intrinsics=intrin,
            meta=_meta(settings, tracker, report, method, passes),
        )

        if settings.estimator == "hybrid":
            model.meta["hybrid"] = _fit_hybrid(samples, template, eyes)

        path = model.save(settings.model_path)
        sp = save_samples(raw, intrin, samples_path(settings.model_path))
        print(f"\n[calibración] guardado -> {path}")
        print(f"[calibración] muestras crudas -> {sp} "
              f"({sp.stat().st_size / 1e3:.0f} kB). Con esto se "
              f"reajusta sin volver a calibrar: `main.py refit`.")
        _advise(report, tracker)
        return path


def _build_template(raw: list[RawSample]) -> face3d.HeadTemplate:
    builder = face3d.TemplateBuilder()
    for s in raw:
        builder.add_observation(s.points_cam, s.valid)
    return builder.build()


def _to_fit_samples(raw: list[RawSample], template: face3d.HeadTemplate,
                    ) -> list[fitting.Sample]:
    """Fit each frame's head pose against the finished template.

    The template may have dropped landmarks that never had depth, so the stored
    point arrays are re-indexed onto whatever survived rather than assumed to
    line up. Skipping that check would misalign template points with
    observations and rotate every head pose by a few degrees - a failure that
    would show up only as mysteriously poor accuracy.
    """
    keep = [list(landmarks.RIGID_INDICES).index(i) for i in template.indices]
    out: list[fitting.Sample] = []

    for s in raw:
        head = face3d.fit_pose(template, s.points_cam[keep], s.valid[keep])
        if head is None:
            continue
        out.append(fitting.Sample(
            target_px=s.target_px,
            rotation=head.rotation, translation=head.translation,
            ray_dirs=s.iris_rays,
            weights={k: min(1.0, v / 12.0) for k, v in s.iris_radius_px.items()},
            target_index=s.target_index, pass_index=s.pass_index,
            head_distance_m=head.distance_m,
        ))
    return out


def _fit_hybrid(samples: list[fitting.Sample], template: face3d.HeadTemplate,
                eyes: eyemodel.EyeGeometry) -> dict:
    """Train the comparison arm on the same samples, for a fair contest.

    Same frames, same head poses, same eye model - the two arms differ only in
    how a gaze ray becomes a screen coordinate. Any other difference would
    confound the comparison this repo exists to make.
    """
    X, Y = [], []
    for s in samples:
        head = face3d.Head(rotation=s.rotation, translation=s.translation,
                           method="kabsch", residual_m=0.0, points_valid=0,
                           distance_m=s.head_distance_m)
        rays = {}
        for key, d in s.ray_dirs.items():
            params = eyes.eye(key)
            centre = head.to_camera(params.centre_head)
            pupil = geometry.ray_sphere_tangent(np.zeros(3), d, centre,
                                                eyes.radius_m)
            optical = geometry.normalize(pupil - centre)
            visual = geometry.apply_kappa(optical, params.kappa_yaw,
                                          params.kappa_pitch, head.basis)
            rays[key] = eyemodel.GazeRay(origin=centre, direction=visual,
                                         optical=optical, pupil=pupil, eye=key,
                                         eccentricity_deg=0.0)
        X.append(estimator.HybridRegressor.features(head, rays))
        Y.append(s.target_px)

    regressor = estimator.HybridRegressor()
    rms = regressor.fit(np.array(X), np.array(Y))
    print(f"[calibración] brazo híbrido (regresión sobre rasgos métricos): "
          f"RMS de entrenamiento {rms:.1f} px")
    return regressor.as_dict()


def _meta(settings: Settings, tracker: GazeTracker, report: fitting.FitReport,
          method: str, passes: int) -> dict:
    return {
        "monitor": {
            "index": tracker.screen.index,
            "name": tracker.screen.name,
            "size": [tracker.screen_w, tracker.screen_h],
            "origin": [tracker.screen.x, tracker.screen.y],
        },
        "camera": {
            "serial": settings.serial,
            "color_size": list(settings.color_size),
            "depth_size": list(settings.depth_size),
        },
        "mount": settings.mount.as_dict(),
        "calibration": {
            "method": method,
            "passes": passes,
            "samples": report.samples,
            "targets": report.targets,
            "rms_px": round(report.rms_px, 2),
            "p95_px": round(report.p95_px, 2),
            "rms_deg": round(report.rms_deg, 3),
            "geometric_rms_px": round(report.geometric_rms_px, 2),
            "head_spread_mm": round(report.head_spread_mm, 1),
            "warnings": report.warnings,
        },
        "estimator": settings.estimator,
    }


def _advise(report: fitting.FitReport, tracker: GazeTracker) -> None:
    """Turn the fit into the one number the board design actually needs."""
    tile = 2.0 * report.p95_px
    cols = max(1, int(tracker.screen_w // max(tile, 1.0)))
    print(f"[calibración] celda mínima segura ~{tile:.0f} px (2x p95) -> "
          f"unas {cols} columna(s) en {tracker.screen_w} px")
    print("[calibración] verifíquelo con `main.py accuracy`: el error de "
          "calibración se mide sobre los mismos puntos que se ajustaron, así "
          "que es optimista por construcción.")
