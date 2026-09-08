"""Re-solve a saved calibration offline, with different assumptions.

The command that exists because of one measured failure. The first real
calibration on hardware came back with *every* mount parameter pinned at its
bound - 52.0 mm of shift is exactly sqrt(3) x 30 mm, and 8.66 degrees is exactly
sqrt(3) x 5 - plus the corneal radius at its maximum and kappa pitch pinned at
+12 degrees on both eyes. The optimiser was telling us, unambiguously, that the
hand-measured mount was wrong by far more than the bounds allowed.

The obvious next step was to refit with wider bounds and see where the geometry
actually wanted to go. That was impossible: `calibrate` had discarded the
samples after fitting. Recovering the information would have cost another
session with the person in the chair.

So the samples are kept now, and this replays the entire post-collection half of
calibration against them: rebuild the template, refit every head pose, re-solve
the eye parameters and the screen pose, with whatever bounds and mount prior you
want to test. No camera, no user, a few seconds per attempt.

It also makes a genuinely useful diagnostic possible: run with the mount bounds
opened wide and read off *where the camera actually was*. If the solve lands 60
mm from the hand measurement, the tape measure was wrong; if it lands on the
measurement and the residual stays high, the problem is somewhere else.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import calibrate, display, estimator, eyemodel, fitting
from .config import CameraMount, Settings, ensure_dirs


def run(settings: Settings, *, mount_shift_mm: float | None = None,
        mount_rotate_deg: float | None = None,
        residual_order: int | None = None,
        no_residual: bool = False,
        no_refine_mount: bool = False,
        out_path: Path | None = None,
        report_mount: bool = True) -> Path | None:
    """Refit from saved samples. Returns the written model path."""
    ensure_dirs()
    samples_file = calibrate.samples_path(settings.model_path)
    raw, intrin = calibrate.load_samples(samples_file)
    print(f"[refit] {len(raw)} muestras de {samples_file.name}")
    print(f"[refit] color {intrin.width}x{intrin.height}, fx={intrin.fx:.1f}")

    screen = display.resolve(settings.monitor)

    # Rebuild the template from the same samples. Not reused from the old model:
    # a changed `bootstrap_basis` or a changed depth band should take effect
    # here, and that is precisely the kind of change worth re-measuring.
    template = calibrate._build_template(raw)
    print(f"[refit] plantilla: {len(template.points)} puntos, "
          f"ojos {template.eye_span_m * 1000:.0f} mm, "
          f"de {template.samples} cuadros")

    samples = calibrate._to_fit_samples(raw, template)
    if len(samples) < 20:
        print(f"[refit] solo {len(samples)} muestras con pose utilizable.")
        return None

    mount = CameraMount.from_screen(
        screen,
        offset_x_mm=settings.mount.offset_x_mm,
        offset_y_mm=settings.mount.offset_y_mm,
        offset_z_mm=settings.mount.offset_z_mm,
        pitch_deg=settings.mount.pitch_deg,
        yaw_deg=settings.mount.yaw_deg,
        roll_deg=settings.mount.roll_deg,
    )
    settings.mount = mount
    screen0 = mount.screen_plane(screen.width, screen.height)
    print(f"[refit] montaje de partida: {mount.describe()}")

    kwargs = {}
    if mount_shift_mm is not None:
        kwargs["mount_shift_m"] = mount_shift_mm / 1000.0
    if mount_rotate_deg is not None:
        kwargs["mount_rotate_deg"] = mount_rotate_deg
    if residual_order is not None:
        kwargs["residual_order"] = residual_order

    eyes, fitted_screen, corrector, report = fitting.fit(
        samples, eyemodel.EyeGeometry.initial(template), screen0,
        refine_mount=not no_refine_mount,
        residual_correction=not no_residual, **kwargs)

    if report_mount:
        _describe_solved_mount(mount, screen0, fitted_screen, screen)

    model = estimator.GazeModel(
        template=template, eyes=eyes, screen=fitted_screen,
        corrector=corrector, mount=mount, intrinsics=intrin,
        meta=calibrate._meta(settings, _ScreenShim(screen), report,
                             "refit", report.passes))
    path = Path(out_path or settings.model_path)
    model.save(path)
    print(f"\n[refit] guardado -> {path}")
    tile = 2.0 * report.p95_px
    print(f"[refit] celda mínima segura ~{tile:.0f} px (2x p95)")
    return path


class _ScreenShim:
    """Adapts a `display.Screen` to what `calibrate._meta` expects.

    `_meta` was written against a GazeTracker. Refit has no tracker - it never
    opens a camera or a window - so rather than build one just to read four
    fields off it, this hands over the same four.
    """

    def __init__(self, screen):
        self.screen = screen
        self.screen_w = screen.width
        self.screen_h = screen.height


def _describe_solved_mount(mount: CameraMount, before, after, screen) -> None:
    """Say where the fit thinks the camera is, in tape-measure units.

    The whole point of a wide-bounds refit. A screen-pose delta expressed as a
    rotation vector and a translation in camera coordinates is unreadable; the
    same thing expressed as "offset_y is 60 mm off" is something you can go and
    check with a ruler.
    """
    shift = after.origin - before.origin
    # Express the shift in the screen's own axes, which is where the
    # hand-measured offsets live.
    d_right = float(np.dot(shift, before.right)) * 1000.0
    d_down = float(np.dot(shift, before.down)) * 1000.0
    d_normal = float(np.dot(shift, before.normal)) * 1000.0

    print("\n--- dónde dice el ajuste que está la cámara ---")
    print(f"  el plano se movió {np.linalg.norm(shift) * 1000:.1f} mm respecto "
          f"de lo medido a mano")
    print(f"  en los ejes de la pantalla:  derecha {d_right:+6.1f} mm   "
          f"abajo {d_down:+6.1f} mm   hacia el usuario {d_normal:+6.1f} mm")
    # The screen moving right/down is the camera moving left/up relative to it,
    # so the implied correction to the measured offsets is the opposite sign.
    print(f"  => implica offset_x ≈ {mount.offset_x_mm - d_right:+.0f} mm, "
          f"offset_y ≈ {mount.offset_y_mm - d_down:+.0f} mm, "
          f"offset_z ≈ {mount.offset_z_mm - d_normal:+.0f} mm")

    angle = np.degrees(np.arccos(np.clip(
        float(np.dot(before.normal, after.normal)), -1.0, 1.0)))
    print(f"  la normal giró {angle:.2f}°")
    print("  Si estos números son grandes, medí de nuevo con la cinta antes de "
          "confiar en el ajuste:\n  un montaje mal medido se absorbe en kappa, "
          "y ahí se pierde la invariancia a la postura.")
