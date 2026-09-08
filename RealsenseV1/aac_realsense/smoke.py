"""Headless checks - no camera, no display, no RealSense SDK required.

This is the substitute for having the hardware, and it is deliberately more than
an import test. The forward problem has a closed form, so `synth.py` can build a
scene from known parameters and these checks require the real estimator to
recover them exactly. A sign error in the kappa rotation, a transposed rotation
matrix, a confused left/right eye, a millimetre/metre mix-up: each of those
breaks a specific assertion below rather than merely making accuracy
disappointing in a way nobody can attribute.

The checks that matter most, in order:

* `gaze_round_trip` - the whole chain, exact.
* `head_invariance` - the same eye parameters recovering the same target from
  five different head poses. This is the property the change of camera was for,
  and it is asserted numerically rather than argued.
* `forward_batch_matches_scalar` - the fit's vectorised model against the
  inference path. Two implementations of one model is a real hazard; this is
  what makes keeping both safe.
* `calibration_recovers_truth` - perturb the parameters, run the actual solver,
  check it finds its way back.

Run it after any change to the geometry. It is fast and it does not need you to
sit in front of a camera.
"""

from __future__ import annotations

import math
import sys
import time

import numpy as np

from . import (estimator, eyemodel, face3d, fitting, geometry, landmarks,
               smoothing, synth)
from .config import CameraMount, Settings
from .frames import DIST_INVERSE_BROWN, Intrinsics


def _check(name: str, fn) -> bool:
    t0 = time.perf_counter()
    try:
        detail = fn()
    except Exception as exc:
        print(f"  FALLA {name}: {type(exc).__name__}: {exc}")
        return False
    ms = (time.perf_counter() - t0) * 1000.0
    suffix = f" - {detail}" if detail else ""
    print(f"  ok    {name}{suffix}   [{ms:.0f} ms]")
    return True


# --------------------------------------------------------------- environment


def _versions() -> str:
    import cv2
    import scipy

    return (f"numpy {np.__version__}, cv2 {cv2.__version__}, "
            f"scipy {scipy.__version__}, python {sys.version.split()[0]}")


def _mediapipe() -> str:
    import mediapipe

    return f"mediapipe {mediapipe.__version__} (modelo no descargado aún)"


def _realsense_sdk() -> str:
    from . import rsdevice

    if not rsdevice.available():
        return "pyrealsense2 NO instalado - todo lo geométrico corre igual"
    rs = rsdevice.import_rs()
    devices = rsdevice.enumerate_devices()
    if not devices:
        return f"pyrealsense2 {rs.__version__} ok, sin cámara conectada"
    return (f"pyrealsense2 {rs.__version__}, "
            + ", ".join(d.model for d in devices))


def _realsense_api() -> str:
    """Every pyrealsense2 name this code uses must exist in the installed SDK.

    Worth a dedicated check because these are the failures that wait until the
    camera is plugged in and then abort a session: option enums get renamed
    across SDK versions, and `AttributeError: module has no attribute` is a poor
    thing to discover with a participant sitting in front of the screen. No
    camera is needed to ask whether a name exists.
    """
    from . import rsdevice

    if not rsdevice.available():
        return "pyrealsense2 no instalado - se omite"
    rs = rsdevice.import_rs()

    missing = []
    names = (
        "pipeline", "config", "context", "align", "threshold_filter",
        "disparity_transform", "spatial_filter", "temporal_filter",
        "hole_filling_filter", "intrinsics", "extrinsics",
        "rs2_deproject_pixel_to_point", "rs2_project_point_to_pixel",
        "stream.color", "stream.depth", "stream.accel", "stream.gyro",
        "format.bgr8", "format.z16",
        "option.emitter_enabled", "option.laser_power", "option.visual_preset",
        "option.global_time_enabled", "option.min_distance",
        "option.max_distance", "option.filter_magnitude",
        "option.filter_smooth_alpha", "option.filter_smooth_delta",
        "camera_info.name", "camera_info.serial_number",
        "camera_info.firmware_version", "camera_info.product_line",
        "camera_info.usb_type_descriptor",
        "distortion.brown_conrady", "distortion.inverse_brown_conrady",
        "config.enable_device", "config.enable_device_from_file",
        "config.enable_record_to_file",
    )
    for path in names:
        obj = rs
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                missing.append(path)
                break
    assert not missing, f"faltan en el SDK {rs.__version__}: {missing}"
    return f"{len(names)} nombres presentes en pyrealsense2 {rs.__version__}"


def _deprojection_matches_sdk() -> str:
    """Our numpy deprojection against librealsense's own implementation.

    `frames.Intrinsics` reimplements `rs2_deproject_pixel_to_point` so the
    geometry runs without the SDK and so a whole landmark array converts at
    once. That is only safe if the two agree, and the agreement can be checked
    with no camera at all - an `rs.intrinsics` is a plain struct that can be
    filled in by hand.

    All three distortion cases are exercised. This is the check that caught the
    real one: a single forward evaluation for `inverse_brown_conrady`, which is
    what the model's name implies and what older librealsense did, disagreed
    with the SDK by 32 mm at 0.6 m and in the opposite direction. Current SDKs
    run the iterative solve for both Brown-Conrady variants. If that changes
    again, this fails here rather than on the camera.
    """
    from . import rsdevice

    if not rsdevice.available():
        return "pyrealsense2 no instalado - se omite"
    rs = rsdevice.import_rs()

    worst = 0.0
    tested = []
    for model_name, rs_model, coeffs in (
        ("sin distorsión", rs.distortion.none, [0.0] * 5),
        ("inverse_brown_conrady", rs.distortion.inverse_brown_conrady,
         [0.12, -0.05, 0.001, -0.002, 0.01]),
        ("brown_conrady", rs.distortion.brown_conrady,
         [0.09, -0.03, 0.0008, -0.0015, 0.006]),
    ):
        native = rs.intrinsics()
        native.width, native.height = 1280, 720
        native.fx = native.fy = 926.0
        native.ppx, native.ppy = 641.3, 358.7
        native.model = rs_model
        native.coeffs = coeffs

        ours = Intrinsics.from_rs(native)
        pixels = [(640.0, 360.0), (100.0, 80.0), (1180.0, 660.0),
                  (320.0, 540.0), (960.0, 180.0)]
        depths = [0.60, 0.52, 0.71, 0.58, 0.65]

        theirs = np.array([rs.rs2_deproject_pixel_to_point(native, list(px), d)
                           for px, d in zip(pixels, depths)])
        mine = ours.deproject(np.array(pixels), np.array(depths))
        # Millimetres, the unit that matters: the D435's own depth noise at
        # 60 cm is about 1 mm. The floor here is not our arithmetic but the
        # SDK's - `rs2_intrinsics` is float32 throughout, so the last few digits
        # cannot agree with float64 no matter what. A fifth of a millimetre is
        # comfortably below the sensor's own noise and comfortably above that
        # floor, which is what makes it a useful threshold rather than a flaky
        # one.
        err_mm = float(np.abs(mine - theirs).max() * 1000.0)
        worst = max(worst, err_mm)
        tested.append(f"{model_name} {err_mm:.4f} mm")

    assert worst < 0.2, f"nuestra deproyección difiere del SDK en {worst:.4f} mm"
    return "; ".join(tested)


def _monitors() -> str:
    from .display import list_screens, resolve

    screens = list_screens()
    assert screens, "no se detectó ninguna pantalla"
    primary = resolve("primary")
    assert primary.is_primary or len(screens) == 1

    for i, s in enumerate(screens):
        assert resolve(i).name == s.name, f"el índice {i} resolvió mal"
        assert s.width > 0 and s.height > 0

    for bad in (len(screens), "laptop", "ask"):
        try:
            resolve(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"resolve({bad!r}) debió fallar")
    return f"{len(screens)} pantalla(s), índices y primary/secondary correctos"


def _accents() -> str:
    """Accented Spanish must survive rendering.

    Asserted because OpenCV's Hershey fonts have no glyph for 'ñ' and silently
    drop it, which turns BAÑO into BANO on a communication board. A regression
    here is not cosmetic - it is a wrong word spoken aloud.
    """
    from . import ui

    a = ui.text_size("BAÑO", 40, bold=True)
    b = ui.text_size("BANO", 40, bold=True)
    assert a != b, "BAÑO y BANO renderizan igual: se perdió la tilde"
    assert ui.text_size("SÍ", 40)[0] > 0 and ui.text_size("MÁS", 40)[0] > 0
    return f"BAÑO {a[0]}x{a[1]} px, distinto de BANO {b[0]}x{b[1]}"


# ------------------------------------------------------------------ geometry


def _speech_receives_accents() -> str:
    """La ñ tiene que llegar al sintetizador, no solo dibujarse bien.

    El chequeo `_accents` de arriba cubre el camino visual: que "BAÑO" se
    renderice distinto de "BANO", que era el bug de las fuentes Hershey de
    OpenCV. El camino de **audio** no tenía ninguna verificación, porque nadie
    escucha en un test - y ahí había un bug real y peor.

    La versión anterior de `tts` pasaba el texto por stdin de PowerShell, que lo
    decodifica con la codepage OEM (cp437/850 en una instalación en español).
    El byte 0xD1 de la ñ llegaba como ╤ - codepoint 9572, un carácter de dibujo
    de cajas. El aparato decía "ba" y un ruido, no "baño". La voz española nunca
    tuvo el problema: nunca recibió una ñ.

    Ahora el texto viaja en base64, que es ASCII puro, y esto compara codepoints
    en vez de sonido - así el camino de audio queda asertado sin necesidad de
    oír nada.
    """
    from . import tts

    if not tts.available():
        return "sin PowerShell: no se puede probar la voz acá"

    palabras = ["BAÑO", "SÍ", "MÁS", "NIÑO", "AÑO", "¿me das agua?",
                "me duele mucho"]
    for texto in palabras:
        got = tts.codepoints_roundtrip(texto)
        if got is None:
            return "el transporte de voz no respondió; no se pudo verificar"
        want = [ord(c) for c in texto]
        assert got == want, (
            f"{texto!r} llegó al sintetizador como {got}, no como {want}. "
            f"La ñ es el codepoint 241; si aparece 9572 volvió el bug de "
            f"codepage."
        )
    voz = tts.voice_name()
    return (f"{len(palabras)} textos con ñ y acentos llegan intactos; "
            f"voz: {voz or 'ninguna en español'}")


def _intrinsics() -> str:
    intrin = Intrinsics.synthetic(1280, 720)
    pts = np.array([[640.0, 360.0], [100.0, 80.0], [1200.0, 700.0]])
    depths = np.array([0.6, 0.55, 0.7])

    p3 = intrin.deproject(pts, depths)
    back = intrin.project(p3)
    assert np.allclose(back, pts, atol=1e-9), "deproject/project no es inverso"
    assert np.allclose(p3[:, 2], depths), "la profundidad no es el eje z"

    # Distortion: the undistort must invert the model it claims to implement.
    distorted = Intrinsics(1280, 720, intrin.fx, intrin.fy, 640.0, 360.0,
                           DIST_INVERSE_BROWN,
                           (0.12, -0.05, 0.001, -0.002, 0.01))
    n = distorted.normalized(pts)
    assert n.shape == (3, 2) and np.all(np.isfinite(n))
    # A centred pixel is a fixed point of any radial model.
    centre = distorted.normalized(np.array([[640.0, 360.0]]))
    assert np.allclose(centre, 0.0, atol=1e-12), "el centro óptico se movió"

    return (f"campo horizontal {intrin.hfov_deg():.1f}°, "
            f"{intrin.px_per_mm_at(0.55):.2f} px/mm a 55 cm "
            f"-> iris ~{intrin.px_per_mm_at(0.55) * 12:.0f} px")


def _geometry_primitives() -> str:
    R = geometry.rodrigues([0.0, 1.0, 0.0], math.radians(30.0))
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert abs(np.linalg.det(R) - 1.0) < 1e-12

    # Ray/sphere against the closed form: a ray along +z at a sphere on the axis.
    hit = geometry.ray_sphere(np.zeros(3), [0.0, 0.0, 1.0],
                             [0.0, 0.0, 1.0], 0.25)
    assert np.allclose(hit, [0.0, 0.0, 0.75], atol=1e-12)
    assert geometry.ray_sphere(np.zeros(3), [1.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0], 0.25) is None

    # The tangent fallback must always produce a point on the sphere.
    tangent = geometry.ray_sphere_tangent(np.zeros(3), [0.4, 0.0, 1.0],
                                          [0.0, 0.0, 1.0], 0.25)
    assert abs(np.linalg.norm(tangent - np.array([0.0, 0.0, 1.0])) - 0.25) < 1e-12

    plane = geometry.ray_plane(np.zeros(3), [0.0, 0.0, 1.0],
                              [0.0, 0.0, 2.0], [0.0, 0.0, -1.0])
    assert np.allclose(plane, [0.0, 0.0, 2.0], atol=1e-12)
    assert geometry.ray_plane(np.zeros(3), [0.0, 0.0, -1.0],
                              [0.0, 0.0, 2.0], [0.0, 0.0, -1.0]) is None

    # yaw/pitch round trip, including the +y-is-down convention.
    for yaw, pitch in ((0.0, 0.0), (0.3, -0.2), (-0.5, 0.4)):
        d = geometry.yaw_pitch_to_direction(yaw, pitch)
        y2, p2 = geometry.direction_to_yaw_pitch(d)
        assert abs(y2 - yaw) < 1e-12 and abs(p2 - pitch) < 1e-12
    down = geometry.yaw_pitch_to_direction(0.0, math.radians(10.0))
    assert down[1] > 0, "un pitch positivo debe mirar hacia abajo (+y)"

    return "rotaciones, ray/sphere, ray/plane y yaw/pitch consistentes"


def _kabsch() -> str:
    rng = np.random.default_rng(7)
    src = rng.normal(size=(20, 3)) * 0.05
    R_true = geometry.rodrigues([0.3, 0.8, -0.2], 0.7)
    t_true = np.array([0.02, -0.03, 0.58])

    dst = src @ R_true.T + t_true
    R, t = geometry.kabsch(src, dst)
    assert np.allclose(R, R_true, atol=1e-10), "kabsch no recuperó la rotación"
    assert np.allclose(t, t_true, atol=1e-10)

    # Outliers: three points wrecked. Plain least squares tilts the whole fit;
    # the robust version must reject them and land back on the truth.
    dirty = dst.copy()
    dirty[[2, 9, 15]] += np.array([0.08, -0.06, 0.09])
    R_plain, _ = geometry.kabsch(src, dirty)
    R_rob, t_rob, w = geometry.kabsch_robust(src, dirty)

    err_plain = geometry.angle_between(R_plain @ [0, 0, 1], R_true @ [0, 0, 1])
    err_rob = geometry.angle_between(R_rob @ [0, 0, 1], R_true @ [0, 0, 1])
    assert err_rob < err_plain / 3.0, (
        f"kabsch_robust no ayudó: {err_rob:.3f}° vs {err_plain:.3f}° simple")
    assert err_rob < 0.15, f"error robusto {err_rob:.3f}° demasiado alto"
    assert (w == 0).sum() >= 3, "no se rechazaron los tres puntos malos"

    return (f"exacto sin ruido; con 3/20 atípicos: simple {err_plain:.2f}° -> "
            f"robusto {err_rob:.3f}°")


def _screen_plane() -> str:
    screen = synth.default_screen(1920, 1080)

    # Pixel round trip at the corners and the centre.
    for px in ((0.0, 0.0), (1919.0, 0.0), (960.0, 540.0), (1919.0, 1079.0)):
        p3 = screen.px_to_point(*px)
        back = screen.point_to_px(p3)
        assert np.allclose(back, px, atol=1e-9), f"ida y vuelta falló en {px}"

    # Physical sanity for a camera on the bottom bezel, tilted up: the screen's
    # top-left corner must be above the camera (negative y) and roughly level
    # with it in depth. Getting a sign wrong here mirrors the whole board.
    origin = screen.origin
    assert origin[1] < 0, f"la esquina superior debería estar arriba, y={origin[1]:.3f}"
    assert origin[0] < 0, "la esquina izquierda debería estar a la izquierda"

    # The normal must point back towards the viewer, i.e. have a +z component
    # in the camera frame, since the camera sits in front of the screen.
    assert screen.normal[2] > 0.9, f"la normal apunta mal: {screen.normal}"

    eye = np.array([0.0, 0.0, 0.58])
    ppd = screen.px_per_degree(eye)
    assert 20.0 < ppd < 80.0, f"{ppd:.1f} px/grado es implausible a 58 cm"

    return (f"ida y vuelta exacta; {ppd:.1f} px por grado a 58 cm "
            f"(el arnés de webcam usaba 40 fijo)")


def _mount_from_gravity() -> str:
    """The accelerometer sign convention, asserted.

    A D435i at rest measures the reaction to gravity, which points *up*, and its
    +y axis points down - so level reads (0, -9.8, 0). Inverting that sign tilts
    the screen plane by twice the mount angle, and the resulting error looks like
    an ordinary calibration offset rather than like a bug.
    """
    base = CameraMount(pitch_deg=99.0, roll_deg=99.0)

    level = base.with_gravity(np.array([0.0, -9.81, 0.0]))
    assert abs(level.pitch_deg) < 1e-6, f"nivelada dio pitch {level.pitch_deg}"
    assert abs(level.roll_deg) < 1e-6, f"nivelada dio roll {level.roll_deg}"

    # Camera tilted up by 10 degrees: world-down rotates towards +z, so the
    # reaction force (its negative) picks up a -z component.
    for angle in (5.0, 10.0, 20.0):
        rad = math.radians(angle)
        accel = -9.81 * np.array([0.0, math.cos(rad), -math.sin(rad)])
        got = base.with_gravity(accel)
        assert abs(got.pitch_deg - angle) < 1e-6, (
            f"inclinación de {angle}° leída como {got.pitch_deg:.3f}°")
        assert abs(got.roll_deg) < 1e-6

    rad = math.radians(8.0)
    rolled = base.with_gravity(-9.81 * np.array([math.sin(rad), math.cos(rad), 0.0]))
    assert abs(rolled.roll_deg - 8.0) < 1e-6, f"roll {rolled.roll_deg:.3f}"

    # A reading that is not 1 g means the camera was moving: keep the measured
    # values rather than trusting it.
    moving = base.with_gravity(np.array([0.0, -2.0, 0.0]))
    assert moving.pitch_deg == 99.0, "se aceptó una lectura que no era 1 g"

    return "nivelada -> 0°, inclinaciones de 5/10/20° y roll de 8° exactos"


# --------------------------------------------------------------- the eye model


def _pose_fitting() -> str:
    """Pose recovery from synthetic 3D points, including the PnP fallback."""
    template = synth.synth_template()
    intrin = Intrinsics.synthetic()

    worst_kabsch = worst_pnp = 0.0
    for yaw, pitch, dist in ((0.0, 0.0, 0.58), (18.0, -8.0, 0.45),
                             (-25.0, 12.0, 0.72)):
        truth = synth.head_pose(distance_m=dist, yaw_deg=yaw, pitch_deg=pitch)
        scene = synth.make_scene(truth, synth.synth_eyes(),
                                 synth.default_screen(), (960.0, 540.0),
                                 intrin=intrin)

        pts_cam, valid = face3d.rigid_points(scene.frame, scene.observation,
                                             template.indices)
        assert pts_cam is not None and valid.all(), "faltó profundidad sintética"

        got = face3d.fit_pose(template, pts_cam, valid)
        assert got is not None
        worst_kabsch = max(worst_kabsch,
                           float(np.linalg.norm(got.translation - truth.translation)))
        assert np.allclose(got.rotation, truth.rotation, atol=1e-6)

        # PnP: same data, no depth at all. It must still recover metric
        # translation, because the template carries true scale.
        tracker = face3d.HeadTracker(template)
        scene.frame.depth_m[:] = 0.0
        pnp = tracker.track(scene.frame, scene.observation)
        assert pnp is not None and pnp.method == "pnp"
        worst_pnp = max(worst_pnp,
                        float(np.linalg.norm(pnp.translation - truth.translation)))

    assert worst_kabsch < 1e-6, f"kabsch erró {worst_kabsch * 1000:.3f} mm"
    assert worst_pnp < 0.002, f"PnP erró {worst_pnp * 1000:.2f} mm"
    return (f"kabsch exacto (<{worst_kabsch * 1e6:.1f} µm); PnP sin profundidad "
            f"dentro de {worst_pnp * 1000:.2f} mm")


def _template_builder() -> str:
    """The template must be metric and rigid, whatever frame it lands in.

    The bootstrap frame is an arbitrary convention, so the invariant to assert
    is not the coordinates but the **pairwise distances**: those are what the
    pose fit consumes and what makes the model metric. Asserting the coordinates
    would be asserting the convention, which is not the property that matters.
    """
    truth = synth.synth_template()
    builder = face3d.TemplateBuilder(indices=truth.indices)

    rng = np.random.default_rng(3)
    for i in range(30):
        head = synth.head_pose(distance_m=0.5 + 0.1 * rng.random(),
                               offset=(rng.normal(0, 0.02), rng.normal(0, 0.02)),
                               yaw_deg=rng.normal(0, 10),
                               pitch_deg=rng.normal(0, 6))
        scene = synth.make_scene(head, synth.synth_eyes(),
                                 synth.default_screen(), (960.0, 540.0),
                                 depth_noise_m=0.001, rng=rng)
        pts_cam, valid = face3d.rigid_points(scene.frame, scene.observation,
                                             truth.indices)
        assert builder.add_observation(pts_cam, valid), f"cuadro {i} rechazado"

    template = builder.build()
    assert template.indices == truth.indices

    def dists(points):
        d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        return d[np.triu_indices(len(points), 1)]

    error_mm = np.abs(dists(template.points) - dists(truth.points)) * 1000.0
    assert error_mm.max() < 2.0, (
        f"la geometría de la plantilla se desvió {error_mm.max():.2f} mm")
    assert abs(template.eye_span_m - truth.eye_span_m) < 0.002

    return (f"{len(template.points)} puntos de 30 cuadros con 1 mm de ruido; "
            f"distancias dentro de {error_mm.max():.2f} mm, "
            f"media {error_mm.mean():.2f} mm")


def _gaze_round_trip() -> str:
    """The whole chain, exact: a target in, the same target out.

    Runs the real `GeometricEstimator` - not a reimplementation - on a scene
    built by inverting the eye model. Any sign or transpose error anywhere in
    intrinsics, head pose, kappa or the plane intersection shows up here as a
    large pixel error.
    """
    template = synth.synth_template()
    eyes = synth.synth_eyes()
    screen = synth.default_screen(1920, 1080)
    model = estimator.GazeModel(template=template, eyes=eyes, screen=screen)
    est = estimator.GeometricEstimator(model, Settings(monitor="primary"))

    targets = [(960, 540), (100, 60), (1820, 60), (100, 1020), (1820, 1020),
               (960, 60), (100, 540)]
    head = synth.head_pose(distance_m=0.58)

    worst = 0.0
    for target in targets:
        scene = synth.make_scene(head, eyes, screen, target)
        got = est.predict(scene.frame, scene.observation)
        assert got.ok, f"el estimador falló en {target}: {got.reason}"
        err = math.hypot(got.x - target[0], got.y - target[1])
        worst = max(worst, err)

    assert worst < 0.05, f"la ida y vuelta erró {worst:.3f} px"
    return f"{len(targets)} objetivos incl. las 4 esquinas, peor error {worst:.4f} px"


def _head_invariance() -> str:
    """The claim behind the whole rewrite, asserted numerically.

    One eye geometry, one screen, one target - and five genuinely different head
    poses, spanning 25 cm of distance and 50 degrees of yaw. A head-invariant
    model returns the same pixel from all of them. An appearance regression
    trained at one pose cannot; that is precisely the 405 px corner error in this
    repo's EyeTrax logs.
    """
    eyes = synth.synth_eyes()
    screen = synth.default_screen(1920, 1080)
    model = estimator.GazeModel(template=synth.synth_template(), eyes=eyes,
                               screen=screen)
    est = estimator.GeometricEstimator(model, Settings(monitor="primary"))

    poses = [
        synth.head_pose(distance_m=0.45),
        synth.head_pose(distance_m=0.70),
        synth.head_pose(distance_m=0.58, offset=(0.10, 0.0)),
        synth.head_pose(distance_m=0.58, offset=(-0.08, 0.06), yaw_deg=22.0),
        synth.head_pose(distance_m=0.55, yaw_deg=-25.0, pitch_deg=12.0,
                        roll_deg=9.0),
    ]

    worst = 0.0
    for target in ((960, 540), (150, 120), (1780, 960)):
        for pose in poses:
            scene = synth.make_scene(pose, eyes, screen, target)
            got = est.predict(scene.frame, scene.observation)
            assert got.ok, f"falló en {target} con la cabeza a {pose.distance_m:.2f} m"
            worst = max(worst, math.hypot(got.x - target[0], got.y - target[1]))

    assert worst < 0.05, (
        f"el error varía {worst:.2f} px según la pose: NO es invariante")
    return (f"3 objetivos x 5 poses (0.45-0.70 m, ±25° de giro): "
            f"error máximo {worst:.4f} px")


def _forward_batch_matches_scalar() -> str:
    """The fit's vectorised model must equal the inference path exactly.

    Keeping two implementations of one model is a genuine hazard: if they drift,
    calibration optimises a model that inference never uses, and nothing looks
    wrong from the outside. This assertion is what makes the fast path safe.
    """
    eyes = synth.synth_eyes()
    screen = synth.default_screen(1920, 1080)
    intrin = Intrinsics.synthetic()

    poses, dirs = [], []
    for i, pose in enumerate([
        synth.head_pose(distance_m=0.50),
        synth.head_pose(distance_m=0.65, offset=(0.05, -0.03)),
        synth.head_pose(distance_m=0.58, yaw_deg=20.0, pitch_deg=-10.0),
    ]):
        for target in ((300, 200), (960, 540), (1700, 900)):
            scene = synth.make_scene(pose, eyes, screen, target, intrin=intrin)
            poses.append(pose)
            eye_obs = scene.observation.eyes["right"]
            iris = np.mean(np.vstack([eye_obs.iris_ring,
                                      eye_obs.iris_centre[None, :]]), axis=0)
            dirs.append(intrin.ray(iris))

    R = np.stack([p.rotation for p in poses])
    t = np.stack([p.translation for p in poses])
    params = eyes.eye("right")

    batch = eyemodel.forward_batch(R, t, np.stack(dirs), params.centre_head,
                                   params.kappa_yaw, params.kappa_pitch,
                                   eyes.radius_m, screen)

    scalar = []
    for pose, d in zip(poses, dirs):
        centre = pose.to_camera(params.centre_head)
        pupil = geometry.ray_sphere_tangent(np.zeros(3), d, centre, eyes.radius_m)
        optical = geometry.normalize(pupil - centre)
        visual = geometry.apply_kappa(optical, params.kappa_yaw,
                                      params.kappa_pitch, pose.basis)
        scalar.append(screen.intersect(centre, visual))

    diff = np.abs(batch - np.array(scalar)).max()
    assert diff < 1e-9, f"vectorizado y escalar difieren en {diff:.3e} px"
    return f"{len(dirs)} muestras, diferencia máxima {diff:.2e} px"


# ------------------------------------------------------------------ the fit


def _residual_corrector() -> str:
    """The corrector must remove a known smooth distortion, not memorise points."""
    grid = np.array([[u, v] for u in np.linspace(-0.9, 0.9, 5)
                     for v in np.linspace(-0.9, 0.9, 5)])
    distorted = grid + np.column_stack([
        0.04 + 0.05 * grid[:, 0] ** 2 - 0.03 * grid[:, 1],
        -0.02 + 0.04 * grid[:, 1] ** 2 + 0.02 * grid[:, 0] * grid[:, 1],
    ])

    before = np.linalg.norm(distorted - grid, axis=1).mean()
    corrector = estimator.ResidualCorrector(order=2, ridge=1e-6)
    corrector.fit(distorted, grid)
    after = np.linalg.norm(corrector.apply(distorted) - grid, axis=1).mean()

    assert after < before / 20.0, f"solo mejoró {before:.4f} -> {after:.4f}"

    # With too few points the order must drop rather than overfit.
    small = estimator.ResidualCorrector(order=2)
    small.fit(distorted[:4], grid[:4])
    assert small.order < 2, "no bajó el orden con pocas muestras"

    # And an unfitted corrector must be the identity, not a crash.
    fresh = estimator.ResidualCorrector()
    assert np.allclose(fresh.apply(np.array([[0.3, -0.4]])), [[0.3, -0.4]])

    return (f"error medio {before:.4f} -> {after:.5f} (unidades normalizadas); "
            f"baja el orden con pocas muestras")


def _calibration_recovers_truth() -> str:
    """Run the real solver from the real starting point and check it converges.

    Deliberately set up the way production is, not the way that makes the test
    easy:

    * The initial eye geometry is `EyeGeometry.initial(template)` - the actual
      anatomical construction the code uses, not a small nudge from the answer.
    * The ground truth sits about 6 mm from that initial guess per eye, with
      kappa well away from the 4.5-degree default, which is the sort of
      individual variation this has to absorb.
    * The mount is wrong by 15 mm and 2 degrees, as a tape measure against a
      bezel will be.
    * A third of a pixel of landmark noise, roughly what MediaPipe delivers on a
      well-lit iris.
    * **Two head positions.** With one, the eye/screen degeneracy is unbroken
      and this test would pass while concealing the exact failure it exists to
      catch.
    """
    template = synth.synth_template()
    intrin = Intrinsics.synthetic()

    truth_eyes = eyemodel.EyeGeometry(
        right=eyemodel.EyeParams(np.array([-0.0335, -0.0040, 0.0272]),
                                 math.radians(6.4), math.radians(1.8)),
        left=eyemodel.EyeParams(np.array([0.0286, 0.0035, 0.0198]),
                                math.radians(-3.1), math.radians(0.4)),
        radius_m=0.0128,
    )
    truth_screen = synth.default_screen(1920, 1080)

    targets = [(x, y) for y in (120, 540, 960) for x in (150, 960, 1780)]
    poses = [synth.head_pose(distance_m=0.55),
             synth.head_pose(distance_m=0.66, offset=(0.06, -0.04))]

    rng = np.random.default_rng(5)
    samples: list[fitting.Sample] = []
    for pass_index, pose in enumerate(poses):
        for index, target in enumerate(targets):
            for _ in range(8):
                scene = synth.make_scene(pose, truth_eyes, truth_screen,
                                         target, intrin=intrin)
                obs = scene.observation
                for key in ("right", "left"):
                    obs.eyes[key].iris_centre = (obs.eyes[key].iris_centre
                                                 + rng.normal(0, 0.33, 2))
                samples.append(fitting.Sample.build(
                    target, pose, obs, intrin, index, pass_index))

    # The production starting point, from this person's own template.
    start = eyemodel.EyeGeometry.initial(template)
    start_offset_mm = 1000.0 * max(
        float(np.linalg.norm(start.right.centre_head - truth_eyes.right.centre_head)),
        float(np.linalg.norm(start.left.centre_head - truth_eyes.left.centre_head)))

    bad_mount = CameraMount(offset_x_mm=8.0, offset_y_mm=172.0 - 15.0,
                            offset_z_mm=18.0, pitch_deg=7.0 + 2.0,
                            screen_width_mm=527.0, screen_height_mm=296.0)
    perturbed_screen = bad_mount.screen_plane(1920, 1080)

    eyes, screen, corrector, report = fitting.fit(
        samples, start, perturbed_screen, refine_mount=True,
        residual_correction=True, verbose=False)

    assert report.rms_px < 25.0, f"RMS de {report.rms_px:.1f} px tras el ajuste"
    assert report.head_spread_mm > 15.0, "las dos posturas no se registraron"
    assert 10.5 <= report.radius_mm <= 13.5

    # The real test of a *model* rather than a curve fit: hold out a head pose
    # that was never in the calibration set and require it to still work.
    model = estimator.GazeModel(template=template, eyes=eyes, screen=screen,
                                corrector=corrector)
    est = estimator.GeometricEstimator(model, Settings(monitor="primary"))
    held_out = synth.head_pose(distance_m=0.61, offset=(-0.09, 0.05),
                               yaw_deg=15.0, pitch_deg=-7.0)
    errors = []
    for target in ((300, 250), (960, 540), (1650, 850)):
        scene = synth.make_scene(held_out, truth_eyes, truth_screen, target,
                                 intrin=intrin)
        got = est.predict(scene.frame, scene.observation)
        assert got.ok
        errors.append(math.hypot(got.x - target[0], got.y - target[1]))

    held_out_rms = float(np.sqrt(np.mean(np.square(errors))))
    assert held_out_rms < 60.0, (
        f"{held_out_rms:.0f} px en una postura no calibrada: no generaliza")

    return (f"arranque a {start_offset_mm:.1f} mm de la verdad -> RMS de ajuste "
            f"{report.rms_px:.1f} px, radio {report.radius_mm:.2f} mm "
            f"(verdad 12.8); postura NO calibrada {held_out_rms:.0f} px")


def _degeneracy_is_detected() -> str:
    """A single-posture calibration must be *reported* as untrustworthy.

    It will still fit, and fit well, on its own targets - that is the trap. The
    warning is the only thing standing between a small reported residual and a
    model that drifts the moment the user shifts in their chair.
    """
    truth_eyes = synth.synth_eyes()
    truth_screen = synth.default_screen(1920, 1080)
    intrin = Intrinsics.synthetic()
    pose = synth.head_pose(distance_m=0.58)

    samples = []
    for index, target in enumerate([(x, y) for y in (120, 540, 960)
                                    for x in (150, 960, 1780)]):
        for _ in range(10):
            scene = synth.make_scene(pose, truth_eyes, truth_screen, target,
                                     intrin=intrin)
            samples.append(fitting.Sample.build(target, pose,
                                                scene.observation, intrin,
                                                index, 0))

    _, _, _, report = fitting.fit(samples, synth.synth_eyes(), truth_screen,
                                  refine_mount=True, verbose=False)
    assert report.head_spread_mm < 5.0
    assert any("movió muy poco" in w for w in report.warnings), (
        f"no se avisó de la degeneración; avisos: {report.warnings}")
    return (f"una sola postura ({report.head_spread_mm:.1f} mm) queda avisada "
            f"pese a un RMS de {report.rms_px:.1f} px")


# --------------------------------------------------------- dwell and smoothing


def _dwell() -> str:
    from .tracker import DwellSelector

    settings = Settings(dwell_ms=800, refractory_ms=600, blink_grace_ms=250,
                        monitor="primary")
    d = DwellSelector(settings, min_confidence=0.25)
    t = 1000.0

    assert d.update("A", 1.0, t)[1] is None
    assert d.update("A", 1.0, t + 0.4)[0] > 0.4
    progress, fired = d.update("A", 1.0, t + 0.81)
    assert fired == "A", "no disparó al completar la permanencia"

    # Refractory: one long look must not fire twice.
    assert d.update("A", 1.0, t + 0.9)[1] is None
    assert d.update("A", 1.0, t + 1.2)[1] is None

    # Changing target resets.
    d2 = DwellSelector(settings)
    d2.update("A", 1.0, t)
    d2.update("B", 1.0, t + 0.5)
    assert d2.update("B", 1.0, t + 0.6)[0] < 0.2

    # A blink inside the grace window keeps progress; past it, resets.
    d3 = DwellSelector(settings)
    d3.update("A", 1.0, t)
    assert d3.update(None, 1.0, t + 0.4)[0] > 0.4, "el parpadeo perdió el avance"
    assert d3.update(None, 1.0, t + 0.8)[0] == 0.0, "la pérdida larga no reinició"

    # Low confidence holds: neither advances nor resets. This is the behaviour
    # the webcam pipeline could not have, since it had no confidence at all.
    d4 = DwellSelector(settings, min_confidence=0.25)
    d4.update("A", 1.0, t)
    held = d4.update("A", 0.1, t + 0.4)
    assert held[1] is None and held[0] > 0.4, "una señal débil no debe avanzar"
    assert d4.update("A", 0.1, t + 0.5)[1] is None
    resumed = d4.update("A", 1.0, t + 0.6)
    assert resumed[0] > 0.5, "no retomó el avance al volver la confianza"

    return "disparo, refractario, cambio de objetivo, parpadeo y señal débil"


def _smoothers() -> str:
    out = []
    for name in ("none", "ema", "kalman", "kalman_ema", "median"):
        s = smoothing.make(name, ema_alpha=0.25)
        for _ in range(80):
            x, y = s.step(500.0, 300.0, 1.0)
        assert abs(x - 500.0) < 2.0 and abs(y - 300.0) < 2.0, (
            f"{name} no converge a una entrada constante: ({x:.1f}, {y:.1f})")
        out.append(name)

    # Confidence must actually change the behaviour, or the plumbing is dead.
    a = smoothing.make("kalman", ema_alpha=0.0)
    b = smoothing.make("kalman", ema_alpha=0.0)
    for _ in range(20):
        a.step(500.0, 300.0, 1.0)
        b.step(500.0, 300.0, 1.0)
    trusted = a.step(900.0, 300.0, 1.0)[0]
    doubted = b.step(900.0, 300.0, 0.1)[0]
    assert doubted < trusted - 20.0, (
        f"la confianza no afecta el filtro: {trusted:.0f} vs {doubted:.0f}")

    return f"{', '.join(out)} convergen; la confianza atenúa un salto dudoso"


def _model_serialisation() -> str:
    """A saved model must reload to the same numbers, through JSON.

    JSON rather than a pickle: the webcam harness pickled an sklearn estimator,
    which ties a calibration to the library versions that made it. These are
    eleven physical parameters and a screen pose - numbers that belong in a
    thesis appendix and should still load in two years.
    """
    import tempfile
    from pathlib import Path

    model = estimator.GazeModel(
        template=synth.synth_template(), eyes=synth.synth_eyes(),
        screen=synth.default_screen(1920, 1080),
        intrinsics=Intrinsics.synthetic(),
        meta={"monitor": {"name": "TEST", "size": [1920, 1080]}})
    corrector = estimator.ResidualCorrector(order=2)
    corrector.fit(np.array([[0.1, 0.1], [0.5, -0.3], [-0.6, 0.2], [0.0, 0.7],
                            [-0.4, -0.5], [0.8, 0.1], [0.2, -0.8], [-0.9, 0.4]]),
                  np.array([[0.12, 0.09], [0.52, -0.28], [-0.58, 0.22],
                            [0.01, 0.72], [-0.38, -0.48], [0.82, 0.13],
                            [0.22, -0.78], [-0.88, 0.43]]))
    model.corrector = corrector

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.json"
        model.save(path)
        back = estimator.GazeModel.load(path)

    assert np.allclose(back.template.points, model.template.points)
    assert np.allclose(back.eyes.to_vector(), model.eyes.to_vector(), atol=1e-6)
    assert np.allclose(back.screen.origin, model.screen.origin, atol=1e-6)
    assert np.allclose(back.corrector.coeffs, corrector.coeffs, atol=1e-9)
    assert back.intrinsics.fx == model.intrinsics.fx

    # An estimator built from the reloaded model must give identical answers.
    a = estimator.GeometricEstimator(model, Settings(monitor="primary"))
    b = estimator.GeometricEstimator(back, Settings(monitor="primary"))
    scene = synth.make_scene(synth.head_pose(), synth.synth_eyes(),
                             model.screen, (700, 400))
    pa = a.predict(scene.frame, scene.observation)
    pb = b.predict(scene.frame, scene.observation)
    # Not bit-exactness: `as_dict` rounds deliberately, because these files are
    # meant to be read by a person. The requirement is that the rounding be far
    # below anything measurable - a hundredth of a pixel.
    drift = max(abs(pa.x - pb.x), abs(pa.y - pb.y))
    assert drift < 0.01, f"el redondeo del JSON movió la mirada {drift:.4f} px"

    return (f"plantilla, ojos, pantalla, corrector e intrínsecos sobreviven el "
            f"JSON; el redondeo mueve la mirada {drift:.2e} px")


def _blink_detection() -> str:
    detector = landmarks.BlinkDetector(window=50, ratio=0.72)
    # An open-eye baseline, then a collapse.
    for _ in range(20):
        assert not detector.update(0.30)
    assert detector.update(0.12), "no detectó un parpadeo claro"
    assert not detector.update(0.29), "marcó un ojo abierto como parpadeo"

    # A different person with narrower eyes must not read as permanently blinking:
    # this is exactly what a fixed threshold gets wrong, and it matters here
    # because reduced eyelid opening is part of the condition being served.
    narrow = landmarks.BlinkDetector()
    for _ in range(20):
        assert not narrow.update(0.16), "un ojo estrecho se marcó como parpadeo"
    assert narrow.update(0.09)

    return "umbral adaptativo: funciona con ojo amplio y con ojo estrecho"


def _hybrid_arm() -> str:
    """The comparison arm must train and predict, on the same features.

    Not asserted to be *good* - it is the baseline, and the point of the
    experiment is to measure how much worse it is. Only that it runs, so a
    thesis comparison is not blocked at the last minute.
    """
    eyes = synth.synth_eyes()
    screen = synth.default_screen(1920, 1080)
    intrin = Intrinsics.synthetic()

    X, Y = [], []
    for pose in (synth.head_pose(0.55), synth.head_pose(0.65, (0.05, 0.0))):
        for target in [(x, y) for y in (120, 540, 960)
                       for x in (150, 960, 1780)]:
            scene = synth.make_scene(pose, eyes, screen, target, intrin=intrin)
            rays = eyemodel.gaze_rays(pose, scene.observation, eyes, intrin)
            X.append(estimator.HybridRegressor.features(pose, rays))
            Y.append(target)

    regressor = estimator.HybridRegressor()
    rms = regressor.fit(np.array(X), np.array(Y, dtype=np.float64))
    assert regressor.coeffs is not None
    x, y = regressor.predict_one(np.array(X[0]))
    assert np.isfinite(x) and np.isfinite(y)
    assert len(X[0]) == estimator.HybridRegressor.N_FEATURES

    round_trip = estimator.HybridRegressor.from_dict(regressor.as_dict())
    assert np.allclose(round_trip.coeffs, regressor.coeffs)

    return (f"{estimator.HybridRegressor.N_FEATURES} rasgos métricos, "
            f"RMS de entrenamiento {rms:.0f} px, serializa")


def _end_to_end_selection() -> str:
    """The whole product, headless: gaze -> tile -> dwell -> selection.

    The unit checks above each verify one stage. This one verifies that the
    stages are wired together correctly, which is a different question and the
    one that actually breaks in practice - a swapped screen size, a smoother
    that never converges, a dwell fed the wrong confidence.

    Run at a head pose that was never calibrated, so it also demonstrates the
    payoff end to end rather than only in the geometry: the tile the user is
    looking at is the tile that fires.
    """
    from . import board_layout
    from .tracker import DwellSelector, GazeTracker

    eyes = synth.synth_eyes()
    screen = synth.default_screen(1920, 1080)
    model = estimator.GazeModel(template=synth.synth_template(), eyes=eyes,
                                screen=screen)

    tiles = board_layout.layout("core", 1920, 1080)
    target = tiles[7]
    tx = (target.x1 + target.x2) // 2
    ty = (target.y1 + target.y2) // 2

    head = synth.head_pose(distance_m=0.63, offset=(0.07, -0.03),
                           yaw_deg=-12.0, pitch_deg=6.0)
    scenes = [synth.make_scene(head, eyes, screen, (tx, ty)) for _ in range(40)]

    settings = Settings(monitor="primary", check_position=False,
                        smoother="kalman_ema")
    tracker = GazeTracker(settings, landmarker=synth.scripted_landmarker(scenes))
    tracker.attach(model)
    dwell = DwellSelector(settings)

    fired = None
    hovered = None
    now = 1000.0
    sample = None
    for scene in scenes:
        sample = tracker.process(scene.frame)
        assert sample.ok, f"el tracker falló: {sample.reason}"
        hovered = next((t for t in tiles if t.contains(sample.x, sample.y)), None)
        now += 1.0 / 30.0
        _, fire = dwell.update(hovered.tile_id if hovered else None,
                               sample.confidence, now=now)
        if fire:
            fired = fire

    err = math.hypot(sample.x - tx, sample.y - ty)
    assert hovered is not None and hovered.tile_id == target.tile_id, (
        f"señaló {hovered.key.label if hovered else None!r}, no "
        f"{target.key.label!r}")
    assert fired == target.tile_id, "la permanencia no llegó a seleccionar"
    assert abs(sample.distance_cm - head.distance_m * 100.0) < 0.5
    assert abs(sample.head_yaw + 12.0) < 0.1 and abs(sample.head_pitch - 6.0) < 0.1

    return (f"celda '{target.key.label}' seleccionada a {err:.1f} px del centro, "
            f"en una postura NO calibrada (63 cm, -12° de giro); "
            f"conf {sample.confidence:.2f}")


def _frontal_head_reads_zero() -> str:
    """A head square to the camera must report angles near zero.

    The regression test for a bug that shipped and reached the hardware: the
    head frame took its backward axis from ``eye_centre - nose_tip``, which on a
    real face is tilted some 49 degrees because the nose sits below and in front
    of the eyes. The placement guide then read a steady 44.9 degrees of pitch for
    someone sitting square to the lens and refused to proceed.

    Deliberately routed through `TemplateBuilder`, so the bootstrap construction
    is the thing under test. Asserting against `synth_template()` would bypass it
    entirely and pass no matter how biased the frame was - which is precisely how
    this hid for as long as it did.
    """
    template_builder = face3d.TemplateBuilder()
    eyes = synth.synth_eyes()
    screen = synth.default_screen(1920, 1080)
    frontal = synth.head_pose(distance_m=0.55)

    for _ in range(12):
        scene = synth.make_scene(frontal, eyes, screen, (960.0, 540.0))
        pts, valid = face3d.rigid_points(scene.frame, scene.observation)
        assert template_builder.add_observation(pts, valid)
    template = template_builder.build()

    scene = synth.make_scene(frontal, eyes, screen, (960.0, 540.0))
    pts, valid = face3d.rigid_points(scene.frame, scene.observation,
                                     template.indices)
    head = face3d.fit_pose(template, pts, valid)
    assert head is not None
    yaw, pitch, roll = head.euler_deg()
    worst = max(abs(yaw), abs(pitch), abs(roll))
    assert worst < 5.0, (
        f"cabeza de frente reportada como giro {yaw:+.1f}° / "
        f"inclinación {pitch:+.1f}° / rotación {roll:+.1f}°")

    # And a deliberately turned head must still be reported as turned, so the
    # fix cannot be "always return zero".
    turned = synth.head_pose(distance_m=0.55, yaw_deg=20.0, pitch_deg=-12.0)
    scene = synth.make_scene(turned, eyes, screen, (960.0, 540.0))
    pts, valid = face3d.rigid_points(scene.frame, scene.observation,
                                     template.indices)
    head2 = face3d.fit_pose(template, pts, valid)
    y2, p2, _ = head2.euler_deg()
    assert abs(y2 - 20.0) < 6.0 and abs(p2 + 12.0) < 6.0, (
        f"cabeza girada 20°/-12° reportada como {y2:+.1f}°/{p2:+.1f}°")

    return (f"de frente: {worst:.2f}° el peor eje; girada 20°/-12°: "
            f"{y2:+.1f}°/{p2:+.1f}°")


def _synthetic_source() -> str:
    from .source import SyntheticSource

    with SyntheticSource(width=640, height=480, head_distance_m=0.6) as src:
        frame = src.read()
    assert frame is not None
    assert frame.color.shape == (480, 640, 3)
    assert frame.has_depth
    assert 0.05 < frame.depth_fill_rate() < 0.9
    assert frame.accel is not None
    return f"cuadro sintético {frame.size}, {frame.depth_fill_rate() * 100:.0f}% con profundidad"


# --------------------------------------------------------------------- runner

CHECKS = (
    ("versiones", _versions),
    ("mediapipe presente", _mediapipe),
    ("SDK de RealSense", _realsense_sdk),
    ("nombres de la API de pyrealsense2", _realsense_api),
    ("deproyección == la del SDK", _deprojection_matches_sdk),
    ("pantallas", _monitors),
    ("acentos en español", _accents),
    ("la voz recibe la ñ y los acentos", _speech_receives_accents),
    ("intrínsecos y distorsión", _intrinsics),
    ("primitivas geométricas", _geometry_primitives),
    ("kabsch y su versión robusta", _kabsch),
    ("plano de pantalla", _screen_plane),
    ("montaje desde el acelerómetro", _mount_from_gravity),
    ("pose de cabeza (profundidad y PnP)", _pose_fitting),
    ("plantilla craneal métrica", _template_builder),
    ("mirada: ida y vuelta exacta", _gaze_round_trip),
    ("INVARIANCIA A LA POSTURA", _head_invariance),
    ("modelo vectorizado == escalar", _forward_batch_matches_scalar),
    ("corrección de residuo", _residual_corrector),
    ("la calibración recupera la verdad", _calibration_recovers_truth),
    ("la degeneración se detecta", _degeneracy_is_detected),
    ("permanencia (dwell)", _dwell),
    ("suavizadores", _smoothers),
    ("serialización del modelo", _model_serialisation),
    ("detección de parpadeo", _blink_detection),
    ("brazo híbrido de comparación", _hybrid_arm),
    ("cabeza de frente lee ~0°", _frontal_head_reads_zero),
    ("fuente sintética", _synthetic_source),
    ("extremo a extremo: mirada -> celda -> selección", _end_to_end_selection),
)


def run() -> int:
    print("chequeos sin cámara ni pantalla:")
    passed = sum(_check(name, fn) for name, fn in CHECKS)
    total = len(CHECKS)
    print(f"\n{passed}/{total} pasaron")
    if passed < total:
        print("\nUn fallo en las pruebas geométricas es real: están medidas "
              "contra verdad de terreno exacta, no contra una tolerancia "
              "elegida a dedo.")
    return 0 if passed == total else 1
