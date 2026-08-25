"""Finding a RealSense, and reporting honestly on what it can do.

Detection is deliberately more than "did `query_devices` return something".
Every failure below has the same symptom - gaze that is subtly worse than it
should be - and none of them raise an exception:

* **USB 2.1 instead of 3.2.** The cable or port silently downgrades and
  librealsense then refuses the high-bandwidth profiles, so you get 640x480
  colour and blame the algorithm. This is by far the most common one.
* **A D435 rather than a D435i.** No IMU, so the camera cannot measure its own
  tilt and the screen plane falls back to hand-measured angles.
* **Stale firmware.** Depth quality and auto-exposure behaviour both changed
  materially across D400 firmware releases.
* **A different D400 model.** A D415 has a narrower field of view and a longer
  stereo baseline: better depth, but it loses the face sooner when the user
  shifts in the chair.

So `probe` returns a full description and the CLI prints it, rather than a
boolean. The import of pyrealsense2 is deferred throughout, so the geometry and
model code stays testable on a machine with no SDK installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Product lines and models this code has been written against. Anything else
# still runs - the streams are requested by resolution, not by model - but the
# report says so, because untested is not the same as broken.
KNOWN_D400 = ("D405", "D415", "D435", "D435i", "D435f", "D455", "D456", "D457")
MIN_FIRMWARE = (5, 13, 0, 50)


class RealSenseUnavailable(RuntimeError):
    """pyrealsense2 is missing, or no camera is attached."""


def import_rs():
    """Import pyrealsense2 with an error message that says what to do.

    The wheel name does not match the import name and the Python version
    constraint is real, so the default ImportError sends people the wrong way.
    """
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RealSenseUnavailable(
            "no se pudo importar pyrealsense2.\n"
            "  pip install pyrealsense2\n"
            "  Hay ruedas oficiales para Windows en CPython 3.9-3.14; si su "
            "intérprete no está en esa lista, pip intentará compilar desde "
            "fuente y va a fallar."
        ) from exc
    return rs


def available() -> bool:
    """True if the SDK imports. Says nothing about a camera being plugged in."""
    try:
        import_rs()
    except RealSenseUnavailable:
        return False
    return True


# ------------------------------------------------------------------- reporting


@dataclass
class StreamMode:
    stream: str
    width: int
    height: int
    fps: int
    fmt: str

    def __str__(self) -> str:
        return f"{self.stream} {self.width}x{self.height}@{self.fps} {self.fmt}"


@dataclass
class DeviceInfo:
    """What a connected camera is and what it can be trusted to do."""

    name: str
    serial: str
    firmware: str
    product_line: str
    usb_descriptor: str

    has_imu: bool
    has_color: bool
    has_depth: bool
    depth_scale: float

    color_modes: list[StreamMode] = field(default_factory=list)
    depth_modes: list[StreamMode] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def model(self) -> str:
        """Short model name, e.g. "D435i", pulled out of the full product name."""
        for known in sorted(KNOWN_D400, key=len, reverse=True):
            if known.lower() in self.name.lower():
                return known
        return self.name

    @property
    def usb3(self) -> bool:
        return self.usb_descriptor.startswith("3")

    def supports(self, stream: str, width: int, height: int, fps: int) -> bool:
        modes = self.color_modes if stream == "color" else self.depth_modes
        return any(m.width == width and m.height == height and m.fps == fps
                   for m in modes)

    def best_color(self, fps: int = 30) -> tuple[int, int] | None:
        """Highest-resolution colour mode at `fps`.

        Iris pixels set the angular precision ceiling, so when in doubt this is
        the mode to take.
        """
        cands = [m for m in self.color_modes if m.fps == fps]
        if not cands:
            return None
        best = max(cands, key=lambda m: m.width * m.height)
        return best.width, best.height

    def describe(self) -> str:
        lines = [
            f"  modelo        {self.model}  ({self.name})",
            f"  serie         {self.serial}",
            f"  firmware      {self.firmware}",
            f"  línea         {self.product_line}",
            f"  USB           {self.usb_descriptor}"
            f"{'' if self.usb3 else '   <-- USB 2: ancho de banda limitado'}",
            f"  color         {'sí' if self.has_color else 'NO'}"
            f"   ({len(self.color_modes)} modos)",
            f"  profundidad   {'sí' if self.has_depth else 'NO'}"
            f"   ({len(self.depth_modes)} modos, escala {self.depth_scale:.6f} m/unidad)",
            f"  IMU           {'sí' if self.has_imu else 'no (modelo sin -i)'}",
        ]
        best = self.best_color()
        if best:
            lines.append(f"  mejor color   {best[0]}x{best[1]} @30")
        if self.warnings:
            lines.append("  avisos:")
            lines += [f"    - {w}" for w in self.warnings]
        return "\n".join(lines)


def _modes(sensor, rs, wanted_stream) -> list[StreamMode]:
    out: list[StreamMode] = []
    for profile in sensor.get_stream_profiles():
        if profile.stream_type() != wanted_stream:
            continue
        try:
            vsp = profile.as_video_stream_profile()
        except Exception:
            continue
        out.append(StreamMode(
            stream=str(profile.stream_type()).split(".")[-1],
            width=vsp.width(), height=vsp.height(), fps=vsp.fps(),
            fmt=str(profile.format()).split(".")[-1],
        ))
    # Deduplicate: the same geometry appears once per pixel format.
    seen, unique = set(), []
    for m in sorted(out, key=lambda m: (-m.width * m.height, -m.fps)):
        key = (m.width, m.height, m.fps)
        if key not in seen:
            seen.add(key)
            unique.append(m)
    return unique


def _firmware_is_old(fw: str) -> bool:
    try:
        parts = tuple(int(p) for p in fw.split("."))
    except ValueError:
        return False
    return parts < MIN_FIRMWARE[: len(parts)]


def _info(dev, rs, key, default: str = "?") -> str:
    """Read a camera_info field that older firmware may not expose."""
    try:
        if not dev.supports(key):
            return default
        return str(dev.get_info(key))
    except Exception:
        return default


def describe_device(dev, rs) -> DeviceInfo:
    """Build a DeviceInfo from a live pyrealsense2 device handle."""
    name = _info(dev, rs, rs.camera_info.name, "cámara desconocida")
    info = DeviceInfo(
        name=name,
        serial=_info(dev, rs, rs.camera_info.serial_number),
        firmware=_info(dev, rs, rs.camera_info.firmware_version),
        product_line=_info(dev, rs, rs.camera_info.product_line),
        usb_descriptor=_info(dev, rs, rs.camera_info.usb_type_descriptor, "?"),
        has_imu=False, has_color=False, has_depth=False, depth_scale=0.0,
    )

    for sensor in dev.query_sensors():
        sname = _info(sensor, rs, rs.camera_info.name, "")
        profiles = sensor.get_stream_profiles()
        streams = {p.stream_type() for p in profiles}

        if rs.stream.color in streams:
            info.has_color = True
            info.color_modes = _modes(sensor, rs, rs.stream.color)
        if rs.stream.depth in streams:
            info.has_depth = True
            info.depth_modes = _modes(sensor, rs, rs.stream.depth)
            # `get_depth_scale` lives on rs.depth_sensor, not on the base
            # rs.sensor that query_sensors hands back, so the cast is required.
            # Without it the call raises, the except swallows it, and the report
            # claims a scale of 0.000000 - which is how this was found.
            try:
                info.depth_scale = float(
                    sensor.as_depth_sensor().get_depth_scale())
            except Exception as exc:
                info.warnings.append(
                    f"no se pudo leer la escala de profundidad ({exc}). Se "
                    f"asumirá 0.001 m/unidad, que es el valor por defecto de "
                    f"la serie D400."
                )
        if rs.stream.accel in streams or "motion" in sname.lower():
            info.has_imu = True

    # ------------------------------------------------- turn facts into warnings
    if not info.usb3:
        info.warnings.append(
            f"conectada por USB {info.usb_descriptor}. Los modos de alta "
            "resolución no van a estar disponibles y la precisión de la mirada "
            "depende de los píxeles sobre el iris. Use un cable USB 3 y un "
            "puerto azul/SS."
        )
    if not info.has_depth:
        info.warnings.append(
            "sin flujo de profundidad: sin escala métrica no hay modelo "
            "geométrico, que es todo el motivo para usar esta cámara."
        )
    if not info.has_imu:
        info.warnings.append(
            "sin IMU (es una D435, no una D435i). La inclinación de la cámara "
            "habrá que medirla a mano en CameraMount.pitch_deg en vez de "
            "leerla del acelerómetro."
        )
    if _firmware_is_old(info.firmware):
        info.warnings.append(
            f"firmware {info.firmware} es anterior a "
            f"{'.'.join(str(p) for p in MIN_FIRMWARE)}. Actualícelo con "
            "RealSense Viewer: la calidad de profundidad y el autoexposición "
            "cambiaron bastante entre versiones."
        )
    if info.model not in KNOWN_D400:
        info.warnings.append(
            f"modelo {info.model!r} no probado con este código. Debería "
            "funcionar - los flujos se piden por resolución, no por modelo - "
            "pero revise los números de precisión antes de confiar en ellos."
        )
    if info.model == "D415":
        info.warnings.append(
            "la D415 tiene un campo de visión más angosto que la D435: mejor "
            "profundidad, pero pierde la cara antes si la persona se mueve en "
            "la silla. Verifique el encuadre con `main.py preview`."
        )
    if info.depth_scale and abs(info.depth_scale - 0.001) > 1e-6:
        info.warnings.append(
            f"la escala de profundidad es {info.depth_scale:.6f} m/unidad, no "
            "los 0.001 habituales. Se usa el valor real de la cámara, pero "
            "vale saberlo al comparar mediciones."
        )
    return info


def enumerate_devices() -> list[DeviceInfo]:
    """Every attached RealSense, described. Empty list if none."""
    rs = import_rs()
    ctx = rs.context()
    return [describe_device(dev, rs) for dev in ctx.query_devices()]


def find(serial: str | None = None) -> DeviceInfo:
    """Pick a camera, defaulting to the only one attached.

    Prefers a model with an IMU when several are connected and no serial was
    given, since the IMU removes a hand-measured angle from the geometry.
    """
    devices = enumerate_devices()
    if not devices:
        raise RealSenseUnavailable(
            "no se detectó ninguna cámara RealSense.\n"
            "  - revise que el cable sea USB 3 (los cables de solo carga no "
            "llevan datos)\n"
            "  - pruebe otro puerto, preferentemente uno azul/SS directo a la "
            "placa, no un hub\n"
            "  - abra RealSense Viewer: si tampoco la ve ahí, el problema es "
            "de driver o de hardware, no de este código"
        )

    if serial:
        for d in devices:
            if d.serial == serial:
                return d
        found = ", ".join(d.serial for d in devices)
        raise RealSenseUnavailable(
            f"no hay ninguna cámara con serie {serial!r}. Conectadas: {found}"
        )

    with_imu = [d for d in devices if d.has_imu]
    return (with_imu or devices)[0]


def report() -> str:
    """Human-readable inventory - what `main.py devices` prints."""
    if not available():
        return (
            "pyrealsense2 no está instalado.\n"
            "  pip install -r requirements.txt"
        )
    try:
        devices = enumerate_devices()
    except RealSenseUnavailable as exc:
        return str(exc)

    if not devices:
        # Reuse find()'s diagnosis so the troubleshooting steps live in one place.
        try:
            find()
        except RealSenseUnavailable as exc:
            return str(exc)

    lines = [f"{len(devices)} cámara(s) RealSense detectada(s):", ""]
    for d in devices:
        lines.append(d.describe())
        lines.append("")
    lines.append("Elija una con --serial <número de serie>.")
    return "\n".join(lines)
