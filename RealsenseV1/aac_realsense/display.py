"""Monitor selection and window placement.

OpenCV fullscreens onto whichever monitor the window currently occupies, and it
offers no way to name a target screen, so a window has to be moved and resized
onto the intended monitor *before* the fullscreen property is set. This module
does that, and picks the monitor explicitly rather than assuming one.

Getting this right matters more here than for an ordinary app: the gaze model
holds a physical screen pose in camera coordinates, so a board that renders on
the wrong panel is not merely misplaced - every gaze ray is being intersected
with a plane that is not the one the user is looking at.

Note that screeninfo's ordering is *not* primary-first and does not match the
numbers Windows shows in Display Settings, so always select via `monitors`
(`python main.py monitors`) rather than guessing an index.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
from screeninfo import get_monitors


@dataclass(frozen=True)
class Screen:
    index: int
    x: int
    y: int
    width: int
    height: int
    is_primary: bool
    name: str
    width_mm: int | None = None
    height_mm: int | None = None

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def diagonal_in(self) -> float | None:
        if not self.width_mm or not self.height_mm:
            return None
        return math.hypot(self.width_mm, self.height_mm) / 25.4

    def label(self) -> str:
        role = "principal" if self.is_primary else "secundaria"
        diag = f", {self.diagonal_in:.0f}\"" if self.diagonal_in else ""
        return (
            f"[{self.index}] {self.width}x{self.height} en ({self.x},{self.y}) "
            f"{role}{diag}  {self.name}"
        )


def list_screens() -> list[Screen]:
    return [
        Screen(
            index=i,
            x=m.x,
            y=m.y,
            width=m.width,
            height=m.height,
            is_primary=bool(m.is_primary),
            name=str(m.name),
            width_mm=m.width_mm,
            height_mm=m.height_mm,
        )
        for i, m in enumerate(get_monitors())
    ]


def describe_screens() -> str:
    screens = list_screens()
    lines = [f"{len(screens)} pantalla(s) detectada(s):"]
    lines += [f"  {s.label()}" for s in screens]
    lines.append("")
    lines.append("Elija con --monitor <índice>, o --monitor primary / secondary.")
    return "\n".join(lines)


def resolve(spec: int | str) -> Screen:
    """Turn a --monitor value into a Screen.

    Accepts an index, "primary", or "secondary" (the first non-primary monitor).
    "ask" is resolved by `pick_screen` in the CLI layer, before Settings is
    built, so it must never reach here.
    """
    screens = list_screens()
    if not screens:
        raise RuntimeError("screeninfo reported no monitors")

    if isinstance(spec, str):
        key = spec.strip().lower()
        if key == "ask":
            raise ValueError(
                "monitor 'ask' must be resolved with pick_screen() before "
                "reaching Settings"
            )
        if key.lstrip("-").isdigit():
            spec = int(key)
        elif key == "primary":
            return next((s for s in screens if s.is_primary), screens[0])
        elif key == "secondary":
            other = [s for s in screens if not s.is_primary]
            if not other:
                raise ValueError(
                    "se pidió --monitor secondary pero solo hay una pantalla "
                    f"conectada.\n\n{describe_screens()}"
                )
            return other[0]
        else:
            raise ValueError(
                f"--monitor debe ser un índice, 'primary' o 'secondary' "
                f"(se recibió {spec!r})"
            )

    if not -len(screens) <= spec < len(screens):
        raise ValueError(
            f"--monitor {spec} fuera de rango.\n\n{describe_screens()}"
        )
    return screens[spec]


def open_fullscreen(window: str, screen: Screen) -> None:
    """Create `window` fullscreen on `screen`.

    Order matters: OpenCV fullscreens onto whichever monitor the window is
    currently on, so the window has to be moved there *first*. The waitKey
    calls give the window manager a chance to apply each step.
    """
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window, screen.x, screen.y)
    cv2.resizeWindow(window, screen.width, screen.height)
    cv2.waitKey(1)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.waitKey(1)


def _draw_picker(screen: Screen, prompt: str):
    from . import ui

    w, h = screen.size
    img = ui.canvas(w, h)

    # A thick frame makes it obvious which physical panel you are looking at.
    cv2.rectangle(img, (14, 14), (w - 15, h - 15), ui.ACCENT, 6)

    ui.text_center(img, prompt, w // 2, int(h * 0.20), 36, ui.TEXT_DIM)
    ui.text_center(img, str(screen.index), w // 2, int(h * 0.46), 260, ui.ACCENT, bold=True)
    ui.text_center(
        img, f"HAGA CLIC AQUÍ  o presione  {screen.index}",
        w // 2, int(h * 0.68), 44, ui.TEXT, bold=True,
    )

    role = "PRINCIPAL" if screen.is_primary else "SECUNDARIA"
    diag = f"  ~{screen.diagonal_in:.0f} pulgadas" if screen.diagonal_in else ""
    ui.text_center(
        img, f"{w} x {h}   {role}{diag}", w // 2, int(h * 0.79), 28, ui.TEXT_DIM
    )
    ui.text_center(img, screen.name, w // 2, int(h * 0.85), 22, ui.TEXT_DIM)
    ui.text_center(img, "ESC para cancelar", w // 2, int(h * 0.93), 22, ui.WARN)
    return img


def pick_screen(prompt: str = "¿En qué pantalla desea usar el sistema?") -> Screen | None:
    """Show a chooser on every monitor and return the one the user picks.

    Puts a labelled fullscreen window on each screen, so the choice is made by
    looking at the physical panels rather than by guessing an index. Returns
    None if the user cancels. With a single monitor it returns immediately.
    """
    screens = list_screens()
    if not screens:
        raise RuntimeError("screeninfo reported no monitors")
    if len(screens) == 1:
        return screens[0]

    clicked: dict[str, int | None] = {"index": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked["index"] = param

    windows = []
    for screen in screens:
        win = f"Pantalla {screen.index}"
        open_fullscreen(win, screen)
        cv2.imshow(win, _draw_picker(screen, prompt))
        cv2.setMouseCallback(win, on_mouse, screen.index)
        windows.append(win)

    print(f"[pantalla] haga clic en la pantalla que desea usar, "
          f"o presione 0-{len(screens) - 1}")

    chosen: Screen | None = None
    try:
        while True:
            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # ESC
                break
            if 0 <= key - ord("0") < len(screens):
                chosen = screens[key - ord("0")]
                break
            if clicked["index"] is not None:
                chosen = screens[clicked["index"]]
                break
            # If the user closed a picker window, bail rather than spin forever.
            if any(
                cv2.getWindowProperty(w, cv2.WND_PROP_VISIBLE) < 1 for w in windows
            ):
                break
    finally:
        for win in windows:
            try:
                cv2.destroyWindow(win)
            except cv2.error:
                pass
        cv2.waitKey(1)

    return chosen


def placement_ok(window: str, screen: Screen, tolerance: int = 4) -> bool:
    """True if `window` actually landed on `screen` (best-effort check)."""
    try:
        x, y, w, h = cv2.getWindowImageRect(window)
    except cv2.error:
        return False
    return (
        abs(x - screen.x) <= tolerance
        and abs(y - screen.y) <= tolerance
        and abs(w - screen.width) <= tolerance
        and abs(h - screen.height) <= tolerance
    )
