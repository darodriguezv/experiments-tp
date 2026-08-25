"""Drawing helpers.

Text goes through Pillow rather than `cv2.putText`, because the Hershey fonts
OpenCV ships cannot render accented characters - "BANO" instead of "BAÑO" is not
acceptable on a Spanish communication board. Rendered glyph runs are cached as
RGBA sprites and alpha-blended, so per-frame cost stays low even with 30+ tile
labels on screen.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# BGR (OpenCV order)
BG = (18, 18, 20)
TILE = (46, 46, 52)
TILE_HOT = (92, 78, 40)
TILE_FIRED = (60, 130, 70)
TEXT = (235, 235, 240)
TEXT_DIM = (150, 150, 158)
ACCENT = (90, 200, 255)
GAZE = (70, 90, 255)
RAW = (110, 110, 110)
WARN = (60, 60, 220)
GOOD = (90, 200, 120)

_FONT_DIR = Path("C:/Windows/Fonts")
_REGULAR = ("segoeui.ttf", "arial.ttf", "calibri.ttf", "tahoma.ttf", "verdana.ttf")
_BOLD = ("seguisb.ttf", "arialbd.ttf", "calibrib.ttf", "tahomabd.ttf")


@lru_cache(maxsize=64)
def _font(size: int, bold: bool) -> ImageFont.FreeTypeFont:
    for name in (_BOLD if bold else _REGULAR):
        path = _FONT_DIR / name
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                continue
    # Bitmap fallback: no scaling and limited glyphs, but never crashes.
    return ImageFont.load_default()


@lru_cache(maxsize=1024)
def _sprite(text: str, size: int, bold: bool, color: tuple[int, int, int]):
    """Render `text` to a tight (bgr, alpha) pair of numpy arrays.

    The colour tuple is written straight into the RGB channels, so passing a BGR
    colour yields a BGR sprite that drops into an OpenCV image unchanged.
    """
    font = _font(size, bold)
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    x0, y0, x1, y1 = probe.textbbox((0, 0), text, font=font)
    w, h = max(1, x1 - x0), max(1, y1 - y0)

    pad = max(2, size // 8)  # room for antialiasing and descenders
    img = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((pad - x0, pad - y0), text, font=font, fill=(*color, 255))

    arr = np.asarray(img)
    return np.ascontiguousarray(arr[:, :, :3]), np.ascontiguousarray(arr[:, :, 3])


def text_size(text: str, size: int, bold: bool = False) -> tuple[int, int]:
    _, alpha = _sprite(text, int(size), bold, (255, 255, 255))
    return alpha.shape[1], alpha.shape[0]


def _blit(img, bgr, alpha, x: int, y: int) -> None:
    """Alpha-blend a sprite at (x, y), clipping at the image edges."""
    ih, iw = img.shape[:2]
    sh, sw = alpha.shape

    sx0, sy0 = max(0, -x), max(0, -y)
    dx0, dy0 = max(0, x), max(0, y)
    dx1, dy1 = min(iw, x + sw), min(ih, y + sh)
    if dx1 <= dx0 or dy1 <= dy0:
        return

    sx1, sy1 = sx0 + (dx1 - dx0), sy0 + (dy1 - dy0)
    a = alpha[sy0:sy1, sx0:sx1, None].astype(np.float32) / 255.0
    dst = img[dy0:dy1, dx0:dx1].astype(np.float32)
    src = bgr[sy0:sy1, sx0:sx1].astype(np.float32)
    img[dy0:dy1, dx0:dx1] = (dst * (1.0 - a) + src * a).astype(np.uint8)


def text_at(img, text: str, x: int, y: int, size: int, color, bold: bool = False):
    """Draw `text` with its top-left corner at (x, y)."""
    if not text:
        return
    bgr, alpha = _sprite(text, int(size), bold, tuple(int(c) for c in color))
    _blit(img, bgr, alpha, int(x), int(y))


def text_center(img, text: str, cx: int, cy: int, size: int, color, bold: bool = False):
    """Draw `text` centred on (cx, cy)."""
    if not text:
        return
    bgr, alpha = _sprite(text, int(size), bold, tuple(int(c) for c in color))
    h, w = alpha.shape
    _blit(img, bgr, alpha, int(cx - w // 2), int(cy - h // 2))


def fit_text_size(
    text: str, max_w: int, max_h: int, bold: bool = False, cap: int = 140
) -> int:
    """Largest font size that keeps `text` inside the box."""
    lo, hi, best = 8, max(8, cap), 8
    while lo <= hi:
        mid = (lo + hi) // 2
        w, h = text_size(text, mid, bold)
        if w <= max_w and h <= max_h:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


# --------------------------------------------------------------------- shapes


def canvas(w: int, h: int):
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[:] = BG
    return img


def rounded_rect(img, x1, y1, x2, y2, color, radius: int = 14, thickness: int = -1):
    """Filled or outlined rectangle with rounded corners."""
    radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    if thickness < 0:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        for cx, cy in ((x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                       (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)):
            cv2.circle(img, (cx, cy), radius, color, -1)
    else:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)


def progress_border(img, x1, y1, x2, y2, progress: float, color, thickness: int = 6):
    """Border filling clockwise from the top-left as progress -> 1.

    Reads better than a spinner on a rectangular tile: the user sees exactly how
    much longer to hold.
    """
    progress = max(0.0, min(1.0, progress))
    if progress <= 0:
        return

    w, h = x2 - x1, y2 - y1
    remaining = 2 * (w + h) * progress
    segments = [
        ((x1, y1), (x2, y1), w),
        ((x2, y1), (x2, y2), h),
        ((x2, y2), (x1, y2), w),
        ((x1, y2), (x1, y1), h),
    ]
    for (sx, sy), (ex, ey), length in segments:
        if remaining <= 0:
            break
        frac = min(1.0, remaining / length)
        px = int(sx + (ex - sx) * frac)
        py = int(sy + (ey - sy) * frac)
        cv2.line(img, (sx, sy), (px, py), color, thickness, cv2.LINE_AA)
        remaining -= length


def gaze_cursor(img, x: int, y: int, color=GAZE, radius: int = 16):
    cv2.circle(img, (x, y), radius, color, 3, cv2.LINE_AA)
    cv2.circle(img, (x, y), 3, color, -1, cv2.LINE_AA)


def camera_inset(img, frame, size=(240, 180), margin: int = 16, anchor: str = "br"):
    """Blit a small camera preview into a corner.

    `anchor` is "br" (bottom-right) or "tr" (top-right) - the board pins it to
    the top so it cannot sit on top of a selectable tile.
    """
    h, w = img.shape[:2]
    x2 = w - margin
    x1 = x2 - size[0]
    if anchor == "tr":
        y1 = margin
        y2 = y1 + size[1]
    else:
        y2 = h - margin
        y1 = y2 - size[1]
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return
    img[y1:y2, x1:x2] = cv2.resize(frame, size)
    cv2.rectangle(img, (x1 - 1, y1 - 1), (x2, y2), TILE, 2)


def status_bar(img, lines: list[str], y: int = 20, color=TEXT_DIM, size: int = 20):
    for i, line in enumerate(lines):
        text_at(img, line, 20, y + i * (size + 6), size, color)


def banner(img, text: str, color=WARN):
    """Centred message for 'no face' style states."""
    h, w = img.shape[:2]
    size = fit_text_size(text, int(w * 0.8), 90, bold=True)
    text_center(img, text, w // 2, h // 2, size, color, bold=True)
