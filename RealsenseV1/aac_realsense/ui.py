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

# Paleta, en BGR porque es el orden de OpenCV.
#
# Clínica y sobria, no de videojuego. El naranja ámbar que había antes
# (ACCENT = RGB 255,200,90) gritaba en una pantalla que alguien va a mirar
# fijo durante horas, y el ámbar sobre fondo oscuro es justo la combinación
# que más fatiga a la vista en uso prolongado.
#
# El reemplazo es un verde azulado de instrumental médico sobre pizarra
# profunda. Tres razones, no solo gusto:
#
# * **Fatiga.** El fondo no es negro puro sino pizarra: el contraste extremo
#   entre texto blanco y negro absoluto cansa más que un fondo apenas
#   levantado.
# * **Una sola familia de tono.** Fondo, superficie y celdas comparten el
#   mismo azul desaturado y solo cambian de luminosidad, así que la jerarquía
#   se lee por brillo y el color queda libre para significar algo.
# * **El color significa.** Verde azulado = el sistema te está siguiendo.
#   Verde salvia = se confirmó. Coral = algo anda mal. Nada más usa color, así
#   que cuando aparece, importa.
BG = (27, 20, 14)
"""Pizarra profunda. El lienzo de todo."""

SURFACE = (42, 32, 23)
"""Paneles sobre el fondo: la barra del mensaje, las tarjetas."""

TILE = (56, 43, 31)
TILE_ACTION = (47, 36, 26)
"""Las celdas de control van un escalón más abajo que las de vocabulario: son
herramientas, no son lo que la persona quiere decir."""

TILE_COMMIT = (62, 56, 34)
"""La celda que EMITE (HABLAR) no es una herramienta más.

Las demás acciones -borrar, espacio, cambiar de página- corrigen y navegan: se
hunden a propósito para que el ojo encuentre primero el vocabulario. HABLAR
hace lo contrario: termina el enunciado y lo dice en voz alta. Hundirla al
mismo nivel que BORRAR la volvía lo más difícil de encontrar siendo lo más
buscado, y una permanencia perdida buscándola es una frase que no se dijo."""

TILE_HOT = (84, 66, 45)
TILE_FIRED = (86, 92, 46)
EDGE = (88, 70, 52)
"""Borde sutil de cada celda. Las separa sin dibujar una grilla, que sería
ruido visual en una pantalla que ya tiene treinta elementos."""

TEXT = (246, 240, 233)
TEXT_DIM = (165, 149, 132)

ACCENT = (190, 182, 94)
"""Verde azulado clínico. El único color que marca «acá está pasando algo»."""

GAZE = (227, 200, 126)
RAW = (124, 110, 94)
WARN = (118, 123, 214)
GOOD = (147, 184, 122)

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


@lru_cache(maxsize=32)
def uniform_text_size(labels: tuple, box_w: int, box_h: int,
                      bold: bool = True, cap: int = 96) -> int:
    """Un solo tamaño de letra para todas las celdas de una página.

    Ajustar cada etiqueta a su celda por separado -que es lo que hacía antes-
    deja «SÍ» gigante al lado de «MEDICINA» diminuta. Se ve amateur, pero lo
    importante es que **rompe el barrido visual**: con tamaños desparejos el ojo
    salta a la palabra más grande en vez de recorrer la grilla, y en un tablero
    que se maneja con la mirada eso es un costo real, no estético.

    Se toma el tamaño más grande que entra en TODAS. Lo manda la etiqueta más
    larga, que es como debe ser: si no entra la peor, no entra la página.
    """
    return min(fit_text_size(l, box_w, box_h, bold=bold, cap=cap)
               for l in labels)


def canvas(w: int, h: int):
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[:] = BG
    return img


def _rounded_poly(x1, y1, x2, y2, radius: int):
    """El contorno del rectángulo redondeado, como polígono."""
    r = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    if r == 0:
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)
    # En sentido HORARIO y arrancando arriba a la izquierda. El orden importa:
    # `progress_border` recorre este mismo polígono, así que el orden de los
    # arcos decide por dónde empieza a llenarse el borde. Antes arrancaba en la
    # esquina inferior derecha y giraba al revés, de modo que una permanencia
    # recién empezada se veía como un trazo suelto en el borde de abajo -justo
    # donde nadie está mirando- mientras el docstring afirmaba lo contrario.
    pts = []
    for cx, cy, a0 in ((x1 + r, y1 + r, 180), (x2 - r, y1 + r, 270),
                       (x2 - r, y2 - r, 0), (x1 + r, y2 - r, 90)):
        pts.extend(cv2.ellipse2Poly((cx, cy), (r, r), 0, a0, a0 + 90, 6))
    return np.array(pts, np.int32)


def rounded_rect(img, x1, y1, x2, y2, color, radius: int = 14, thickness: int = -1):
    """Rectángulo redondeado, relleno o de contorno, con antialiasing real.

    La versión anterior componía el relleno con dos rectángulos y cuatro
    círculos. Es rápido y deja las esquinas dentadas, porque `cv2.rectangle`
    rellenado no admite antialiasing. En una pantalla con treinta celdas esos
    bordes escalonados son lo primero que delata que la interfaz está hecha a
    mano.

    Con `fillPoly` sobre el contorno y `LINE_AA` se pagan unos microsegundos
    más y las celdas dejan de verse como un prototipo.
    """
    poly = _rounded_poly(x1, y1, x2, y2, radius)
    if thickness < 0:
        cv2.fillPoly(img, [poly], color, cv2.LINE_AA)
    else:
        cv2.polylines(img, [poly], True, color, thickness, cv2.LINE_AA)


def tile(img, x1, y1, x2, y2, base, *, edge=None, radius: int = 16,
         progress: float = 0.0, accent=None):
    """Una celda del tablero: relleno, borde sutil y avance de permanencia.

    El avance se muestra de **dos formas a la vez**, y es deliberado: el borde
    que se completa dice cuánto falta, y el relleno que se aclara dice que el
    sistema te está mirando *a vos*. Con solo el borde, alguien que mira el
    centro de la celda -que es lo que uno hace- percibe el progreso por el
    rabillo del ojo; con solo el relleno no se sabe cuánto queda.
    """
    rounded_rect(img, x1, y1, x2, y2, base, radius)
    if progress > 0 and accent is not None:
        # El relleno se acerca al acento proporcionalmente, sin llegar nunca:
        # si la celda terminara del color del acento, el borde completo dejaría
        # de distinguirse del fondo de la celda justo en el instante que más
        # importa.
        k = 0.30 * min(1.0, progress)
        mezcla = tuple(int(b + (a - b) * k) for b, a in zip(base, accent))
        rounded_rect(img, x1, y1, x2, y2, mezcla, radius)
    if edge is not None:
        rounded_rect(img, x1, y1, x2, y2, edge, radius, thickness=1)
    if progress > 0 and accent is not None:
        progress_border(img, x1, y1, x2, y2, progress, accent, 5, radius)


def progress_border(img, x1, y1, x2, y2, progress: float, color,
                    thickness: int = 6, radius: int = 16):
    """Border filling clockwise from the top-left as progress -> 1.

    Reads better than a spinner on a rectangular tile: the user sees exactly how
    much longer to hold.
    """
    progress = max(0.0, min(1.0, progress))
    if progress <= 0:
        return

    # Se recorre el MISMO contorno redondeado que dibuja la celda, así el trazo
    # sigue las esquinas en vez de cortarlas. Con el rectángulo recto de antes,
    # el avance se despegaba del borde justo en las esquinas y parecía un
    # error de dibujo.
    # Metido hacia adentro media pincelada: un trazo centrado en el contorno
    # se derrama por fuera de la celda y se mete en el espacio de la vecina,
    # que es exactamente la celda que no hay que sugerir.
    adentro = thickness // 2
    poly = _rounded_poly(x1 + adentro, y1 + adentro,
                         x2 - adentro, y2 - adentro, radius)
    pts = np.vstack([poly, poly[:1]])
    tramos = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    objetivo = float(tramos.sum()) * progress

    acumulado = 0.0
    for i, largo in enumerate(tramos):
        if acumulado >= objetivo:
            break
        a, b = pts[i], pts[i + 1]
        if acumulado + largo <= objetivo:
            fin = b
        else:
            t = (objetivo - acumulado) / max(largo, 1e-6)
            fin = a + (b - a) * t
        cv2.line(img, tuple(a.astype(int)), tuple(fin.astype(int)),
                 color, thickness, cv2.LINE_AA)
        acumulado += largo


def gaze_cursor(img, x: int, y: int, color=GAZE, radius: int = 16):
    """El punto donde el sistema cree que está mirando.

    Va perfilado en oscuro porque cae encima de la etiqueta de la celda -el
    centro de la celda es justamente donde uno mira- y un aro claro sobre una
    letra clara se confunde con la letra. El perfil cuesta dos circunferencias
    más por cuadro y hace que el cursor se lea sobre cualquier fondo.
    """
    cv2.circle(img, (x, y), radius, BG, 5, cv2.LINE_AA)
    cv2.circle(img, (x, y), radius, color, 2, cv2.LINE_AA)
    cv2.circle(img, (x, y), 4, BG, -1, cv2.LINE_AA)
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
