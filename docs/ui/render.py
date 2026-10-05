"""Regenera las figuras de la interfaz, sin cámara ni pantalla.

    python docs/ui/render.py .

Las capturas de este directorio van a la tesis, así que tienen que poder
rehacerse: una figura que no se puede regenerar queda desactualizada en
silencio en cuanto alguien toca la paleta, y nadie se entera hasta la defensa.

Dibuja llamando a los mismos `_draw_board` / `_draw_candidates` que corren en
el dispositivo -no a una maqueta aparte-, así que lo que se ve acá es
exactamente lo que ve la persona.
"""
import sys
from pathlib import Path
raiz = Path(sys.argv[1])
for sib in ("RealsenseV1", "Eyetrax", "slm"):
    sys.path.insert(0, str(raiz / sib))

import cv2
from aac_realsense import board_layout, ui
from aac_semantic import demo, gaze_board

W, H = 1920, 1080
salida = raiz / "docs" / "ui"
salida.mkdir(parents=True, exist_ok=True)

# --- tablero, con una celda en permanencia -------------------------------
st = demo.State()
st.words = ["POR FAVOR", "AGUA"]
tiles = board_layout.layout("core", W, H)
objetivo = next(t for t in tiles if t.key.label == "BAÑO")
img = ui.canvas(W, H)
gaze_board._draw_board(img, st, tiles, objetivo.tile_id, 0.62, W, H)
ui.gaze_cursor(img, (objetivo.x1 + objetivo.x2)//2 + 18,
               (objetivo.y1 + objetivo.y2)//2 - 10, ui.RAW, 7)
ui.gaze_cursor(img, (objetivo.x1 + objetivo.x2)//2,
               (objetivo.y1 + objetivo.y2)//2)
cv2.imwrite(str(salida / "tablero.png"), img)

# --- pagina de letras -----------------------------------------------------
st2 = demo.State(); st2.words = ["QUIERO"]
tiles2 = board_layout.layout("abc", W, H)
obj2 = next(t for t in tiles2 if t.key.label == "M")
img2 = ui.canvas(W, H)
gaze_board._draw_board(img2, st2, tiles2, obj2.tile_id, 0.35, W, H)
ui.gaze_cursor(img2, (obj2.x1+obj2.x2)//2, (obj2.y1+obj2.y2)//2)
cv2.imwrite(str(salida / "letras.png"), img2)

# --- candidatos ----------------------------------------------------------
st3 = demo.State()
st3.words = ["POR FAVOR", "AGUA"]
st3.candidates = ["por favor, quiero agua", "por favor, necesito agua",
                  "por favor, tengo sed", "por favor agua"]
st3.latency_s = 0.31
zones = gaze_board._candidate_zones(st3, W, H)
img3 = ui.canvas(W, H)
gaze_board._draw_candidates(img3, st3, zones, W, H, 0.45,
                            f"say:{st3.candidates[1]}")
cv2.imwrite(str(salida / "candidatos.png"), img3)

for f in ("tablero.png", "letras.png", "candidatos.png"):
    print(f"  {salida / f}")
