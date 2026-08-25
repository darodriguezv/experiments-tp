# EyeTrax evaluation harness (AAC / eye tracking)

A test project for [`eyetrax`](https://pypi.org/project/eyetrax/) `0.4.0`, built to answer one
question for the AAC system: **is webcam-only gaze tracking accurate and stable enough to
drive a dwell-selected communication board?**

EyeTrax gives you MediaPipe face/iris landmarks → a regression model → a gaze point in screen
pixels. It does *not* give you dwell selection, a board, or accuracy metrics — those are the
parts this harness adds.

## Status of this checkout

Verified on this machine (Windows 11, 1920x1080, built-in webcam at 640x480):

| Check | Result |
| --- | --- |
| `python main.py smoke` | 18/18 passing |
| Face detection on live camera | 78/83 frames (5 blinks, 0 dropouts) |
| Per-frame latency (features + predict + smooth) | mean 26.9 ms, p95 35.1 ms → ~37 fps ceiling |
| Feature vector | 486 dims (161 landmarks x 3 + yaw/pitch/roll) |
| Fullscreen placement on either monitor | exact, verified against `getWindowImageRect` |
| Monitor picker (key + click + ESC cancel) | verified with injected keystrokes |
| Accented Spanish (`BAÑO`, `SÍ`, `MÁS`) | renders correctly via Pillow; asserted in smoke |
| Board render cost | 11.7 ms/frame at 1080p with the tile cache (was 36 ms) |
| Spanish TTS voice | Microsoft Sabina Desktop (es-MX) found |
| Head-pose measurement | roll tracks real tilt 1:1, no leakage into yaw/pitch (synthetic-face asserts) |
| Regression backends | ridge, elastic_net, linear_svr, svr, tiny_mlp all train/predict |
| Camera capture | webcam supports **1280x720**; eyetrax was running it at 640x480 |

Still to do by hand, because it needs a person in front of the screen:
`calibrate`, then `accuracy` and `board`.

## Inicio rápido (en español)

Todo el sistema en pantalla está en español: el tablero, la calibración, la guía de
colocación y los mensajes de consola.

```powershell
.\run.cmd pipeline
```

Ese único comando recorre todo el proceso: elegir pantalla → colocar la cara → calibrar →
medir la precisión → abrir el tablero CAA. Entre paso y paso aparece una pantalla que
explica qué viene; **ENTER** continúa y **ESC** sale sin dejar nada a medias.

Opciones útiles:

| Comando | Para qué |
| --- | --- |
| `.\run.cmd pipeline --monitor 0 --capture 1280x720` | fija pantalla y resolución, sin preguntar |
| `.\run.cmd pipeline --skip-calibration` | reutiliza la calibración guardada |
| `.\run.cmd pipeline --no-accuracy` | salta la medición y va directo al tablero |
| `.\run.cmd board --dwell 1000` | solo el tablero, con 1 s de permanencia |
| `.\run.cmd accuracy` | mide el error y dice qué conviene cambiar |

Para editar el vocabulario del tablero, modifique `PAGES` en
[board_layout.py](aac_eyetrax/board_layout.py) y nada más.

La voz usa la que ya trae Windows y prefiere una en español si existe — en esta máquina
encontró **Microsoft Sabina Desktop (es-MX)**. Si no hubiera ninguna, las frases se
imprimen en consola en lugar de hablarse.

> Todos los textos usan español neutro y trato de **usted**, sin voseo ni regionalismos.
> Los comentarios del código y las notas técnicas de más abajo están en inglés. Avíseme si
> desea que traduzca también el README completo.

## Guía de colocación de la cara

Un modelo de mirada solo vale para, más o menos, la posición de cabeza con la que se
calibró. Si calibrás sentado derecho y después te recostás, el error es **sistemático** y
no hay suavizado que lo arregle.

Por eso, antes de calibrar y antes de usar el tablero aparece una pantalla con la cámara
**en espejo** y un óvalo objetivo encima:

- Antes de calibrar solo se exige un rango razonable (centrado, distancia dentro de una
  banda amplia, cabeza derecha y de frente), y se registra la posición lograda.
- Después, el objetivo es **exactamente** la posición registrada durante la calibración,
  que queda guardada en `models\gaze_model.meta.json` junto al modelo.

La imagen va en espejo a propósito: moverse hacia el óvalo se vuelve intuitivo y nadie
tiene que razonar sobre izquierda y derecha. Los avisos ("ACÉRCATE A LA CÁMARA",
"ENDEREZA LA CABEZA") aparecen en orden de importancia, y cuando todo está bien hay que
sostener la posición un momento para continuar.

Se puede desactivar con `--no-position-check`, aunque no es recomendable salvo para
pruebas rápidas.

Las medidas se filtran con una mediana temporal de 7 cuadros y el estado alineado tiene
histéresis, para que los avisos no parpadeen cuadro a cuadro.

## Setup

**EyeTrax pins `numpy<2`, and numpy 1.x has no Python 3.13 wheels** — on 3.13 pip tries to
compile numpy from source and fails without MSVC. Use Python 3.10–3.12.

A ready environment is already in `.conda/` (Python 3.12.13). To recreate it:

```powershell
conda create -y -p .\.conda python=3.12 pip
.\.conda\python.exe -m pip install -r requirements.txt
```

First run downloads MediaPipe's `face_landmarker.task` (~3 MB) to
`~/.cache/eyetrax/mediapipe/`.

## Usage

`run.cmd` forwards to the env's interpreter, so you don't need to activate anything:

```powershell
.\run.cmd monitors     # list screens, pick the index you want
.\run.cmd smoke        # headless checks - no camera or display needed
.\run.cmd calibrate    # 9-point calibration -> models\gaze_model.pkl
.\run.cmd live         # gaze cursor overlay - sanity-check the calibration
.\run.cmd accuracy     # measure gaze error, get a minimum tile size
.\run.cmd board        # the actual AAC board: dwell to select, SPEAK to talk
```

Run `calibrate` first — everything else loads `models\gaze_model.pkl` and exits with a clear
error if it is missing. Recalibrate whenever you move, change seat distance, change lighting, or
**switch screens**.

**Stopping a run:** every fullscreen view exits on **ESC** (`live` and `board` also take **Q**),
and the key hint is drawn on screen. The window must have focus, so click it first if a keypress
seems ignored. Failing that, **Ctrl+C** in the launching terminal.

### If precision is poor

Start with `accuracy` — it separates the two failures that both feel like "imprecise",
because they have opposite fixes:

- **Jitter (precision)** — the point dances while you hold a fixation. Signal quality: too few
  pixels on the iris, dim light, sitting too far back.
- **Offset (accuracy)** — the point is steady but lands in the wrong place. Calibration or model:
  stale calibration, head moved since, or a linear model that cannot reach the screen edges.

It prints both, a per-target breakdown, and a ranked list of what to change. Then work down it.

**The first thing to fix on this machine:** eyetrax never sets a capture resolution, so it takes
the driver default of **640x480** — while this webcam supports **1280x720**. That is 4x the pixel
area on the iris, and iris pixels are precisely what gaze precision is made of. Now handled by
`--capture`, defaulting to 720p:

```powershell
.\run.cmd calibrate --capture 1280x720 --monitor 0   # recalibrate at the higher res
.\run.cmd accuracy  --capture 1280x720 --monitor 0
```

Calibrate and run at the **same** resolution. `--capture default` keeps whatever the driver gives.

After that, in rough order of payoff:

1. **Light on your face, not behind you.** A backlit face makes the iris landmarks noisy;
   `accuracy` reports mean frame brightness so you can see this.
2. **Sit closer.** Aim for your face filling most of the camera preview. Iris pixels again.
3. **More calibration data + a nonlinear model.** Ridge is linear in the 486 landmark features,
   so it typically tracks the centre and collapses at the edges — `accuracy` detects that pattern
   explicitly. Try `--method lissajous --model svr`, which pairs many samples with a model that
   can use them.
4. **Hold the posture you calibrated in.** A constant x/y bias almost always means your head sits
   differently now than during calibration.
5. **Stronger smoothing, if jitter is the problem** — `--smoother kde`, or
   `--smoother kalman_ema --ema-alpha 0.6`. This trades lag for steadiness; it hides jitter rather
   than fixing it, so do it last.

### Choosing a monitor

**By default every command asks.** A chooser appears on *all* screens at once — each showing its
own index, size and physical role — and you click the screen you want, or press its number. ESC
cancels. With one monitor connected it selects it and skips the prompt.

Picking by looking at the physical panel avoids the whole problem, because the index numbers are
not guessable (see below).

To skip the picker, name the screen explicitly — an index, `primary`, or `secondary`:

```powershell
.\run.cmd monitors                        # just list them
.\run.cmd calibrate --monitor secondary   # or --monitor 0
.\run.cmd board --monitor 0
```

After you pick, the run prints the flag that would have skipped the prompt, so you can paste it
into later commands.

This machine reports:

```
[0] 1920x1080 at (1920,301) secondary, 16"  \\.\DISPLAY1
[1] 1920x1080 at (0,0) primary, 24"  \\.\DISPLAY5
```

**screeninfo's ordering is not primary-first** and does not match the numbers in Windows Display
Settings — here index 0 is the *secondary* 16" laptop panel and index 1 is the 24" primary. This
is exactly why the picker is the default: `--monitor 1` is not "the second screen" in any sense a
person would expect.

Two things this works around in eyetrax:

- It sizes its canvas from `get_monitors()[0]`, which is not necessarily the primary monitor — so
  on this machine stock eyetrax sizes for DISPLAY1 while showing the window on DISPLAY5.
- It never positions its windows. OpenCV fullscreens onto whichever monitor the window is
  currently on, so [display.py](aac_eyetrax/display.py) moves and resizes the window to the
  target monitor *before* setting the fullscreen property. Verified exact for every combination
  of monitor and window, including the calibration window eyetrax manages itself.

Gaze coordinates stay monitor-local (0..width, 0..height), so nothing downstream needs an offset.

**A calibration is only valid for the screen it was made on.** `calibrate` writes a
`models\gaze_model.meta.json` sidecar recording the monitor and camera; loading that model on a
different screen prints a warning telling you to recalibrate, instead of silently giving you
skewed gaze. Monitors are matched by device name, so this fires even when both screens are the
same resolution — as they are here.

> **Put the webcam on the screen you are using.** Gaze accuracy depends on the camera sitting
> near the display you look at. On this laptop-plus-external setup the built-in webcam is on the
> laptop lid (the 16" screen, index 0), so running the board on the 24" external while tracking
> from the lid camera costs real accuracy. If you want the board on the external screen, mount a
> webcam on it and pass the matching `--camera` index.

### Commands in detail

**`calibrate`** — walks 9 targets; look at each green dot until it stops pulsing. ESC aborts
without saving. `--method 5point` is quicker and less accurate; `--method lissajous` sweeps a
continuous curve and collects far more samples.

**`live`** — fullscreen gaze cursor on a thirds grid. `R` overlays the unsmoothed prediction so
you can see how much jitter the smoother is absorbing. This is the fastest way to tell a bad
calibration from a good one.

**`accuracy`** — the diagnostic tool; see [If precision is poor](#if-precision-is-poor). Holds a grid of known targets, discards the first `--settle` seconds (saccade +
smoothing lag), then records error for `--hold` seconds. Prints mean/median/p95/max error and
per-axis bias, writes every sample to `logs\accuracy-*.csv`, and reports a **minimum safe tile
size** of 2x the p95 error. That number is the design input for the AAC grid: the current board
uses 5 columns on 1920 px = 384 px tiles, so a p95 error above ~190 px means the board needs
fewer, larger tiles.

**`board`** — the AAC board itself. Two pages (core vocabulary, A–Z), dwell to select, an amber
border fills around the tile as the dwell completes. Live keys: `[` / `]` shift dwell time by
100 ms, `S` speech, `C` camera preview, `X` clear, `ESC` quit. Every selection is logged to
`logs\aac-session-*.csv` with timestamp, gaze position and settings — that CSV is what you graph
for a selection-rate / error-rate write-up.

Speech uses the Windows SAPI voice via PowerShell, so there is no extra dependency; on other
platforms selections print to the console instead.

### Tuning knobs

| Flag | Trade-off |
| --- | --- |
| `--smoother none` | raw output — use it to see true jitter |
| `--smoother kalman` | eyetrax's Kalman filter; note it returns the *predicted* state, so it leads slightly |
| `--smoother kalman_ema` (default) | Kalman + EMA; `--ema-alpha 0` disables the EMA, values near 1 are very smooth but laggy |
| `--smoother kde` | picks the mode of a 0.5 s window — most stable for dwell, worst for fast movement |
| `--model ridge` (default) | linear, trains instantly, extrapolates predictably |
| `--model svr` / `tiny_mlp` | can fit non-linear head-pose effects, needs more calibration samples |
| `--dwell 800` | lower = faster typing, more mis-selections. 600–1000 ms is the usual AAC range |
| `--monitor ask` (default) | on-screen picker; or an index / `primary` / `secondary` to skip it |
| `--capture 1280x720` (default) | capture resolution — the biggest single precision lever |
| `--no-position-check` | skip the face-placement guide (not recommended) |

Dwell timing lives in `Settings` ([config.py](aac_eyetrax/config.py)) — `blink_grace_ms` is the
one worth knowing about: a blink makes EyeTrax report no face, so without a grace window every
natural blink would reset the dwell.

## Layout

| File | Role |
| --- | --- |
| [main.py](main.py) | CLI |
| [pipeline.py](aac_eyetrax/pipeline.py) | the full run: calibrate → measure → board, with step screens |
| [positioning.py](aac_eyetrax/positioning.py) | face-placement guide, tied to the calibration pose |
| [display.py](aac_eyetrax/display.py) | monitor enumeration, the picker UI, fullscreen window placement |
| [video.py](aac_eyetrax/video.py) | camera opening at an explicit resolution |
| [config.py](aac_eyetrax/config.py) | all tunables in one dataclass, plus the derived feature dim |
| [tracker.py](aac_eyetrax/tracker.py) | wraps `GazeEstimator`: smoothing, clamping, `DwellSelector` |
| [board_layout.py](aac_eyetrax/board_layout.py) | vocabulary pages + tile geometry (edit the board here) |
| [aac_board.py](aac_eyetrax/aac_board.py) | the dwell-select board app |
| [accuracy.py](aac_eyetrax/accuracy.py) | error measurement + CSV export |
| [live_gaze.py](aac_eyetrax/live_gaze.py) | cursor overlay |
| [smoke.py](aac_eyetrax/smoke.py) | headless checks |
| [ui.py](aac_eyetrax/ui.py) | drawing helpers; Pillow text so accents render, with sprite caching |

The AAC layer only talks to `tracker.py`, so swapping EyeTrax for another gaze library later
means rewriting one file.

## Notes for the project

- **~37 fps is the ceiling** on this machine and it is CPU-bound in MediaPipe, not in the UI.
  Fine for dwell selection; too slow for anything gesture-based.
- **Calibration is per-session.** EyeTrax normalises landmarks by head pose, which helps, but
  the saved model still assumes roughly the seat position you calibrated at. For a real AAC
  deployment plan on recalibration being a first-class, one-tap action.
- **Blinks read as "no face"**, not as a distinct signal. If you want blink-to-select as an
  alternative to dwell, `extract_features` already returns a blink flag — the EAR threshold is
  adaptive (80% of a rolling 50-frame mean).
- **Multi-monitor is not supported upstream.** eyetrax assumes a single screen; the workaround
  lives in `display.py` and the `_force_screen_size` patch in
  [tracker.py](aac_eyetrax/tracker.py). Worth knowing if you upgrade eyetrax - the patch rebinds
  `get_screen_size` inside each calibration module, so a refactor there could silently break it
  (the `monitors` smoke check and a visual pass on `live --monitor ...` will catch it).
- **Do not reuse eyetrax's head-pose angles for anything user-facing.** Its feature
  extractor builds a basis whose x axis *is* the eye line, then reads ZYX Euler angles off
  it. An in-plane tilt rotates that whole basis, so it vanishes from the extracted "roll",
  which comes out a constant **180 degrees for any head, level or tilted** — and the tilt
  reappears in the value labelled "yaw". Harmless as a model feature (a constant is just a
  constant), fatal for a guide: it made the placement screen say "straighten your head"
  permanently while never measuring real yaw at all.
  [tracker.py](aac_eyetrax/tracker.py) now measures roll, yaw and pitch directly in image
  space via `pose_from_landmarks`, which is a pure function of the landmarks and is
  asserted in the smoke test against synthetic faces at known angles.
- **Text must go through Pillow, not `cv2.putText`.** OpenCV's Hershey fonts have no
  glyphs for `ñ` or accented vowels, so a Spanish board rendered with `putText` shows
  "BANO" instead of "BAÑO". [ui.py](aac_eyetrax/ui.py) renders via Pillow and caches glyph
  runs as sprites; [aac_board.py](aac_eyetrax/aac_board.py) caches whole composed tiles,
  which is what keeps the board at ~12 ms/frame instead of ~36 ms. The smoke test asserts
  that `BAÑO` and `BANO` render differently, so a font regression cannot pass silently.
- **No gaze confidence.** EyeTrax's regression always returns a point, however far off. The
  clamping in `tracker.process` keeps it on-screen; there is no way to know it was a bad frame.
  Tile-dwell tolerates this reasonably well, small targets would not.
