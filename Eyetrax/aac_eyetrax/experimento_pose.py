"""¿El estimador consciente de pose aguanta mejor que la persona se mueva?

Es la pregunta que quedó abierta cuando se construyó `headaware.py` y nunca se
midió. Lo que **sí** está probado es el mecanismo: los 486 rasgos de eyetrax son
ciegos a dónde está la cabeza. Medido con landmarks sintéticos, trasladarla un
5% del ancho de la imagen los cambia en 2e-07 -ruido de punto flotante- y
alejarla un 20% en 6e-07. Los 6 términos nuevos cambian 0.31 y 1.56.

Que los rasgos vean el movimiento no implica que el error baje. Eso es lo que
mide esto.

## Por qué no alcanza con correr `accuracy` dos veces

Dos corridas dan dos calibraciones distintas, en dos momentos, con dos
posturas. La diferencia entre ellas mezcla el estimador con todo lo demás, y
las sesiones guardadas de este proyecto muestran que "todo lo demás" vale
mucho: el sesgo medido varía de 79 px a 828 px entre sesiones del mismo
estimador. Un efecto de 50 px quedaría enterrado.

## El diseño: la degradación es lo comparable

Para cada estimador, en una sola apertura:

1. calibrar;
2. medir en la postura de la calibración;
3. pedirle a la persona que **se corra unos centímetros**;
4. medir de nuevo, sin recalibrar.

Lo que se compara entre estimadores no es el error absoluto -cada uno tiene su
propia calibración y eso no es comparable- sino **cuánto se degrada de 2 a 4**.
La hipótesis dice que el de pose se degrada menos. Medir la degradación dentro
de cada estimador controla la iluminación, la cámara, el ánimo y el día.

## Lo que este experimento NO puede decidir

Con una persona y dos posturas, un resultado favorable es indicio y no prueba.
Y la persona sabe qué estimador está corriendo, así que puede moverse distinto
sin darse cuenta. Por eso el orden de los estimadores se alterna y la consigna
de movimiento es la misma frase textual en los dos casos.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from . import accuracy, calibrate, ui
from .config import LOGS_DIR, Settings
from .tracker import GazeTracker

VENTANA = "CAA - experimento de pose"

CONSIGNA = (
    "Córrase unos 10 cm hacia un costado, o acérquese o aléjese un poco.",
    "Quédese cómodo en la posición nueva. No vuelva a la anterior.",
)
"""Textual e idéntica para los dos estimadores.

Si a uno se le pidiera "muévase un poco" y al otro "muévase bastante", la
comparación mediría la consigna. Y como la persona sabe cuál está corriendo,
alcanzar con que la frase sea la misma es lo mínimo defendible."""


def _pedir_movimiento(settings: Settings, segundos: float = 8.0) -> bool:
    """Pantalla de transición entre la postura A y la B. False si cancela."""
    w, h = settings.screen_w or 1920, settings.screen_h or 1080
    t0 = time.time()
    cv2.namedWindow(VENTANA, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(VENTANA, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)
    try:
        while True:
            queda = segundos - (time.time() - t0)
            img = ui.canvas(w, h)
            ui.text_center(img, "AHORA MUÉVASE", w // 2, int(h * 0.30), 64,
                           ui.ACCENT, bold=True)
            for i, linea in enumerate(CONSIGNA):
                ui.text_center(img, linea, w // 2,
                               int(h * 0.44) + i * 46, 30, ui.TEXT)
            ui.text_center(img, "NO se va a recalibrar: esa es la prueba.",
                           w // 2, int(h * 0.60), 26, ui.TEXT_DIM)
            if queda > 0:
                ui.text_center(img, f"{queda:.0f}", w // 2, int(h * 0.74), 70,
                               ui.TEXT_DIM, bold=True)
            else:
                ui.text_center(img, "Listo: apriete ESPACIO para seguir",
                               w // 2, int(h * 0.74), 34, ui.GOOD, bold=True)
            cv2.imshow(VENTANA, img)
            k = cv2.waitKey(30) & 0xFF
            if k == 27:
                return False
            if k == 32 and queda <= 0:
                return True
    finally:
        cv2.destroyWindow(VENTANA)
        cv2.waitKey(1)


def _pose_actual(tracker) -> dict:
    """Dónde está la cabeza ahora, para dejar constancia de que se movió.

    Sin esto, un resultado nulo es imposible de interpretar: no se sabría si el
    estimador aguantó el movimiento o si la persona no se movió.
    """
    pose = getattr(tracker, "last_pose", None)
    if pose is None:
        return {}
    d = pose.to_dict() if hasattr(pose, "to_dict") else dict(pose)
    return {k: float(v) for k, v in d.items() if isinstance(v, (int, float))}


def _medir(settings: Settings, tracker_factory, etiqueta: str,
           rows: int, cols: int, hold: float) -> dict | None:
    print(f"\n[pose] midiendo postura {etiqueta.upper()} ...")
    # check_position apagado a propósito en la postura B: la guía pide volver a
    # la posición de la calibración, que es exactamente lo que acá NO queremos.
    previo = settings.check_position
    settings.check_position = (etiqueta == "a")
    try:
        return accuracy.run(settings, rows=rows, cols=cols, hold_s=hold,
                            make_tracker=tracker_factory)
    finally:
        settings.check_position = previo


def correr_un_estimador(settings: Settings, nombre: str, tracker_factory,
                        rows: int = 3, cols: int = 3,
                        hold: float = 2.0) -> dict | None:
    """Calibra una vez y mide en dos posturas. Devuelve el par de resultados."""
    print(f"\n{'=' * 64}")
    print(f"  {nombre}")
    print(f"{'=' * 64}")
    print("[pose] calibración (obligatoria, como siempre)")
    if calibrate.run(settings, "9point", tracker_factory) is None:
        print("[pose] cancelado en la calibración.")
        return None

    a = _medir(settings, tracker_factory, "a", rows, cols, hold)
    if a is None:
        return None
    if not _pedir_movimiento(settings):
        print("[pose] cancelado antes de la segunda postura.")
        return None
    b = _medir(settings, tracker_factory, "b", rows, cols, hold)
    if b is None:
        return None

    return {"estimador": nombre, "a": a, "b": b,
            "degradacion_px": b["mean_px"] - a["mean_px"],
            "degradacion_rel": (b["mean_px"] / a["mean_px"] - 1.0)
                               if a["mean_px"] else float("nan")}


def run(settings: Settings, head_aware_factory, *, rows: int = 3,
        cols: int = 3, hold: float = 2.0, primero: str = "base") -> int:
    """El experimento completo: los dos estimadores, dos posturas cada uno."""
    orden = [("eyetrax (base)", None),
             ("eyetrax + pose", head_aware_factory)]
    if primero == "pose":
        orden.reverse()
    print("[pose] Son DOS calibraciones y CUATRO mediciones: unos 10 minutos.")
    print(f"[pose] Orden: {orden[0][0]} y después {orden[1][0]}.")

    resultados = []
    for nombre, factory in orden:
        r = correr_un_estimador(settings, nombre, factory, rows, cols, hold)
        if r is None:
            print("[pose] experimento incompleto; no se guarda nada.")
            return 130
        resultados.append(r)

    destino = guardar(resultados)
    print()
    print(informe(resultados))
    print(f"  guardado -> {destino}")
    return 0


def guardar(resultados: list[dict]) -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    p = LOGS_DIR / f"pose-{time.strftime('%Y%m%d-%H%M%S')}.json"
    p.write_text(json.dumps(resultados, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    return p


def cargar(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def informe(resultados: list[dict]) -> str:
    out = ["Degradación al moverse (lo comparable entre estimadores)", ""]
    out.append(f"  {'estimador':<22}{'postura A':>11}{'postura B':>11}"
               f"{'degrada':>10}{'relativo':>10}")
    out.append("  " + "-" * 64)
    for r in resultados:
        out.append(f"  {r['estimador']:<22}{r['a']['mean_px']:>9.0f} px"
                   f"{r['b']['mean_px']:>9.0f} px"
                   f"{r['degradacion_px']:>+8.0f} px"
                   f"{r['degradacion_rel']:>+9.0%}")
    out.append("")

    if len(resultados) == 2:
        base = next((r for r in resultados if "base" in r["estimador"]), None)
        pose = next((r for r in resultados if "pose" in r["estimador"]), None)
        if base and pose:
            out += _veredicto(base, pose)
    out.append("")
    out.append("  El error ABSOLUTO entre estimadores no es comparable: cada "
               "uno tiene su")
    out.append("  propia calibración, hecha en otro momento. Lo comparable es "
               "cuánto se")
    out.append("  degrada cada uno de A a B.")
    return "\n".join(out) + "\n"


def _veredicto(base: dict, pose: dict) -> list[str]:
    db, dp = base["degradacion_px"], pose["degradacion_px"]
    out = []

    # Una degradación chica en los DOS significa que la persona no se movió lo
    # suficiente, no que los dos aguanten. Es el modo de fallar más probable y
    # el que daría un falso negativo.
    if max(abs(db), abs(dp)) < 25:
        out.append("  SIN CONCLUSIÓN: ninguno de los dos se degradó "
                   "apreciablemente.")
        out.append("  Lo más probable es que el movimiento haya sido muy "
                   "chico. Repetir el")
        out.append("  experimento corriéndose más -10 cm de verdad- antes de "
                   "sacar conclusiones.")
        return out

    mejora = db - dp
    if mejora > 0.25 * abs(db):
        out.append(f"  A FAVOR del estimador con pose: se degrada "
                   f"{mejora:.0f} px menos que el base")
        out.append(f"  ({dp:+.0f} px contra {db:+.0f} px).")
        out.append("")
        out.append("  Con UNA persona y DOS posturas esto es un indicio, no "
                   "una prueba. Para")
        out.append("  afirmarlo hacen falta varias repeticiones alternando el "
                   "orden.")
    elif mejora < -0.25 * abs(db):
        out.append(f"  EN CONTRA: el estimador con pose se degrada "
                   f"{-mejora:.0f} px MÁS que el base.")
        out.append("  Los 6 términos de cabeza agregan capacidad que con 19 "
                   "puntos de")
        out.append("  calibración puede estar sobreajustando. La hipótesis "
                   "queda refutada")
        out.append("  en esta configuración.")
    else:
        out.append(f"  EMPATE: {dp:+.0f} px contra {db:+.0f} px, una "
                   f"diferencia de {abs(mejora):.0f} px.")
        out.append("  Los rasgos de pose SÍ ven el movimiento -eso está "
                   "probado aparte- pero")
        out.append("  verlo no se traduce en menos error acá. Es un resultado, "
                   "no una falta de")
        out.append("  resultado: dice que el cuello de botella está en otro "
                   "lado.")
    return out
