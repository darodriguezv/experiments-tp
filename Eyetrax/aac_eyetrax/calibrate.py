"""Ejecuta una calibración de EyeTrax y guarda el modelo resultante."""

from __future__ import annotations

from pathlib import Path

from . import positioning
from .config import Settings, ensure_dirs
from .tracker import GazeTracker


def run(settings: Settings, method: str = "9point") -> Path | None:
    ensure_dirs()
    print(f"[calibración] método={method} modelo={settings.model_name} "
          f"cámara={settings.camera_index}")

    with GazeTracker(settings) as tracker:
        print(f"[calibración] pantalla: {tracker.screen_w}x{tracker.screen_h}")

        # La posición de la cabeza durante la calibración es la que el modelo
        # va a asumir después, así que se fija antes de empezar y se guarda.
        pose = None
        if settings.check_position:
            pose = positioning.run(
                settings, tracker,
                title="COLOQUE SU CARA EN LA GUÍA",
                subtitle="Esta será la posición de referencia de la calibración",
            )
            if pose is None:
                print("[calibración] cancelada en la colocación - no se guardó nada.")
                return None

        print("[calibración] Mire cada punto verde hasta que deje de latir. "
              "ESC cancela.")
        tracker.calibrate(method)

        if not tracker.trained:
            print("[calibración] cancelada o sin muestras útiles - no se guardó nada.")
            return None

        tracker.reference_pose = pose
        path = tracker.save()
        print(f"[calibración] guardado -> {path}")
        if pose is not None:
            print(f"[calibración] posición de referencia registrada "
                  f"(ojos={pose.eye_span:.3f}, centro=({pose.cx:.2f},{pose.cy:.2f}))")
        return path
