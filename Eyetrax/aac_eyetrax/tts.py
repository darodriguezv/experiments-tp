"""Salida de voz, sin bloquear la interfaz.

Usa la voz SAPI que ya trae Windows, así el proyecto no necesita dependencias
extra. Prefiere una voz en español si hay alguna instalada (por ejemplo
"Microsoft Sabina Desktop", es-MX); si no, usa la predeterminada. En otros
sistemas no habla y solo imprime en consola.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import threading
from functools import lru_cache

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# Selecciona la primera voz cuya cultura empiece con "es"; si no hay, deja la
# predeterminada. Se lee el texto por stdin para no pelearse con las comillas.
_PS_SPEAK = (
    "Add-Type -AssemblyName System.Speech; "
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
    "$v = $s.GetInstalledVoices() | "
    "Where-Object { $_.VoiceInfo.Culture.Name -like 'es*' } | "
    "Select-Object -First 1; "
    "if ($v) { $s.SelectVoice($v.VoiceInfo.Name) }; "
    "$s.Speak([Console]::In.ReadToEnd())"
)

_PS_LIST = (
    "Add-Type -AssemblyName System.Speech; "
    "(New-Object System.Speech.Synthesis.SpeechSynthesizer).GetInstalledVoices() | "
    "ForEach-Object { $_.VoiceInfo.Name + '|' + $_.VoiceInfo.Culture.Name }"
)


def available() -> bool:
    return platform.system() == "Windows" and shutil.which("powershell") is not None


@lru_cache(maxsize=1)
def voice_name() -> str | None:
    """Nombre de la voz en español que se va a usar, si existe."""
    if not available():
        return None
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", _PS_LIST],
            capture_output=True, text=True, timeout=25, creationflags=_NO_WINDOW,
        ).stdout
    except Exception:
        return None
    spanish = [
        line.split("|")[0].strip()
        for line in out.splitlines()
        if "|" in line and line.split("|")[1].strip().lower().startswith("es")
    ]
    return spanish[0] if spanish else None


def _speak_blocking(text: str) -> None:
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", _PS_SPEAK],
            input=text, text=True, timeout=60, creationflags=_NO_WINDOW,
        )
    except Exception as exc:  # hablar es un extra, nunca debe romper la app
        print(f"[voz] falló: {exc}")


def speak(text: str) -> None:
    """Dice `text` en un hilo aparte; vuelve enseguida."""
    text = text.strip()
    if not text:
        return
    print(f"[voz] {text}")
    if not available():
        return
    threading.Thread(target=_speak_blocking, args=(text,), daemon=True).start()
