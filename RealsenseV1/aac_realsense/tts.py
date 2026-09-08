"""Salida de voz, sin bloquear la interfaz.

Usa la voz SAPI que ya trae Windows, así el proyecto no necesita dependencias
extra. Prefiere una voz en español si hay alguna instalada (por ejemplo
"Microsoft Sabina Desktop", es-MX); si no, usa la predeterminada. En otros
sistemas no habla y solo imprime en consola.
"""

from __future__ import annotations

import base64
import platform
import shutil
import subprocess
import threading
from functools import lru_cache

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# Selecciona la primera voz cuya cultura empiece con "es"; si no hay, deja la
# predeterminada.
#
# El texto viaja en **base64**, y esa decisión es el arreglo de un bug real. La
# versión anterior lo pasaba por stdin, y la ñ nunca llegaba al sintetizador:
# PowerShell decodifica stdin con la codepage OEM (cp437/850 en una instalación
# en español), así que el byte 0xD1 de "BAÑO" se convertía en ╤ - codepoint 9572,
# un carácter de dibujo de cajas. La voz española pronuncia la ñ sin problema;
# lo que recibía era basura.
#
# Base64 es ASCII puro, así que ninguna codepage lo puede corromper, y no hace
# falta archivo temporal ni pelear con el encoding de la consola. El costo es
# una línea de decodificación del otro lado.
#
# Vale registrar por qué tardó en aparecer: el smoke test verificaba que "BAÑO"
# se *dibuje* distinto de "BANO" (que era el bug de cv2.putText), y eso pasaba.
# El camino de audio no tenía ninguna verificación, porque nadie escucha en un
# test. `_PS_ROUNDTRIP` de abajo lo cubre sin necesidad de oír.
_PS_SPEAK = (
    "Add-Type -AssemblyName System.Speech; "
    "$t = [Text.Encoding]::UTF8.GetString("
    "[Convert]::FromBase64String([Console]::In.ReadToEnd())); "
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
    "$v = $s.GetInstalledVoices() | "
    "Where-Object { $_.VoiceInfo.Culture.Name -like 'es*' } | "
    "Select-Object -First 1; "
    "if ($v) { $s.SelectVoice($v.VoiceInfo.Name) }; "
    "$s.Speak($t)"
)

# El mismo transporte, pero devolviendo los codepoints en vez de hablar. Es la
# forma de asertar el camino de audio en un test: si la ñ llega como 241, el
# sintetizador recibió una ñ.
_PS_ROUNDTRIP = (
    "$t = [Text.Encoding]::UTF8.GetString("
    "[Convert]::FromBase64String([Console]::In.ReadToEnd())); "
    "Write-Output ((($t.ToCharArray() | ForEach-Object { [int]$_ }) -join ','))"
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


def _encode(text: str) -> bytes:
    """El texto como base64 ASCII, listo para stdin de PowerShell."""
    return base64.b64encode(text.encode("utf-8"))


def _speak_blocking(text: str) -> None:
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", _PS_SPEAK],
            input=_encode(text), timeout=60, creationflags=_NO_WINDOW,
        )
    except Exception as exc:  # hablar es un extra, nunca debe romper la app
        print(f"[voz] falló: {exc}")


def codepoints_roundtrip(text: str) -> list[int] | None:
    """Qué codepoints recibe realmente PowerShell. None si no se pudo probar.

    Existe para que el camino de audio sea asertable. Sin esto, un bug de
    codificación entre Python y el sintetizador es invisible en cualquier test
    -nadie escucha- y solo aparece cuando alguien nota que el aparato dice
    "bao" en vez de "baño".
    """
    if not available():
        return None
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", _PS_ROUNDTRIP],
            input=_encode(text), capture_output=True, timeout=30,
            creationflags=_NO_WINDOW,
        )
        out = result.stdout.decode("ascii", "replace").strip()
        return [int(n) for n in out.split(",") if n.strip()]
    except Exception:
        return None


def speak(text: str) -> None:
    """Dice `text` en un hilo aparte; vuelve enseguida."""
    text = text.strip()
    if not text:
        return
    print(f"[voz] {text}")
    if not available():
        return
    threading.Thread(target=_speak_blocking, args=(text,), daemon=True).start()
