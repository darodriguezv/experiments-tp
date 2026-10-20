"""Lo que cuesta un modelo: disco, RAM, arranque y velocidad de generación.

La grilla de calidad responde *qué tan bien* contesta cada modelo. No responde
si entra en el equipo, ni cuánto hay que esperarlo después de una pausa. Para
un dispositivo que vive en una mesa de luz esas dos cosas deciden tanto como el
recall, y hasta ahora se declaraban a mano en `CANDIDATE_MODELS["size_gb"]`
-un número copiado de la página del modelo, nunca medido en esta máquina.

## Las tres latencias distintas que todo el mundo mezcla

Decir "el modelo tarda X" no significa nada hasta decir en qué estado estaba:

1. **Caliente**: ya cargado en RAM. Es el caso normal mientras se conversa, y
   es la única que mide la grilla de calidad.
2. **Recarga tibia**: Ollama lo descargó (`keep_alive` vencido) pero el archivo
   sigue en la caché de páginas del sistema operativo. Es lo que pasa tras una
   pausa de unos minutos.
3. **Fría de verdad**: primera lectura desde el disco, con la caché vacía.

La diferencia entre 2 y 3 es enorme -se midió 4.3 s contra 39 s para el mismo
modelo- y confundirlas lleva a conclusiones opuestas sobre si un modelo es
usable. Este módulo mide 1 y 2, y **no puede medir 3**: vaciar la caché de
páginas requiere privilegios que un experimento no debería pedir, y reiniciar
la máquina entre modelos no es automatizable. Se reporta como lo que es, un
límite declarado, en vez de pasar la recarga tibia por fría.

## Por qué la huella de RAM se le pregunta a Ollama

`/api/ps` informa cuánta memoria tiene reservada el modelo *cargado*, separando
RAM de VRAM. Medir el proceso desde afuera daría el total de Ollama, que
incluye cualquier otro modelo que haya quedado residente, y para un equipo
dedicado lo que importa es cuánto pide ESTE modelo.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .backends import OLLAMA_URL, OllamaBackend

GB = 1024 ** 3


@dataclass
class Footprint:
    """El costo medido de un modelo en ESTA máquina."""

    model: str
    disk_gb: float = 0.0
    """Tamaño del archivo de pesos, según Ollama."""

    ram_gb: float = 0.0
    """RAM reservada con el modelo cargado, sin contar VRAM."""

    vram_gb: float = 0.0
    quantization: str = ""
    """Por ejemplo Q4_K_M. Decide la huella tanto como el número de parámetros:
    el mismo 7b en q8 ocupa casi el doble que en q4."""

    params: str = ""
    warm_p50_s: float = 0.0
    """Mediana del modelo ya cargado. Comparable con la grilla de calidad."""

    reload_s: float = 0.0
    """Recarga tibia: descargado de RAM, archivo todavía en caché del SO."""

    tokens_per_s: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def _api(url: str, path: str, payload: dict | None = None,
         timeout: float = 30.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url.rstrip("/") + path, data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return json.loads(fh.read().decode("utf-8"))


def _disco_y_cuantizacion(url: str, model: str) -> tuple[float, str, str]:
    """Disco, cuantización y parámetros, de los metadatos de Ollama."""
    disco = 0.0
    try:
        for m in _api(url, "/api/tags", timeout=5.0).get("models", []):
            # `/api/tags` lista "qwen2.5:3b"; una descarga sin etiqueta queda
            # como "qwen2.5:latest" y no coincidiría por igualdad estricta.
            if m.get("name") in (model, f"{model}:latest"):
                disco = float(m.get("size", 0)) / GB
                break
    except Exception:
        pass

    cuant = params = ""
    try:
        info = _api(url, "/api/show", {"model": model}, timeout=15.0)
        detalles = info.get("details", {})
        cuant = detalles.get("quantization_level", "") or ""
        params = detalles.get("parameter_size", "") or ""
    except Exception:
        pass
    return disco, cuant, params


def _residente(url: str, model: str) -> tuple[float, float]:
    """RAM y VRAM del modelo cargado ahora mismo."""
    try:
        for m in _api(url, "/api/ps", timeout=5.0).get("models", []):
            if m.get("name") in (model, f"{model}:latest"):
                total = float(m.get("size", 0))
                vram = float(m.get("size_vram", 0))
                # `size` es el total reservado y `size_vram` la parte en GPU;
                # la diferencia es lo que realmente pesa en la RAM del equipo,
                # que es la restricción en una máquina sin placa dedicada.
                return max(0.0, total - vram) / GB, vram / GB
    except Exception:
        pass
    return 0.0, 0.0


def _descargar(url: str, model: str) -> None:
    """Saca el modelo de memoria. `keep_alive: 0` es la forma documentada."""
    try:
        _api(url, "/api/generate",
             {"model": model, "prompt": "", "keep_alive": 0}, timeout=30.0)
    except Exception:
        pass
    # Ollama libera de forma asíncrona: sin esta espera la medición siguiente
    # encuentra el modelo todavía cargado y mide una recarga que no ocurrió.
    for _ in range(40):
        try:
            cargados = {m.get("name") for m in
                        _api(url, "/api/ps", timeout=3.0).get("models", [])}
        except Exception:
            return
        if model not in cargados and f"{model}:latest" not in cargados:
            return
        time.sleep(0.25)


def measure(model: str, *, url: str = OLLAMA_URL, repeticiones: int = 5,
            medir_recarga: bool = True) -> Footprint:
    """Mide un modelo. Devuelve un `Footprint` con `error` si algo falla."""
    import numpy as np

    fp = Footprint(model=model)
    backend = OllamaBackend(model, url=url, keep_alive="5m")
    if not backend.available():
        fp.error = "Ollama no responde"
        return fp

    fp.disk_gb, fp.quantization, fp.params = _disco_y_cuantizacion(url, model)

    # --- recarga tibia -----------------------------------------------------
    # Se mide ANTES de las repeticiones calientes y con el modelo recién
    # descargado. Al revés mediría una carga que no ocurrió, porque la primera
    # repetición ya lo habría dejado residente.
    if medir_recarga:
        _descargar(url, model)
        t0 = time.perf_counter()
        primera = backend.complete("Responde OK.", "OK", max_tokens=4)
        fp.reload_s = time.perf_counter() - t0
        if not primera.ok:
            fp.error = primera.error
            return fp
    else:
        backend.warm()

    # --- caliente ----------------------------------------------------------
    # Un prompt con algo de salida: medir tokens/s sobre una respuesta de
    # cuatro tokens da un número dominado por el prefill y no por la
    # generación, que es lo que se quiere comparar entre modelos.
    lat, tps = [], []
    for _ in range(max(1, repeticiones)):
        c = backend.complete(
            "Respondé en una sola frase corta, en español.",
            "Contá hasta cinco con palabras.", max_tokens=48)
        if not c.ok:
            fp.error = c.error
            return fp
        lat.append(c.model_s or c.latency_s)
        if c.generate_s > 0 and c.output_tokens > 0:
            tps.append(c.output_tokens / c.generate_s)

    fp.warm_p50_s = float(np.percentile(lat, 50))
    fp.tokens_per_s = float(np.median(tps)) if tps else 0.0
    fp.ram_gb, fp.vram_gb = _residente(url, model)
    return fp


def measure_all(models: list[str], *, url: str = OLLAMA_URL,
                repeticiones: int = 5, verbose: bool = True) -> list[Footprint]:
    salida = []
    for m in models:
        if verbose:
            print(f"  midiendo {m} ...", flush=True)
        fp = measure(m, url=url, repeticiones=repeticiones)
        if verbose:
            print(f"    {'FALLA: ' + fp.error if not fp.ok else ''}"
                  f"{'' if not fp.ok else f'{fp.disk_gb:.1f} GB disco, '
                     f'{fp.ram_gb:.1f} GB RAM, recarga {fp.reload_s:.1f} s'}",
                  flush=True)
        salida.append(fp)
    return salida


def save(footprints: list[Footprint], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "\n".join(json.dumps(asdict(f), ensure_ascii=False)
                  for f in footprints) + "\n", encoding="utf-8")
    return p


def load(path: str | Path) -> list[Footprint]:
    out = []
    for linea in Path(path).read_text(encoding="utf-8").splitlines():
        if linea.strip():
            out.append(Footprint(**json.loads(linea)))
    return out


LIMITE_CACHE = (
    "La recarga medida es TIBIA: el modelo salió de RAM pero su archivo sigue "
    "en la caché de páginas del sistema. Una primera lectura real desde disco "
    "es varias veces peor (se midió 4.3 s contra 39 s en el mismo modelo) y no "
    "se puede automatizar sin vaciar la caché del SO. Para el dispositivo lo "
    "que importa es la tibia, porque el equipo queda encendido; la fría solo "
    "aparece tras reiniciar."
)


def table(footprints: list[Footprint]) -> str:
    """La tabla de costo, ordenada por disco."""
    filas = [f for f in footprints if f.ok]
    fallidos = [f for f in footprints if not f.ok]

    def _fallos() -> list[str]:
        return [f"  sin medir  {f.model}: {f.error}" for f in fallidos]

    if not filas:
        # Antes se devolvía solo "no hay medición válida" y la lista de
        # fallidos se perdía. Quien corría el comando veía que no salió nada y
        # ninguna pista de por qué: justo el caso en que el motivo es lo único
        # que importa, porque suele ser el mismo para todos (Ollama apagado).
        return "\n".join(["No hay ninguna medición válida.", ""] + _fallos()
                         + [""]) + "\n"

    out = ["Costo medido en esta máquina", ""]
    out.append(f"  {'modelo':<20} {'params':>7} {'cuant':>8} {'disco':>7} "
               f"{'RAM':>7} {'VRAM':>7} {'calien':>7} {'recarga':>8} {'tok/s':>7}")
    out.append("  " + "-" * 90)
    for f in sorted(filas, key=lambda x: x.disk_gb):
        out.append(f"  {f.model:<20} {f.params:>7} {f.quantization:>8} "
                   f"{f.disk_gb:>6.1f}G {f.ram_gb:>6.1f}G {f.vram_gb:>6.1f}G "
                   f"{f.warm_p50_s:>6.2f}s {f.reload_s:>7.1f}s "
                   f"{f.tokens_per_s:>7.1f}")
    out.append("")
    for linea in _envolver(LIMITE_CACHE, 86):
        out.append(f"  {linea}")
    if fallidos:
        out.append("")
        out += _fallos()
    return "\n".join(out) + "\n"


def _envolver(texto: str, ancho: int) -> list[str]:
    palabras, linea, salida = texto.split(), "", []
    for p in palabras:
        if len(linea) + len(p) + 1 > ancho:
            salida.append(linea)
            linea = p
        else:
            linea = f"{linea} {p}".strip()
    if linea:
        salida.append(linea)
    return salida
