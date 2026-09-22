"""Where completions come from: Ollama, or a scripted stand-in.

Same shape as the camera project's `FrameSource`, for the same reason: the
experiment harness, the corpus and every metric can then be written and tested
before any model is installed, and swapping model runtimes later touches one
file.

Only Ollama is implemented as a real backend, deliberately. This module is not
trying to be a model gateway - it needs to run **small open-weight models
locally**, because that is what the finished device has to do. An AAC device for
someone with ALS cannot make a network call to say "me duele": no internet, no
speech, and the content is medical. So local inference on modest hardware is the
target, not a budget compromise, and Ollama is the shortest path to it on
Windows.

`options` are pinned low-temperature on purpose. This is not creative writing:
the same telegraphic input should produce the same candidates today and next
week, or the user cannot build muscle memory and the experiment cannot be
replicated.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

OLLAMA_URL = "http://127.0.0.1:11434"
"""The numeric address, never the hostname `localhost`.

Measured, not assumed: on this Windows machine `http://localhost:11434` costs
**2.06 s per request** against **0.03 s** for `http://127.0.0.1:11434` - a factor
of 68. Ollama binds to 127.0.0.1 only, so resolving `localhost` sends Windows to
the IPv6 `::1` first, where it waits out a connection timeout before falling
back to IPv4.

That two seconds was pure measurement artefact, and it was the dominant term in
every latency number the first grid produced: the model itself takes 0.18-0.36 s
and the harness was reporting 2.2-2.6 s. It also flattened the differences
between models, which would have supported a wrong conclusion - "latency does
not depend on model size, so use the largest one that fits".
"""

# Small language models: everything here is 4B parameters or less.
#
# SLM rather than LLM is a requirement of the device, not a budget compromise.
# Three constraints push the same way:
#
# * **Offline.** A device that says "me duele" cannot depend on a network call.
#   If the wifi is down, the user has no voice.
# * **Latency.** Each dwell costs ~800 ms of the user's effort. A model that
#   takes eight seconds to propose candidates spends more of their time than the
#   typing did.
# * **Privacy.** The content is medical and it is spoken by someone who cannot
#   easily consent again later. It should not leave the room.
#
# The Qwen family appears at four sizes on purpose: same training recipe, only
# the parameter count changes, which is the only clean way to ask "does size
# matter for *this* task" rather than "is model A better than model B".
#
# qwen2.5:7b is the one entry that breaks the 4B ceiling, and it is here to
# *measure the ceiling*, not to replace it. The SLM constraint above is a
# design claim, and a design claim that is never tested is just a preference.
# The 7b arm answers the only question that can falsify it: how much fidelity
# are we giving up by refusing to go bigger? If it wins clearly, the constraint
# is expensive and the thesis has to say so out loud. If it does not, the
# constraint is free - which is a stronger result than never having asked.
#
# Either way it is not the default. It costs ~4.7 GB of RAM and the latency
# budget is the binding one on this hardware.
CANDIDATE_MODELS: dict[str, dict] = {
    "qwen2.5:0.5b": {
        "size_gb": 0.4, "params": "0.5B", "license": "Apache-2.0",
        "why": "el piso: ¿puede algo tan chico hacer esta tarea? Si sí, corre "
               "en cualquier cosa, sin GPU",
    },
    "qwen2.5:1.5b": {
        "size_gb": 1.0, "params": "1.5B", "license": "Apache-2.0",
        "why": "el candidato realista: Apache limpio, español decente, entra "
               "en RAM sin GPU",
    },
    "qwen2.5:3b": {
        "size_gb": 1.9, "params": "3B", "license": "Qwen Research",
        "why": "el techo de la familia; con 0.5b y 1.5b forma la curva de "
               "escala a receta constante",
    },
    "qwen2.5:7b": {
        "size_gb": 4.7, "params": "7B", "license": "Apache-2.0",
        "why": "el brazo que rompe el techo de 4B a propósito: mide cuánto "
               "cuesta la restricción SLM en vez de asumirla. Misma receta que "
               "0.5b/1.5b/3b, así que extiende la curva de escala a un punto "
               "más sin cambiar ninguna otra variable",
    },
    "llama3.2:1b": {
        "size_gb": 1.3, "params": "1B", "license": "Llama 3.2",
        "why": "segunda familia al mismo tamaño; Llama 3.2 declara español "
               "como idioma soportado oficialmente",
    },
    "llama3.2:3b": {
        "size_gb": 2.0, "params": "3B", "license": "Llama 3.2",
        "why": "el par de qwen2.5:3b: mismo tamaño, otra receta",
    },
    "gemma2:2b": {
        "size_gb": 1.6, "params": "2B", "license": "Gemma Terms",
        "why": "tercera familia en el medio del rango",
    },
    "phi3.5": {
        "size_gb": 2.2, "params": "3.8B", "license": "MIT",
        "why": "el más grande que sigue siendo SLM, y la licencia más limpia "
               "de todas para un dispositivo que se va a distribuir",
    },
    "smollm2:1.7b": {
        "size_gb": 1.8, "params": "1.7B", "license": "Apache-2.0",
        "why": "control: entrenado sobre todo en inglés. Si va notablemente "
               "peor, el resultado es que el español en el entrenamiento pesa "
               "más que el tamaño",
    },
}

@dataclass
class Completion:
    """One model response, with the timing the experiment cares about."""

    text: str
    model: str
    latency_s: float
    prompt_tokens: int = 0
    output_tokens: int = 0
    error: str = ""

    model_s: float = 0.0
    """Time Ollama itself reports, excluding our transport. The honest figure
    for what the device would experience with a sane client."""

    prefill_s: float = 0.0
    """Cost of reading the prompt. This is the price the few-shot examples
    charge, and it is what makes the recall-versus-latency trade measurable
    instead of assumed."""

    generate_s: float = 0.0
    raw: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def tokens_per_s(self) -> float:
        return self.output_tokens / self.latency_s if self.latency_s > 0 else 0.0


class Backend:
    """Interface: a prompt in, a `Completion` out."""

    name = "backend"

    def complete(self, system: str, user: str, *, json_mode: bool = False,
                 max_tokens: int = 300) -> Completion:
        raise NotImplementedError

    def available(self) -> bool:
        return True

    def models(self) -> list[str]:
        return []


class OllamaBackend(Backend):
    """Local inference through Ollama's HTTP API.

    Uses `urllib` rather than `requests` so the harness has no dependency beyond
    the standard library plus numpy - one less thing to install on a machine
    that will eventually be a dedicated device.
    """

    def __init__(self, model: str, url: str = OLLAMA_URL,
                 temperature: float = 0.2, timeout_s: float = 180.0,
                 seed: int = 7, keep_alive: str = "30m"):
        self.model = model
        self.name = model
        self.url = url.rstrip("/")
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.seed = seed
        self.keep_alive = keep_alive
        """Cuánto sigue el modelo en RAM tras la última petición.

        Ollama usa 5 minutos por defecto y es demasiado poco para hablar:
        recargar qwen2.5:3b cuesta 4.3 s y el 7b 6.9 s, así que una pausa hace
        que la frase siguiente se demore sin que nada lo explique. Para el
        tablero conviene `-1`, que lo deja cargado mientras Ollama viva."""

    # ------------------------------------------------------------- discovery

    def available(self) -> bool:
        try:
            self._get("/api/tags", timeout=3.0)
        except Exception:
            return False
        return True

    def models(self) -> list[str]:
        try:
            data = self._get("/api/tags", timeout=5.0)
        except Exception:
            return []
        return sorted(m["name"] for m in data.get("models", []))

    def has_model(self) -> bool:
        installed = self.models()
        # Ollama reports "qwen2.5:7b" but a bare "qwen2.5" pull lands as
        # "qwen2.5:latest", so match on the family too rather than exact string.
        return any(m == self.model or m.split(":")[0] == self.model.split(":")[0]
                   for m in installed)

    # ------------------------------------------------------------ completion

    def complete(self, system: str, user: str, *, json_mode: bool = False,
                 max_tokens: int = 300) -> Completion:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": max_tokens,
                # A fixed seed makes a run reproducible. Without it, comparing
                # two prompts means comparing two samples of noise as much as
                # two prompts.
                "seed": self.seed,
                # Enough context for a few-shot prompt plus the input, and no
                # more: a large window costs VRAM that the model needs.
                "num_ctx": 2048,
            },
        }
        if json_mode:
            # Ollama constrains decoding to valid JSON. Worth using: a small
            # model that answers correctly but in prose is a parsing failure,
            # and that confounds "can it do the task" with "can it follow a
            # format". Constrained decoding separates the two.
            payload["format"] = "json"

        t0 = time.perf_counter()
        try:
            data = self._post("/api/chat", payload, timeout=self.timeout_s)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:300]
            return Completion("", self.model, time.perf_counter() - t0,
                              error=f"HTTP {exc.code}: {body}")
        except Exception as exc:
            return Completion("", self.model, time.perf_counter() - t0,
                              error=f"{type(exc).__name__}: {exc}")

        latency = time.perf_counter() - t0
        # Ollama reports where the time went, in nanoseconds. Kept because
        # wall-clock alone cannot distinguish "the model is slow" from "the
        # prompt is long" from "our transport is broken" - and it was the third
        # one, by a factor of 68, until the URL above was fixed.
        ns = 1e9
        return Completion(
            text=data.get("message", {}).get("content", ""),
            model=self.model,
            latency_s=latency,
            prompt_tokens=int(data.get("prompt_eval_count", 0)),
            output_tokens=int(data.get("eval_count", 0)),
            model_s=float(data.get("total_duration", 0)) / ns,
            prefill_s=float(data.get("prompt_eval_duration", 0)) / ns,
            generate_s=float(data.get("eval_duration", 0)) / ns,
            raw=data,
        )

    def warm(self) -> Completion:
        """One throwaway call so the first measured latency is not the load time.

        Loading a 5 GB model into VRAM takes seconds. Including that in the
        first sample would make whichever model ran first look terrible, which
        is a measurement artefact rather than a finding.
        """
        return self.complete("Responde OK.", "OK", max_tokens=4)

    # --------------------------------------------------------------- plumbing

    def _get(self, path: str, timeout: float) -> dict:
        req = urllib.request.Request(self.url + path, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as fh:
            return json.loads(fh.read().decode("utf-8"))

    def _post(self, path: str, payload: dict, timeout: float) -> dict:
        req = urllib.request.Request(
            self.url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as fh:
            return json.loads(fh.read().decode("utf-8"))


class ScriptedBackend(Backend):
    """Replays canned responses. The test double.

    Lets the corpus, every metric and the whole report be exercised with known
    inputs and known outputs, so a metric that miscounts an invention fails
    here rather than being mistaken for a bad model later.
    """

    name = "scripted"

    def __init__(self, responses: dict[str, str] | list[str],
                 latency_s: float = 0.05):
        self._responses = responses
        self._order: list[str] = (list(responses) if isinstance(responses, list)
                                  else [])
        self._i = 0
        self.latency_s = latency_s

    def complete(self, system: str, user: str, *, json_mode: bool = False,
                 max_tokens: int = 300) -> Completion:
        if isinstance(self._responses, dict):
            text = self._responses.get(user.strip(), "")
        else:
            text = self._order[self._i % len(self._order)] if self._order else ""
            self._i += 1
        return Completion(text=text, model=self.name, latency_s=self.latency_s,
                          prompt_tokens=len(system.split()) + len(user.split()),
                          output_tokens=len(text.split()))


def ollama_report(url: str = OLLAMA_URL) -> str:
    """What is installed and what is missing - printed by `main.py models`."""
    probe = OllamaBackend("qwen2.5:1.5b", url=url)
    if not probe.available():
        return (
            "Ollama no responde en " + url + ".\n"
            "  1. Instalalo desde https://ollama.com/download (Windows, ~700 MB)\n"
            "  2. Queda corriendo como servicio; verificá con `ollama list`\n"
            "  3. Bajá los modelos con `main.py pull`\n\n"
            "Todo lo que no necesita modelo - el corpus, las métricas, el "
            "reporte - se puede ejercitar igual con `main.py smoke`."
        )

    installed = set(probe.models())
    lines = [f"Ollama responde en {url}.", ""]
    lines.append(f"{'modelo':<16}{'params':<8}{'tamaño':<9}{'estado':<14}licencia")
    for name, info in CANDIDATE_MODELS.items():
        here = any(m == name or m.split(":")[0] == name.split(":")[0]
                   for m in installed)
        lines.append(f"{name:<16}{info['params']:<8}{info['size_gb']:<9.1f}"
                     f"{'instalado' if here else 'falta':<14}{info['license']}")
    extra = sorted(installed - {n for n in CANDIDATE_MODELS})
    if extra:
        lines += ["", "otros modelos instalados: " + ", ".join(extra)]
    lines += ["", "Por qué cada uno:"]
    lines += [f"  {n}: {i['why']}" for n, i in CANDIDATE_MODELS.items()]
    return "\n".join(lines)
