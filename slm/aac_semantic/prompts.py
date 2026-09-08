"""The prompt variants under test, from bare instruction to full contract.

Ordered so each one adds exactly one thing to the previous. That is the point:
if V4 beats V0 it is worth knowing *which* of the four additions bought the
improvement, and a set of prompts that differ in five ways at once cannot tell
you. Each variant's docstring says what it adds and what it is meant to fix.

The output contract throughout is a **list of candidates**, never a single
sentence. The reason is not hedging, it is the safety argument in the README: at
roughly 800 ms per dwell the user paid dearly for every word, and a device that
commits to one reading of an ambiguous input can speak something they did not
mean. Candidates keep the choice with the person whose voice it is.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

N_CANDIDATES = 3
"""How many options to generate.

Three, not five. Each extra candidate is another tile the user must read and
another dwell of scanning before choosing, and reading is not free for someone
whose gaze is also their only pointer. Three plus the literal text is four
things on screen, which fits one row of the existing board.
"""

# The constraints that matter, written once and reused, so a variant differs
# from its predecessor only in what its docstring claims.
_RULES_ES = """Reglas, en orden de importancia:
1. NO agregues información que la entrada no diga. Si dice AGUA, no digas "agua fría" ni "un vaso de agua": no sabés la temperatura ni el recipiente.
2. NO cambies la polaridad. Si hay NO, el resultado niega. Si dice BASTA, el resultado ordena detenerse.
3. NO suavices urgencia ni dolor. "DOLOR" es "me duele", no "tengo una molestia leve".
4. Hablás EN PRIMERA PERSONA, como si fueras la persona. No le preguntes nada ni la saludes.
5. Español neutro, natural, breve. Como habla alguien, no como escribe un manual.
6. Si la entrada es ambigua, que los candidatos cubran lecturas DISTINTAS en vez de repetir una.
7. Si la entrada no alcanza para una frase (una sola letra, un verbo sin objeto, vacío), devolvé la entrada tal cual y nada más. No adivines la palabra que falta."""

# Los ejemplos NO deben aparecer en el corpus de evaluación. Cinco de los cinco
# originales sí aparecían, y el efecto fue inmediato y grande: qwen2.5:0.5b
# devolvía los candidatos de los ejemplos palabra por palabra en las entradas
# que coincidían. Eso no mide generalización, mide recuperación desde el prompt,
# y habría inflado toda variante con ejemplos haciendo parecer que los ejemplos
# ayudan enormemente.
#
# Estos cinco salen del mismo vocabulario del tablero y demuestran las mismas
# propiedades difíciles -pedido positivo, negación ambigua, palabra sola, verbo
# sin objeto, deletreo con acciones opuestas- sin tocar ningún ítem evaluado.
# `smoke.py` verifica el solapamiento cero en cada corrida.
_EXAMPLES_ES = """Ejemplos:

entrada: NECESITO AGUA
{"candidatos": ["necesito agua", "necesito agua, por favor", "tengo sed"]}

entrada: COMIDA NO
{"candidatos": ["no quiero comer", "no hay comida", "comida no"]}

entrada: BIEN
{"candidatos": ["estoy bien", "bien", "me siento bien"]}

entrada: NECESITO
{"candidatos": ["necesito"]}

entrada: LUZ
{"candidatos": ["prendé la luz", "apagá la luz", "la luz"]}"""

_GLOSARIO_ES = """El tablero tiene 20 palabras fijas y cada una significa una cosa concreta en este sistema. No las interpretes de otra forma:

SÍ, NO — respuestas. POR FAVOR, GRACIAS, HOLA — cortesía.
QUIERO, NECESITO — piden algo. ME SIENTO — introduce un estado.
AYUDA — pide asistencia de otra persona.
BASTA — orden de detener lo que está pasando.
AGUA — para beber. COMIDA — para comer.
BAÑO — el inodoro. NO significa bañarse ni ducharse.
DOLOR — dolor físico propio. MEDICINA — su medicación.
MÁS, MENOS — cantidad o intensidad de lo que ya estaba pasando.
BIEN, MAL — cómo está. TAL VEZ — duda, ni sí ni no.

Cualquier otra palabra vino deletreada letra por letra, así que es exactamente lo que la persona quiso decir."""

_JSON_SCHEMA_ES = """Devolvé SOLO un objeto JSON con esta forma exacta, sin texto alrededor:
{"candidatos": ["frase 1", "frase 2", "frase 3"]}"""


@dataclass(frozen=True)
class Prompt:
    variant: str
    adds: str
    system: str
    json_mode: bool = False

    def user(self, telegraphic: str) -> str:
        return f"entrada: {telegraphic}"


def _v0() -> Prompt:
    """Bare instruction. The floor.

    Exists to answer "does any of this prompting effort pay for itself?". A
    model that already behaves well here needs none of the machinery below.
    """
    return Prompt(
        variant="v0-minimo",
        adds="nada: instrucción desnuda",
        system=(
            "Convertís entradas telegráficas de un tablero de comunicación en "
            f"frases naturales en español. Devolvé {N_CANDIDATES} opciones, una "
            "por línea."
        ),
    )


def _v1() -> Prompt:
    """+ the rules. Tests whether stating the constraints is enough.

    This is where the safety-relevant behaviour is asked for explicitly:
    no invention, polarity preserved, urgency not softened.
    """
    return Prompt(
        variant="v1-reglas",
        adds="las 7 reglas (no inventar, polaridad, urgencia, primera persona)",
        system=(
            "Sos el módulo de expansión de un dispositivo de comunicación "
            "aumentativa. Una persona con ELA arma frases telegráficas mirando "
            "un tablero: cada palabra le cuesta casi un segundo de esfuerzo. "
            "Tu trabajo es proponer cómo decir eso mismo de forma natural, sin "
            "poner palabras que no dijo.\n\n"
            f"{_RULES_ES}\n\n"
            f"Devolvé {N_CANDIDATES} opciones, una por línea, sin numerarlas."
        ),
    )


def _v2() -> Prompt:
    """+ few-shot examples. Tests whether showing beats telling.

    The examples are chosen to demonstrate the hard cases rather than the easy
    ones: an ambiguous input, a negation, an urgent symptom, and an input too
    incomplete to expand.
    """
    return Prompt(
        variant="v2-ejemplos",
        adds="5 ejemplos, elegidos sobre los casos difíciles",
        system=(
            "Sos el módulo de expansión de un dispositivo de comunicación "
            "aumentativa para una persona con ELA. Cada palabra de la entrada "
            "le costó casi un segundo de esfuerzo.\n\n"
            f"{_RULES_ES}\n\n{_EXAMPLES_ES}\n\n"
            f"Devolvé {N_CANDIDATES} opciones, una por línea."
        ),
    )


def _v3() -> Prompt:
    """+ a JSON contract and constrained decoding.

    Separates two failures that a line-based format confuses: a model that got
    the task right but answered in prose, and a model that got the task wrong.
    Ollama's `format: json` constrains decoding, so parse failures stop being a
    confound and format compliance becomes measurable on its own.
    """
    return Prompt(
        variant="v3-json",
        adds="contrato JSON + decodificación restringida",
        system=(
            "Sos el módulo de expansión de un dispositivo de comunicación "
            "aumentativa para una persona con ELA.\n\n"
            f"{_RULES_ES}\n\n{_EXAMPLES_ES}\n\n{_JSON_SCHEMA_ES}"
        ),
        json_mode=True,
    )


def _v4() -> Prompt:
    """+ the literal reading must be one of the candidates.

    The escape hatch, made the model's job instead of the UI's. If every
    generated candidate is wrong, the user must still be able to say exactly
    what they typed - otherwise the expansion module has taken away an ability
    the plain board already had.
    """
    return Prompt(
        variant="v4-literal",
        adds="obliga a incluir la lectura literal como una opción",
        system=(
            "Sos el módulo de expansión de un dispositivo de comunicación "
            "aumentativa para una persona con ELA.\n\n"
            f"{_RULES_ES}\n"
            "8. Una de las opciones debe ser la lectura LITERAL: la entrada con "
            "el mínimo arreglo gramatical y nada más. Si las otras dos se "
            "equivocan, la persona tiene que poder decir exactamente lo que "
            "escribió.\n\n"
            f"{_EXAMPLES_ES}\n\n{_JSON_SCHEMA_ES}"
        ),
        json_mode=True,
    )


def _v5() -> Prompt:
    """+ think about ambiguity first, then expand.

    Two steps in one call: name the distinct readings, then write one candidate
    per reading. The hypothesis is that a model asked to enumerate readings
    first will stop producing three paraphrases of the same guess - which is the
    failure that makes N candidates worth no more than one.
    """
    return Prompt(
        variant="v5-lecturas",
        adds="primero enumera lecturas distintas, después expande una por lectura",
        system=(
            "Sos el módulo de expansión de un dispositivo de comunicación "
            "aumentativa para una persona con ELA.\n\n"
            f"{_RULES_ES}\n"
            "8. Una de las opciones debe ser la lectura LITERAL.\n\n"
            "Procedimiento: primero identificá qué lecturas DISTINTAS admite la "
            "entrada (puede ser una sola). Después escribí una frase por "
            "lectura. Si sobran lugares, agregá variantes de registro de la "
            "lectura más probable.\n\n"
            f"{_EXAMPLES_ES}\n\n"
            'Devolvé SOLO JSON con esta forma:\n'
            '{"lecturas": ["descripción breve de cada lectura"], '
            '"candidatos": ["frase 1", "frase 2", "frase 3"]}'
        ),
        json_mode=True,
    )


def _v6() -> Prompt:
    """+ un glosario de las 20 palabras del tablero.

    Información, no restricciones - y la distinción importa, porque la ablación
    ya mostró que agregar reglas en prosa *baja* el recall 15 puntos mientras
    agregar ejemplos lo sube 31. El glosario es del segundo tipo: le dice al
    modelo qué significan las palabras, que es algo que no puede inferir.

    El caso que lo motivó: "POR FAVOR BAÑO" salía como "por favor báñate". En
    este tablero BAÑO es el inodoro, no la ducha, y nada en el prompt lo decía.
    El modelo eligió el sentido más frecuente del español general, que acá es el
    equivocado - y ninguna cantidad de "no inventes" lo arregla, porque no
    estaba inventando: estaba desambiguando mal.
    """
    return Prompt(
        variant="v6-glosario",
        adds="glosario de las 20 palabras del tablero (información, no reglas)",
        system=f"""Sos el módulo de expansión de un dispositivo de comunicación aumentativa para una persona con ELA.

{_GLOSARIO_ES}

{_RULES_ES}
8. Una de las opciones debe ser la lectura LITERAL.

{_EXAMPLES_ES}

{_JSON_SCHEMA_ES}""",
        json_mode=True,
    )


VARIANTS: dict[str, Prompt] = {
    p.variant: p for p in (_v0(), _v1(), _v2(), _v3(), _v4(), _v5(),
                           _v6())
}


# ------------------------------------------------------------------- parsing


_NUMBERING = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def parse_candidates(text: str, limit: int = N_CANDIDATES
                     ) -> tuple[list[str], str]:
    """Pull the candidate list out of a response. Returns (candidates, format).

    `format` is one of "json", "lineas" or "vacio", and it is reported rather
    than hidden: how a model failed matters. A model that reasons well but
    ignores the JSON contract is fixable with constrained decoding; one that
    returns three paraphrases of a wrong guess is not.

    Tolerant on purpose - code fences, numbering, stray prose - because the
    experiment is measuring the quality of the expansion, and a parser that
    gives up on a leading "Claro, aquí tienes:" would score that as a content
    failure when it is a formatting one.
    """
    if not text or not text.strip():
        return [], "vacio"

    cleaned = _FENCE.sub("", text).strip()

    # JSON first, including a bare array and JSON buried in prose.
    for blob in _json_candidates(cleaned):
        try:
            data = json.loads(blob)
        except (ValueError, TypeError):
            continue
        items = None
        if isinstance(data, dict):
            for key in ("candidatos", "candidates", "opciones", "options"):
                if isinstance(data.get(key), list):
                    items = data[key]
                    break
        elif isinstance(data, list):
            items = data
        if items is not None:
            out = _clean([str(i) for i in items if isinstance(i, (str, int, float))])
            if out:
                return out[:limit], "json"

    lines = [_NUMBERING.sub("", ln) for ln in cleaned.splitlines()]
    out = _clean(lines)
    return (out[:limit], "lineas") if out else ([], "vacio")


def _json_candidates(text: str):
    """The whole string, then the outermost {...} and [...] found inside it."""
    yield text
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if 0 <= start < end:
            yield text[start:end + 1]


_PREAMBLE = re.compile(
    r"^(claro|por supuesto|aqu[íi]|aca|acá|estas|estás|opciones|candidatos|"
    r"respuesta|entrada|salida)\b.{0,40}[:：]\s*$",
    re.IGNORECASE)


def _clean(items: list[str]) -> list[str]:
    """Drop empties, quotes, preambles and duplicates, preserving order."""
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        s = str(item).strip().strip('"').strip("'").strip()
        if not s or _PREAMBLE.match(s):
            continue
        key = normalise(s)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


_PUNCT = re.compile(r"[^\wáéíóúüñ ]+", re.UNICODE)
_SPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    The comparison key used by every metric. Accented vowels and ñ survive,
    because in Spanish they are letters and not decoration - stripping them
    would make "años" and "anos" the same string, which is a different word.
    """
    return _SPACE.sub(" ", _PUNCT.sub(" ", str(text).lower())).strip()
