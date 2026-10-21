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


_CUERPO_ES = """Qué puede hacer sola y qué no:

La ELA le paralizó el cuerpo. No puede moverse, incorporarse, acomodarse, darse vuelta ni alcanzar nada. Para cualquier cosa física depende de quien la cuida.

Por eso, cuando nombra una parte del cuerpo, una postura o un objeto de la habitación, casi siempre está PIDIENDO QUE ALGUIEN LO HAGA, no contando lo que va a hacer ella:

MOVER BRAZO es "quiero que me muevan el brazo" o "movéme el brazo", no "quiero mover el brazo".
SENTAR es "quiero que me sienten" o "sentáme", no "me voy a sentar".
ALMOHADA es "quiero la almohada" o "acomodáme la almohada".
NO MOVER es "no me muevas", no "no quiero moverme".

Eso no contradice hablar en primera persona: sigue siendo ella la que habla, pero de algo que otro tiene que hacer."""


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


_EXTRA_V7_ES = """Además, dos cosas más:

10. Lo que la entrada nombra es PARA la persona que habla. AGUA es "quiero agua", nunca "tomá agua": eso sería darle una orden a quien la cuida en vez de pedirle algo. Usá el imperativo solo cuando la entrada pida una acción sobre el entorno -una puerta, una luz, una ventana-, nunca sobre el cuerpo o las necesidades de la persona.
11. Devolvé siempre 3 opciones que digan cosas DISTINTAS, no la misma frase reescrita tres veces. Menos de 3 solo si la entrada está incompleta (regla 7).

Un ejemplo más, del caso de la regla 10:

entrada: MENOS DOLOR
{"candidatos": ["quiero menos dolor", "el dolor bajó", "menos dolor"]}"""


def _v7() -> Prompt:
    """+ para quién es lo que se pide, y la exigencia de tres opciones.

    Dos fallos vistos en uso real, no en el corpus:

    * **Registro invertido.** "POR FAVOR BAÑO" salía como "por favor, use el
      inodoro" y "POR FAVOR AGUA" como "por favor, tome agua": órdenes a quien
      cuida en vez de pedidos de la persona. La regla 4 ya decía "primera
      persona" y no alcanzaba, y hay una explicación plausible: el ejemplo de
      LUZ enseña imperativos dirigidos a otro ("prendé la luz"), que ahí son
      correctos porque se trata del entorno. El modelo generaliza ese
      imperativo a las necesidades del cuerpo, donde invierte quién pide y
      quién recibe. La regla 10 marca esa frontera en vez de prohibir el
      imperativo, que perdería el caso de la luz.

    * **Una sola opción.** "ME SIENTO MAL" devolvía un único candidato. Sin
      elección el módulo no aporta nada: el tablero pelado ya podía decir eso.
      La regla 11 pide tres distintas y ata el caso de una sola a la entrada
      incompleta, que es el único donde corresponde.

    El ejemplo es MENOS DOLOR y muestra tres cosas a la vez: primera persona,
    lecturas realmente distintas -un pedido y un reporte, no dos formas de
    decir lo mismo- y cero invención. La primera versión de este ejemplo usaba
    "es hora de mi medicación", que **inventa un horario y un posesivo**: un
    ejemplo que viola la regla 1 mientras pretende reforzar la 10. La medición
    lo detectó -la invención subió de 2 % a 5 %- antes de que llegara a nadie.

    Todo se **agrega al final** del prompt de v6 en vez de intercalarse, y eso
    no es comodidad: la escalera de variantes existe para poder atribuir cada
    cambio de métrica a un agregado concreto, y eso solo vale si cada prompt
    contiene literalmente al anterior. Intercalar las reglas donde "quedaban
    mejor" rompía esa propiedad, y `smoke.py` lo detectó.

    **Medido, y NO mejora.** Con filtro, contra v6 sobre qwen2.5:3b: recall
    85 % contra 88 %, seguridad 95 % contra 100 %, invención 2 % contra 0 %. Sin
    filtro la invención sube de 2 % a 8 %.

    El resultado es consistente con lo que la ablación original ya había
    medido: agregar reglas en prosa **baja** el recall 15 puntos mientras
    agregar ejemplos lo sube 31. Los modelos chicos no absorben más
    instrucciones; se ahogan. Se conserva la variante porque un resultado
    negativo medido vale más que la intuición que lo motivó.
    """
    base = _v6()
    return Prompt(
        variant="v7-registro",
        adds="para quién se pide + exigir 3 opciones distintas",
        system=base.system + chr(10) * 2 + _EXTRA_V7_ES,
        json_mode=True,
    )


_EXTRA_V8_ES = """Dos ejemplos más:

entrada: MENOS DOLOR
{"candidatos": ["quiero menos dolor", "el dolor bajó", "menos dolor"]}

entrada: POR FAVOR MEDICINA
{"candidatos": ["por favor, quiero la medicina", "por favor, necesito la medicina", "por favor, medicina"]}"""


def _v8() -> Prompt:
    """+ dos ejemplos más, y NINGUNA regla nueva.

    Es la intervención que la evidencia del propio módulo respalda. La ablación
    original ya había medido que agregar reglas en prosa **baja** el recall 15
    puntos mientras agregar ejemplos lo sube 31, y v7 lo confirmó por las malas:
    sus dos reglas nuevas dejaron el recall en 85 % contra 88 % y subieron la
    invención de 2 % a 8 %.

    Así que acá se enseñan las mismas dos cosas **mostrándolas**:

    * `MENOS DOLOR` cubre el registro y la diversidad a la vez: primera persona,
      y dos lecturas realmente distintas -un pedido y un reporte- en vez de la
      misma frase escrita de tres formas.
    * `POR FAVOR MEDICINA` es el contraejemplo directo del fallo visto en uso
      real, donde "POR FAVOR AGUA" salía como "por favor, tome agua": una orden
      a quien cuida en lugar de un pedido de la persona.

    Ninguno de los dos inventa nada, que es donde v7 se tropezó: su ejemplo
    decía "es hora de mi medicación" e inventaba un horario y un posesivo
    mientras pretendía reforzar la regla de no inventar.

    **Medido, y tampoco mejora.** Con filtro, contra v6 sobre qwen2.5:3b:
    empata en recall (88 %), seguridad (100 %) e invención (0 %), pero baja la
    diversidad de 0.56 a 0.45 y las opciones de 3.1 a 2.9. O sea que empeora
    justo donde más duele: menos opciones distintas es exactamente la queja que
    motivó todo esto.

    Se conserva como resultado negativo. Dos intentos bien motivados -uno por
    reglas, otro por ejemplos- y ninguno le gana a v6, que sigue siendo la
    única configuración que pasa todas las puertas.
    """
    base = _v6()
    return Prompt(
        variant="v8-ejemplos",
        adds="2 ejemplos más (registro y diversidad), sin reglas nuevas",
        system=base.system + chr(10) * 2 + _EXTRA_V8_ES,
        json_mode=True,
    )


_SITUACION_ES = """La situación, para que entiendas qué estás haciendo:

Una persona con ELA no puede hablar ni escribir. Mira un tablero en una pantalla y cada celda que elige le cuesta alrededor de un segundo de mirada sostenida. Lo que te llega es TODO lo que alcanzó a decir, junto: es UN SOLO MENSAJE, no una lista de palabras sueltas. "POR FAVOR AGUA" es una frase, no dos cosas.

La opción que elija se va a decir EN VOZ ALTA por un sintetizador, ahí mismo, a quien esté en la habitación: casi siempre quien la cuida o alguien de su familia. Esa persona va a actuar según lo que escuche.

Si tu frase dice algo que ella no quiso decir, corregirlo le cuesta otro minuto de mirada. Por eso una opción fiel y sosa siempre le gana a una elegante que agregó algo."""


def _v9() -> Prompt:
    """+ la situación: qué pasa con lo que sale, y quién lo escucha.

    El prompt decía qué **es** el sistema -"módulo de expansión de un
    dispositivo de comunicación aumentativa"- y nada de lo que **pasa** con su
    salida: que se dice en voz alta, que hay alguien escuchando que va a actuar,
    que corregir un malentendido cuesta otro minuto de mirada, y sobre todo que
    lo que llega es un mensaje entero y no una lista de palabras.

    Ese último punto sale de un fallo concreto y feo: qwen2.5:7b contestaba
    "POR FAVOR AGUA" con `["por favor agua", "agua", "por favor"]`, o sea que
    devolvía los pedazos de la entrada como si fueran tres mensajes distintos.
    No estaba inventando ni perdiendo la polaridad: estaba leyendo la entrada
    como una lista, y ninguna regla del prompt decía que no lo fuera.

    Es información y no una regla más, que es la distinción que la ablación de
    este módulo ya había medido: el glosario -información- subió el recall, y
    las reglas en prosa de v7 lo bajaron.
    """
    base = _v6()
    cabecera = "Sos el módulo de expansión de un dispositivo de comunicación aumentativa para una persona con ELA."
    return Prompt(
        variant="v9-situacion",
        adds="la situación de uso: se dice en voz alta y alguien va a actuar",
        system=base.system.replace(
            cabecera, cabecera + chr(10) * 2 + _SITUACION_ES, 1),
        json_mode=True,
    )


def _v10() -> Prompt:
    """+ qué puede y qué no puede hacer con su cuerpo.

    Sale de una falla medida y concentrada: sobre los 133 ítems, la etiqueta
    `postura` daba **1 acierto de 12** con qwen2.5:7b, la peor del corpus por
    lejos. Mirando las salidas, el patrón era siempre el mismo:

        MOVER BRAZO -> "quiero mover el brazo"   (esperado: "que me muevan")
        SENTAR      -> "me quiero sentar"        (esperado: "que me sienten")
        NO MOVER    -> "no quiero moverme"       (esperado: "no me muevas")

    El modelo no estaba inventando ni perdiendo polaridad: producía primera
    persona ACTIVA, que es exactamente lo que la regla 4 le pide. El problema
    es que para alguien que no puede mover el brazo, "quiero mover el brazo"
    no es una frase útil - necesita que OTRO se lo mueva, y nada en el prompt
    decía que no pudiera hacerlo ella.

    Va como INFORMACIÓN y no como regla nueve, que es la distinción que la
    ablación de este módulo ya midió dos veces: el glosario -información- subió
    el recall, y las reglas en prosa de v7 lo bajaron. Se le dice qué puede y
    qué no puede hacer su cuerpo, y se deja que el modelo saque la conclusión.
    """
    base = _v4()
    return Prompt(
        variant="v10-cuerpo",
        adds="qué puede y qué no puede hacer con su cuerpo",
        system=base.system.replace(
            _EXAMPLES_ES, _CUERPO_ES + chr(10) * 2 + _EXAMPLES_ES, 1),
        json_mode=True,
    )


def _v11() -> Prompt:
    """v10 sobre la base del glosario, para saber si se suman o se pisan.

    v10 cuelga de v4 y el glosario de v6. Si las dos ayudan por separado pero
    juntas no suman, es que estaban arreglando el mismo fallo por caminos
    distintos, y eso es tan informativo como la mejora.
    """
    base = _v6()
    return Prompt(
        variant="v11-cuerpo-glosario",
        adds="cuerpo + glosario",
        system=base.system.replace(
            _EXAMPLES_ES, _CUERPO_ES + chr(10) * 2 + _EXAMPLES_ES, 1),
        json_mode=True,
    )


VARIANTS: dict[str, Prompt] = {
    p.variant: p for p in (_v0(), _v1(), _v2(), _v3(), _v4(), _v5(),
                           _v6(), _v7(), _v8(), _v9(),
                           _v10(), _v11())
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
