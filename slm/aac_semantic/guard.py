"""Un filtro determinista entre el modelo y la persona.

La medición dice que la única puerta que ningún arreglo pasa es la polaridad:
entre 5 % y 15 % de los conjuntos de candidatos pierden la negación o la orden
de detenerse. El mejor arreglo -qwen2.5:3b con v4-literal, 88 % de recall- falla
el 12 %.

Eso **no se arregla con más prompting**, y hay una razón de fondo: un modelo
generativo puede producir cualquier cadena, así que cualquier restricción
expresada en el prompt es una tendencia, no una garantía. Para una propiedad
donde el fallo tiene consecuencias -alguien sigue haciendo algo que causa dolor
porque el "basta" se diluyó- una tendencia no alcanza.

Un filtro sí garantiza. Si la entrada niega, todo candidato que no niegue se
descarta antes de llegar a la pantalla. La polaridad pasa a 0 % **por
construcción**, no por entrenamiento, y el resultado es verificable leyendo
treinta líneas de código en vez de confiando en una tabla.

El precio, que es real y se mide: el filtro puede dejar menos de dos candidatos,
y ahí el usuario pierde la elección. Cuando eso pasa cae a la lectura literal,
que es siempre segura. `rescore --guard` mide exactamente cuánto cuesta.

Lo que este módulo deliberadamente **no** hace es arreglar el texto. No reescribe
un candidato malo en uno bueno: eso sería el filtro poniendo palabras, que es lo
mismo que se le critica al modelo. Solo descarta.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .metrics import NEGATION, STOP_MARKERS, _has_any, token_f1
from .prompts import normalise


@dataclass
class GuardResult:
    """Qué sobrevivió al filtro y por qué se cayó el resto."""

    candidates: list[str]
    dropped: list[tuple[str, str]] = field(default_factory=list)
    fell_back: bool = False
    """True cuando el filtro se quedó sin candidatos y tuvo que usar el literal.
    Es el costo del filtro, y se reporta en vez de esconderse."""


# Los sustantivos de contenido del tablero. Cerrados, pocos, y confundirlos es
# el peor fallo posible del módulo: pedir agua cuando la persona necesita el
# baño manda a alguien a hacer lo incorrecto con total confianza.
#
# Medido: qwen2.5:3b contestó "POR FAVOR BAÑO" con "por favor, tome agua". El
# filtro de polaridad lo dejaba pasar entero, porque la polaridad estaba bien -
# lo que estaba mal era el sustantivo.
BOARD_NOUNS: dict[str, tuple[str, ...]] = {
    "agua": ("agua", "sed", "beber", "tomar agua"),
    "comida": ("comida", "comer", "hambre", "almuerzo", "cena", "desayuno"),
    "baño": ("baño", "inodoro", "orinar", "hacer pis"),
    "dolor": ("dolor", "duele", "duelen"),
    "medicina": ("medicina", "medicamento", "remedio", "remedios", "pastilla"),
}


def foreign_nouns(telegraphic: str, candidate: str) -> list[str]:
    """Sustantivos del tablero que el candidato dice y la entrada no.

    Una garantía dura sobre un vocabulario cerrado: si la entrada dice BAÑO, un
    candidato que menciona agua o comida está hablando de otra cosa. No hay
    lectura razonable en la que eso sea una expansión.

    Deliberadamente limitado a estos cinco. Un chequeo general de "toda palabra
    que la entrada no licencie" sería demasiado frágil - descartaría giros
    legítimos del español que nadie anticipó, como "¿me acompañás?" - y el
    precio de sobre-filtrar es quitarle opciones a la persona. Estos cinco son
    inequívocos y son los que importan.
    """
    entrada = normalise(telegraphic)
    texto = normalise(candidate)
    intrusos = []
    for noun, forms in BOARD_NOUNS.items():
        # La entrada autoriza el sustantivo si menciona CUALQUIERA de sus
        # formas, no solo la del tablero. Antes se miraba únicamente la clave, y
        # eso descartaba expansiones correctas: para la entrada `NO HAMBRE`
        # -deletreada, porque HAMBRE no está en el tablero- el candidato "no
        # tengo hambre" se caía por "hablar de comida". Quedaba solo el literal,
        # o sea que deletrear una palabra le quitaba a la persona las opciones
        # que el módulo existe para darle.
        if noun in entrada or any(_has_any(entrada, (f,)) for f in forms):
            continue
        if any(_has_any(texto, (f,)) for f in forms):
            intrusos.append(noun)
    return intrusos


def _negates(text: str) -> bool:
    return _has_any(text, NEGATION)


def _stops(text: str) -> bool:
    return _has_any(text, STOP_MARKERS) or _has_any(text, NEGATION)


def needs_negation(telegraphic: str) -> bool:
    """La entrada niega, así que la salida tiene que negar.

    Se decide sobre la entrada, no sobre lo que el modelo entendió. Es lo que
    hace al filtro una garantía: no depende de que el modelo haya captado nada.
    """
    return _has_any(telegraphic, ("no",))


def needs_stop(telegraphic: str) -> bool:
    return _has_any(telegraphic, ("basta",))


def is_uncertain(telegraphic: str) -> bool:
    """La entrada expresa duda, así que la salida no puede resolverla.

    Dos formas, y la segunda la destapó el corpus ampliado: `TAL VEZ` es la
    palabra del tablero, pero alguien que duda también escribe `SI NO` -las dos
    respuestas seguidas-, y ahí el filtro no hacía nada. Los candidatos
    resolvían la duda a "no" y eso pasaba como válido.

    Es el peor tipo de fallo de este módulo: la persona logró comunicar
    exactamente una cosa -que no está segura- y el sistema la reemplaza por una
    decisión que ella no tomó. Y no es raro: dudar es lo más humano que puede
    pasar en una conversación sobre el propio cuerpo.
    """
    t = normalise(telegraphic)
    if "tal vez" in t:
        return True
    tokens = t.split()
    return "si" in tokens and "no" in tokens


def apply(telegraphic: str, candidates: list[str],
          min_candidates: int = 2) -> GuardResult:
    """Filtra un conjunto de candidatos. Nunca reescribe, solo descarta.

    El orden importa: primero se descarta lo que viola la polaridad, después se
    garantiza el literal, y solo al final se mira si quedó una elección. Al
    revés, un literal agregado podría ser descartado por su propia regla - y el
    literal es, por definición, lo que la persona escribió, así que nunca puede
    ser una violación.
    """
    literal = telegraphic.strip().lower()
    kept: list[str] = []
    dropped: list[tuple[str, str]] = []

    require_stop = needs_stop(telegraphic)
    # Cuando la entrada pide detenerse, NO se exige además una negación léxica.
    # "NO BASTA DOLOR" contiene las dos palabras, pero expresan una sola cosa:
    # "basta, me duele" es una expansión correcta y no tiene partícula negativa.
    # Pedir las dos por separado descartaba referencias válidas del propio
    # corpus - y dejaba al filtro más estricto que la métrica, que ya usaba un
    # chequeo disyuntivo (marcador de parada O negación). Que discrepen es peor
    # que cualquiera de los dos criterios: significa que lo que se mide no es lo
    # que se aplica.
    require_neg = needs_negation(telegraphic) and not require_stop
    uncertain = is_uncertain(telegraphic)

    for candidate in candidates:
        text = candidate.strip()
        if not text:
            continue
        if require_stop and not _stops(text):
            dropped.append((text, "perdió la orden de detenerse"))
            continue
        if require_neg and not _negates(text):
            dropped.append((text, "perdió la negación"))
            continue
        if uncertain and normalise(text) in ("si", "sí", "no"):
            # Resolver una duda a sí o no destruye lo único que la persona
            # logró comunicar.
            dropped.append((text, "resolvió la duda a sí/no"))
            continue
        intrusos = foreign_nouns(telegraphic, text)
        if intrusos:
            dropped.append((text, "habla de " + ", ".join(intrusos)
                            + " y la entrada no lo dice"))
            continue
        kept.append(text)

    # El literal siempre disponible. Es lo que la persona escribió, así que no
    # puede violar su propia polaridad, y es la salida de escape si todo lo
    # generado se descartó.
    if literal and not any(normalise(c) == normalise(literal) for c in kept):
        kept.append(literal)

    result = GuardResult(candidates=kept, dropped=dropped)
    if len(kept) < min_candidates:
        # No se inventa nada para rellenar. Menos opciones es un costo honesto
        # del filtro, y `rescore --guard` lo mide.
        result.fell_back = True
    return result


def _is_pure_repetition(text: str) -> bool:
    """"mañana mañana mañana" - una sola palabra repetida y nada más.

    Nunca es una expansión válida. Como *entrada* la repetición es énfasis
    ("AGUA AGUA AGUA"), pero como *salida* es el modelo trabado. Solo se
    descarta cuando el candidato entero es una palabra repetida: "no, no" o
    "sí, sí" son enfáticos legítimos de dos tokens distintos por puntuación, y
    esos no caen acá porque el chequeo pide más de una repetición.
    """
    words = normalise(text).split()
    return len(words) > 1 and len(set(words)) == 1


def dedupe(candidates: list[str], threshold: float = 0.9) -> list[str]:
    """Quita candidatos que son la misma idea escrita distinto.

    Tres reglas, y la segunda existe porque la primera no alcanzaba:

    1. **F1 de tokens alto** - el mismo texto con otra puntuación.
    2. **Mismo conjunto de tokens** - `token_f1("mañana", "mañana mañana")` da
       0.667, por debajo de cualquier umbral razonable, porque el solapamiento
       de multiconjuntos crece con la repetición. Comparando *conjuntos* las dos
       son la misma idea y una se va. Medido: qwen2.5:1.5b contestó "MAÑANA" con
       `["mañana", "mañana mañana", "mañana mañana mañana"]` - tres celdas de
       lectura para una palabra.
    3. **Repetición pura** - se descarta directamente, no es español.

    Es cuestión de no gastarle al usuario una celda en algo que ya leyó, no de
    seguridad; por eso está separado de `apply`.
    """
    out: list[str] = []
    for candidate in candidates:
        if _is_pure_repetition(candidate):
            continue
        tokens_here = set(normalise(candidate).split())
        if any(token_f1(candidate, k) >= threshold
               or (tokens_here and tokens_here == set(normalise(k).split()))
               for k in out):
            continue
        out.append(candidate)
    return out


def guarded(telegraphic: str, candidates: list[str],
            min_candidates: int = 2) -> GuardResult:
    """El pipeline completo: deduplicar, después filtrar."""
    result = apply(telegraphic, dedupe(candidates), min_candidates)
    return result
