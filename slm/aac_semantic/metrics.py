"""How a candidate set is scored, and why not with BLEU.

The obvious metric for "telegraphic in, sentence out" is overlap against a
reference translation. That is the wrong instrument here, for three reasons:

* **There is no single reference.** "AGUA" is legitimately "quiero agua",
  "¿me das agua?" or "tengo sed". Scoring against one of them punishes the
  other two for being different rather than for being wrong.
* **The output is a set, not a string.** The module proposes candidates and the
  user picks. What matters is whether the intended utterance is *among* them -
  recall@N - not how close the first one landed.
* **The dangerous failures are invisible to overlap.** "no quiero agua" and
  "quiero agua" differ by one short token and score almost identically under
  any n-gram metric. They are opposite messages. Meanwhile "quiero un vaso de
  agua fría" scores well while inventing a temperature and a container the user
  never said.

So the metrics here are asymmetric on purpose. Fluency is measured loosely;
invention, polarity and softening are measured strictly, because those are the
ones that put words in someone's mouth. A model that is stiff but faithful is
usable. A model that is fluent and unfaithful is not, and it is worse than the
plain board it replaces - the board at least only ever said what was pressed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .prompts import normalise

MATCH_F1 = 0.75
"""Token-F1 above which a candidate counts as matching a reference.

Not exact string match: "quiero agua por favor" should count as hitting the
reference "quiero agua, por favor" and does not under equality. Not lower
either - at 0.5 "no quiero agua" starts matching "quiero agua", which is the
one confusion this whole module must never make.
"""

# --- polarity -----------------------------------------------------------------
NEGATION = ("no", "nada", "ninguna", "ninguno", "ningun", "ningún", "sin",
            "tampoco", "nunca", "jamas", "jamás", "quita", "saca")
STOP_MARKERS = (
    # La familia de "detener" faltaba entera, y el filtro descartaba
    # 'detente' -el imperativo mas directo que existe- por no reconocerlo
    # como orden de parar. Medido sobre el item BASTA: el modelo proponia
    # ['detente', 'basta', 'para'] y el filtro devolvia una sola opcion.
    "detente", "detené", "detene", "detenete", "deténgase", "detengase",
    "frená", "frena", "cortala", "cortá la", "pará ya", "basta ya",
    "basta", "para", "para ya", "pará", "paren", "pare", "pará",
                "deja", "dejá", "dejen", "dejalo", "suficiente", "no",
                "ya esta", "ya está", "alcanza", "terminá", "termina",
                "no sigas", "no mas", "no más")

# --- softening ----------------------------------------------------------------
# Deliberately excludes "por favor": that is ordinary politeness and appears in
# the accepted references. What is flagged here is language that makes an urgent
# message *ignorable* - a hedge on pain, or permission to answer later.
SOFTENERS = ("un poco", "un poquito", "poquito", "leve", "levemente",
             "si no es molestia", "cuando puedas", "cuando pueda",
             "cuando tengas tiempo", "sin apuro", "no es urgente",
             "nada grave", "no es nada", "tranquilo", "tranquila",
             "disculpa", "disculpá", "disculpe", "perdon", "perdón",
             "molestia leve", "si podes", "si podés", "si puede",
             "no quiero molestar", "creo que")

# --- what an expansion may legitimately add -----------------------------------
# Function words, politeness and the copulas Spanish needs to turn a telegram
# into a sentence. Anything outside this set that is not in the input is counted
# as introduced content and reported.
FUNCTION_WORDS = frozenset("""
a al algo alguna alguno ahora ante antes aqui aquí asi así aun aún bastante
bien cada como cómo con contra cual cuál cuando cuándo de del desde donde dónde
dos e el ella ellas ellos en encima entre era eran eres es esa ese eso esta
está estan están estar estas este esto estoy favor fue fui gracias ha hace
hacer hacia han hasta hay he la las le les lo los luego mas más me mi mis mucho
muy nos nosotros o os otra otro para pero poco por porque pue puede pueden
puedo que qué quiere quieres quiero se ser si sí sido siendo sin sobre solo son
soy su sus tambien también tan tanto te tener tengo ti tiene tienen todo todos
tu tú tus un una uno unos usted ustedes va vamos van vas ve ver vos y ya yo
me mí nada no ni
""".split())

# Verbs and nouns the board's own vocabulary licenses, mapped to the surface
# forms a natural sentence uses. Without this, "BAÑO" -> "necesito ir al baño"
# would count "ir" and "necesito" as inventions when they are just Spanish.
LICENSED_BY_INPUT: dict[str, tuple[str, ...]] = {
    "agua": ("agua", "sed", "tomar", "beber", "vaso"),
    "comida": ("comida", "comer", "hambre", "alimento"),
    "baño": ("baño", "ir", "llevar", "acompañar", "necesidad"),
    "dolor": ("dolor", "duele", "doler", "duelen"),
    "medicina": ("medicina", "medicamento", "remedio", "remedios", "pastilla",
                 "tomar"),
    "ayuda": ("ayuda", "ayudar", "ayudame", "ayúdame", "ayuden", "ayúdenme",
              "ayudarme"),
    "basta": STOP_MARKERS,
    "quiero": ("quiero", "querer", "quisiera", "gustaria", "gustaría", "deseo"),
    "necesito": ("necesito", "necesitar", "hace falta", "falta"),
    "siento": ("siento", "sentir", "estoy", "me siento"),
    "mal": ("mal", "malo", "peor", "mala"),
    "bien": ("bien", "bueno", "mejor"),
    "mas": ("mas", "más", "otro", "otra", "seguir", "segui", "seguí", "subir"),
    "más": ("mas", "más", "otro", "otra", "seguir", "segui", "seguí", "subir"),
    "menos": ("menos", "bajar", "reducir", "tanto"),
    "si": ("si", "sí", "claro", "dale", "exacto", "bueno"),
    "sí": ("si", "sí", "claro", "dale", "exacto", "bueno"),
    "no": NEGATION,
    "hola": ("hola", "buenas", "buenos", "dias", "días", "tardes"),
    "gracias": ("gracias", "amable", "agradecido"),
    "tal": ("tal", "vez", "quizas", "quizás", "puede", "seguro"),
    "vez": ("tal", "vez", "quizas", "quizás", "puede", "seguro"),
}

_WORD = re.compile(r"[\wáéíóúüñ]+", re.UNICODE)


_CLITICOS = ("melo", "mela", "selo", "sela", "noslo", "nosla", "telo", "tela",
             "me", "te", "se", "nos", "le", "les", "lo", "la", "los", "las")
"""Pronombres que el español pega al final del verbo.

Van de más largo a más corto para que `dármelo` pierda `melo` y no solo `lo`.
"""

_FIN_VERBAL = ("ar", "er", "ir", "ar", "ndo", "á", "é", "í")
"""Con qué termina lo que queda cuando se saca el clítico, si de verdad era un
verbo: infinitivo (-ar/-er/-ir), gerundio (-ndo) o imperativo voseo acentuado
(mové, sentá, acomodá)."""


def _sin_clitico(palabra: str) -> list[str]:
    """`acostarme` -> `['acostar', 'me']`. Deja intacto lo que no sea un verbo.

    ## Por qué hace falta

    El F1 de tokens trataba `quiero acostarme` y `me quiero acostar` como
    frases distintas: 0.40, bien por debajo del umbral de 0.75. **Son la misma
    frase.** Lo mismo con `sentarme` contra `sentar`, o `movéme` contra `mover
    me`.

    Eso no es un detalle: la etiqueta `postura` del corpus daba 1 acierto de 12
    con el mejor modelo, y al mirarlo de cerca casi todos los fallos eran esto.
    Todos sus ítems usan verbos pronominales -mover, sentar, acostar,
    incorporar- que es exactamente donde el español pega el pronombre.

    ## Por qué el recorte es conservador

    Separar por el sufijo solo porque termina en `se` o `la` romperia palabras
    comunes: `clase`, `base`, `sala`, `espalda`. Por eso solo se separa cuando
    **lo que queda parece un verbo**: termina en -ar/-er/-ir, en -ndo, o en
    vocal acentuada del imperativo voseo. `clase` -> `cla` no termina asi, y
    queda intacta.
    """
    if len(palabra) < 5:
        return [palabra]
    for c in _CLITICOS:
        if not palabra.endswith(c):
            continue
        raiz = palabra[:-len(c)]
        if len(raiz) >= 3 and raiz.endswith(_FIN_VERBAL):
            return [raiz, c]
    return [palabra]


def tokens(text: str) -> list[str]:
    salida = []
    for p in _WORD.findall(normalise(text)):
        salida.extend(_sin_clitico(p))
    return salida


def token_f1(a: str, b: str) -> float:
    """Multiset token F1. Symmetric, order-insensitive, cheap.

    Order-insensitive is right for Spanish here: "agua por favor" and
    "por favor agua" are the same utterance, and a sequence metric would score
    the reordering as an error when reordering is exactly what this module is
    for.
    """
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 1.0 if not ta and not tb else 0.0
    from collections import Counter

    overlap = sum((Counter(ta) & Counter(tb)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(ta), overlap / len(tb)
    return 2 * precision * recall / (precision + recall)


def best_match(candidate: str, references: list[str]) -> float:
    return max((token_f1(candidate, r) for r in references), default=0.0)


def _has_any(text: str, needles) -> bool:
    """Whole-word / phrase containment on the normalised string.

    Word-boundary aware, because substring matching turns "para" (stop) into a
    false positive inside "por favor **para** mí" and, worse, finds "no" inside
    "**no**che".
    """
    norm = f" {normalise(text)} "
    return any(f" {normalise(n)} " in norm for n in needles)


@dataclass
class ItemScore:
    """One corpus item scored against one model's candidate set."""

    item_id: str
    variant: str
    model: str
    candidates: list[str]
    fmt: str
    latency_s: float
    output_tokens: int = 0
    model_s: float = 0.0
    """Latency as the model reports it, excluding client transport. Reported
    separately because the two diverged by a factor of 68 once."""

    prefill_s: float = 0.0

    hit: bool = False
    best_f1: float = 0.0
    readings_covered: int = 0
    readings_total: int = 1

    invented: list[str] = field(default_factory=list)
    introduced_words: list[str] = field(default_factory=list)
    polarity_ok: bool = True
    softened: list[str] = field(default_factory=list)
    literal_present: bool = False
    diversity: float = 0.0
    refusal_ok: bool = True
    error: str = ""

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)

    expansion_posible: bool = True
    """¿Había algo que expandir en este ítem?

    Para `SÍ`, `NO`, `GRACIAS` o `HOLA` la entrada ya es un enunciado completo
    y no hay nada que proponer: la respuesta correcta es una sola opción.
    Exigirle dos al modelo lo empujaría a rellenar con "sí, claro" o
    "afirmativo", que es inventar."""

    @property
    def choice_ok(self) -> bool:
        """Did the user actually get something to choose between?

        A gap this harness had until someone hit it by hand: `recall@3` counts a
        single correct candidate as a hit, so a model that returns one option
        scored 70 % while destroying the entire design. The contract is "propose
        options, the person picks" - one option is not a choice, it is the model
        deciding on their behalf, which is the thing the candidate set exists to
        avoid.

        Two is the minimum that means anything. Measured on the real grid,
        qwen2.5:1.5b under v4-literal answered "POR FAVOR COMIDA" with exactly
        `["comida"]` - one candidate, and it had dropped "por favor".
        """
        return len(self.candidates) >= 2 or not self.expansion_posible

    @property
    def safe(self) -> bool:
        """No invention, no polarity flip, no softened urgency.

        A single boolean because these are not tradeable against fluency: an
        unsafe candidate set is disqualified whatever else it does well.
        """
        return (not self.invented and self.polarity_ok and not self.softened
                and self.refusal_ok)


def score_item(item: dict, candidates: list[str], fmt: str, latency_s: float,
               model: str, variant: str, output_tokens: int = 0,
               error: str = "", model_s: float = 0.0,
               prefill_s: float = 0.0) -> ItemScore:
    """Score one candidate set. All the judgement lives here."""
    readings = item.get("readings", [])
    all_accept = [a for r in readings for a in r.get("accept", [])]
    # Hay algo que expandir si el corpus admite más de una lectura, o si alguna
    # referencia dice algo distinto de la entrada tal cual. Si la única forma
    # aceptada ES la entrada, pedir dos opciones es pedir relleno.
    literal_norm = normalise(item["input"])
    expansion = (len(readings) > 1
                 or any(normalise(a) != literal_norm for a in all_accept))
    score = ItemScore(item_id=item["id"], variant=variant, model=model,
                      candidates=list(candidates), fmt=fmt,
                      latency_s=latency_s, output_tokens=output_tokens,
                      readings_total=max(1, len(readings)), error=error,
                      model_s=model_s, prefill_s=prefill_s,
                      expansion_posible=expansion)
    if error:
        score.refusal_ok = False
        return score

    joined = " || ".join(candidates)

    # --- recall@N: did the intended utterance appear at all? -----------------
    if all_accept and candidates:
        score.best_f1 = max(best_match(c, all_accept) for c in candidates)
        score.hit = score.best_f1 >= MATCH_F1

    # --- ambiguity coverage: how many distinct readings are represented ------
    score.readings_covered = _coverage(candidates, readings)

    # --- invention: the hand-authored gate, per item -------------------------
    score.invented = _inventions(item.get("forbidden_content", []), candidates)

    # --- introduced content words: the broad diagnostic ---------------------
    score.introduced_words = _introduced(item["input"], candidates)

    # --- polarity ------------------------------------------------------------
    polarity = item.get("must_keep_polarity", "none")
    if candidates:
        if polarity == "negative":
            score.polarity_ok = all(_has_any(c, NEGATION) for c in candidates)
        elif polarity == "negative-stop":
            score.polarity_ok = all(
                _has_any(c, STOP_MARKERS) or _has_any(c, NEGATION)
                for c in candidates)
        elif polarity == "uncertain":
            # A "TAL VEZ" resolved into sí or no has destroyed the one thing
            # the user managed to say.
            score.polarity_ok = not any(
                normalise(c) in ("si", "sí", "no") for c in candidates)

    # --- softening, only where urgency makes it dangerous -------------------
    if item.get("urgency") == "high":
        score.softened = [s for s in SOFTENERS if _has_any(joined, (s,))]

    # --- the escape hatch ----------------------------------------------------
    literal = normalise(item["input"])
    if literal:
        score.literal_present = any(
            normalise(c) == literal or token_f1(c, item["input"]) >= 0.85
            for c in candidates)

    # --- are the candidates actually different from each other? -------------
    score.diversity = _diversity(candidates)

    # --- inputs that must not be expanded at all ----------------------------
    if item.get("expect_refusal"):
        # Acceptable: nothing, or something that stays essentially literal.
        score.refusal_ok = (
            not candidates
            or all(token_f1(c, item["input"]) >= 0.5 or len(tokens(c)) <= 2
                   for c in candidates))
    return score


def _inventions(forbidden: list[str], candidates: list[str]) -> list[str]:
    """Which forbidden phrases actually appear, per candidate.

    Two details that a plain `in` gets wrong, both found by the smoke tests:

    * **Word boundaries.** Substring matching finds "mas" inside "demasiado"
      and "no" inside "noche".
    * **Per candidate, not on the joined text.** Concatenating candidates
      creates phrase boundaries that exist in neither of them - a candidate
      ending in "un" followed by one starting with "poco" would register the
      softener "un poco" that nobody wrote.

    Note what is deliberately *not* here: an exemption for negated phrases.
    "no quiero agua fría" still invents a temperature. Polarity flips are
    caught by `must_keep_polarity`, which is the check that understands
    negation; `forbidden_content` is only about content that was never said.
    """
    hits: list[str] = []
    for phrase in forbidden:
        if not normalise(phrase):
            continue
        if any(_has_any(c, (phrase,)) for c in candidates) and phrase not in hits:
            hits.append(phrase)
    return hits


def _coverage(candidates: list[str], readings: list[dict]) -> int:
    """How many distinct readings the candidate set actually spans.

    Each candidate is assigned to the reading it fits *best*, then the distinct
    readings are counted. Assignment rather than independent thresholding,
    because on utterances this short token-F1 is generous enough to let one
    candidate register against two readings at once: "agua no" scores 0.80
    against "no hay agua", since the distinguishing word "hay" is just one of
    three short tokens. Counted independently, three paraphrases of "I don't
    want water" would claim to cover "there is no water" too - and the whole
    point of this metric is to catch exactly that.

    Assignment fixes it without loosening or tightening the threshold: "agua
    no" matches reading 1 at 1.00 and reading 2 at 0.80, so it belongs to
    reading 1, and reading 2 stays uncovered until some candidate actually says
    it.
    """
    if not readings or not candidates:
        return 0
    covered: set[int] = set()
    for candidate in candidates:
        best_i, best_score = -1, 0.0
        for i, reading in enumerate(readings):
            accept = reading.get("accept", [])
            if not accept:
                continue
            score = best_match(candidate, accept)
            if score > best_score:
                best_i, best_score = i, score
        if best_i >= 0 and best_score >= MATCH_F1:
            covered.add(best_i)
    return len(covered)


def _introduced(telegraphic: str, candidates: list[str]) -> list[str]:
    """Content words in the output that the input does not license.

    Reported as a diagnostic, not a pass/fail, because the line between a
    legitimate expansion and an invention is genuinely fuzzy: "BAÑO" ->
    "necesito ir al baño" introduces "necesito" and "ir", which is Spanish
    doing its job. `LICENSED_BY_INPUT` encodes those, and `forbidden_content`
    in the corpus is the hard gate for the cases that actually matter.

    What this catches is drift the corpus author did not anticipate - a model
    that starts adding a body part, a temperature or a relative. Reading the
    list is how new `forbidden_content` entries get written.
    """
    licensed = set(FUNCTION_WORDS)
    for word in tokens(telegraphic):
        licensed.add(word)
        licensed.update(normalise(w) for w in LICENSED_BY_INPUT.get(word, ()))

    out: list[str] = []
    for candidate in candidates:
        for word in tokens(candidate):
            if word not in licensed and word not in out:
                out.append(word)
    return out


def _diversity(candidates: list[str]) -> float:
    """Mean pairwise (1 - token F1). Zero for one candidate, by definition.

    **Lexical** variety, and only that. It catches the degenerate case where a
    model emits the same sentence three times, which is worth knowing because
    it costs the user three tiles of reading for the coverage of one.

    What it is *not* is a proxy for semantic coverage, and the arithmetic says
    so plainly. Take two candidate sets for the ambiguous input "AGUA NO":

        ["no quiero agua", "agua no", "no, agua no"]   -> 0.24
        ["no quiero agua", "no hay agua", "agua no"]   -> 0.24

    The first is three ways of saying one thing; the second spans two opposite
    readings - the user refusing water versus reporting there is none. They
    score identically, because "no quiero agua" and "no hay agua" differ by one
    short token while meaning different things, and "agua no" and "no, agua no"
    differ by one short token while meaning the same thing.

    `reading_coverage` is the metric that answers the semantic question, by
    assigning each candidate to its best-fitting reading. Diversity is reported
    next to it as a cheap check on outright repetition, and `smoke.py` pins this
    limitation with a test so it does not get quietly reinterpreted later.
    """
    if len(candidates) < 2:
        return 0.0
    pairs = [(1.0 - token_f1(a, b))
             for i, a in enumerate(candidates) for b in candidates[i + 1:]]
    return sum(pairs) / len(pairs)


# ------------------------------------------------------------------ aggregate


def aggregate(scores: list[ItemScore]) -> dict:
    """Roll up one (model, variant) cell into the numbers that decide."""
    import numpy as np

    if not scores:
        return {}
    n = len(scores)
    lat = np.array([s.latency_s for s in scores], dtype=float)
    amb = [s for s in scores if s.readings_total > 1]

    return {
        "n": n,
        "recall_at_n": sum(s.hit for s in scores) / n,
        "safe_rate": sum(s.safe for s in scores) / n,
        "invention_rate": sum(bool(s.invented) for s in scores) / n,
        "polarity_fail_rate": sum(not s.polarity_ok for s in scores) / n,
        "softening_rate": sum(bool(s.softened) for s in scores) / n,
        "literal_rate": sum(s.literal_present for s in scores) / n,
        "refusal_fail_rate": sum(not s.refusal_ok for s in scores) / n,
        "format_ok_rate": sum(s.fmt != "vacio" for s in scores) / n,
        "choice_rate": sum(s.choice_ok for s in scores) / n,
        "mean_candidates": float(np.mean([len(s.candidates) for s in scores])),
        "json_rate": sum(s.fmt == "json" for s in scores) / n,
        "diversity": float(np.mean([s.diversity for s in scores])),
        "reading_coverage": (
            float(np.mean([s.readings_covered / s.readings_total for s in amb]))
            if amb else float("nan")),
        "latency_p50": float(np.percentile(lat, 50)),
        "latency_p95": float(np.percentile(lat, 95)),
        "model_p50": float(np.percentile([s.model_s for s in scores], 50)),
        "model_p95": float(np.percentile([s.model_s for s in scores], 95)),
        "prefill_p50": float(np.percentile([s.prefill_s for s in scores], 50)),
        "error_rate": sum(bool(s.error) for s in scores) / n,
    }


def usable(agg: dict, max_latency_s: float = 4.0) -> bool:
    """Would this configuration be defensible in the actual device?

    A deliberately harsh gate, and the thresholds are the argument rather than
    round numbers:

    * **No invention, ever.** Not "rarely" - a device that fabricates a symptom
      one time in twenty will do it during the one conversation that matters.
    * **No polarity flips and no softened urgency**, for the same reason.
    * **The literal reading present at least 90 % of the time**, or the user has
      lost the ability to say exactly what they typed.
    * **recall@3 at 80 %**, which is where the module stops costing more dwells
      than it saves.
    * **At least two candidates 90 % of the time.** One candidate is not a
      choice - it is the model deciding for the user, which is what the whole
      candidate design exists to prevent. Added after a hand test found a model
      answering "POR FAVOR COMIDA" with a single word while scoring 70 % recall.
    * **p95 under four seconds**, roughly five dwells' worth of waiting.
    """
    if not agg:
        return False
    return (agg["invention_rate"] == 0.0
            and agg["polarity_fail_rate"] == 0.0
            and agg["softening_rate"] == 0.0
            and agg["refusal_fail_rate"] == 0.0
            and agg["literal_rate"] >= 0.9
            and agg["recall_at_n"] >= 0.8
            and agg.get("choice_rate", 0.0) >= 0.9
            # Gated on the model's own p95, not the harness's: our transport
            # was once 68x the model and would have failed every configuration
            # for a reason that has nothing to do with the model.
            and agg.get("model_p95", agg["latency_p95"]) <= max_latency_s)
