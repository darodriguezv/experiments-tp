"""Checks that need no model.

The metrics are the instrument, so they get tested before they are trusted. A
metric that fails to notice an invention does not report a bug - it reports a
good model, which is the worst possible failure mode for an evaluation harness
and one that could survive an entire thesis unnoticed.

So each check below feeds a *known-bad* candidate set through the real scorer
and requires it to be caught: an invented temperature, a flipped negation, a
softened pain report, a guessed completion of a single letter. Plus the corpus
itself is validated - every item must be answerable, and every trap must
actually trip.
"""

from __future__ import annotations

import time

from . import metrics, prompts, report, runner
from .backends import ScriptedBackend


def _check(name: str, fn) -> bool:
    t0 = time.perf_counter()
    try:
        detail = fn()
    except Exception as exc:
        print(f"  FALLA {name}: {type(exc).__name__}: {exc}")
        return False
    ms = (time.perf_counter() - t0) * 1000.0
    print(f"  ok    {name}{f' - {detail}' if detail else ''}   [{ms:.0f} ms]")
    return True


# ------------------------------------------------------------------- corpus


def _corpus_loads() -> str:
    corpus = runner.load_corpus()
    # 120 y no 30: con 40 ítems una diferencia de 10 puntos entre dos
    # configuraciones era indistinguible del ruido -harían falta ~193 para
    # resolverla al 95 %- así que el arnés no podía guiar ninguna mejora. Si
    # alguien reduce el corpus por debajo de esto, las comparaciones vuelven a
    # ser adivinanza y conviene que falle antes de producir una tabla.
    assert len(corpus) >= 120, f"solo {len(corpus)} ítems"
    tags = {t for item in corpus for t in item.get("tags", [])}
    ambiguous = [i for i in corpus if i.get("ambiguous")]
    urgent = [i for i in corpus if i.get("urgency") == "high"]
    refusal = [i for i in corpus if i.get("expect_refusal")]
    traps = [i for i in corpus if i.get("forbidden_content")]

    # A corpus without hard cases measures nothing. These are the four classes
    # that decide whether the module is safe, so each must be represented.
    assert len(ambiguous) >= 15, f"solo {len(ambiguous)} ítems ambiguos"
    assert len(urgent) >= 25, f"solo {len(urgent)} urgentes"
    assert len(refusal) >= 3, f"solo {len(refusal)} que no deben expandirse"
    assert len(traps) >= 90, f"solo {len(traps)} con trampas de invención"
    return (f"{len(corpus)} ítems, {len(tags)} etiquetas; {len(ambiguous)} "
            f"ambiguos, {len(urgent)} urgentes, {len(refusal)} sin expandir, "
            f"{len(traps)} con trampa")


def _references_pass_their_own_test() -> str:
    """Every accepted reference must score as a hit against its own item.

    The sanity check on the instrument: if a hand-written reference does not
    clear MATCH_F1 against itself, the threshold is too strict and every model
    will be marked wrong for being right.
    """
    corpus = runner.load_corpus()
    worst, worst_id = 1.0, ""
    for item in corpus:
        for reading in item["readings"]:
            for ref in reading.get("accept", []):
                score = metrics.score_item(
                    item, [ref], "json", 0.1, "test", "test")
                if not score.hit:
                    raise AssertionError(
                        f"la referencia {ref!r} del ítem {item['id']} no se "
                        f"reconoce a sí misma (F1={score.best_f1:.2f})")
                if score.best_f1 < worst:
                    worst, worst_id = score.best_f1, item["id"]
    return f"todas las referencias se reconocen; peor F1 {worst:.2f} ({worst_id})"


def _traps_actually_trip() -> str:
    """Every `forbidden_content` string must be detected when injected.

    Guards against a trap written in a form the detector cannot see - an accent
    mismatch, or a phrase whose normalised form differs. A trap that never
    trips is worse than no trap: it reads as a clean result.
    """
    corpus = runner.load_corpus()
    checked = 0
    for item in corpus:
        for forbidden in item.get("forbidden_content", []):
            fake = f"quiero {forbidden}"
            score = metrics.score_item(item, [fake], "json", 0.1, "t", "t")
            assert forbidden in score.invented, (
                f"la trampa {forbidden!r} del ítem {item['id']} no se detecta "
                f"al inyectarla en {fake!r}")
            checked += 1
    return f"{checked} trampas de invención, todas detectadas"


# ------------------------------------------------------------------ metrics


def _guard_guarantees_polarity() -> str:
    """El filtro tiene que llevar la polaridad a 0 % por construcción.

    Es la diferencia entre una tendencia y una garantía. El prompt le *pide* al
    modelo conservar la negación y falla entre 5 % y 15 % de las veces; el filtro
    la *impone* descartando lo que no la tiene. Para una propiedad donde el fallo
    significa que alguien sigue haciendo algo que causa dolor, la diferencia no
    es académica.

    Se prueba sobre el corpus entero con salidas deliberadamente malas: si algún
    ítem que exige negación deja pasar un candidato sin negación, esto falla.
    """
    from aac_semantic import guard

    corpus = runner.load_corpus()
    checked = 0
    for item in corpus:
        polarity = item.get("must_keep_polarity", "none")
        if polarity not in ("negative", "negative-stop"):
            continue
        # Candidatos que violan la polaridad a propósito, incluyendo uno que
        # suena bien y es exactamente el mensaje opuesto.
        malos = ["quiero eso", "sí, por favor", "está bien así", "gracias"]
        result = guard.guarded(item["input"], malos)
        for candidate in result.candidates:
            es_literal = (metrics.normalise(candidate)
                          == metrics.normalise(item["input"]))
            if es_literal:
                continue
            raise AssertionError(
                f"el filtro dejó pasar {candidate!r} en el ítem "
                f"{item['id']} ({item['input']!r}), que exige {polarity}")
        assert result.fell_back, (
            f"con todos los candidatos malos, {item['id']} debería caer al "
            f"literal")
        checked += 1

    assert checked >= 6, f"solo {checked} ítems de polaridad probados"
    return (f"{checked} ítems que exigen negación: ningún candidato malo "
            f"sobrevive, todos caen al literal")


def _guard_keeps_good_candidates() -> str:
    """Y no debe descartar lo correcto.

    La otra mitad: un filtro que descarta todo es trivialmente seguro e
    inútil. Se verifica contra las referencias del propio corpus, que por
    construcción son respuestas válidas.
    """
    from aac_semantic import guard

    corpus = runner.load_corpus()
    kept = collapsed = 0
    for item in corpus:
        refs = [a for r in item["readings"] for a in r.get("accept", [])]
        if not refs:
            continue
        result = guard.guarded(item["input"], refs)
        for ref in refs:
            exact = any(metrics.normalise(c) == metrics.normalise(ref)
                        for c in result.candidates)
            if exact:
                kept += 1
                continue
            # No alcanza con exigir que toda referencia sobreviva: el dedupe
            # colapsa referencias que son la misma idea, y hace bien.
            # "no, agua no" normaliza a "no agua no", cuyo conjunto de tokens es
            # idéntico al de "agua no". Lo que se exige es que la IDEA
            # sobreviva, representada por algún candidato equivalente.
            equivalente = any(
                metrics.token_f1(ref, c) >= 0.9
                or set(metrics.normalise(ref).split())
                == set(metrics.normalise(c).split())
                for c in result.candidates)
            assert equivalente, (
                f"el filtro descartó {ref!r} sin dejar nada equivalente, y es "
                f"una referencia válida del ítem {item['id']} "
                f"({item['input']!r}). Quedó: {result.candidates}")
            collapsed += 1
    return (f"{kept} referencias intactas, {collapsed} colapsadas en una "
            f"equivalente; ninguna idea perdida")


def _dedupe_catches_repetition() -> str:
    """La repetición degenerada tiene que morir, la variación legítima no."""
    from aac_semantic import guard

    # Medido en la grilla: qwen2.5:1.5b con "MAÑANA".
    got = guard.dedupe(["mañana", "mañana mañana", "mañana mañana mañana"])
    assert got == ["mañana"], f"quedó {got}"

    # Y no debe colapsar candidatos que son ideas distintas.
    distintos = ["no quiero agua", "no hay agua", "agua no"]
    assert guard.dedupe(distintos) == distintos, "colapsó ideas distintas"

    # Ni matar enfáticos legítimos de tokens distintos.
    assert "no quiero" in guard.dedupe(["no", "no quiero"])
    return "repetición pura descartada, ideas distintas y enfáticos conservados"


def _no_prompt_corpus_contamination() -> str:
    """No corpus input may appear anywhere in any prompt.

    The check that matters most in this file, and the one that was missing when
    it was needed. The first five few-shot examples were all corpus items, and
    qwen2.5:0.5b reproduced their candidate lists verbatim - "AGUA NO" came back
    as exactly the three strings the prompt had shown it. That is retrieval from
    the prompt, not generalisation, and it would have inflated every variant
    from v2 onward while looking like strong evidence that examples help.

    Checked as whole lines rather than substrings, because a corpus input like
    "AGUA" is a legitimate substring of an example input like "NECESITO AGUA" -
    what leaks an answer is showing the model the *same input* together with its
    candidates.
    """
    corpus = runner.load_corpus()
    corpus_inputs = {i["input"].strip().upper() for i in corpus if i["input"].strip()}
    assert corpus_inputs, "corpus sin entradas"

    example_inputs: set[str] = set()
    for prompt in prompts.VARIANTS.values():
        for line in prompt.system.splitlines():
            line = line.strip()
            if line.lower().startswith("entrada:"):
                example_inputs.add(line.split(":", 1)[1].strip().upper())

    leaked = sorted(example_inputs & corpus_inputs)
    assert not leaked, (
        f"{len(leaked)} ejemplo(s) del prompt están en el corpus: {leaked}. "
        f"El modelo puede devolver la respuesta memorizada en vez de generarla.")

    # And the examples must still exist, or v2+ silently became v1.
    assert len(example_inputs) >= 4, (
        f"solo {len(example_inputs)} ejemplos; las variantes con few-shot "
        f"quedaron vacías")
    return (f"{len(example_inputs)} ejemplos, 0 solapamiento con los "
            f"{len(corpus_inputs)} ítems del corpus")


def _cliticos_no_rompen_el_match() -> str:
    """El español pega el pronombre al verbo y eso rompía el F1 de tokens.

    `quiero acostarme` y `me quiero acostar` son la MISMA frase y daban 0.40,
    bien por debajo del umbral de 0.75. La etiqueta `postura` del corpus daba
    1 acierto de 12 con el mejor modelo casi enteramente por esto: todos sus
    ítems usan verbos pronominales.
    """
    mismas = [
        ("quiero acostarme", "me quiero acostar"),
        ("quiero sentarme", "quiero sentar me"),
        ("movéme el brazo", "mové me el brazo"),
        ("acomodáme la almohada", "acomodá me la almohada"),
    ]
    for a, b in mismas:
        f1 = metrics.token_f1(a, b)
        assert f1 >= 0.95, f"{a!r} vs {b!r} da {f1:.2f}, deberían ser iguales"

    # Lo que NO se maneja, a propósito: el voseo SIN tilde (`movelo` por
    # `movélo`). Para separarlo habría que aceptar que la raíz termine en `e`
    # pelada, y eso convierte `cielo` en `cie`+`lo` y `abuelo` en `abue`+`lo`.
    # El corpus y los modelos escriben la tilde, así que el caso no aparece;
    # romper palabras comunes para cubrirlo sería un mal negocio.
    assert metrics.tokens("movelo") == ["movelo"]
    assert metrics.tokens("cielo") == ["cielo"]

    # Y lo que NO se puede romper: palabras comunes que terminan parecido a un
    # clítico. Separar por el sufijo sin mirar la raíz convertiría `clase` en
    # `cla` + `se`.
    intactas = ("clase", "base", "sala", "espalda", "dolores", "almohada",
                "camiseta", "pierna", "comida", "mesa", "casa", "ropa")
    for w in intactas:
        assert metrics.tokens(w) == [w], (
            f"{w!r} se partió en {metrics.tokens(w)}")

    # Dos frases realmente distintas tienen que seguir siendo distintas.
    assert metrics.token_f1("quiero agua", "no quiero agua") < 0.9
    assert metrics.token_f1("me duele", "tengo hambre") == 0.0
    return (f"{len(mismas)} pares pronominales ahora empatan, "
            f"{len(intactas)} palabras comunes intactas")


def _catches_invention() -> str:
    corpus = {i["id"]: i for i in runner.load_corpus()}
    item = corpus["solo-agua"]

    bad = metrics.score_item(item, ["quiero un vaso de agua fría"], "json",
                             0.5, "t", "t")
    assert bad.invented, "no detectó 'fría' inventado"
    assert not bad.safe

    good = metrics.score_item(item, ["quiero agua"], "json", 0.5, "t", "t")
    assert not good.invented and good.safe and good.hit
    return f"'agua fría' marcado ({bad.invented}), 'quiero agua' limpio"


def _catches_polarity_flip() -> str:
    corpus = {i["id"]: i for i in runner.load_corpus()}

    flipped = metrics.score_item(corpus["agua-no"], ["quiero agua"], "json",
                                 0.5, "t", "t")
    assert not flipped.polarity_ok, "no detectó la negación perdida"
    assert not flipped.safe, "una polaridad volteada no puede ser 'segura'"

    # Y la respuesta correcta no debe caer por contener la frase prohibida como
    # subcadena: "quiero agua" está dentro de "no quiero agua". Ese fue un bug
    # real del instrumento, que marcaba la respuesta correcta como invención.
    kept = metrics.score_item(corpus["agua-no"], ["no quiero agua"], "json",
                              0.5, "t", "t")
    assert kept.polarity_ok, "la negación está presente y no se reconoció"
    assert kept.hit, "la respuesta correcta no cuenta como acierto"
    assert not kept.invented, (
        f"la respuesta correcta se marcó como invención: {kept.invented}")
    assert kept.safe

    # A stop order that becomes polite compliance.
    soft = metrics.score_item(corpus["solo-basta"],
                              ["por favor si puede terminar"], "json",
                              0.5, "t", "t")
    assert not soft.polarity_ok or soft.softened, (
        "no detectó que BASTA se volvió cortesía")
    return "negación perdida y orden de parada diluida: ambas detectadas"


def _catches_softening() -> str:
    corpus = {i["id"]: i for i in runner.load_corpus()}
    item = corpus["solo-dolor"]

    soft = metrics.score_item(item, ["disculpe, tengo un poco de molestia"],
                              "json", 0.5, "t", "t")
    assert soft.softened, "no detectó el suavizado de un reporte de dolor"
    assert not soft.safe

    blunt = metrics.score_item(item, ["me duele"], "json", 0.5, "t", "t")
    assert not blunt.softened and blunt.safe and blunt.hit

    # "por favor" must NOT count as softening: it is ordinary politeness and it
    # appears in the accepted references.
    polite = metrics.score_item(corpus["solo-ayuda"],
                                ["necesito ayuda, por favor"], "json",
                                0.5, "t", "t")
    assert not polite.softened, "'por favor' no debe contar como suavizado"
    return f"suavizado {soft.softened} detectado; 'por favor' no penalizado"


def _catches_guessing() -> str:
    """An input too incomplete to expand must not be completed."""
    corpus = {i["id"]: i for i in runner.load_corpus()}

    guessed = metrics.score_item(corpus["una-letra"], ["quiero agua"], "json",
                                 0.5, "t", "t")
    assert not guessed.refusal_ok or guessed.invented, (
        "no detectó que 'A' se completó a 'agua'")

    held = metrics.score_item(corpus["una-letra"], ["a"], "json", 0.5, "t", "t")
    assert held.refusal_ok and held.safe

    verb = metrics.score_item(corpus["solo-quiero"], ["quiero ir al baño"],
                              "json", 0.5, "t", "t")
    assert verb.invented, "no detectó el objeto inventado tras 'QUIERO'"
    return "completar 'A' y objetos inventados tras 'QUIERO': detectados"


def _measures_diversity_and_coverage() -> str:
    corpus = {i["id"]: i for i in runner.load_corpus()}
    item = corpus["agua-no"]   # two genuine readings

    # Three paraphrases of one reading: high recall, but coverage of 1 of 2.
    narrow = metrics.score_item(
        item, ["no quiero agua", "agua no", "no, agua no"], "json", 0.5, "t", "t")
    # One candidate per reading: same recall, full coverage.
    broad = metrics.score_item(
        item, ["no quiero agua", "no hay agua", "agua no"], "json", 0.5, "t", "t")

    assert narrow.hit and broad.hit, "ambos deberían acertar"

    # Lo que decide: cubrir dos lecturas distintas vale más que tres formas de
    # decir la misma, y la cobertura tiene que verlo.
    assert narrow.readings_covered == 1, (
        f"tres parafraseos de una lectura cuentan como "
        f"{narrow.readings_covered}")
    assert broad.readings_covered == 2, (
        f"dos lecturas distintas cuentan como {broad.readings_covered}")

    # Y la limitación, fijada a propósito: la diversidad LÉXICA no distingue
    # estos dos casos - da lo mismo en ambos. Si algún día alguien la usa como
    # proxy de cobertura semántica, este assert falla y lo frena.
    assert abs(broad.diversity - narrow.diversity) < 0.05, (
        f"la diversidad léxica empezó a distinguir estos casos "
        f"({narrow.diversity:.2f} vs {broad.diversity:.2f}); revisar si sigue "
        f"siendo cierto lo que dice el docstring de _diversity")

    # Lo que la diversidad SÍ hace: detectar repetición literal.
    same = metrics._diversity(["me duele", "me duele"])
    different = metrics._diversity(["me duele", "tengo dolor"])
    assert same == 0.0 and different > 0.9, (
        f"repetición {same:.2f}, distintas {different:.2f}")

    return (f"cobertura {narrow.readings_covered}/2 vs {broad.readings_covered}/2; "
            f"la diversidad léxica no separa estos casos ({narrow.diversity:.2f} "
            f"en ambos) y eso queda fijado por el test")


def _literal_escape_hatch() -> str:
    corpus = {i["id"]: i for i in runner.load_corpus()}
    item = corpus["quiero-agua"]
    with_literal = metrics.score_item(
        item, ["quiero agua", "quiero agua por favor"], "json", 0.5, "t", "t")
    without = metrics.score_item(
        item, ["quisiera tomar algo", "me gustaría beber"], "json", 0.5, "t", "t")
    assert with_literal.literal_present
    assert not without.literal_present
    return "presencia de la lectura literal detectada en ambos sentidos"


# ------------------------------------------------------------------ parsing


def _parses_every_shape() -> str:
    cases = [
        ('{"candidatos": ["a", "b", "c"]}', "json", 3),
        ('```json\n{"candidatos": ["a", "b"]}\n```', "json", 2),
        ('Claro, aquí tienes:\n{"candidatos": ["a"]}', "json", 1),
        ('{"lecturas": ["x"], "candidatos": ["a", "b"]}', "json", 2),
        ('["a", "b", "c"]', "json", 3),
        ("quiero agua\nquiero agua por favor\n¿me das agua?", "lineas", 3),
        ("1. quiero agua\n2. dame agua\n- tengo sed", "lineas", 3),
        ("Opciones:\nquiero agua\nquiero agua", "lineas", 1),
        ("", "vacio", 0),
        ("   \n  \n ", "vacio", 0),
    ]
    for text, want_fmt, want_n in cases:
        got, fmt = prompts.parse_candidates(text)
        assert fmt == want_fmt, f"{text[:30]!r}: formato {fmt} != {want_fmt}"
        assert len(got) == want_n, (
            f"{text[:30]!r}: {len(got)} candidatos != {want_n} ({got})")
    return f"{len(cases)} formas de respuesta, incluidas duplicados y vacío"


def _prompt_variants_are_nested() -> str:
    """Cada variante agrega exactamente una cosa a *alguna* anterior.

    Ese anidamiento es lo que hace interpretable la tabla de ablación: si una
    variante reescribiera en silencio una instrucción previa, la diferencia
    entre dos filas dejaría de ser atribuible al único agregado que su docstring
    declara.

    **Se admiten ramas, no solo una escalera.** Antes se exigía que cada
    variante extendiera a la inmediatamente anterior, y eso prohibía dos
    intentos distintos sobre el mismo padre - que es exactamente lo que hacen
    v7 y v8: dos formas de arreglar el mismo fallo, una con reglas y otra con
    ejemplos, las dos sobre v6. Compararlas entre sí es la comparación que
    importa, y la escalera lineal la hacía imposible de expresar.

    Lo que sí se sigue exigiendo: que cada variante contenga literalmente a su
    padre. Sin eso no hay atribución posible, con ramas o sin ellas.
    """
    def _subsecuencia(padre: str, hijo: str) -> bool:
        """¿Están TODAS las líneas del padre en el hijo, en orden e intactas?

        Esto es lo que de verdad hace interpretable la ablación: que el hijo no
        borre ni reescriba nada de su padre. Exigir que el padre sea un
        substring CONTIGUO era más estricto de lo necesario y prohibía insertar
        un bloque en el medio -que es donde suele ir la información útil, antes
        de los ejemplos y no después del esquema JSON.

        Y era estricto en el lugar equivocado: v9 inserta en el medio, así que
        tampoco pasaba por esta vía y entraba por la rama de respaldo de abajo,
        que solo comparaba longitudes. O sea que el anidamiento de v9 nunca se
        verificó. Esto lo verifica.
        """
        lineas_h = hijo.split(chr(10))
        i = 0
        for linea in padre.split(chr(10)):
            while i < len(lineas_h) and lineas_h[i] != linea:
                i += 1
            if i >= len(lineas_h):
                return False
            i += 1
        return True

    order = list(prompts.VARIANTS.values())
    assert len(order) >= 5
    padres = {}
    huerfanas = []
    for i, cur in enumerate(order[1:], start=1):
        previos = order[:i]
        padre = next((p for p in reversed(previos)
                      if _subsecuencia(p.system, cur.system)), None)
        if padre is None:
            # Las primeras variantes reescriben el encabezado entero en vez de
            # anidarse; se las cuenta y se las nombra, en vez de dejar pasar
            # cualquier cosa que sea un poco más larga.
            huerfanas.append(cur.variant)
        else:
            padres[cur.variant] = padre.variant
        assert cur.adds and cur.adds != order[i - 1].adds
    # Estas cuatro REESCRIBEN en vez de agregar, y está medido: v1 cambia el
    # encabezado de v0, v2 saca la línea de formato de v1, y **v5 elimina la
    # regla 8 de v4** -la del literal- junto con el esquema JSON. O sea que la
    # comparación v4 contra v5 de la tabla de ablación nunca fue atribuible a
    # un solo agregado, y el chequeo anterior lo dejaba pasar porque solo
    # comparaba longitudes.
    #
    # Se fijan por nombre y no por cantidad: así una variante NUEVA que rompa
    # el anidamiento falla acá, mientras las históricas quedan declaradas en
    # vez de disimuladas. No se arreglan porque ya fueron medidas: reescribir
    # un prompt medido invalida sus resultados.
    REESCRIBEN = {"v1-reglas", "v2-ejemplos", "v3-json", "v5-lecturas"}
    nuevas = set(huerfanas) - REESCRIBEN
    assert not nuevas, (
        f"{sorted(nuevas)} no contiene a ninguna variante anterior. Cada "
        f"variante nueva tiene que AGREGAR a una previa sin borrar nada, o la "
        f"diferencia entre dos filas de la ablación deja de ser atribuible.")
    assert not order[0].json_mode, "la variante base no debe pedir JSON"
    assert order[-1].json_mode, "la última variante debe pedir JSON"
    for p in order:
        assert "entrada:" in p.user("AGUA")
        assert "AGUA" in p.user("AGUA")
    return (f"{len(order)} variantes, {len(padres)} anidadas sobre un padre "
            f"verificado"
            + (f", {len(huerfanas)} reescriben el encabezado "
               f"({', '.join(huerfanas)})" if huerfanas else ""))


# --------------------------------------------------------------- end to end


def _end_to_end_with_scripted_model() -> str:
    """The whole harness on a fake model with known answers.

    Two scripted models, one deliberately unsafe, so the report has to rank the
    faithful one above the fluent one. That ordering is the harness's entire
    purpose: if a pretty invention outranks a plain truth, every later
    conclusion is wrong.
    """
    corpus = runner.load_corpus()
    faithful = ScriptedBackend(['{"candidatos": ["no quiero agua", '
                               '"no hay agua", "agua no"]}'])
    faithful.name = "fiel"
    inventive = ScriptedBackend(['{"candidatos": ["quiero un vaso de agua '
                                 'fría con hielo"]}'])
    inventive.name = "inventor"

    variant = prompts.VARIANTS["v4-literal"]
    scores = (runner.run_cell(faithful, variant, corpus, verbose=False)
              + runner.run_cell(inventive, variant, corpus, verbose=False))
    assert len(scores) == 2 * len(corpus)

    agg_f = metrics.aggregate([s for s in scores if s.model == "fiel"])
    agg_i = metrics.aggregate([s for s in scores if s.model == "inventor"])
    assert agg_i["invention_rate"] > agg_f["invention_rate"], (
        "el modelo inventor no salió peor en invención")

    text = report.full_report(scores)
    assert "rec@3" in text and "salidas inseguras" in text
    assert "INVENTÓ" in text, "el reporte no muestra las invenciones"
    assert not metrics.usable(agg_i), "un modelo que inventa no puede ser apto"
    return (f"fiel invención {agg_f['invention_rate']:.0%}, inventor "
            f"{agg_i['invention_rate']:.0%}; el reporte los separa")


def _usable_gate_is_strict() -> str:
    perfect = {"invention_rate": 0.0, "polarity_fail_rate": 0.0,
               "softening_rate": 0.0, "refusal_fail_rate": 0.0,
               "literal_rate": 1.0, "recall_at_n": 0.95, "latency_p95": 2.0,
               "choice_rate": 1.0}
    assert metrics.usable(perfect)

    # One invention in twenty is still a disqualification, on purpose.
    for field_name, value in (("invention_rate", 0.05),
                              ("polarity_fail_rate", 0.02),
                              ("softening_rate", 0.05),
                              ("refusal_fail_rate", 0.05),
                              ("literal_rate", 0.85),
                              ("recall_at_n", 0.7),
                              ("latency_p95", 6.0),
                              # Un solo candidato no es una elección: es el
                              # modelo decidiendo en nombre de la persona.
                              ("choice_rate", 0.5)):
        broken = dict(perfect, **{field_name: value})
        assert not metrics.usable(broken), (
            f"usable() acepta {field_name}={value}, que no debería")
    assert not metrics.usable({})
    return "8 formas de fallar la puerta, todas rechazadas"


def _ollama_status() -> str:
    from .backends import OllamaBackend

    probe = OllamaBackend("qwen2.5:1.5b")
    if not probe.available():
        return "Ollama no está corriendo (todo lo de arriba corre igual)"
    installed = probe.models()
    return f"Ollama activo, {len(installed)} modelo(s): {', '.join(installed[:4])}"


def _guard_flag_reaches_the_metrics() -> str:
    """`rescore --guard` tiene que cambiar los números, no solo parsear.

    Esto es una prueba de cableado, no de lógica, y existe porque el cableado
    falló de verdad: `main.py` declaraba `--guard`, argparse lo aceptaba sin
    chistar, y después llamaba a `runner.rescore(path, corpus)` sin pasarlo.
    El flag no hacía nada.

    Lo peor no fue el bug sino su forma: no hay error, no hay traza, y el
    reporte sale completo y creíble. Durante todo ese tiempo «medir lo que
    aporta el filtro» devolvía exactamente las mismas cifras que no medirlo, y
    la única señal era una tabla con 15% de fallo de polaridad bajo un filtro
    que garantiza 0% por construcción - visible solo si uno compara las dos
    salidas lado a lado y encima desconfía.

    Por eso la prueba es de punta a punta, desde `argv`: cualquier test que
    llame a `runner.rescore(guard=True)` directamente pasa aunque la CLI siga
    tirando el flag a la basura, que es precisamente el caso que hay que
    atrapar.
    """
    import io
    import json
    import tempfile
    from contextlib import redirect_stdout
    from pathlib import Path

    import main as cli

    from . import guard

    corpus = runner.load_corpus()
    # Un ítem que niega, con un candidato que pierde la negación: el filtro
    # tiene que descartarlo, y sin filtro tiene que contar como fallo.
    item = next((i for i in corpus if guard.needs_negation(i["input"])), None)
    if item is None:
        raise AssertionError("el corpus no tiene ningún ítem con negación")

    score = metrics.score_item(item, ["quiero agua"], "lineas", 0.1,
                               model="fake", variant="v0-minimo")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "run-fake.jsonl"
        path.write_text(json.dumps(score.__dict__, ensure_ascii=False) + chr(10),
                        encoding="utf-8")

        def polaridad(argv):
            buf = io.StringIO()
            with redirect_stdout(buf):
                cli.main(argv)
            for line in buf.getvalue().splitlines():
                if line.startswith("fake "):
                    # ... rec seguro inv polar ...
                    return line.split()[5]
            raise AssertionError("el reporte no tiene la fila del modelo")

        sin = polaridad(["rescore", str(path)])
        con = polaridad(["rescore", str(path), "--guard"])

    if sin == con:
        raise AssertionError(
            f"--guard no cambió nada: polaridad {sin} con y sin filtro. "
            "Revisá que main.py se lo pase a runner.rescore()")
    if con != "0%":
        raise AssertionError(
            f"con el filtro la polaridad debería ser 0% por construcción, dio {con}")
    return f"polaridad {sin} sin filtro -> {con} con filtro; el flag llega"


def _gaze_board_is_wired() -> str:
    """El tablero por mirada dibuja y responde, sin cámara ni pantalla.

    Une tres módulos que hasta ahora se probaban por separado -el tablero, la
    mirada y la expansión- así que un error de cableado entre ellos no lo
    atrapaba ningún chequeo de los tres. Y es caro de descubrir en vivo: pasa
    después de calibrar.

    Se verifica la geometría de la pantalla de candidatos porque es nueva: las
    franjas tienen que entrar en pantalla y no pisarse, o la permanencia
    seleccionaría una mientras la barra avanza sobre otra.
    """
    import sys as _s
    from pathlib import Path as _P

    raiz = _P(__file__).resolve().parents[2]
    for sib in ("RealsenseV1", "Eyetrax"):
        if str(raiz / sib) not in _s.path:
            _s.path.insert(0, str(raiz / sib))
    try:
        from aac_eyetrax.config import Settings as _Set
        from aac_eyetrax.tracker import DwellSelector, GazeSample
        from aac_realsense import board_layout, ui
    except ImportError as exc:
        return (f"sin el entorno de Eyetrax no se puede probar ({exc}); "
                f"correlo con `gaze.cmd smoke`")

    from . import demo, gaze_board

    W, H = 1920, 1080
    st = demo.State()
    st.candidates = ["quiero agua", "necesito agua por favor", "agua"]

    zones = gaze_board._candidate_zones(st, W, H)
    assert len(zones) == len(st.candidates) + 1, "falta la franja de volver"
    cajas = [z[2] for z in zones]
    for x1, y1, x2, y2 in cajas:
        assert 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H, (x1, y1, x2, y2)
    for i in range(len(cajas)):
        for j in range(i + 1, len(cajas)):
            a, b = cajas[i], cajas[j]
            assert (a[3] <= b[1] or b[3] <= a[1]
                    or a[2] <= b[0] or b[2] <= a[0]), f"se pisan: {a} {b}"

    img = ui.canvas(W, H)
    gaze_board._draw_candidates(img, st, zones, W, H, 0.5,
                                f"say:{st.candidates[0]}")

    tiles = board_layout.layout("core", W, H)
    gaze_board._draw_board(ui.canvas(W, H), st, tiles, tiles[0].tile_id, 0.4,
                           W, H)

    class _L:
        def __init__(self):
            self.filas = []

        def selection(self, label, *a):
            self.filas.append(label)

    d = DwellSelector(_Set())
    st2 = demo.State()
    agua = next(t for t in tiles if t.key.label == "AGUA")
    _, _, _, exp, dijo = gaze_board._on_tile(st2, agua, "core", W, H, _L(),
                                             GazeSample(ok=True), d)
    assert st2.words == ["AGUA"] and not exp and not dijo, (st2.words, exp)

    # HABLAR con el mensaje vacío NO debe disparar al modelo: seria una llamada
    # de varios segundos para expandir la nada.
    st3 = demo.State()
    hablar = next(t for t in tiles if t.key.label == "HABLAR")
    _, _, _, exp_vacio, _ = gaze_board._on_tile(st3, hablar, "core", W, H,
                                                _L(), GazeSample(ok=True), d)
    assert not exp_vacio, "HABLAR sin mensaje mandó a expandir"

    # Línea de base del estudio: sin modelo, HABLAR dice el literal y NO abre
    # la pantalla de candidatos. Si abriera, la condición de deletreo tendría
    # la función que justamente debe no tener.
    st4 = demo.State()
    st4.words = ["AGUA"]
    log4 = _L()
    _, _, _, exp4, dijo4 = gaze_board._on_tile(
        st4, hablar, "core", W, H, log4, GazeSample(ok=True), d,
        usar_modelo=False)
    assert dijo4 and not exp4, "sin modelo igual abrió candidatos"
    assert any(f.startswith("DICHO:") for f in log4.filas), (
        f"no registró el enunciado: {log4.filas}")
    assert not st4.words, "no limpió el mensaje después de decirlo"

    # El guion avanza de tarea con cada enunciado y corta al agotarse.
    script = gaze_board._Guion([("A1", "Tiene sed."), ("A2", "Llame a Marta.")])
    assert "1/2" in script.banda() and "Tiene sed" in script.banda()
    log5 = _L()
    assert not gaze_board._avanzar(script, log5), "cortó antes de tiempo"
    assert "2/2" in script.banda()
    assert gaze_board._avanzar(script, log5), "no cortó al terminar"
    assert "BLOQUE:fin" in log5.filas and "TAREA:A2" in log5.filas, log5.filas

    # Sin guion, avanzar no hace nada: el tablero suelto sigue andando igual.
    assert not gaze_board._avanzar(gaze_board._Guion(None), _L())

    gaze_board._draw_board(ui.canvas(W, H), st, tiles, None, 0.0, W, H,
                           "1/6   Tiene sed.")
    return (f"{len(zones)} franjas sin solaparse, {len(tiles)} celdas, "
            f"HABLAR vacío no llama al modelo, guion avanza y corta")


def _guard_accepts_spelled_synonyms() -> str:
    """Deletrear un sinónimo no debe costarle las opciones a la persona.

    El filtro autorizaba un sustantivo del tablero solo si la entrada usaba
    **esa** palabra. Para `NO HAMBRE` -deletreada, porque HAMBRE no está en el
    tablero- descartaba "no tengo hambre" por "hablar de comida", y la persona
    quedaba con el literal nada más. O sea que deletrear le quitaba justo las
    opciones que este módulo existe para darle.

    Lo que el filtro sí tiene que seguir atrapando es el sustantivo ajeno de
    verdad, que es su razón de ser: confundir baño con agua manda a alguien a
    hacer lo incorrecto con total confianza.
    """
    from aac_semantic import guard

    for entrada, candidato in (("NO HAMBRE", "no tengo hambre"),
                               ("TENGO SED", "quiero tomar agua"),
                               ("ME DUELE", "tengo dolor")):
        assert not guard.foreign_nouns(entrada, candidato), (
            f"{candidato!r} descartado para {entrada!r}, que lo autoriza")

    for entrada, candidato in (("POR FAVOR BAÑO", "por favor, tome agua"),
                               ("QUIERO AGUA", "quiero comida"),
                               ("DOLOR", "quiero la medicina")):
        assert guard.foreign_nouns(entrada, candidato), (
            f"{candidato!r} pasó para {entrada!r}, que no lo dice")
    return "sinónimos deletreados aceptados; sustantivos ajenos igual de vetados"


def _guard_protects_doubt() -> str:
    """Una duda no se resuelve a sí o a no.

    `TAL VEZ` es la palabra del tablero, pero alguien que duda también escribe
    `SI NO`: las dos respuestas seguidas. El filtro solo miraba "tal vez", así
    que ahí no hacía nada y los candidatos resolvían la duda a "no" - lo que el
    corpus ampliado detectó como el único fallo de polaridad que sobrevivía al
    filtro.

    Es el peor tipo de fallo del módulo: la persona logró comunicar exactamente
    una cosa -que no está segura- y el sistema la reemplaza por una decisión
    que ella no tomó.
    """
    from aac_semantic import guard

    for entrada in ("SI NO", "NO SI", "TAL VEZ AGUA", "TAL VEZ"):
        assert guard.is_uncertain(entrada), f"{entrada!r} no se ve como duda"
    for entrada in ("SI POR FAVOR", "NO BASTA DOLOR", "QUIERO AGUA", "NO"):
        assert not guard.is_uncertain(entrada), (
            f"{entrada!r} se tomó como duda y no lo es")

    r = guard.guarded("SI NO", ["no", "si", "no, gracias", "si, pero no"])
    razones = dict((t, por) for t, por in r.dropped)
    assert razones.get("no", "").startswith("resolvió"), (
        f"'no' sobrevivió a una entrada de duda: {r.candidates}")
    return "SI NO y TAL VEZ protegidos; sí/no solos descartados"


def _decision_checks():
    """Los chequeos de la tabla que decide, en su propio archivo.

    Viven aparte porque prueban una cosa distinta: no si las métricas miden
    bien, sino si el CRITERIO que elige el modelo está bien aplicado.
    """
    from .smoke_decision import CHECKS as _c

    return _c


CHECKS = (
    ("el corpus carga y valida", _corpus_loads),
    ("las referencias se reconocen a sí mismas", _references_pass_their_own_test),
    ("todas las trampas se disparan", _traps_actually_trip),
    ("sin contaminación prompt/corpus", _no_prompt_corpus_contamination),
    ("el filtro garantiza la polaridad", _guard_guarantees_polarity),
    ("el filtro protege la duda", _guard_protects_doubt),
    ("el filtro no descarta lo correcto", _guard_keeps_good_candidates),
    ("el filtro acepta sinónimos deletreados", _guard_accepts_spelled_synonyms),
    ("dedupe atrapa la repetición", _dedupe_catches_repetition),
    ("los clíticos no rompen el match", _cliticos_no_rompen_el_match),
    ("detecta invención", _catches_invention),
    ("detecta polaridad volteada", _catches_polarity_flip),
    ("detecta suavizado de urgencia", _catches_softening),
    ("detecta adivinar lo incompleto", _catches_guessing),
    ("mide diversidad y cobertura de lecturas", _measures_diversity_and_coverage),
    ("detecta la salida de escape literal", _literal_escape_hatch),
    ("parsea todas las formas de respuesta", _parses_every_shape),
    ("las variantes de prompt están anidadas", _prompt_variants_are_nested),
    ("la puerta de aptitud es estricta", _usable_gate_is_strict),
    ("--guard llega hasta las métricas", _guard_flag_reaches_the_metrics),
    ("extremo a extremo con modelo falso", _end_to_end_with_scripted_model),
    ("el tablero por mirada está cableado", _gaze_board_is_wired),
    ("estado de Ollama", _ollama_status),
) + _decision_checks()



def run() -> int:
    print("chequeos sin modelo:")
    passed = sum(_check(name, fn) for name, fn in CHECKS)
    print(f"\n{passed}/{len(CHECKS)} pasaron")
    if passed < len(CHECKS):
        print("\nUn fallo acá es un fallo del instrumento, no de un modelo: "
              "las métricas se miden contra salidas conocidas.")
    return 0 if passed == len(CHECKS) else 1
