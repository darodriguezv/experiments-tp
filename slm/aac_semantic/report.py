"""Turning a run into tables that support a decision.

Two tables, because there are two decisions. The first ranks configurations by
whether they are *admissible* at all - a fluent, fast model that invents
symptoms is not a candidate, however good its recall. The second isolates what
each prompt addition bought, which is the only way to know whether the few-shot
examples or the JSON contract were worth their tokens.

Everything prints in the units that matter for a device someone speaks through:
rates, and seconds at p95 rather than the mean. The mean latency of a model that
is usually fast and occasionally takes nine seconds looks fine and feels awful.
"""

from __future__ import annotations

from collections import defaultdict

from . import metrics, prompts


def _delta_verdict(p_a: float, p_b: float, n_a: int, n_b: int) -> tuple[str, float]:
    """Is a difference between two rates bigger than the corpus can explain?

    A two-proportion z, printed next to every delta, because the corpus is small
    enough that most deltas are noise and a table that hides that invites
    exactly the wrong conclusion. Measured on the first real run: the ablation
    showed recall dropping 10 points when the rules were added, which reads as
    "the rules hurt" and is in fact a four-item difference at n=40 - z of -0.9.
    The one delta that survived was +28 points from the few-shot examples.

    At n=40 the standard error on a rate near 0.5 is about 8 points, so a
    10-point difference needs roughly 190 items to resolve. Until the corpus
    grows, only large effects are readable, and this makes that explicit rather
    than leaving it to whoever reads the table.
    """
    import math

    se = math.sqrt(max(p_a * (1 - p_a) / max(n_a, 1)
                       + p_b * (1 - p_b) / max(n_b, 1), 1e-12))
    z = (p_b - p_a) / se
    if abs(z) > 2.58:
        return "real", z
    if abs(z) > 1.96:
        return "real*", z
    return "ruido", z


def _cells(scores: list[metrics.ItemScore]) -> dict[tuple[str, str], list]:
    cells: dict[tuple[str, str], list] = defaultdict(list)
    for s in scores:
        cells[(s.model, s.variant)].append(s)
    return cells


def leaderboard(scores: list[metrics.ItemScore], max_latency_s: float = 4.0
                ) -> str:
    """Every configuration, ranked. Safety first, then recall."""
    cells = _cells(scores)
    rows = [(model, variant, metrics.aggregate(cell))
            for (model, variant), cell in cells.items()]
    # Sort by admissibility, then by recall, then by speed. Not a weighted
    # score: a composite number would let a high recall buy off an invention,
    # and that trade is not available.
    rows.sort(key=lambda r: (metrics.usable(r[2], max_latency_s),
                             r[2]["recall_at_n"], -r[2]["latency_p95"]),
              reverse=True)

    head = (f"{'modelo':<14}{'variante':<16}{'rec@3':>7}{'seguro':>8}"
            f"{'inv':>6}{'polar':>7}{'suav':>6}{'liter':>7}{'divers':>8}"
            f"{'modelo s':>9}{'prefill':>8}{'json':>6}  apto")
    out = [head, "-" * len(head)]
    for model, variant, agg in rows:
        out.append(
            f"{model:<14}{variant:<16}"
            f"{agg['recall_at_n']:>6.0%} {agg['safe_rate']:>7.0%} "
            f"{agg['invention_rate']:>5.0%} {agg['polarity_fail_rate']:>6.0%} "
            f"{agg['softening_rate']:>5.0%} {agg['literal_rate']:>6.0%} "
            f"{agg['diversity']:>7.2f} {agg['mean_candidates']:>5.1f} "
            f"{agg['model_p50']:>8.3f} "
            f"{agg['prefill_p50']:>7.3f} {agg['json_rate']:>5.0%}  "
            f"{'SÍ' if metrics.usable(agg, max_latency_s) else 'no'}")

    out += [
        "",
        "  rec@3   la frase buscada aparece entre los 3 candidatos",
        "  seguro  sin invención, sin voltear polaridad, sin suavizar urgencia",
        "  inv     inventó contenido que la entrada no dice  (debe ser 0%)",
        "  polar   perdió la negación o la orden de detenerse (debe ser 0%)",
        "  suav    suavizó un mensaje urgente                (debe ser 0%)",
        "  liter   ofreció la lectura literal como salida de escape",
        "  divers  0 = tres parafraseos de lo mismo, 1 = tres cosas distintas",
        "  opc     candidatos devueltos en promedio (1.0 = no hay elección)",
        "  modelo s   latencia p50 que reporta el modelo, sin nuestro transporte",
        "  prefill    de esos, cuánto cuesta leer el prompt (el precio de los "
        "ejemplos)",
        "  apto    pasa TODAS las puertas de metrics.usable()",
    ]
    return "\n".join(out)


def prompt_ablation(scores: list[metrics.ItemScore]) -> str:
    """What each prompt addition bought, averaged over models.

    The variants are nested - each adds one thing to the previous - so the
    difference between consecutive rows is attributable to that one addition.
    That is the whole reason they were built that way instead of as six
    unrelated prompts.
    """
    by_variant: dict[str, list] = defaultdict(list)
    for s in scores:
        by_variant[s.variant].append(s)

    order = [v for v in prompts.VARIANTS if v in by_variant]
    n = len(by_variant[order[0]]) if order else 0
    head = (f"{'variante':<16}{'agrega':<52}{'rec@3':>7}{'seguro':>8}"
            f"{'divers':>8}{'json':>6}")
    out = [head, "-" * len(head)]
    import math

    resolvable = (math.ceil(2 * (1.96 ** 2) * 0.25 / 0.10 ** 2) if n else 0)
    out.append(f"(n={n} por celda. Para resolver una diferencia de 10 puntos al "
               f"95% harían falta ~{resolvable} ítems, así que abajo solo son "
               f"legibles los efectos grandes.)")
    previous = None
    for variant in order:
        agg = metrics.aggregate(by_variant[variant])
        adds = prompts.VARIANTS[variant].adds
        out.append(f"{variant:<16}{adds[:50]:<52}"
                   f"{agg['recall_at_n']:>6.0%} {agg['safe_rate']:>7.0%} "
                   f"{agg['diversity']:>7.2f} {agg['json_rate']:>5.0%}")
        if previous is not None:
            d_rec = agg["recall_at_n"] - previous["recall_at_n"]
            d_safe = agg["safe_rate"] - previous["safe_rate"]
            v_rec, z_rec = _delta_verdict(previous["recall_at_n"],
                                          agg["recall_at_n"],
                                          previous["n"], agg["n"])
            v_safe, _ = _delta_verdict(previous["safe_rate"], agg["safe_rate"],
                                       previous["n"], agg["n"])
            label = f"  -> cambio ({v_rec}, z={z_rec:+.1f})"
            out.append(f"{'':<16}{label:<52}"
                       f"{d_rec:>+6.0%} {d_safe:>+7.0%}"
                       f"{'':>8}{'':>6}   seguro: {v_safe}")
        previous = agg
    return "\n".join(out)


def failures(scores: list[metrics.ItemScore], limit: int = 12) -> str:
    """The unsafe outputs, verbatim. The part worth reading.

    Aggregates say a configuration invents 8 % of the time. Only the actual
    sentences say whether it invented a temperature or a diagnosis, and that
    difference decides whether the fix is a prompt line or a different model.
    """
    bad = [s for s in scores if not s.safe]
    if not bad:
        return ("Ninguna salida insegura. (Revisá igual que recall@3 no sea "
                "bajo por otras razones.)")

    # Worst first: inventions before polarity before softening, because that is
    # the order in which they mislead someone.
    bad.sort(key=lambda s: (bool(s.invented), not s.polarity_ok,
                            bool(s.softened)), reverse=True)
    out = [f"{len(bad)} salida(s) insegura(s); las {min(limit, len(bad))} peores:"]
    for s in bad[:limit]:
        reasons = []
        if s.invented:
            reasons.append("INVENTÓ " + ", ".join(f"{w!r}" for w in s.invented))
        if not s.polarity_ok:
            reasons.append("POLARIDAD")
        if s.softened:
            reasons.append("SUAVIZÓ " + ", ".join(f"{w!r}" for w in s.softened))
        if not s.refusal_ok:
            reasons.append("EXPANDIÓ LO QUE NO DEBÍA")
        out.append(f"\n  [{s.model} / {s.variant}]  ítem {s.item_id}   "
                   f"{' + '.join(reasons)}")
        for c in s.candidates:
            out.append(f"      - {c}")
    return "\n".join(out)


def drift(scores: list[metrics.ItemScore], limit: int = 20) -> str:
    """Content words the models introduced that the corpus did not forbid.

    Reading this is how new `forbidden_content` entries get written. A word that
    shows up here repeatedly is either a legitimate expansion that belongs in
    `LICENSED_BY_INPUT`, or an invention nobody thought to forbid yet - and
    which of the two it is takes a human to decide.
    """
    counts: dict[str, int] = defaultdict(int)
    for s in scores:
        for word in s.introduced_words:
            counts[word] += 1
    if not counts:
        return "Sin palabras introducidas fuera de lo licenciado."

    ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:limit]
    width = max(len(w) for w, _ in ranked)
    out = ["Palabras introducidas, por frecuencia "
           "(decidir a mano: ¿expansión legítima o invención?):"]
    out += [f"   {w:<{width}}  {n}" for w, n in ranked]
    return "\n".join(out)


def full_report(scores: list[metrics.ItemScore], max_latency_s: float = 4.0
                ) -> str:
    parts = [
        "=" * 78,
        "  EXPANSIÓN SEMÁNTICA CAA - resultados",
        "=" * 78,
        "",
        leaderboard(scores, max_latency_s),
        "",
        "--- qué aportó cada agregado al prompt ---",
        prompt_ablation(scores),
        "",
        "--- salidas inseguras ---",
        failures(scores),
        "",
        "--- deriva léxica ---",
        drift(scores),
    ]
    return "\n".join(parts)
