"""Running the grid: every model x every prompt variant x every corpus item.

Two things this does that matter more than they look:

* **Warms each model before timing it.** Loading a 5 GB model into VRAM takes
  seconds; charging that to the first item would make whichever model ran first
  look terrible for reasons that have nothing to do with the model.
* **Writes every raw response to disk as it goes.** Re-scoring is then free.
  When a metric turns out to be wrong - and one will - the fix is to re-run the
  scoring over the saved responses in a second, not to re-run the grid for
  twenty minutes. That is the same lesson the camera work paid for the hard way
  by discarding its calibration samples.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from . import metrics, prompts
from .backends import Backend, Completion

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "telegraphic_es.jsonl"
RESULTS = ROOT / "results"


def load_corpus(path: Path | None = None) -> list[dict]:
    """Read the corpus, checking the fields the metrics rely on.

    Validated on load rather than trusted: a typo in a corpus tag would
    silently disable a whole class of check - an item whose
    `must_keep_polarity` is misspelled is an item where polarity is never
    tested, and it would look like a pass.
    """
    path = Path(path or DATA)
    if not path.exists():
        raise FileNotFoundError(f"no hay corpus en {path}")

    valid_polarity = {"positive", "negative", "negative-stop", "uncertain", "none"}
    items: list[dict] = []
    seen: set[str] = set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            item = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"{path.name}:{n} no es JSON válido: {exc}") from None

        for field_name in ("id", "input", "readings"):
            if field_name not in item:
                raise ValueError(f"{path.name}:{n} le falta '{field_name}'")
        if item["id"] in seen:
            raise ValueError(f"{path.name}:{n} id repetido: {item['id']!r}")
        seen.add(item["id"])

        polarity = item.get("must_keep_polarity", "none")
        if polarity not in valid_polarity:
            raise ValueError(
                f"{path.name}:{n} must_keep_polarity={polarity!r} no es uno de "
                f"{sorted(valid_polarity)}. Un valor mal escrito desactiva el "
                f"chequeo de polaridad para ese ítem sin avisar."
            )
        items.append(item)
    return items


def run_cell(backend: Backend, prompt: prompts.Prompt, corpus: list[dict],
             *, verbose: bool = True) -> list[metrics.ItemScore]:
    """One model, one prompt variant, the whole corpus."""
    scores: list[metrics.ItemScore] = []
    for i, item in enumerate(corpus, 1):
        completion: Completion = backend.complete(
            prompt.system, prompt.user(item["input"]),
            json_mode=prompt.json_mode, max_tokens=300)

        candidates, fmt = prompts.parse_candidates(completion.text)
        scores.append(metrics.score_item(
            item, candidates, fmt, completion.latency_s,
            model=backend.name, variant=prompt.variant,
            output_tokens=completion.output_tokens, error=completion.error,
            model_s=completion.model_s, prefill_s=completion.prefill_s))

        if verbose and (i % 10 == 0 or i == len(corpus)):
            done = scores
            print(f"    {i}/{len(corpus)}  recall@N "
                  f"{sum(s.hit for s in done) / len(done):.0%}  "
                  f"seguros {sum(s.safe for s in done) / len(done):.0%}",
                  flush=True)
    return scores


def run_grid(backends: list[Backend], variants: list[prompts.Prompt],
             corpus: list[dict], out_dir: Path | None = None,
             warm: bool = True) -> Path:
    """Run every cell and append every response to a JSONL. Returns its path."""
    out_dir = Path(out_dir or RESULTS)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"run-{stamp}.jsonl"

    total = len(backends) * len(variants)
    print(f"[grid] {len(backends)} modelo(s) x {len(variants)} variante(s) x "
          f"{len(corpus)} ítems = {total * len(corpus)} llamadas")

    with path.open("w", encoding="utf-8", newline="\n") as fh:
        cell = 0
        for backend in backends:
            if warm and hasattr(backend, "warm"):
                t0 = time.perf_counter()
                backend.warm()
                print(f"[grid] {backend.name}: cargado en "
                      f"{time.perf_counter() - t0:.1f}s")
            for prompt in variants:
                cell += 1
                print(f"[grid] ({cell}/{total}) {backend.name} / "
                      f"{prompt.variant}", flush=True)
                for score in run_cell(backend, prompt, corpus):
                    fh.write(json.dumps(asdict(score), ensure_ascii=False) + "\n")
                fh.flush()

    print(f"[grid] respuestas crudas -> {path}")
    return path


def load_scores(path: Path) -> list[metrics.ItemScore]:
    """Read a run back. This is what makes re-scoring cheap."""
    out: list[metrics.ItemScore] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(metrics.ItemScore(**json.loads(line)))
    return out


def rescore(path: Path, corpus: list[dict], *, guard: bool = False,
            ) -> list[metrics.ItemScore]:
    """Re-apply the current metrics to a saved run, without any model.

    The point of saving raw candidates. A metric bug found on Friday gets fixed
    and re-measured against Thursday's run instead of costing another pass over
    the grid.
    """
    by_id = {item["id"]: item for item in corpus}
    out: list[metrics.ItemScore] = []
    for old in load_scores(path):
        item = by_id.get(old.item_id)
        if item is None:
            continue
        candidates = old.candidates
        if guard:
            # Aplicar el filtro a las respuestas guardadas es lo que permite
            # medir su efecto sin volver a inferir: mismas salidas del modelo,
            # distinta etapa de post-proceso, comparación exacta.
            from . import guard as guard_mod

            candidates = guard_mod.guarded(item["input"], candidates).candidates
        out.append(metrics.score_item(
            item, candidates, old.fmt, old.latency_s, model=old.model,
            variant=old.variant, output_tokens=old.output_tokens,
            error=old.error, model_s=old.model_s, prefill_s=old.prefill_s))
    return out
