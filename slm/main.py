"""Expansión y enriquecimiento semántico para el tablero CAA.

    python main.py smoke                # chequeos sin modelo (valida las métricas)
    python main.py models               # qué hay instalado en Ollama y qué falta
    python main.py pull                 # comandos para bajar los modelos
    python main.py corpus               # muestra el corpus de evaluación
    python main.py try --model qwen2.5:1.5b --input "AGUA NO"
    python main.py run                  # la grilla completa -> results/run-*.jsonl
    python main.py report results/run-*.jsonl
    python main.py rescore results/run-*.jsonl   # re-mide sin volver a inferir
    python main.py demo                 # el tablero clickeable, para probarlo a mano
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _utf8_console() -> None:
    try:
        import ctypes

        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
    except Exception:
        pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_utf8_console()

from aac_semantic import backends, prompts, report, runner  # noqa: E402

# La curva de escala de Qwen (misma receta, tres tamaños) mas una familia
# rival al mismo tamaño y la de licencia mas limpia. El resto se pide a mano.
DEFAULT_MODELS = ["qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b",
                  "llama3.2:3b", "phi3.5"]


def _resolve_run(spec: str) -> Path:
    """Accept a path or a glob, taking the newest match.

    So `report results/run-*.jsonl` works from PowerShell, which does not
    expand globs before handing them to a program.
    """
    path = Path(spec)
    if path.exists():
        return path
    matches = sorted(Path(path.parent or ".").glob(path.name),
                     key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"no hay resultados que coincidan con {spec!r}")
    return matches[-1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Expansión semántica telegráfica -> habla natural, con "
                    "modelos abiertos locales.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("smoke", help="chequeos sin modelo")
    sub.add_parser("models", help="inventario de modelos en Ollama")
    sub.add_parser("pull", help="imprime los comandos para bajar los modelos")

    p_corpus = sub.add_parser("corpus", help="muestra el corpus de evaluación")
    p_corpus.add_argument("--tag", default=None,
                          help="filtra por etiqueta, p.ej. urgente o ambiguo")

    p_try = sub.add_parser("try", help="una sola entrada, para mirar la salida")
    p_try.add_argument("--model", default="qwen2.5:1.5b")
    p_try.add_argument("--variant", default="v4-literal",
                       choices=list(prompts.VARIANTS))
    p_try.add_argument("--input", required=True)
    p_try.add_argument("--temperature", type=float, default=0.2)

    p_demo = sub.add_parser(
        "demo",
        help="el tablero clickeable con la expansión enchufada: armá una frase "
             "y escuchá los candidatos")
    p_demo.add_argument("--model", default=None,
                        help="con qué modelo arrancar (M cicla en vivo)")
    p_demo.add_argument("--variant", default=None, choices=list(prompts.VARIANTS),
                        help="con qué prompt arrancar (V cicla en vivo)")

    p_run = sub.add_parser("run", help="la grilla completa")
    p_run.add_argument("--models", nargs="+", default=None,
                       help=f"por defecto: {' '.join(DEFAULT_MODELS)}")
    p_run.add_argument("--variants", nargs="+", default=None,
                       choices=list(prompts.VARIANTS),
                       help="por defecto todas")
    p_run.add_argument("--limit", type=int, default=None,
                       help="usa solo los primeros N ítems (para una prueba rápida)")
    p_run.add_argument("--temperature", type=float, default=0.2)
    p_run.add_argument("--max-latency", type=float, default=4.0)

    for name in ("report", "rescore"):
        p = sub.add_parser(name, help=("tabla desde un run guardado"
                                       if name == "report"
                                       else "re-mide un run con las métricas actuales"))
        p.add_argument("run", help="ruta o glob de results/run-*.jsonl")
        p.add_argument("--max-latency", type=float, default=4.0)
        if name == "rescore":
            p.add_argument(
                "--guard", action="store_true",
                help="aplicar el filtro determinista de polaridad antes de "
                     "medir. Compará con y sin: la diferencia es exactamente "
                     "lo que aporta el filtro, sobre las mismas respuestas")

    args = parser.parse_args(argv)

    if args.command == "smoke":
        from aac_semantic import smoke

        return smoke.run()

    if args.command == "models":
        print(backends.ollama_report())
        return 0

    if args.command == "pull":
        print("Instalá Ollama primero: https://ollama.com/download\n")
        print("Después, bajá los modelos (el total es unos 14 GB):\n")
        for name, info in backends.CANDIDATE_MODELS.items():
            print(f"  ollama pull {name:<14}# {info['size_gb']:.1f} GB, "
                  f"{info['params']}, {info['license']}")
        print("\nSi el espacio aprieta, con estos tres ya se puede comparar:")
        print("  ollama pull qwen2.5:7b    # el candidato principal")
        print("  ollama pull qwen2.5:3b    # el brazo de latencia")
        print("  ollama pull phi3.5        # el brazo sin GPU, licencia MIT")
        return 0

    if args.command == "corpus":
        corpus = runner.load_corpus()
        if args.tag:
            corpus = [i for i in corpus if args.tag in i.get("tags", [])]
        print(f"{len(corpus)} ítem(s)"
              + (f" con la etiqueta {args.tag!r}" if args.tag else ""))
        for item in corpus:
            flags = []
            if item.get("ambiguous"):
                flags.append("AMBIGUO")
            if item.get("urgency") == "high":
                flags.append("URGENTE")
            if item.get("expect_refusal"):
                flags.append("NO-EXPANDIR")
            print(f"\n  [{item['id']}] {item['input']!r}"
                  + (f"   {' '.join(flags)}" if flags else ""))
            for reading in item["readings"]:
                accept = reading.get("accept", [])
                print(f"      · {reading['gloss']}: "
                      + (", ".join(f"{a!r}" for a in accept[:3]) or "(nada)"))
            if item.get("forbidden_content"):
                print("      ✗ no debe decir: "
                      + ", ".join(f"{f!r}" for f in item["forbidden_content"]))
            if item.get("note"):
                print(f"      » {item['note']}")
        return 0

    if args.command == "demo":
        from aac_semantic import demo

        return demo.run(model=args.model, variant=args.variant)

    if args.command == "try":
        backend = backends.OllamaBackend(args.model, temperature=args.temperature)
        if not backend.available():
            print(backends.ollama_report())
            return 1
        variant = prompts.VARIANTS[args.variant]
        completion = backend.complete(variant.system, variant.user(args.input),
                                      json_mode=variant.json_mode)
        if not completion.ok:
            print(f"error: {completion.error}")
            return 1
        candidates, fmt = prompts.parse_candidates(completion.text)
        print(f"entrada    {args.input!r}")
        print(f"modelo     {args.model}   variante {args.variant}")
        print(f"latencia   {completion.latency_s:.2f}s   "
              f"{completion.output_tokens} tokens   "
              f"{completion.tokens_per_s:.0f} tok/s   formato {fmt}")
        print("candidatos:")
        for c in candidates:
            print(f"  - {c}")
        if not candidates:
            print(f"  (nada parseable)\n  respuesta cruda: {completion.text[:300]!r}")
        return 0

    if args.command == "run":
        corpus = runner.load_corpus()
        if args.limit:
            corpus = corpus[:args.limit]
        names = args.models or DEFAULT_MODELS
        variants = [prompts.VARIANTS[v] for v in
                    (args.variants or list(prompts.VARIANTS))]

        chosen: list[backends.Backend] = []
        for name in names:
            backend = backends.OllamaBackend(name, temperature=args.temperature)
            if not backend.available():
                print(backends.ollama_report())
                return 1
            if not backend.has_model():
                print(f"[grid] {name} no está instalado; se saltea. "
                      f"(`ollama pull {name}`)")
                continue
            chosen.append(backend)
        if not chosen:
            print("Ningún modelo disponible. Ver `main.py pull`.")
            return 1

        path = runner.run_grid(chosen, variants, corpus)
        scores = runner.load_scores(path)
        print()
        print(report.full_report(scores, args.max_latency))
        return 0

    if args.command in ("report", "rescore"):
        path = _resolve_run(args.run)
        if args.command == "rescore":
            scores = runner.rescore(path, runner.load_corpus(),
                                    guard=args.guard)
            print(f"[re-medido] {len(scores)} respuestas de {path.name} con las "
                  f"métricas actuales\n")
        else:
            scores = runner.load_scores(path)
        print(report.full_report(scores, args.max_latency))
        return 0

    parser.error(f"comando desconocido {args.command}")
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nerror: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\ninterrumpido")
        sys.exit(130)
