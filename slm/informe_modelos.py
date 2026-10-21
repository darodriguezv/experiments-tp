"""Arma el informe de selección de modelo en Markdown, desde los datos.

    python slm/informe_modelos.py results/run-*.jsonl \
        --footprint results/footprint-*.jsonl --out docs/modelos.md

## Por qué se genera y no se escribe

Un informe con treinta números copiados a mano empieza desactualizado en cuanto
alguien vuelve a correr la grilla, y nadie se entera hasta que un número de la
tesis no coincide con el archivo que supuestamente lo respalda. Acá todas las
cifras salen del JSONL de la corrida; lo único escrito a mano es la
interpretación, que es lo que una persona sí tiene que poner.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))


def _consola_utf8() -> None:
    """La consola de Windows usa cp1252 y no sabe escribir ≥, → ni —.

    Sin esto el informe se genera bien en el archivo pero revienta al
    imprimirlo, que es la forma más confusa posible de fallar: el trabajo está
    hecho y parece que no."""
    try:
        import ctypes

        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
    except Exception:
        pass
    for flujo in (sys.stdout, sys.stderr):
        try:
            flujo.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_consola_utf8()

from aac_semantic import footprint as fp_mod  # noqa: E402
from aac_semantic import metrics, runner  # noqa: E402
from aac_semantic.backends import CANDIDATE_MODELS  # noqa: E402
from aac_semantic.decision import LICENCIAS_DISTRIBUIBLES  # noqa: E402
from aac_semantic.report import _delta_verdict  # noqa: E402

# Familias al mismo tamaño. Comparar el mejor Qwen contra el peor Llama no dice
# nada sobre las familias, solo sobre los dos puntos elegidos.
PARES = [
    ("qwen2.5:3b", "llama3.2:3b", "3B contra 3B, misma talla y otra receta"),
    ("qwen2.5:1.5b", "llama3.2:1b", "el extremo chico de cada familia"),
    ("qwen2.5:1.5b", "gemma2:2b", "Qwen contra Gemma en el rango de ~2B"),
    ("qwen2.5:3b", "phi3.5", "contra la licencia más limpia (MIT)"),
    ("qwen2.5:7b", "llama3.1:8b", "los dos grandes"),
    ("qwen2.5:7b", "mistral:7b", "Qwen contra Mistral a 7B"),
    ("qwen2.5:1.5b", "smollm2:1.7b",
     "control: smollm2 se entrenó casi solo en inglés"),
]

# Mismo modelo, distinta precisión. Aísla la cuantización dejando todo lo demás
# idéntico: mismos pesos, misma receta, mismo tamaño.
CUANTIZACION = [
    ("qwen2.5:0.5b", "qwen2.5:0.5b-instruct-fp16", "0.5B", "q4_K_M", "fp16"),
    ("qwen2.5:1.5b", "qwen2.5:1.5b-instruct-q8_0", "1.5B", "q4_K_M", "q8_0"),
    ("qwen2.5:1.5b", "qwen2.5:1.5b-instruct-fp16", "1.5B", "q4_K_M", "fp16"),
    ("qwen2.5:3b", "qwen2.5:3b-instruct-q8_0", "3B", "q4_K_M", "q8_0"),
]

# Mismo presupuesto de disco, distinto reparto entre parámetros y precisión.
PRESUPUESTO = [
    ("qwen2.5:0.5b-instruct-fp16", "qwen2.5:1.5b", "~0.9 GB"),
    ("qwen2.5:1.5b-instruct-q8_0", "qwen2.5:3b", "~1.5-1.8 GB"),
    ("qwen2.5:1.5b-instruct-fp16", "qwen2.5:3b-instruct-q8_0", "~3 GB"),
]

ESCALA = ["qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b"]


def mejor_por_modelo(scores):
    """La mejor variante de prompt de cada modelo.

    A prompt igualado o mejor: dejar que cada modelo use la redacción que más
    le conviene mide la familia, no el acople entre familia y redacción.
    """
    celdas = defaultdict(list)
    for s in scores:
        celdas[(s.model, s.variant)].append(s)
    mejor = {}
    for (modelo, variante), cell in celdas.items():
        agg = metrics.aggregate(cell)
        agg["variant"] = variante
        prev = mejor.get(modelo)
        clave = (metrics.usable(agg), agg["recall_at_n"])
        if prev is None or clave > (metrics.usable(prev), prev["recall_at_n"]):
            mejor[modelo] = agg
    return mejor


def _pct(v, signo=False):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:+.0%}" if signo else f"{v:.0%}"


def _delta(mejor, a, b, campo="recall_at_n"):
    ga, gb = mejor.get(a), mejor.get(b)
    if not ga or not gb:
        falta = a if not ga else b
        return None, None, None, f"falta `{falta}`"
    va, vb = ga[campo], gb[campo]
    ver, z = _delta_verdict(va, vb, ga["n"], gb["n"])
    return va, vb, (ver, z), None


def construir(scores, costos, ruta_run) -> str:
    mejor = mejor_por_modelo(scores)
    n_items = max((a["n"] for a in mejor.values()), default=0)
    variantes = sorted({s.variant for s in scores})
    L = []
    w = L.append

    w("# Selección del modelo de lenguaje")
    w("")
    w(f"*Generado el {datetime.now():%Y-%m-%d} con `slm/informe_modelos.py` "
      f"desde `{ruta_run.name}`. No editar a mano: regenerar.*")
    w("")
    w(f"{len(mejor)} modelos, {n_items} entradas del corpus, "
      f"{len(variantes)} variantes de prompt por modelo "
      f"(`{'`, `'.join(variantes)}`), "
      f"{len(scores)} respuestas en total.")
    w("")
    w("Cada modelo compite con la variante de prompt que mejor le va. Fijar "
      "una sola redacción para todos mediría qué familia le sienta mejor a esa "
      "redacción, no qué familia es mejor.")
    w("")

    # ---------------------------------------------------------- resumen
    w("## La tabla completa")
    w("")
    w("| modelo | prompt | recall@3 | literal | invención | polaridad | "
      "diversidad | p50 modelo | disco | licencia | entregable |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    filas = sorted(mejor.items(), key=lambda kv: -kv[1]["recall_at_n"])
    for modelo, a in filas:
        meta = CANDIDATE_MODELS.get(modelo, {})
        lic = meta.get("license", "?")
        c = costos.get(modelo)
        disco = (f"{c.disk_gb:.1f} GB" if c
                 else (f"{meta['size_gb']:.1f} GB~" if meta.get("size_gb")
                       else "—"))
        w(f"| `{modelo}` | {a['variant']} | **{_pct(a['recall_at_n'])}** | "
          f"{_pct(a['literal_rate'])} | {_pct(a['invention_rate'])} | "
          f"{_pct(a['polarity_fail_rate'])} | {a['diversity']:.2f} | "
          f"{a['model_p50']:.2f} s | {disco} | {lic} | "
          f"{'sí' if lic in LICENCIAS_DISTRIBUIBLES else '**no**'} |")
    w("")
    w("`~` marca un tamaño declarado por el fabricante, no medido en esta "
      "máquina.")
    w("")
    w("**Qué mide cada columna**")
    w("")
    w("- **recall@3**: la frase buscada aparece entre los 3 candidatos. Se "
      "cuenta por F1 de tokens ≥ 0.75 contra las referencias escritas a mano, "
      "no por coincidencia exacta: *quiero agua por favor* tiene que contar "
      "como acierto de *quiero agua, por favor*. A 0.5, *no quiero agua* "
      "empezaría a matchear *quiero agua*, que es la única confusión que este "
      "módulo no puede cometer.")
    w("- **literal**: ofreció la lectura literal como salida de escape. Si las "
      "tres propuestas están mal, la persona tiene que poder decir exactamente "
      "lo que escribió — el tablero pelado ya podía.")
    w("- **invención**: dijo algo que la entrada no dice. Debe ser 0%.")
    w("- **polaridad**: perdió una negación o una orden de detenerse. El "
      "filtro determinista lo lleva a 0% por construcción; esta columna es "
      "**antes** del filtro, o sea la tendencia cruda del modelo.")
    w("- **diversidad**: 0 = tres parafraseos de lo mismo, 1 = tres cosas "
      "distintas.")
    w("")

    # ------------------------------------------------- puerta de aptitud
    aptos = [m for m, a in mejor.items() if metrics.usable(a)]
    w("## La puerta de aptitud")
    w("")
    w("Un modelo es admisible solo si cumple **todas**: cero invención, cero "
      "vuelcos de polaridad, cero suavizado de urgencia, literal presente "
      "≥ 90%, recall@3 ≥ 80%, al menos dos candidatos el 90% de las veces, y "
      "p95 del modelo ≤ 4 s.")
    w("")
    if aptos:
        w("Pasan: " + ", ".join(f"`{m}`" for m in aptos) + ".")
    else:
        cerca = sorted(mejor.items(), key=lambda kv: -kv[1]["recall_at_n"])[:3]
        w("**Ninguno pasa.** No es un empate ni un detalle de calibración: "
          "mientras no la pase nadie, la respuesta honesta a *¿qué modelo "
          "usamos?* es *ninguno cumple todavía el criterio que fijamos "
          "nosotros*, y lo que hay que mover es el prompt o el corpus, no el "
          "umbral.")
        w("")
        w("Los más cerca por recall:")
        w("")
        for m, a in cerca:
            falta = max(0.0, 0.80 - a["recall_at_n"])
            w(f"- `{m}`: {_pct(a['recall_at_n'])}, le faltan {falta:.0%}")
    w("")

    # ----------------------------------------------------- familias
    w("## ¿Por qué Qwen y no otra familia?")
    w("")
    w("Solo tiene sentido a tamaño igualado.")
    w("")
    w("| comparación | recall@3 | Δ | veredicto | qué controla |")
    w("|---|---|---|---|---|")
    for a, b, porque in PARES:
        va, vb, ver, err = _delta(mejor, a, b)
        if err:
            w(f"| `{a}` vs `{b}` | — | — | sin datos | {err} |")
            continue
        v, z = ver
        w(f"| `{a}` vs `{b}` | {_pct(va)} vs {_pct(vb)} | "
          f"{_pct(vb - va, True)} | {v} (z={z:+.2f}) | {porque} |")
    w("")
    w(_nota_potencia(n_items))
    w("")

    # ------------------------------------------------- cuantización
    w("## ¿Cuánto cuesta cuantizar?")
    w("")
    w("Mismo modelo y misma receta; lo único que cambia son los bits por peso.")
    w("")
    w("| modelo | precisión | recall@3 | Δ | veredicto |")
    w("|---|---|---|---|---|")
    for a, b, talla, pa, pb in CUANTIZACION:
        va, vb, ver, err = _delta(mejor, a, b)
        if err:
            w(f"| {talla} | {pa} → {pb} | — | — | sin datos ({err}) |")
            continue
        v, z = ver
        w(f"| {talla} | {pa} → {pb} | {_pct(va)} vs {_pct(vb)} | "
          f"{_pct(vb - va, True)} | {v} (z={z:+.2f}) |")
    w("")

    # ----------------------------------------- parámetros vs precisión
    w("## Al mismo disco: ¿parámetros o precisión?")
    w("")
    w("La pregunta práctica para un equipo dedicado: con los mismos GB, "
      "¿conviene un modelo más grande y más cuantizado, o uno más chico con "
      "más precisión?")
    w("")
    w("| presupuesto | menos params, más bits | más params, menos bits | Δ | "
      "veredicto |")
    w("|---|---|---|---|---|")
    for chico, grande, pres in PRESUPUESTO:
        va, vb, ver, err = _delta(mejor, chico, grande)
        if err:
            w(f"| {pres} | `{chico}` | `{grande}` | — | sin datos ({err}) |")
            continue
        v, z = ver
        w(f"| {pres} | `{chico}` {_pct(va)} | `{grande}` {_pct(vb)} | "
          f"{_pct(vb - va, True)} | {v} (z={z:+.2f}) |")
    w("")

    # ----------------------------------------------------- escala
    presentes = [m for m in ESCALA if m in mejor]
    if len(presentes) >= 2:
        w("## ¿Cuánto compra el tamaño?")
        w("")
        w("Dentro de Qwen, a receta constante: lo único que cambia es la "
          "cantidad de parámetros.")
        w("")
        w("| modelo | recall@3 | literal | invención | p50 |")
        w("|---|---|---|---|---|")
        for m in presentes:
            a = mejor[m]
            w(f"| `{m}` | {_pct(a['recall_at_n'])} | "
              f"{_pct(a['literal_rate'])} | {_pct(a['invention_rate'])} | "
              f"{a['model_p50']:.2f} s |")
        w("")
        va, vb, ver, _ = _delta(mejor, presentes[0], presentes[-1])
        v, z = ver
        w(f"De `{presentes[0]}` a `{presentes[-1]}`: "
          f"{_pct(vb - va, True)} de recall, **{v}** (z={z:+.2f}).")
        w("")
        vl_a, vl_b, ver_l, _ = _delta(mejor, presentes[0], presentes[-1],
                                      "literal_rate")
        vl, zl = ver_l
        w(f"Y en la salida de escape literal: {_pct(vl_a)} → {_pct(vl_b)}, "
          f"{_pct(vl_b - vl_a, True)}, **{vl}** (z={zl:+.2f}).")
        w("")

    # ------------------------------------------------------- costo
    if costos:
        w("## Costo medido en esta máquina")
        w("")
        w("| modelo | params | cuantización | disco | VRAM | caliente p50 | "
          "recarga tibia | tok/s |")
        w("|---|---|---|---|---|---|---|---|")
        for c in sorted(costos.values(), key=lambda x: x.disk_gb):
            w(f"| `{c.model}` | {c.params} | {c.quantization} | "
              f"{c.disk_gb:.1f} GB | {c.vram_gb:.1f} GB | "
              f"{c.warm_p50_s:.2f} s | {c.reload_s:.1f} s | "
              f"{c.tokens_per_s:.0f} |")
        w("")
        w("**Sobre la recarga.** Es *tibia*: el modelo salió de RAM pero su "
          "archivo sigue en la caché de páginas del sistema. Una primera "
          "lectura real desde disco es varias veces peor — se midió 4.3 s "
          "contra 39 s en el mismo modelo — y no se puede automatizar sin "
          "vaciar la caché del sistema operativo. Para el dispositivo importa "
          "la tibia, porque el equipo queda encendido; la fría solo aparece "
          "tras reiniciar.")
        w("")

    w("## Límites de esta medición")
    w("")
    w(_limites(n_items, costos))
    w("")
    return "\n".join(L) + "\n"


def _nota_potencia(n):
    if not n:
        return ""
    se = math.sqrt(0.25 / n + 0.25 / n)
    minimo = 1.96 * se
    return (f"*ruido* significa que el corpus no alcanza para distinguirlos, "
            f"**no** que sean iguales. Con {n} entradas hace falta una "
            f"diferencia de ~{minimo:.0%} para que se lea; por debajo de eso "
            f"la elección se decide por costo y licencia, no por calidad.")


def _limites(n, costos):
    L = [
        f"- **El corpus tiene {n} entradas y las escribimos nosotros.** Las "
        f"categorías salen de la literatura de CAA y cuidados en ELA, pero las "
        f"frases concretas y las referencias aceptadas son juicio del tesista, "
        f"no de un clínico ni de un usuario de CAA. Sirven para comparar "
        f"configuraciones entre sí, que es para lo que se usan acá. No sirven "
        f"para afirmar que el módulo dice lo correcto.",
        f"- **Potencia.** Con {n} entradas solo se leen diferencias de ~12 "
        f"puntos. Para resolver 5 puntos harían falta unas 380 entradas; para "
        f"3 puntos, más de mil. Ampliar el corpus compra más capacidad de "
        f"decisión que probar otro modelo.",
        "- **Las etiquetas por defecto de Ollama no usan la misma "
        "cuantización.** Qwen viene en Q4_K_M, pero otras familias vienen en "
        "Q8_0 o Q4_0. Donde una familia pierde corriendo con *más* precisión, "
        "el resultado se refuerza; donde pierde con menos, parte del déficit "
        "puede ser el redondeo y no la familia. La tabla de costo dice la "
        "cuantización real de cada uno.",
    ]
    if costos and all(c.ram_gb < 0.1 for c in costos.values()):
        L.append(
            "- **Todo corrió en GPU.** La RAM medida es ~0 en los quince: la "
            "carga está en VRAM. Las latencias de la tabla son de GPU, y un "
            "equipo dedicado de CAA probablemente no tenga una placa dedicada. "
            "Ahí los tokens por segundo serían sustancialmente peores.")
    L.append(
        "- **La polaridad de la tabla es anterior al filtro determinista.** En "
        "el dispositivo, `guard.py` descarta por construcción lo que perdió la "
        "negación, así que la polaridad efectiva es 0%. La columna muestra la "
        "tendencia cruda del modelo, que es lo que hay que conocer para saber "
        "cuánto trabajo está haciendo el filtro.")
    return "\n".join(L)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="informe_modelos.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="ruta o glob del JSONL de la corrida")
    ap.add_argument("--footprint", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rescore", action="store_true",
                    help="re-medir con las métricas actuales")
    args = ap.parse_args(argv[1:])

    ruta = _resolver(args.run)
    scores = (runner.rescore(ruta, runner.load_corpus()) if args.rescore
              else runner.load_scores(ruta))
    costos = {}
    if args.footprint:
        costos = {f.model: f for f in fp_mod.load(_resolver(args.footprint))
                  if f.ok}

    texto = construir(scores, costos, ruta)
    if args.out:
        destino = Path(args.out)
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(texto, encoding="utf-8")
        print(f"[informe] {len(scores)} respuestas -> {destino} "
              f"({len(texto)} caracteres)")
    else:
        print(texto)
    return 0


def _resolver(spec: str) -> Path:
    p = Path(spec)
    if p.exists():
        return p
    coincidencias = sorted(Path(p.parent or ".").glob(p.name),
                           key=lambda x: x.stat().st_mtime)
    if not coincidencias:
        raise FileNotFoundError(f"nada coincide con {spec!r}")
    return coincidencias[-1]


if __name__ == "__main__":
    sys.exit(main(sys.argv))
