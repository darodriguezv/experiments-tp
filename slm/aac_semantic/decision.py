"""Una sola tabla que decide qué modelo va en el dispositivo, y por qué.

El `leaderboard` ordena configuraciones por calidad. No alcanza para sostener
"usamos este modelo y no otro", porque esa frase involucra cuatro cosas que
viven en lugares distintos: qué tan bien contesta, cuánto ocupa, cuánto hay que
esperarlo, y si se puede **entregar** el dispositivo con ese modelo adentro.

Este módulo las junta y aplica un criterio explícito. Que sea explícito es el
punto: un puntaje compuesto -0.4·recall + 0.3·latencia + 0.3·tamaño- pondría un
número al final y escondería el único juicio que importa, que es cuánto vale
cada cosa. Peor: dejaría que un recall alto compre una invención, y esa es
exactamente la transacción que no está disponible.

## El criterio, en orden

1. **Seguridad, eliminatoria.** `metrics.usable()`. Un modelo que inventa un
   síntoma una vez cada veinte lo va a hacer en la única conversación que
   importa. No hay recall que lo compense.
2. **Licencia, eliminatoria para distribuir.** Un modelo que no se puede
   redistribuir no puede ir dentro de un dispositivo que se le entrega a
   alguien. Es una restricción legal, no técnica, y por eso no se negocia con
   dos puntos de recall.
3. **Costo, desempate.** Entre los que pasan, gana el más chico. Arriba de la
   puerta la calidad extra es invisible para la persona -la frase ya aparecía
   entre los candidatos- mientras que la RAM es la restricción que ata en un
   equipo dedicado sin placa de video.

## Las dos preguntas que la tabla contesta aparte

- **¿Por qué Qwen y no Llama?** Solo se puede responder comparando a tamaño
  igualado: qwen2.5:3b contra llama3.2:3b es misma talla y distinta receta.
  Comparar el mejor Qwen contra el peor Llama no dice nada sobre las familias.
- **¿Cuánto compra el tamaño?** La curva 0.5b→1.5b→3b→7b a receta constante.
  Si la curva es plana, la restricción de usar modelos chicos es gratis, que es
  un resultado más fuerte que no haber preguntado.

Ambas se imprimen con el veredicto estadístico al lado, porque con 133 ítems la
mayoría de las diferencias son ruido y una tabla que lo oculta invita
justamente a la conclusión equivocada.
"""

from __future__ import annotations

from collections import defaultdict

from . import metrics
from .backends import CANDIDATE_MODELS
from .footprint import Footprint
from .report import _delta_verdict

LICENCIAS_DISTRIBUIBLES = frozenset({"Apache-2.0", "MIT"})
"""Licencias bajo las que el dispositivo se puede entregar sin más trámite.

No es purismo. Las otras no son "malas": `Qwen Research` prohíbe el uso
comercial, y las de Llama y Gemma agregan condiciones de uso aceptable y de
atribución que hay que leer antes de poner el modelo adentro de algo que se le
da a una persona. Para una tesis eso se puede reportar; para un dispositivo que
se distribuye, alguien tiene que decidirlo, y la tabla lo pone a la vista en
lugar de que aparezca al final del proyecto."""

PARES_FAMILIA = (
    ("qwen2.5:3b", "llama3.2:3b", "3B contra 3B: misma talla, otra receta"),
    ("qwen2.5:1.5b", "llama3.2:1b", "el extremo chico de cada familia"),
    ("qwen2.5:1.5b", "gemma2:2b", "Qwen contra Gemma, ~2B"),
    ("qwen2.5:1.5b", "smollm2:1.7b",
     "control: smollm2 se entrenó casi solo en inglés"),
    ("qwen2.5:3b", "phi3.5", "contra la licencia más limpia (MIT)"),
)

CURVA_QWEN = ("qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b")


def _mejor_por_modelo(scores) -> dict[str, dict]:
    """La mejor variante de prompt de cada modelo.

    Comparar familias tiene que ser a prompt igualado o mejor: dejar que cada
    modelo use el prompt que más le conviene mide la familia, no el acople
    entre familia y redacción. Si se comparara con un prompt fijo, el ganador
    sería el modelo al que esa redacción le sienta mejor.
    """
    celdas: dict[tuple[str, str], list] = defaultdict(list)
    for s in scores:
        celdas[(s.model, s.variant)].append(s)

    mejor: dict[str, dict] = {}
    for (modelo, variante), cell in celdas.items():
        agg = metrics.aggregate(cell)
        agg["variant"] = variante
        actual = mejor.get(modelo)
        # Apto primero, después recall: un modelo apto con recall menor sigue
        # siendo mejor candidato que uno inapto con recall mayor.
        clave = (metrics.usable(agg), agg["recall_at_n"])
        if actual is None or clave > (metrics.usable(actual),
                                      actual["recall_at_n"]):
            mejor[modelo] = agg
    return mejor


def comparacion_familias(scores) -> str:
    """Qwen contra las otras familias, a tamaño igualado."""
    mejor = _mejor_por_modelo(scores)
    out = ["¿Por qué Qwen y no otra familia? (a tamaño igualado)", ""]
    out.append(f"  {'par':<30} {'rec@3':>14} {'delta':>8} {'veredicto':>10}")
    out.append("  " + "-" * 66)

    vistos = 0
    for a, b, por_que in PARES_FAMILIA:
        ag, bg = mejor.get(a), mejor.get(b)
        if not ag or not bg:
            out.append(f"  {a} vs {b}: sin datos "
                       f"({'falta ' + a if not ag else 'falta ' + b})")
            continue
        vistos += 1
        ra, rb = ag["recall_at_n"], bg["recall_at_n"]
        veredicto, z = _delta_verdict(ra, rb, ag["n"], bg["n"])
        out.append(f"  {a + ' vs ' + b:<30} {ra:>6.0%} vs {rb:>5.0%} "
                   f"{rb - ra:>+7.0%} {veredicto:>10}  z={z:+.1f}")
        out.append(f"    {por_que}")

    if not vistos:
        out.append("  Ninguna pareja completa en esta corrida.")
        out.append("  Para contestar la pregunta hay que correr AMBOS modelos "
                   "de al menos un par.")
        return "\n".join(out) + "\n"

    out += [
        "",
        "  'ruido' significa que el corpus no alcanza para distinguirlos, NO "
        "que sean iguales.",
        "  Con 133 ítems hace falta una diferencia de ~12 puntos para que se "
        "lea; por debajo",
        "  de eso la elección entre familias se decide por costo y licencia, "
        "no por calidad.",
    ]
    return "\n".join(out) + "\n"


def curva_de_escala(scores) -> str:
    """Qué compra el tamaño dentro de Qwen, a receta constante."""
    mejor = _mejor_por_modelo(scores)
    presentes = [m for m in CURVA_QWEN if m in mejor]
    out = ["¿Cuánto compra el tamaño? (Qwen, misma receta)", ""]
    if len(presentes) < 2:
        out.append("  Hacen falta al menos dos tamaños de Qwen en la corrida.")
        return "\n".join(out) + "\n"

    out.append(f"  {'modelo':<16} {'rec@3':>7} {'invent':>7} {'divers':>8} "
               f"{'modelo s':>9}")
    out.append("  " + "-" * 52)
    for m in presentes:
        a = mejor[m]
        out.append(f"  {m:<16} {a['recall_at_n']:>6.0%} "
                   f"{a['invention_rate']:>6.0%} {a['diversity']:>7.2f} "
                   f"{a['model_p50']:>8.2f}s")

    chico, grande = mejor[presentes[0]], mejor[presentes[-1]]
    veredicto, z = _delta_verdict(chico["recall_at_n"], grande["recall_at_n"],
                                  chico["n"], grande["n"])
    salto = grande["recall_at_n"] - chico["recall_at_n"]
    out += ["", f"  De {presentes[0]} a {presentes[-1]}: {salto:+.0%} de "
                f"recall, veredicto {veredicto} (z={z:+.1f})."]
    if veredicto == "ruido":
        out.append("  La curva es PLANA en este corpus: el tamaño extra no se "
                   "paga en recall, así que")
        out.append("  la restricción de usar modelos chicos sale gratis. Es un "
                   "resultado, no una falta")
        out.append("  de resultado.")
    else:
        out.append("  El tamaño SÍ compra recall acá. La restricción de usar "
                   "modelos chicos tiene un")
        out.append("  costo medible y la tesis tiene que decirlo en voz alta.")
    return "\n".join(out) + "\n"


def tabla_decision(scores, footprints: list[Footprint] | None = None,
                   max_latency_s: float = 4.0) -> str:
    """Calidad, costo y licencia en una sola tabla, con el criterio aplicado."""
    mejor = _mejor_por_modelo(scores)
    costos = {f.model: f for f in (footprints or []) if f.ok}

    filas = []
    for modelo, agg in mejor.items():
        meta = CANDIDATE_MODELS.get(modelo, {})
        lic = meta.get("license", "?")
        fp = costos.get(modelo)
        filas.append({
            "modelo": modelo,
            "variante": agg["variant"],
            "recall": agg["recall_at_n"],
            "apto": metrics.usable(agg, max_latency_s),
            "invencion": agg["invention_rate"],
            "licencia": lic,
            "distribuible": lic in LICENCIAS_DISTRIBUIBLES,
            # Sin medición de huella se cae al número declarado por el
            # fabricante, y se MARCA, porque un número copiado de una página
            # web no es una medición de esta máquina.
            "gb": fp.disk_gb if fp else float(meta.get("size_gb", 0.0)),
            "medido": fp is not None,
            "ram": fp.ram_gb if fp else 0.0,
            "recarga": fp.reload_s if fp else 0.0,
            "lat": agg["model_p50"],
        })

    filas.sort(key=lambda r: (r["apto"], r["distribuible"], -r["gb"]),
               reverse=True)

    out = ["La tabla que decide", ""]
    out.append(f"  {'modelo':<16}{'variante':<12}{'rec@3':>7}{'inv':>6}"
               f"{'apto':>6}{'licencia':>16}{'dist':>6}{'disco':>7}"
               f"{'RAM':>7}{'recarga':>8}")
    out.append("  " + "-" * 91)
    for r in filas:
        gb = f"{r['gb']:.1f}G" + ("" if r["medido"] else "~")
        ram = f"{r['ram']:.1f}G" if r["medido"] else "  -"
        rec = f"{r['recarga']:.1f}s" if r["medido"] else "  -"
        out.append(
            f"  {r['modelo']:<16}{r['variante']:<12}{r['recall']:>6.0%} "
            f"{r['invencion']:>5.0%} {'SÍ' if r['apto'] else 'no':>5}"
            f"{r['licencia']:>16}{'sí' if r['distribuible'] else 'NO':>6}"
            f"{gb:>7}{ram:>7}{rec:>8}")

    out += ["", "  dist    se puede entregar el dispositivo con ese modelo "
                "adentro sin más trámite",
            "  ~       tamaño DECLARADO por el fabricante, no medido acá "
            "(corré `footprint`)"]
    out.append("")
    out.append(_recomendacion(filas, max_latency_s))
    return "\n".join(out) + "\n"


def _recomendacion(filas: list[dict], max_latency_s: float) -> str:
    """Aplica el criterio y nombra al ganador, o explica por qué no hay."""
    aptos = [r for r in filas if r["apto"]]
    out = ["Recomendación", ""]

    if not aptos:
        out.append("  NINGUNA configuración pasa la puerta de aptitud.")
        out.append("")
        out.append("  Eso no es un empate ni un detalle de calibración: la "
                   "puerta exige cero invención,")
        out.append("  cero vuelcos de polaridad, la lectura literal presente y "
                   "recall@3 ≥ 80%. Mientras")
        out.append("  no la pase nadie, la respuesta honesta a '¿qué modelo "
                   "usamos?' es 'ninguno todavía")
        out.append("  cumple el criterio que nosotros mismos fijamos', y lo que "
                   "hay que mover es el")
        out.append("  prompt o el corpus, no el umbral.")
        cerca = sorted(filas, key=lambda r: -r["recall"])[:3]
        if cerca:
            out.append("")
            out.append("  Los más cerca, por recall:")
            for r in cerca:
                out.append(f"    {r['modelo']:<16} {r['recall']:.0%} "
                           f"(falta {max(0.0, 0.80 - r['recall']):.0%} para la "
                           f"puerta)")
        return "\n".join(out)

    distribuibles = [r for r in aptos if r["distribuible"]]
    if not distribuibles:
        out.append("  Hay modelos aptos, pero NINGUNO con licencia "
                   "redistribuible.")
        out.append("  Sirven para medir y reportar en la tesis; no para "
                   "entregar el dispositivo.")
        out.append("  Alguien tiene que leer la licencia y decidir antes de "
                   "que esto sea un producto.")
        return "\n".join(out)

    # Criterio 3: entre los que pasan, el más chico. La calidad arriba de la
    # puerta no se percibe; la RAM sí.
    ganador = min(distribuibles, key=lambda r: (r["gb"], -r["recall"]))
    out.append(f"  {ganador['modelo']}  con el prompt {ganador['variante']}")
    out.append("")
    out.append(f"  Pasa la puerta de seguridad, tiene licencia "
               f"{ganador['licencia']} -o sea que el")
    out.append(f"  dispositivo se puede entregar- y es el más chico de los que "
               f"cumplen ambas")
    out.append(f"  cosas: {ganador['gb']:.1f} GB en disco con recall@3 de "
               f"{ganador['recall']:.0%}.")
    otros = [r for r in distribuibles if r["modelo"] != ganador["modelo"]]
    if otros:
        arriba = max(otros, key=lambda r: r["recall"])
        if arriba["recall"] > ganador["recall"]:
            out.append("")
            out.append(f"  {arriba['modelo']} saca {arriba['recall']:.0%}, "
                       f"{arriba['recall'] - ganador['recall']:+.0%} más, a "
                       f"cambio de")
            out.append(f"  {arriba['gb'] - ganador['gb']:+.1f} GB. Arriba de la "
                       f"puerta ese recall extra no se ve:")
            out.append("  la frase ya aparecía entre los candidatos. Los GB sí "
                       "se ven, en un equipo sin")
            out.append("  placa de video.")
    return "\n".join(out)


def informe(scores, footprints: list[Footprint] | None = None,
            max_latency_s: float = 4.0) -> str:
    """Las tres secciones, en el orden en que se leen."""
    partes = [
        tabla_decision(scores, footprints, max_latency_s),
        comparacion_familias(scores),
        curva_de_escala(scores),
    ]
    return ("\n" + "=" * 78 + "\n\n").join(partes)
