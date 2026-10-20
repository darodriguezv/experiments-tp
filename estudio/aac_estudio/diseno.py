"""Quién hace qué condición, en qué orden, y con qué frases.

Esta es la parte del estudio donde se gana o se pierde la posibilidad de
afirmar algo, y no se puede arreglar después: si el orden está mal asignado, no
hay análisis que recupere la comparación.

## El problema del orden

Tres condiciones -deletreo, sistema sin contexto, sistema contextualizado- y la
tentación obvia es que todo el mundo las haga en ese orden, que además es el
orden en que cuentan la historia. Sería un error que invalida el resultado.

Para cuando la persona llega a la tercera condición lleva veinte minutos
manejando un cursor con los ojos. Es sustancialmente más diestra que al
empezar. Si la tercera condición gana, no hay forma de distinguir cuánto fue el
contexto y cuánto fue la práctica, porque las dos cosas crecen juntas. El
efecto de aprendizaje en control por mirada es grande justamente en los
primeros minutos, que es donde cae la primera condición.

La solución es contrabalancear: distintas personas hacen las condiciones en
distinto orden, de modo que la práctica se reparta parejo entre las tres en vez
de acumularse siempre en la misma.

## Por qué seis secuencias y no tres

Un cuadrado latino de 3x3 (ABC, BCA, CAB) equilibra la **posición**: cada
condición aparece una vez primera, una vez segunda y una vez tercera. No
equilibra el **arrastre**: en esas tres secuencias, B siempre viene después de
A, y C siempre después de B. Si estar en el sistema con contexto deja a la
persona predispuesta de alguna manera para lo que sigue, ese efecto queda
confundido con la condición siguiente.

Las seis permutaciones (diseño de Williams para tres condiciones) equilibran
las dos cosas: cada condición aparece igual de seguido en cada posición Y cada
condición precede a cada otra el mismo número de veces.

## Diez participantes no se puede equilibrar

Seis secuencias necesitan un múltiplo de seis. Con diez, cuatro secuencias se
usan dos veces y dos se usan una sola: el diseño queda **desbalanceado**, y
`balance()` lo reporta en vez de dejarlo pasar en silencio.

No es fatal -el desbalance es chico y conocido- pero hay que declararlo en la
tesis, y conviene saber que **con doce participantes el equilibrio es exacto**.
La diferencia entre diez y doce es dos sesiones.

## Las frases también rotan

Si las tres condiciones usan las mismas frases, para la tercera la persona ya
sabe de memoria dónde están las letras de "tengo frío". Por eso hay tres
conjuntos de frases emparejados en dificultad, y rotan contra las condiciones
con su propio cuadrado latino. Así ninguna condición queda pegada a un conjunto
de frases.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

CONDICIONES = ("letra", "sin_contexto", "con_contexto")
"""Las tres condiciones, en orden narrativo. NO es el orden de aplicación."""

NOMBRES = {
    "letra": "Letra a letra",
    "sin_contexto": "Sistema sin contexto",
    "con_contexto": "Sistema contextualizado",
}

SECUENCIAS: tuple[tuple[str, ...], ...] = tuple(
    tuple(p) for p in sorted(permutations(CONDICIONES)))
"""Las seis permutaciones. Ordenadas para que la asignación sea reproducible:
el participante 7 recibe siempre la misma secuencia, corra esto quien lo corra.
"""

CONJUNTOS_FRASES = (0, 1, 2)
"""Tres conjuntos de frases emparejados. Ver `protocolo.TAREAS`."""


@dataclass(frozen=True)
class Asignacion:
    """Lo que le toca a un participante. Determinista a partir del número."""

    participante: int
    secuencia: tuple[str, ...]
    conjuntos: tuple[int, ...]
    """Qué conjunto de frases usa en cada condición, en el orden de
    `secuencia`."""

    def bloques(self) -> list[tuple[int, str, int]]:
        """(posición 1-3, condición, conjunto de frases) para cada bloque."""
        return [(i + 1, c, s)
                for i, (c, s) in enumerate(zip(self.secuencia, self.conjuntos))]


def asignar(participante: int) -> Asignacion:
    """La asignación del participante N (1-indexado).

    Determinista a propósito, sin azar. Un sorteo con semilla daría lo mismo y
    agregaría una semilla que alguien tiene que recordar; un sorteo sin semilla
    haría que el diseño no se pueda reconstruir si se pierde la planilla. Acá
    el número de participante **es** la asignación.
    """
    if participante < 1:
        raise ValueError("los participantes se numeran desde 1")
    i = participante - 1
    secuencia = SECUENCIAS[i % len(SECUENCIAS)]

    # El conjunto de frases se decide por la CONDICIÓN, no por la posición en
    # la secuencia. La primera versión rotaba por posición -el primer bloque
    # llevaba F0, el segundo F1, el tercero F2- y el resultado medido fue que
    # `letra` recibía F0 en 8 de cada 12 casos y `sin_contexto` no lo recibía
    # nunca: los conjuntos quedaban pegados a las condiciones, que es
    # exactamente el confundido que rotarlos venía a evitar.
    #
    # Al indexar por condición, tres participantes consecutivos le dan a cada
    # condición los tres conjuntos, y ninguno repite conjunto dentro de su
    # propia sesión.
    conjuntos = tuple((CONDICIONES.index(c) + i) % len(CONJUNTOS_FRASES)
                      for c in secuencia)
    return Asignacion(participante, secuencia, conjuntos)


def plan(n: int) -> list[Asignacion]:
    return [asignar(p) for p in range(1, n + 1)]


def balance(n: int) -> dict:
    """Qué tan equilibrado queda el diseño con N participantes.

    Devuelve los conteos crudos para que la tesis pueda citarlos, más el
    veredicto. Se cuenta sobre la asignación real, no sobre la teoría: si
    `asignar` cambiara, esto lo detecta.
    """
    asignaciones = plan(n)
    por_posicion = {c: [0] * len(CONDICIONES) for c in CONDICIONES}
    arrastre = {(a, b): 0 for a in CONDICIONES for b in CONDICIONES if a != b}
    for a in asignaciones:
        for idx, cond in enumerate(a.secuencia):
            por_posicion[cond][idx] += 1
        for antes, despues in zip(a.secuencia, a.secuencia[1:]):
            arrastre[(antes, despues)] += 1

    esperado_pos = n / len(CONDICIONES)
    desvio_pos = max(abs(v - esperado_pos)
                     for fila in por_posicion.values() for v in fila)
    esperado_arr = (n * (len(CONDICIONES) - 1)) / len(arrastre)
    desvio_arr = max(abs(v - esperado_arr) for v in arrastre.values())

    exacto = n % len(SECUENCIAS) == 0
    return {
        "n": n,
        "posicion": por_posicion,
        "arrastre": arrastre,
        "desvio_posicion": desvio_pos,
        "desvio_arrastre": desvio_arr,
        "exacto": exacto,
        "n_exacto_mas_cercano": _siguiente_multiplo(n, len(SECUENCIAS)),
    }


def _siguiente_multiplo(n: int, k: int) -> int:
    return n if n % k == 0 else ((n // k) + 1) * k


def informe_balance(n: int) -> str:
    b = balance(n)
    out = [f"Contrabalanceo con {n} participantes", ""]
    out.append("  Cada participante hace las tres condiciones en un orden "
               "distinto, para que la")
    out.append("  práctica no se acumule siempre en la misma.")
    out.append("")
    out.append(f"  {'condición':<22}{'1ra':>6}{'2da':>6}{'3ra':>6}")
    out.append("  " + "-" * 40)
    for cond, fila in b["posicion"].items():
        out.append(f"  {NOMBRES[cond]:<22}" + "".join(f"{v:>6}" for v in fila))
    out.append("")

    if b["exacto"]:
        out.append(f"  EXACTO: {n} es múltiplo de {len(SECUENCIAS)}, así que "
                   f"cada secuencia se usa igual")
        out.append("  de seguido y no hay nada que declarar.")
    else:
        out.append(f"  DESBALANCEADO. Seis secuencias necesitan un múltiplo de "
                   f"seis; con {n} algunas")
        out.append(f"  se usan dos veces y otras una. El desvío máximo es de "
                   f"{b['desvio_posicion']:.0f} participante(s) en")
        out.append(f"  posición y {b['desvio_arrastre']:.0f} en arrastre.")
        out.append("")
        out.append(f"  Hay que declararlo en la tesis. Con "
                   f"{b['n_exacto_mas_cercano']} participantes el equilibrio "
                   f"es exacto:")
        out.append(f"  son {b['n_exacto_mas_cercano'] - n} sesión(es) más.")
    out.append("")
    out.append("  Asignación:")
    out.append(f"    {'part':>5}  {'1ro':<30}{'2do':<30}{'3ro':<30}")
    for a in plan(n):
        celdas = "".join(
            "{} (F{})".format(NOMBRES[c], s).ljust(30)
            for c, s in zip(a.secuencia, a.conjuntos))
        out.append(f"    {a.participante:>5}  {celdas}")
    return "\n".join(out) + "\n"
