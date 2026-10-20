"""Las tres condiciones, el caso del roleplay y las frases que se piden.

## Qué separa a cada condición

| | tablero | predicción | SLM | contexto del caso |
|-|-|-|-|-|
| `letra` | deletreo | no | no | no |
| `sin_contexto` | palabras + deletreo | no | sí | no |
| `con_contexto` | palabras + deletreo | sí | sí | sí |

La línea de base es `letra` a propósito: es lo que hoy usa mucha gente con ELA
cuando el tablero no tiene nada más, y es el único punto de comparación que no
depende de ninguna decisión nuestra. Comparar contra una versión degradada de
nuestro propio sistema mediría qué tan bien apagamos una función, no qué aporta
el sistema.

## Por qué el caso es el mismo para todos

Diez participantes no alcanzan para que el caso sea una variable. Con un caso
por persona, cada comparación tendría un caso distinto adentro y la varianza
entre casos se mezclaría con la diferencia entre condiciones. Fijarlo convierte
al caso en una **constante**, que es lo único defendible con esta muestra.

El precio hay que decirlo: los resultados valen **para este caso**. Que el
sistema contextualizado ayude a Rosa no demuestra que ayude a cualquiera; lo
demuestra para alguien como Rosa. Eso es una limitación del estudio, no un
defecto del diseño, y con n=10 la alternativa es no poder afirmar nada.

## Nadie con ELA participa

Los participantes son voluntarios sanos haciendo un rol. Eso cambia lo que el
estudio puede afirmar, y conviene decirlo antes que lo pregunten:

* **Sí mide** el costo de interacción: cuántas permanencias y cuánto tiempo
  cuesta decir algo, y cuánto de eso ahorra cada condición. La mirada de una
  persona sana sobre este tablero se comporta parecido.
* **No mide** la experiencia de quien depende del dispositivo. Un voluntario
  que se cansa deja el experimento; alguien con ELA se queda sin voz. La
  frustración, la urgencia real y la fatiga de un día entero no están acá y
  ningún número de esta tesis puede hablar de eso.

## Las frases: tres conjuntos emparejados

Si las tres condiciones usaran las mismas frases, para la tercera la persona ya
sabría de memoria dónde está cada letra. Por eso hay tres conjuntos que rotan.

Están emparejados en lo que de verdad cuesta: **cantidad de caracteres** (lo
que paga la condición `letra`) y **cuántas de sus palabras están en el tablero**
(lo que paga la condición con palabras). Si un conjunto fuera más corto, la
condición que le tocara ganaría por el conjunto y no por el sistema.
`verificar_emparejamiento()` lo chequea en vez de confiar en que quedó bien.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Caso:
    """El personaje que el participante interpreta, y que el SLM conoce."""

    nombre: str
    resumen: str
    guion: str
    """Lo que lee el participante antes de empezar. Segunda persona: es él
    quien lo va a interpretar."""

    contexto_modelo: str
    """Lo que se le agrega al prompt del SLM en la condición contextualizada.

    Es DISTINTO del guion a propósito: el participante necesita motivación y
    circunstancia para actuar; el modelo necesita vocabulario y relaciones.
    Pasarle el guion entero al modelo le daría las frases objetivo escritas, y
    entonces el estudio mediría si el modelo sabe copiar."""


CASO = Caso(
    nombre="Rosa",
    resumen="72 años, jubilada, vive con su hija Marta. ELA bulbar: entiende "
            "todo y no puede hablar.",
    guion=(
        "Usted es Rosa, tiene 72 años y vive con su hija Marta.\n\n"
        "Hace ocho meses le cuesta hablar y desde hace dos no puede nada. "
        "Entiende todo lo que pasa a su alrededor, se acuerda de todo y tiene "
        "opiniones sobre todo: lo único que perdió es la voz.\n\n"
        "Es media tarde. Marta está en la cocina. Usted está en el sillón del "
        "living, con el tablero. Hace un rato que quiere decirle algunas "
        "cosas.\n\n"
        "Diga cada frase que le vayan pidiendo, como la diría Rosa. No hace "
        "falta que sea palabra por palabra: lo que importa es que se entienda "
        "lo mismo."
    ),
    contexto_modelo=(
        "La persona que usa el tablero se llama Rosa, tiene 72 años y está "
        "jubilada. Vive con su hija Marta. Entiende todo y no puede hablar.\n"
        "Personas de su entorno: Marta (su hija), Elena (su hermana), "
        "el doctor Ferreyra.\n"
        "Cosas de su casa que suele nombrar: el sillón, la ventana, la radio, "
        "la manta, el jardín."
    ),
)


@dataclass(frozen=True)
class Tarea:
    """Una frase que se le pide decir. `intencion` es lo que debe entenderse."""

    id: str
    intencion: str
    pista: str
    """Cómo se le muestra al participante. Describe la INTENCIÓN, nunca las
    palabras exactas: si se le dicta la frase, el estudio mide transcripción y
    no comunicación, y la condición con SLM pierde todo sentido porque no hay
    nada que interpretar."""


# Tres conjuntos emparejados. Cada uno: seis tareas, mezcla de cotidiano y
# urgente, con una ambigua donde los candidatos del SLM tienen algo que hacer.
TAREAS: tuple[tuple[Tarea, ...], ...] = (
    (
        Tarea("A1", "quiero agua", "Tiene sed."),
        Tarea("A2", "llama a marta", "Quiere que venga su hija."),
        Tarea("A3", "tengo mucho frio", "Tiene mucho frío."),
        Tarea("A4", "no quiero comer ahora",
              "Le ofrecen comida y no quiere todavía."),
        Tarea("A5", "me duele mucho la espalda",
              "Le duele la espalda, y bastante."),
        Tarea("A6", "quiero salir al jardin", "Quiere salir al jardín."),
    ),
    (
        Tarea("B1", "quiero dormir", "Tiene sueño."),
        Tarea("B2", "llama a elena", "Quiere que venga su hermana."),
        Tarea("B3", "tengo calor", "Tiene calor."),
        Tarea("B4", "no quiero ver television",
              "Le ofrecen poner la tele y no quiere."),
        Tarea("B5", "me duele mucho la cabeza",
              "Le duele la cabeza, y bastante."),
        Tarea("B6", "quiero abrir la ventana", "Quiere que abran la ventana."),
    ),
    (
        Tarea("C1", "quiero la manta", "Quiere que le traigan la manta."),
        Tarea("C2", "llama al doctor", "Quiere que llamen al médico."),
        Tarea("C3", "estoy cansada", "Está cansada."),
        Tarea("C4", "no quiero bañarme ahora",
              "Le proponen bañarla y no quiere ahora."),
        Tarea("C5", "me duele mucho la garganta",
              "Le duele la garganta, y bastante."),
        Tarea("C6", "quiero la radio", "Quiere escuchar la radio."),
    ),
)

PALABRAS_TABLERO = frozenset({
    "si", "no", "por favor", "gracias", "hola", "quiero", "necesito",
    "me siento", "ayuda", "basta", "agua", "comida", "baño", "dolor",
    "medicina", "mas", "menos", "bien", "mal", "tal vez",
})
"""Lo que está como celda en la página de palabras. Se usa para verificar el
emparejamiento de los conjuntos, no para puntuar."""


def _sin_tildes(s: str) -> str:
    import unicodedata

    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def costo_deletreo(tarea: Tarea) -> int:
    """Caracteres a seleccionar si se deletrea todo, espacios incluidos.

    Es el piso de permanencias de la condición `letra`, y por eso la medida
    natural para emparejar conjuntos."""
    return len(tarea.intencion)


def cubierto_por_tablero(tarea: Tarea) -> int:
    """Cuántas palabras de la frase son celdas del tablero."""
    return sum(1 for p in _sin_tildes(tarea.intencion).split()
               if p in {_sin_tildes(w) for w in PALABRAS_TABLERO})


def verificar_emparejamiento(tolerancia_chars: float = 0.05) -> list[str]:
    """Chequea que los conjuntos cuesten parecido. Devuelve las quejas.

    Sin esto, un conjunto más corto le regalaría la victoria a la condición que
    le tocara, y como los conjuntos rotan contra las condiciones el sesgo no se
    cancelaría: se repartiría de forma desigual entre participantes.
    """
    quejas = []
    largos = [sum(costo_deletreo(t) for t in conj) for conj in TAREAS]
    medio = sum(largos) / len(largos)
    for i, largo in enumerate(largos):
        if abs(largo - medio) / medio > tolerancia_chars:
            quejas.append(
                f"el conjunto F{i} cuesta {largo} caracteres contra "
                f"{medio:.0f} de promedio ({(largo - medio) / medio:+.0%}): "
                f"la condición que le toque arranca con ventaja o desventaja")

    tamanos = {len(c) for c in TAREAS}
    if len(tamanos) > 1:
        quejas.append(f"los conjuntos no tienen la misma cantidad de tareas: "
                      f"{sorted(len(c) for c in TAREAS)}")

    cubiertos = [sum(cubierto_por_tablero(t) for t in conj) for conj in TAREAS]
    if max(cubiertos) - min(cubiertos) > 2:
        quejas.append(
            f"las palabras que ya están en el tablero se reparten desparejo "
            f"entre conjuntos: {cubiertos}. La condición con palabras rinde "
            f"distinto según qué conjunto le toque")
    return quejas


def resumen_conjuntos() -> str:
    out = ["Conjuntos de frases", ""]
    out.append(f"  {'conj':<6}{'tareas':>7}{'chars':>7}{'en tablero':>12}")
    out.append("  " + "-" * 32)
    for i, conj in enumerate(TAREAS):
        out.append(f"  F{i:<5}{len(conj):>7}"
                   f"{sum(costo_deletreo(t) for t in conj):>7}"
                   f"{sum(cubierto_por_tablero(t) for t in conj):>12}")
    quejas = verificar_emparejamiento()
    out.append("")
    if quejas:
        out.append("  EMPAREJAMIENTO ROTO:")
        out += [f"    - {q}" for q in quejas]
    else:
        out.append("  Emparejados: ningún conjunto le da ventaja a la "
                   "condición que le toque.")
    return "\n".join(out) + "\n"


CONFIG_CONDICION = {
    # `predecir` enciende el módulo de predicción personalizada; `expandir` el
    # SLM; `contexto` le pasa `CASO.contexto_modelo` al prompt.
    "letra": {"expandir": False, "predecir": False, "contexto": False,
              "pagina_inicial": "abc"},
    "sin_contexto": {"expandir": True, "predecir": False, "contexto": False,
                     "pagina_inicial": "core"},
    "con_contexto": {"expandir": True, "predecir": True, "contexto": True,
                     "pagina_inicial": "core"},
}

INSTRUCCIONES = {
    "letra": "Va a escribir cada frase letra por letra. El tablero no le va a "
             "proponer nada: lo que usted deletree es lo que se dice.",
    "sin_contexto": "Ahora el tablero tiene palabras enteras, y cuando toque "
                    "HABLAR le va a proponer algunas formas de decirlo. Elija "
                    "la que más se parezca a lo que quiso decir, o la última "
                    "opción para que diga exactamente lo que escribió.",
    "con_contexto": "Igual que antes, pero el sistema ya sabe quién es Rosa y "
                    "quiénes la rodean. Va a notar que las propuestas están "
                    "más cerca de su situación.",
}
