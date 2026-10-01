"""Onboarding opcional: quién es la persona, para arrancar sin estar en blanco.

## El problema que resuelve

El módulo de predicción arranca **vacío a propósito**: sin evidencia no propone
nada, porque cada casilla cuesta una celda más de lectura en toda selección.
Eso es correcto y es lo que evita el impuesto de proponer ruido.

Pero tiene un costo real: las primeras sesiones no ahorran nada. Alguien que
recién recibe el dispositivo -justo cuando más le cuesta todo- usa un tablero
pelado hasta que repite una frase dos veces.

Un perfil rompe ese arranque en frío **sin inventar evidencia**: no se finge que
la persona dijo algo que no dijo, se declara qué vocabulario le corresponde y
se marca como venido del perfil y no del uso.

## Qué aporta, en orden de valor medible

1. **Nombres propios.** Si su hija se llama Ana, decir ANA cuesta hoy cinco
   permanencias deletreadas. Como frase sembrada cuesta una. Es el ahorro más
   grande que existe en el tablero y no necesita ningún modelo para calcularse.
2. **Vocabulario de su vida.** Un carpintero y una maestra no nombran las
   mismas cosas, y ninguna de las dos está en las veinte palabras del tablero.
3. **Glosario del SLM.** Agregar sus términos al glosario es *información*, que
   es exactamente el único tipo de agregado al prompt que la ablación midió
   como útil. Las cuatro variantes que agregaron reglas o contexto no mejoraron.

Lo que deliberadamente **no** hace: escribir frases sobre su salud, su estado de
ánimo o sus necesidades. El perfil aporta *sustantivos*, no afirmaciones. Que
alguien sea diabético no autoriza a sembrar "necesito insulina" como algo que
dijo.

## Advertencia de datos personales

Un perfil con edad, sexo, profesión y nombres de familiares **es dato personal
identificable**, y de los nombres de terceros que nunca consintieron. Cambia la
declaración del proyecto: hasta ahora no había ninguno.

Por eso: es **opcional**, vive en un archivo aparte (`*.perfil.json`), está
excluido del repositorio, y `main.py perfil --borrar` lo elimina. Nada del
perfil se mezcla con las métricas ni sale del equipo.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

MAX_TERMINOS = 40
"""Tope de términos sembrados. No es arbitrario: cada frase sembrada compite
por las casillas de la franja con lo que la persona dijo de verdad, y el uso
real debe ganarle al perfil apenas exista. Cuarenta cubre nombres, oficio e
intereses sin ahogar la evidencia."""

PESO_PERFIL = 2.0
"""Con cuánto peso entra un término del perfil.

Igual a `min_evidence`, o sea el mínimo para ser ofrecido: el perfil dice "esto
es plausible", no "esto ya pasó". Una sola vez que la persona diga algo de
verdad lo empuja por encima de cualquier cosa sembrada, que es el orden
correcto."""


@dataclass
class Perfil:
    """Lo que la persona (o quien la acompaña) quiera declarar. Todo opcional."""

    nombre: str = ""
    edad: int | None = None
    sexo: str = ""
    profesion: str = ""
    region: str = ""
    """Para el trato: en el Río de la Plata es «quedate», en México «quédate».
    El tablero habla por la persona, así que tiene que sonar como ella."""

    personas: list[str] = field(default_factory=list)
    """Nombres de quienes la acompañan. Son los que más permanencias ahorran y
    los más sensibles: son datos de terceros que no consintieron nada."""

    intereses: list[str] = field(default_factory=list)
    objetos: list[str] = field(default_factory=list)
    """Cosas concretas de su entorno: su silla, su radio, su perro."""

    notas: str = ""

    def resumen(self) -> str:
        """Una línea para el prompt. Sin datos de salud, a propósito."""
        partes = []
        if self.edad:
            partes.append(f"{self.edad} años")
        if self.sexo:
            partes.append(self.sexo)
        if self.profesion:
            partes.append(f"trabajó de {self.profesion}")
        if self.region:
            partes.append(f"de {self.region}")
        return ", ".join(partes)

    def terminos(self) -> list[str]:
        """Los sustantivos que el perfil aporta, sin duplicados ni vacíos."""
        bruto = list(self.personas) + list(self.intereses) + list(self.objetos)
        vistos, salida = set(), []
        for t in bruto:
            t = " ".join(str(t).split()).strip().lower()
            if not t or t in vistos:
                continue
            vistos.add(t)
            salida.append(t)
        return salida[:MAX_TERMINOS]

    # ------------------------------------------------------------ persistencia

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(p)

    @classmethod
    def load(cls, path: str | Path) -> "Perfil":
        p = Path(path)
        if not p.exists():
            return cls()
        d = json.loads(p.read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})

    @property
    def vacio(self) -> bool:
        return not any((self.nombre, self.edad, self.sexo, self.profesion,
                        self.region, self.personas, self.intereses,
                        self.objetos, self.notas))


def _sin_tildes(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def frases_semilla(perfil: Perfil) -> list[list[str]]:
    """Frases plausibles a partir del perfil, SIN modelo.

    Deliberadamente mecánico y no generativo. Un modelo podría escribir frases
    más naturales, y también podría escribir una sobre un síntoma que nadie
    mencionó - y una frase sembrada se ofrece como si la persona ya la hubiera
    dicho. El riesgo no está balanceado, así que la siembra se arma con
    plantillas que solo combinan verbos del tablero con sustantivos declarados.

    El modelo sí se usa, pero para *proponerle términos a la persona* antes de
    guardar nada: ahí ella revisa y acepta. Ver `sugerir_terminos`.
    """
    frases: list[list[str]] = []
    for nombre in perfil.personas:
        n = nombre.strip().lower()
        if not n:
            continue
        frases += [["llama", n], ["quiero", "ver", "a", n], [n]]
    for cosa in perfil.intereses + perfil.objetos:
        c = cosa.strip().lower()
        if not c:
            continue
        frases += [["quiero", c], [c]]
    return frases[:MAX_TERMINOS * 3]


def sembrar(modelo, perfil: Perfil, ahora: float | None = None) -> int:
    """Carga el perfil en el modelo de usuario. Devuelve cuántas frases sembró.

    Se siembra con el peso mínimo (`PESO_PERFIL`), así que una sola vez que la
    persona diga algo real lo supera. El perfil es un punto de partida, no una
    creencia sobre lo que va a decir.
    """
    import time as _t

    ahora = _t.time() if ahora is None else ahora
    n = 0
    for tokens in frases_semilla(perfil):
        # `observe` suma 1.0 por vez; se llama las veces necesarias para llegar
        # al umbral en vez de tocar los pesos a mano, así el decaimiento y la
        # marca de tiempo quedan consistentes con todo lo demás.
        for _ in range(int(PESO_PERFIL)):
            modelo.observe(tokens, ahora)
        n += 1
    return n


PROMPT_SUGERENCIAS = """Sos un asistente que arma el vocabulario de un tablero de comunicación para una persona que no puede hablar.

Te doy una descripción breve de quién es. Devolvé SOLO sustantivos y nombres concretos que esa persona probablemente necesite nombrar en el día a día: objetos de su entorno, actividades suyas, lugares.

Reglas:
1. NADA de salud, síntomas, medicación ni estados de ánimo. Eso lo dice la persona, no vos.
2. Sustantivos sueltos, en minúscula, uno por línea. Nada de frases ni verbos.
3. Como máximo 15. Concretos y cotidianos, no abstractos.
4. Si la descripción no alcanza para sugerir nada, devolvé una línea vacía."""


def sugerir_terminos(perfil: Perfil, backend) -> list[str]:
    """Le pide términos al SLM, para que la PERSONA los revise antes de guardar.

    El modelo propone, nunca decide. Lo que devuelve va a una pantalla donde se
    aceptan o se descartan de a uno, y recién lo aceptado entra al perfil. Es la
    misma forma que el resto del sistema: candidatos, no respuestas.

    El filtro de abajo es duro y a propósito. Un término sobre salud sembrado
    como frase se ofrecería después como si la persona lo hubiera dicho, y eso
    es precisamente lo que el módulo entero trabaja para que no pase.
    """
    descripcion = perfil.resumen()
    if perfil.notas:
        descripcion += f". {perfil.notas}"
    if not descripcion.strip():
        return []

    completion = backend.complete(PROMPT_SUGERENCIAS, descripcion)
    if completion.error:
        return []
    return _filtrar(completion.text)


VETADO = (
    # Raíces, no palabras completas: la lista anterior tenía "medicina" y
    # "medicamento" y dejaba pasar "la medicación", que es la forma que el
    # modelo usó de verdad. Enumerar flexiones es una carrera que se pierde.
    "dolor", "duele", "medic", "remedio", "pastilla", "farmac",
    "sintoma", "enferm", "diagnost", "tratamiento", "terapia", "clinic",
    "hospital", "triste", "miedo", "angustia", "depres", "ansied", "animo",
    "morir", "muerte", "sonda", "oxigeno", "respirador", "silla de ruedas",
)
"""Nada de salud ni de estado de ánimo en el vocabulario sembrado.

Esos términos sí pueden aparecer -y aparecen- cuando la PERSONA los escribe.
La diferencia es quién los dice: sembrarlos desde un perfil pondría en su boca
un tema que nadie planteó, y el tablero ofrecería "me duele" a alguien que no
dijo nada parecido."""


def _filtrar(texto: str) -> list[str]:
    salida, vistos = [], set()
    for linea in texto.splitlines():
        t = linea.strip().strip("-*•·0123456789. ").lower()
        t = " ".join(t.split())
        if not t or len(t) > 24 or len(t.split()) > 3:
            continue
        plano = _sin_tildes(t)
        if any(v in plano for v in VETADO):
            continue
        if plano in vistos:
            continue
        vistos.add(plano)
        salida.append(t)
    return salida[:15]
