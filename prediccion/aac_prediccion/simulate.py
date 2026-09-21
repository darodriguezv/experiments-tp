"""Un usuario sintético, construido para que el arnés no se mienta solo.

## La trampa que este archivo evita

Si el simulador generara enunciados con un modelo de n-gramas y después
midiéramos un predictor de n-gramas, el resultado sería precioso y no
significaría nada: estaríamos midiendo que dos copias del mismo modelo
coinciden. Es la forma más fácil de producir un gráfico convincente y vacío.

Así que este generador **no** es el modelo. Es una persona con rutinas:
frases ligadas a la hora del día, una cola larga de cosas que dice poco, y dos
propiedades que ponen un techo por debajo del 100 %:

* **Novedad.** Una fracción de lo que dice no está en su repertorio. Es
  impredecible por construcción, y tiene que serlo: una persona real dice
  cosas nuevas todos los días.
* **Deriva.** A mitad del período aparecen frases nuevas y se apagan otras,
  como cuando cambia un síntoma o un cuidador. Un modelo que solo acumula
  frecuencia se degrada acá; uno con recencia se recupera. Esa diferencia es
  justamente lo que el parámetro de vida media tiene que ganarse.

## Lo que este archivo NO puede decir

**No predice el ahorro sobre una persona real.** Sirve para validar el
mecanismo, para barrer parámetros y para atrapar errores del arnés. El número
que la tesis puede citar sale de sesiones reales - y dado que este proyecto no
prueba con pacientes, «real» significa las sesiones del propio tesista usando
el tablero. Confundir una cosa con la otra sería el error más grave que este
módulo puede cometer, así que queda escrito acá y en el README.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .replay import Utterance

DAY = 86400.0

# Frases del repertorio. Las etiquetas de varias palabras -«por favor», «me
# siento», «tal vez»- son UNA celda del tablero y por eso van como un solo
# token; las palabras sueltas que el tablero no tiene hay que deletrearlas, y
# son justamente donde la predicción puede ahorrar más.
MORNING = [
    (["quiero", "agua"], 3.0),
    (["necesito", "medicina"], 2.0),
    (["quiero", "comida"], 2.0),
    (["por favor", "baño"], 2.5),
    (["llama", "hermana"], 0.6),
]

EVENING = [
    (["me siento", "mal"], 1.5),
    (["basta", "dolor"], 1.2),
    (["quiero", "television"], 1.0),
    (["mueve", "almohada"], 0.8),
    (["tengo", "frio"], 0.9),
]

ANYTIME = [
    (["gracias"], 4.0),
    (["sí"], 5.0),
    (["no"], 5.0),
    (["ayuda"], 1.5),
    (["por favor", "agua"], 1.8),
    (["me siento", "bien"], 1.0),
    (["tal vez"], 0.7),
    (["necesito", "kinesiologa"], 0.4),
]

# Lo que aparece recién a mitad del período: un síntoma nuevo y su vocabulario.
DRIFT_IN = [
    (["necesito", "aspirar"], 2.0),
    (["me siento", "ahogado"], 1.5),
    (["llama", "enfermera"], 1.2),
]

# Vocabulario para lo impredecible. Combinaciones que nadie repite.
NOVEL_POOL = [
    "ventana", "puerta", "musica", "radio", "manta", "silla", "lentes",
    "telefono", "carta", "foto", "perro", "jardin", "cansado", "aburrido",
]


@dataclass
class Persona:
    """Los hábitos de una persona sintética, con semilla para reproducir."""

    days: int = 60
    novelty: float = 0.15
    """Fracción de enunciados fuera del repertorio. 15 % es un supuesto, no una
    medición, y mueve el techo del ahorro más que casi cualquier otra cosa."""
    drift_day: int | None = 30
    seed: int = 7
    t0: float = 1_700_000_000.0
    rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def _repertoire(self, day: int) -> list[tuple[list[str], float]]:
        items = list(ANYTIME)
        if self.drift_day is not None and day >= self.drift_day:
            items += DRIFT_IN
            # Lo viejo no desaparece de golpe: se usa menos.
            items += [(t, w * 0.3) for t, w in MORNING]
            items += [(t, w * 0.3) for t, w in EVENING]
        else:
            items += MORNING + EVENING
        return items

    def _novel(self) -> list[str]:
        a, b = self.rng.sample(NOVEL_POOL, 2)
        return self.rng.choice([["quiero", a], ["necesito", a], [a, b]])

    def generate(self) -> list[Utterance]:
        out: list[Utterance] = []
        for day in range(self.days):
            n = self.rng.randint(12, 22)
            for _ in range(n):
                hour = self.rng.uniform(7.0, 22.0)
                if self.rng.random() < self.novelty:
                    tokens = self._novel()
                else:
                    pool = self._repertoire(day)
                    if hour < 12:
                        pool = pool + [(t, w * 2) for t, w in MORNING
                                       if (t, w) in MORNING]
                    elif hour > 18:
                        pool = pool + [(t, w * 2) for t, w in EVENING
                                       if (t, w) in EVENING]
                    weights = [w for _, w in pool]
                    tokens = list(self.rng.choices(pool, weights=weights)[0][0])
                out.append(Utterance(tokens, self.t0 + day * DAY + hour * 3600))
        out.sort(key=lambda u: u.t)
        return out
