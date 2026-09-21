"""El modelo del usuario: qué dice esta persona, y hace cuánto.

## Por qué conteos y no una red

La tentación es afinar un modelo de lenguaje sobre lo que la persona escribió.
Es la herramienta equivocada acá, por tres razones que empujan igual:

* **Una sola vez tiene que alcanzar.** Si alguien dice «llamá a mi hermana» el
  martes, el miércoles tiene que estar disponible. Un modelo neuronal necesita
  muchos ejemplos y un reentrenamiento; una tabla de conteos lo aprende en la
  primera observación. Para alguien que paga ~800 ms por palabra, esperar a la
  décima repetición es inaceptable.
* **El SLM ya generaliza.** El módulo de expansión convierte `AGUA NO` en
  frases que nunca vio. Este módulo no tiene que hacer eso: tiene que
  **memorizar**. Son trabajos distintos y conviene que los hagan piezas
  distintas - si la personalización también generalizara, sus errores serían
  invenciones, que es justo lo que el otro módulo trabaja para evitar.
* **Se puede auditar.** «Esta predicción apareció porque la dijiste 7 veces, la
  última anteayer» es explicable en una frase. El contenido es médico y lo dice
  alguien que quizá no pueda corregir un malentendido; que el sistema sea
  inspeccionable no es un lujo.

## Recencia: implementada, pero NO validada

Los conteos decaen con vida media, con el argumento de que un hábito de hace
seis meses no debería ganarle a uno de la semana pasada. El decaimiento es
exacto y perezoso - se guarda `(peso, instante)` y se descuenta al leer - así
que no cuesta casi nada.

**Y sobre el usuario sintético no se gana nada con él.** Medido: vida media de
30 días da 27.1 % de ahorro y no olvidar nunca da 26.8 %, que es la misma cifra;
olvidar rápido (3 días) *empeora* mucho, a 14.0 %. O sea que el parámetro
importa solo por el lado de romperlo.

Se mantiene por dos razones honestas, ninguna de ellas «funciona mejor»: cuesta
casi nada, y una deriva real - un síntoma que aparece, un cuidador que cambia -
es probablemente más brusca que la que el simulador produce. Pero mientras no
se mida sobre sesiones reales, **afirmar que la recencia ayuda sería inventar un
resultado**. Hoy lo defendible es que 30 días no hace daño.

## Nada de arranque en frío inventado

Un modelo sin datos **no propone**. Es deliberado, y va contra el reflejo de
llenar las casillas con una lista de frecuencias del español: cada casilla de
predicción cuesta una celda más que leer en toda selección, así que una
predicción sin evidencia no es neutral, es un impuesto sobre la vista a cambio
de nada. `min_evidence` es ese umbral.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

SEP = "\x1f"
"""Separador para serializar contextos a claves JSON. Unit Separator: no puede
aparecer en una etiqueta del tablero."""

DEFAULT_HALFLIFE_DAYS = 30.0
"""Cuánto tarda una observación en valer la mitad. Treinta días es un punto de
partida, no un hallazgo: es el parámetro que `main.py sweep` barre."""

MAX_ORDER = 3
"""Hasta trigramas. Más orden necesita más datos de los que una persona genera
en meses, y el respaldo a bigramas ya cubre el caso."""


@dataclass
class _Count:
    """Un peso con fecha, para poder descontarlo sin recorrer la tabla."""

    w: float = 0.0
    t: float = 0.0

    def value(self, now: float, halflife_s: float) -> float:
        if self.w <= 0.0:
            return 0.0
        return self.w * math.pow(0.5, (now - self.t) / halflife_s)

    def bump(self, now: float, halflife_s: float, amount: float = 1.0) -> None:
        self.w = self.value(now, halflife_s) + amount
        self.t = now


@dataclass
class UserModel:
    """Lo que sabemos de esta persona. Se guarda en un JSON y nada más."""

    halflife_days: float = DEFAULT_HALFLIFE_DAYS
    min_evidence: float = 2.0
    """Peso mínimo para que algo se ofrezca. Por debajo, la casilla queda vacía:
    ver la nota sobre arranque en frío arriba."""

    ngrams: dict[str, dict[str, _Count]] = field(default_factory=dict)
    phrases: dict[str, _Count] = field(default_factory=dict)
    utterances_seen: int = 0

    # ------------------------------------------------------------------ util

    @property
    def _halflife_s(self) -> float:
        return self.halflife_days * 86400.0

    @staticmethod
    def _ctx_key(context: tuple[str, ...]) -> str:
        return SEP.join(context)

    # --------------------------------------------------------------- escribir

    def observe(self, tokens: list[str], now: float | None = None) -> None:
        """Registra un enunciado que la persona efectivamente dijo.

        Solo lo hablado, nunca lo tecleado y borrado. Un borrador abandonado es
        justamente lo que la persona decidió **no** decir; aprenderlo empujaría
        al sistema hacia sus propios errores.
        """
        if not tokens:
            return
        now = time.time() if now is None else now
        hl = self._halflife_s

        self.phrases.setdefault(SEP.join(tokens), _Count()).bump(now, hl)
        self.utterances_seen += 1

        # El símbolo de fin importa: sin él el modelo no aprende que «gracias»
        # suele cerrar y nunca sabría cuándo dejar de proponer.
        seq = list(tokens) + ["\x03"]
        for i, symbol in enumerate(seq):
            for order in range(MAX_ORDER):
                if i - order < 0:
                    continue
                context = tuple(seq[i - order:i])
                key = self._ctx_key(context)
                table = self.ngrams.setdefault(key, {})
                table.setdefault(symbol, _Count()).bump(now, hl)

    # ------------------------------------------------------------------- leer

    def next_symbols(self, context: tuple[str, ...], now: float | None = None,
                     limit: int = 5) -> list[tuple[str, float]]:
        """Símbolos probables tras `context`, con respaldo de orden.

        Se prueba el contexto más largo primero y se baja mientras no haya
        evidencia suficiente. No se interpola entre órdenes: una mezcla daría
        números más suaves y una explicación mucho peor, y acá poder explicar
        una predicción vale más que el último punto de precisión.
        """
        now = time.time() if now is None else now
        hl = self._halflife_s
        for order in range(min(len(context), MAX_ORDER - 1), -1, -1):
            table = self.ngrams.get(self._ctx_key(context[len(context) - order:]
                                                  if order else ()))
            if not table:
                continue
            ranked = sorted(
                ((s, c.value(now, hl)) for s, c in table.items() if s != "\x03"),
                key=lambda p: -p[1])
            ranked = [(s, w) for s, w in ranked if w >= self.min_evidence]
            if ranked:
                return ranked[:limit]
        return []

    def phrase_completions(self, prefix: tuple[str, ...],
                           now: float | None = None, limit: int = 5
                           ) -> list[tuple[list[str], float]]:
        """Enunciados guardados que empiezan con `prefix`.

        Es la parte que de verdad ahorra: una permanencia entrega la frase
        entera. Un prefijo vacío devuelve los enunciados más habituales, que es
        lo que hace falta al empezar a hablar.
        """
        now = time.time() if now is None else now
        hl = self._halflife_s
        out: list[tuple[list[str], float]] = []
        for key, count in self.phrases.items():
            tokens = key.split(SEP)
            if len(tokens) <= len(prefix):
                continue
            if tuple(tokens[:len(prefix)]) != prefix:
                continue
            w = count.value(now, hl)
            if w >= self.min_evidence:
                out.append((tokens, w))
        out.sort(key=lambda p: (-p[1], len(p[0])))
        return out[:limit]

    @property
    def last_seen(self) -> float:
        """Instante de la observación más nueva, o 0 si no hay ninguna.

        Sirve para explicar el silencio: un modelo con mil enunciados de hace
        dos años no propone nada, y sin este dato eso parece un bug en vez de
        el decaimiento haciendo exactamente su trabajo.
        """
        return max((c.t for c in self.phrases.values()), default=0.0)

    # ------------------------------------------------------------ persistencia

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "halflife_days": self.halflife_days,
            "min_evidence": self.min_evidence,
            "utterances_seen": self.utterances_seen,
            "ngrams": {k: {s: [c.w, c.t] for s, c in v.items()}
                       for k, v in self.ngrams.items()},
            "phrases": {k: [c.w, c.t] for k, c in self.phrases.items()},
        }
        # Archivo temporal y reemplazo atómico: el modelo se guarda mientras la
        # persona usa el tablero, y un corte de luz a mitad de escritura no
        # puede dejarla sin su historial.
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(blob, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "UserModel":
        path = Path(path)
        if not path.exists():
            return cls()
        blob = json.loads(path.read_text(encoding="utf-8"))
        model = cls(halflife_days=blob.get("halflife_days",
                                           DEFAULT_HALFLIFE_DAYS),
                    min_evidence=blob.get("min_evidence", 2.0),
                    utterances_seen=blob.get("utterances_seen", 0))
        model.ngrams = {k: {s: _Count(w, t) for s, (w, t) in v.items()}
                        for k, v in blob.get("ngrams", {}).items()}
        model.phrases = {k: _Count(w, t)
                         for k, (w, t) in blob.get("phrases", {}).items()}
        return model
