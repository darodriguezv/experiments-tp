"""El registro del español: voseo o tú.

Dos cosas, las dos necesarias para medir un prompt en español neutro sin
hacerle trampa a nadie:

1. **Referencias neutras.** 58 de las 455 frases aceptadas del corpus están en
   voseo ("¿me pasás agua?", "acomodáme la almohada", "llamá a la enfermera").
   La métrica compara palabras con acento incluido, así que "pásame" no
   coincide con "pasame" y "llama a la enfermera" pierde contra "llamá". Un
   modelo que contesta bien en español neutro perdía aciertos por el registro,
   no por el contenido.

   `con_referencias_neutras` AGREGA el equivalente con tú y conserva el
   original. No le quita nada al voseo: lo que se mide es si el contenido es el
   correcto, y el registro se mide aparte.

2. **Cuánto voseo produce un modelo.** `tiene_voseo` es una heurística sobre
   formas concretas, no un analizador morfológico. Se valida en
   `smoke_registro` contra frases escritas a mano en los dos registros.
"""

from __future__ import annotations

import copy
import re

from .prompts import normalise

# Forma de vos -> forma de tú. Sale de mirar una por una las 58 frases del
# corpus. Las que parecían voseo y no lo son quedaron afuera a propósito:
# "me atraganté" y "me equivoqué" son primera persona del pretérito.
VOS_A_TU: dict[str, str] = {
    "ayudame": "ayúdame", "dejá": "deja", "llamá": "llama", "abrí": "abre",
    "podés": "puedes", "prendé": "prende", "movéme": "muéveme",
    "moveme": "muéveme", "acomodáme": "acomódame", "acomodame": "acomódame",
    "pará": "para", "bajá": "baja", "cerrá": "cierra", "tapame": "tápame",
    "quedate": "quédate", "esperá": "espera", "pasás": "pasas",
    "llevame": "llévame", "seguí": "sigue", "subí": "sube", "traés": "traes",
    "mirá": "mira", "acomodá": "acomoda", "corré": "corre",
    "sentáme": "siéntame", "sentame": "siéntame", "limpiá": "limpia",
    "lavame": "lávame", "cambiá": "cambia", "andá": "ve", "destapame":
    "destápame", "poné": "pon", "pasame": "pásame", "hacés": "haces",
    "repetí": "repite", "sos": "eres", "tenés": "tienes", "querés":
    "quieres", "sabés": "sabes", "vení": "ven", "decí": "di", "tené": "ten",
    "apagá": "apaga", "frená": "frena", "detené": "detén", "terminá":
    "termina", "dame": "dame", "sentate": "siéntate", "acostame":
    "acuéstame", "dejame": "déjame", "dejalo": "déjalo", "escuchá":
    "escucha", "vení acá": "ven aquí", "girame": "gírame", "levantame":
    "levántame", "rascame": "ráscame", "tocame": "tócame", "secame":
    "sécame", "abrime": "ábreme", "traeme": "tráeme", "sacame": "sácame",
    "subime": "súbeme", "bajame": "bájame", "poneme": "ponme",
    "avisale": "avísale", "llamalo": "llámalo", "llamala": "llámala",
    "decile": "dile", "pedile": "pídele", "hablá": "habla", "sacá": "saca",
    "tomá": "toma", "vos": "tú",
}
"""Acotado a lo que se vio. Agregar una forma acá cambia qué cuenta como
acierto y qué como voseo, así que se agrega mirando una salida real."""

# Palabras que terminan en -ás/-és/-ís o en vocal acentuada y NO son voseo.
_NO_VOSEO = frozenset("""
más jamás atrás además detrás estás das vas después través país quizás compás
está mamá papá sofá allá acá café bebé té sé aquí allí ahí así sí mí tú ya
también qué cómo dónde cuándo cuál cuáles quién quiénes atraganté equivoqué
comí bebí dormí sentí perdí caí oí leí tosí vomité desperté acosté levanté
mareé ahogué aún ojalá jabón inglés francés interés estrés revés ciprés
""".split())

_PALABRA = re.compile(r"[a-záéíóúüñ]+")
# Presente de vos: -ás, -és, -ís (hablás, tenés, sentís). E imperativo de vos:
# termina en vocal acentuada (hablá, tené, vení).
_FINAL_VOS = re.compile(r"^[a-záéíóúüñ]{2,}(?:ás|és|ís|á|é|í)$")


def tiene_voseo(texto: str) -> list[str]:
    """Las formas de vos que aparecen en el texto. Vacío si no hay."""
    halladas = []
    for p in _PALABRA.findall(normalise(texto)):
        if p in _NO_VOSEO:
            continue
        if p in VOS_A_TU and VOS_A_TU[p] != p:
            halladas.append(p)
        elif _FINAL_VOS.match(p):
            halladas.append(p)
    return halladas


def a_neutro(texto: str) -> str:
    """La misma frase con las formas de vos cambiadas por las de tú."""
    def cambiar(m: re.Match) -> str:
        p = m.group(0)
        q = VOS_A_TU.get(p.lower())
        if q is None or q == p.lower():
            return p
        return q.capitalize() if p[0].isupper() else q
    return re.sub(r"[A-Za-záéíóúüñÁÉÍÓÚÜÑ]+", cambiar, texto)


def con_referencias_neutras(corpus: list[dict]) -> tuple[list[dict], int]:
    """Copia del corpus con el equivalente neutro de cada frase en voseo.

    Devuelve también cuántas frases se agregaron, para poder decirlo.
    """
    salida = copy.deepcopy(corpus)
    agregadas = 0
    for item in salida:
        for lectura in item.get("readings", []):
            aceptadas = lectura.get("accept", [])
            ya = {normalise(a) for a in aceptadas}
            nuevas = []
            for a in aceptadas:
                n = a_neutro(a)
                if normalise(n) not in ya:
                    ya.add(normalise(n))
                    nuevas.append(n)
            lectura["accept"] = aceptadas + nuevas
            agregadas += len(nuevas)
    return salida, agregadas


def smoke_registro() -> None:
    """Frases escritas a mano en los dos registros. Si esto falla, el detector
    está midiendo otra cosa."""
    vos = ["¿me pasás agua?", "acomodáme la almohada", "llamá a la enfermera",
           "prendé la luz", "sos muy amable", "vení acá", "no podés irte",
           "pará, me duele", "quedate conmigo", "tenés que ayudarme"]
    # Emparejadas con `vos` una a una: es lo que tiene que dar `a_neutro`.
    # "prende" y no "enciende", y "ven acá" y no "ven aquí": las dos son
    # neutras, y el mapeo cambia la forma verbal, no el vocabulario.
    tu = ["¿me pasas agua?", "acomódame la almohada", "llama a la enfermera",
          "prende la luz", "eres muy amable", "ven acá", "no puedes irte",
          "para, me duele", "quédate conmigo", "tienes que ayudarme",
          "quiero más agua", "me atraganté", "estás ahí?", "ya está",
          "después", "me duele mucho", "así está bien", "quizás",
          "necesito ir al baño", "no quiero comer", "enciende la luz",
          "ven aquí"]
    faltan = [v for v in vos if not tiene_voseo(v)]
    sobran = [(t, tiene_voseo(t)) for t in tu if tiene_voseo(t)]
    assert not faltan, f"no detecta voseo en {faltan}"
    assert not sobran, f"ve voseo donde no hay: {sobran}"
    for v, t in zip(vos, tu):
        assert normalise(a_neutro(v)) == normalise(t), (v, a_neutro(v), t)


if __name__ == "__main__":
    smoke_registro()
    print("registro: ok")
