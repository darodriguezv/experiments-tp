"""Construye y valida el corpus de evaluación.

## Por qué el corpus se genera en vez de editarse a mano

Cada ítem tiene que cumplir reglas que no son obvias al escribirlo y que, si se
rompen, hacen que el arnés mida mal en vez de fallar:

* Una referencia que menciona un sustantivo del tablero que la entrada no dice
  (`agua` en un ítem de `COMIDA`) la descarta el filtro, y entonces el modelo
  aparece fallando cuando quien falla es el corpus.
* Una referencia sin negación en un ítem marcado `negative` es contradictoria:
  el arnés exige negar y el corpus da un ejemplo que no niega.
* Una trampa de invención que no se detecta al inyectarla es una trampa que
  nunca se dispara, y eso se lee como un resultado limpio.

Escribir 170 ítems a mano respetando eso es una garantía de error silencioso.
Acá se declaran y se **validan** antes de escribir el archivo, así que un ítem
mal formado se detiene en esta pantalla y no seis pasos después, disfrazado de
resultado.

## Qué se buscó al ampliarlo

El corpus original tenía 40 ítems y casi todos eran del tablero pelado: agua,
comida, baño, dolor. Eso mide la parte fácil. Lo que una persona con ELA
necesita decir de verdad incluye cosas que el tablero no tiene y hay que
deletrear -aspirar secreciones, acomodar una pierna, llamar a alguien- y es ahí
donde el módulo se gana el lugar o no.

Las categorías nuevas salen de lo que la literatura de CAA y de cuidados en ELA
reporta como necesidades frecuentes: respiración, postura y comodidad, dolor
localizado, higiene, deglución, temperatura, personas, entorno, estado
emocional, y la metacomunicación -«esperá», «no me entendiste»- que aparece
apenas la comunicación es lenta.

**Advertencia que vale para todo el corpus:** las referencias son juicio del
tesista, no de un clínico ni de un usuario de CAA. Sirven para comparar
configuraciones entre sí, que es para lo que se usan. No sirven para afirmar
que el módulo dice lo correcto, y eso necesita revisión de alguien con
experiencia clínica.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))

from aac_semantic.guard import foreign_nouns  # noqa: E402
from aac_semantic.metrics import NEGATION, SOFTENERS, STOP_MARKERS  # noqa: E402
from aac_semantic.prompts import normalise  # noqa: E402


def item(id_, entrada, lecturas, *, prohibido=(), polaridad="none",
         urgencia="normal", tags=(), ambiguo=False, sin_expandir=False,
         nota=""):
    d = {
        "id": id_,
        "input": entrada,
        "tags": list(tags),
        "readings": [{"gloss": g, "accept": list(a)} for g, a in lecturas],
        "forbidden_content": list(prohibido),
        "must_keep_polarity": polaridad,
        "urgency": urgencia,
    }
    if ambiguo:
        d["ambiguous"] = True
    if sin_expandir:
        d["expect_refusal"] = True
    if nota:
        d["note"] = nota
    return d


def _tiene(texto, marcas):
    t = f" {normalise(texto)} "
    return any(f" {normalise(m)} " in t for m in marcas)


def validar(items, ejemplos_prompt):
    """Devuelve la lista de problemas. Vacía = el corpus es coherente."""
    problemas = []
    vistos = set()
    for it in items:
        iid, entrada = it["id"], it["input"]
        if iid in vistos:
            problemas.append(f"{iid}: id repetido")
        vistos.add(iid)

        ent = normalise(entrada)
        accepts = [a for r in it["readings"] for a in r["accept"]]
        if not accepts and not it.get("expect_refusal"):
            problemas.append(f"{iid}: sin referencias")

        for a in accepts:
            # 1. sustantivos del tablero que la entrada no autoriza.
            # Se pregunta al FILTRO REAL en vez de reimplementar su criterio:
            # una copia se desincroniza, y cuando eso pasa el corpus valida
            # contra una regla que el sistema ya no aplica. Pasó: esta función
            # tenía su propia versión y seguía rechazando ítems que el filtro
            # ya había aprendido a aceptar.
            intrusos = foreign_nouns(entrada, a)
            for noun in intrusos:
                problemas.append(
                    f"{iid}: la referencia {a!r} dice '{noun}' y la entrada "
                    f"{entrada!r} no lo autoriza")
            # 2. polaridad
            if it["must_keep_polarity"] == "negative" and not _tiene(a, NEGATION):
                problemas.append(f"{iid}: {a!r} no niega y el ítem exige negar")
            if it["must_keep_polarity"] == "negative-stop" and not (
                    _tiene(a, STOP_MARKERS) or _tiene(a, NEGATION)):
                problemas.append(f"{iid}: {a!r} no detiene ni niega")
            # 3. urgencia suavizada
            if it["urgency"] == "high" and _tiene(a, SOFTENERS):
                problemas.append(f"{iid}: {a!r} suaviza un ítem urgente")
            # 4. una referencia no puede contener su propia trampa
            for f in it["forbidden_content"]:
                if _tiene(a, (f,)):
                    problemas.append(
                        f"{iid}: la referencia {a!r} contiene la trampa {f!r}")
        # 5. las trampas tienen que ser detectables al inyectarlas
        for f in it["forbidden_content"]:
            if not _tiene(f"quiero {f}", (f,)):
                problemas.append(f"{iid}: la trampa {f!r} no se detecta")
        # 6. sin contaminación con los ejemplos del prompt
        if ent in ejemplos_prompt:
            problemas.append(f"{iid}: {entrada!r} aparece en los ejemplos del prompt")
    return problemas


BASE = "telegraphic_es.base.jsonl"


def cargar_items():
    """Los 40 originales tal cual, más los bloques nuevos.

    Los originales se leen de un archivo congelado en vez de retipearse: 40
    ítems con sus referencias y sus trampas copiados a mano es una garantía de
    introducir un error silencioso en datos que ya estaban validados.
    """
    import corpus_items
    import corpus_items_b
    import corpus_items_c

    base = [json.loads(l) for l in
            (AQUI / BASE).read_text(encoding="utf-8").splitlines() if l.strip()]
    nuevos = (corpus_items.NUEVOS_A + corpus_items_b.NUEVOS_B
              + corpus_items_c.NUEVOS_C)
    return base + [item(i, e, l, **kw) for i, e, l, kw in nuevos]


def main() -> int:
    from aac_semantic import prompts

    ITEMS = cargar_items()

    ejemplos = set()
    for p in prompts.VARIANTS.values():
        for linea in p.system.splitlines():
            if linea.startswith("entrada:"):
                ejemplos.add(normalise(linea.split(":", 1)[1]))

    problemas = validar(ITEMS, ejemplos)
    if problemas:
        print(f"{len(problemas)} problema(s); no se escribe nada:\n")
        for p in problemas[:40]:
            print("  -", p)
        if len(problemas) > 40:
            print(f"  ... y {len(problemas) - 40} más")
        return 1

    destino = AQUI / "telegraphic_es.jsonl"
    with destino.open("w", encoding="utf-8") as fh:
        for it in ITEMS:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")

    urg = sum(1 for i in ITEMS if i["urgency"] == "high")
    amb = sum(1 for i in ITEMS if i.get("ambiguous"))
    ref = sum(1 for i in ITEMS if i.get("expect_refusal"))
    trampas = sum(len(i["forbidden_content"]) for i in ITEMS)
    refs = sum(len(r["accept"]) for i in ITEMS for r in i["readings"])
    etiquetas = {t for i in ITEMS for t in i["tags"]}
    print(f"{len(ITEMS)} ítems escritos en {destino.name}")
    print(f"  {refs} referencias, {trampas} trampas de invención")
    print(f"  {urg} urgentes, {amb} ambiguos, {ref} que no deben expandirse")
    print(f"  {len(etiquetas)} etiquetas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
