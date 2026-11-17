"""Las decisiones del filtro de Python, para comprobar el puerto de JavaScript.

    python web/generar_fixture_filtro.py

El filtro que protege a la persona de un vuelco de polaridad existe dos veces:
en `slm/aac_semantic/guard.py`, con el que se midió todo, y en
`web/js/expansion.js`, que es lo que corre en la página. Si divergen, las
tablas de la tesis describen un filtro que no es el desplegado.

Esto vuelca, para miles de casos, lo que decidió el de Python. La prueba
`web/pruebas/expansion.test.mjs` le pide al de JavaScript que decida
exactamente lo mismo en cada uno.

Los casos salen de dos lados:

* **todas las respuestas reales** guardadas en `slm/results/`, de quince
  modelos: es la distribución de entradas que el filtro va a ver de verdad;
* **casos construidos** para pegarle a cada regla en su borde: negación dentro
  de otra palabra, clíticos, acentos, puntuación, repeticiones, sustantivos
  ajenos. Las respuestas reales casi no los tocan, y ahí es donde un puerto se
  equivoca.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "slm"))

from aac_semantic import guard, metrics, prompts  # noqa: E402
from aac_semantic.runner import load_corpus  # noqa: E402

DESTINO = RAIZ / "web" / "pruebas" / "expansion.fixture.json"

CONSTRUIDOS = [
    # Negación como palabra y como parte de otra palabra.
    ("NO AGUA", ["no quiero agua", "quiero agua", "agua de noche", "nada de agua",
                 "sin agua", "no, gracias", "No quiero AGUA."]),
    ("AGUA NO", ["agua no", "agua", "nunca agua", "jamás agua", "quita el agua"]),
    # Parada, con y sin negación léxica.
    ("BASTA", ["basta", "detente", "para ya", "alto", "ya está", "suficiente",
               "sigue", "deténgase", "no sigas", "ya basta, por favor"]),
    ("NO BASTA DOLOR", ["basta, me duele", "me duele", "no, me duele",
                        "para, me duele", "dolor"]),
    # Duda.
    ("TAL VEZ", ["tal vez", "sí", "no", "Sí.", "quizás", "puede ser"]),
    ("SI NO", ["sí o no", "si", "no lo sé", "no"]),
    # Sustantivos ajenos, en el borde: formas, subcadenas, la entrada deletreada.
    ("BAÑO", ["necesito ir al baño", "tengo sed", "quiero comer",
              "me duele", "quiero ir al inodoro", "necesito orinar"]),
    ("NO HAMBRE", ["no tengo hambre", "no quiero comida", "no tengo sed"]),
    ("MEDICINA DOLOR", ["dame la medicina, me duele", "el remedio me causa dolor",
                        "quiero agua"]),
    ("AGUACATE", ["quiero aguacate", "quiero agua"]),
    # Repeticiones y duplicados.
    ("MAÑANA", ["mañana", "mañana mañana", "mañana mañana mañana", "Mañana."]),
    ("AGUA", ["quiero agua", "Quiero agua.", "quiero agua por favor",
              "agua quiero", "agua agua"]),
    # Clíticos: el F1 de la deduplicación depende de partirlos bien.
    ("ACOSTAR", ["quiero acostarme", "me quiero acostar", "acostame",
                 "acuéstame", "clase", "dármelo"]),
    # Vacíos y espacios.
    ("  AGUA  ", ["", "   ", "agua"]),
    # Cortesía con palabra negativa: "de nada" no niega ni detiene.
    ("BASTA", ["de nada", "de nada, basta", "basta", "nada más"]),
    ("NO AGUA", ["de nada", "no, de nada", "de nada de nada"]),
    ("", ["algo"]),
]


def _casos_reales(corpus: list[dict]) -> list[tuple[str, list[str]]]:
    entrada = {it["id"]: it["input"] for it in corpus}
    vistos: set[tuple[str, tuple[str, ...]]] = set()
    casos = []
    for ruta in sorted((RAIZ / "slm" / "results").glob("run-*.jsonl")):
        for linea in ruta.read_text(encoding="utf-8").splitlines():
            if not linea.strip():
                continue
            r = json.loads(linea)
            e = entrada.get(r.get("item_id"))
            if e is None:
                continue
            clave = (e, tuple(r.get("candidates") or []))
            if clave in vistos:
                continue
            vistos.add(clave)
            casos.append((e, list(clave[1])))
    return casos


def main() -> int:
    corpus = load_corpus()
    reales = _casos_reales(corpus)
    casos = []
    for origen, lista in (("real", reales), ("construido", CONSTRUIDOS)):
        for entrada, candidatos in lista:
            r = guard.guarded(entrada, candidatos)
            casos.append({
                "origen": origen,
                "entrada": entrada,
                "candidatos": candidatos,
                "quedan": r.candidates,
                "descartados": [list(d) for d in r.dropped],
                "sinEleccion": r.fell_back,
            })

    # Las primitivas por separado: si una falla, el error se ve ahí y no
    # escondido en una decisión del filtro.
    textos = sorted({c for caso in casos for c in caso["candidatos"]}
                    | {caso["entrada"] for caso in casos})
    primitivas = [{"texto": t, "normal": prompts.normalise(t),
                   "tokens": metrics.tokens(t)} for t in textos]
    pares = [(casos[i]["entrada"], c) for i in range(0, len(casos), 7)
             for c in casos[i]["candidatos"]]
    f1 = [{"a": a, "b": b, "f1": metrics.token_f1(a, b)} for a, b in pares]

    salida = {
        "nota": "Generado por web/generar_fixture_filtro.py. No editar a mano.",
        "constantes": {
            "NEGACION": list(metrics.NEGATION),
            "PARADA": list(metrics.STOP_MARKERS),
            "SUSTANTIVOS_TABLERO": {k: list(v)
                                    for k, v in guard.BOARD_NOUNS.items()},
            "CORTESIA_SIN_NEGACION": list(metrics.CORTESIA_SIN_NEGACION),
        },
        "casos": casos,
        "primitivas": primitivas,
        "f1": f1,
    }
    DESTINO.write_text(json.dumps(salida, ensure_ascii=False, indent=0),
                       encoding="utf-8")
    descartes = sum(bool(c["descartados"]) for c in casos)
    print(f"[fixture] {len(reales)} casos reales + {len(CONSTRUIDOS)} construidos,"
          f" {descartes} con algún descarte, {len(primitivas)} textos,"
          f" {len(f1)} pares de F1 -> {DESTINO} "
          f"({DESTINO.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
