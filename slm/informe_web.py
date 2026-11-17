"""Analiza lo que midió el banco de la página (`slm/banco_web/`).

    python slm/informe_web.py results/web-cribado.jsonl [otros.jsonl ...]

Cada respuesta del banco trae tres listas de candidatos, y las tres se puntúan:

* **modelo**: lo que el modelo propuso, sin nada encima;
* **sin filtro**: lo que mostraba la página hasta este cambio (modelo + literal);
* **con filtro**: lo que muestra desde ahora (modelo, filtro, literal).

## Sobre las repeticiones

La página muestrea a temperatura 0.9, así que el banco pasa el corpus entero
varias veces (k). Las k respuestas de una misma entrada no son independientes:
si una entrada es difícil, lo es las k veces. Los intervalos de confianza se
calculan entonces remuestreando ENTRADAS -con sus k respuestas adentro- y no
respuestas sueltas, que daría intervalos falsamente angostos.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

from aac_semantic import guard, metrics, prompts  # noqa: E402
from aac_semantic.metrics import NEGATION, MATCH_F1, _has_any, best_match  # noqa: E402
from aac_semantic.prompts import normalise  # noqa: E402
from aac_semantic.registro import con_referencias_neutras, tiene_voseo  # noqa: E402
from aac_semantic.runner import load_corpus  # noqa: E402

# Lo que la página puede cargar: WebLLM 0.2.79. La memoria es la que declara
# WebLLM para cada modelo; la descarga, la suma de los pesos publicados en
# huggingface.co/mlc-ai; la licencia, la del modelo.
MODELOS: dict[str, dict] = {
    "Qwen2.5-0.5B-Instruct-q4f16_1-MLC": dict(
        corto="qwen2.5-0.5b", descarga_gb=0.28, params=0.5, vram_mb=945, licencia="Apache-2.0",
        entregable=True),
    "Qwen2.5-1.5B-Instruct-q4f16_1-MLC": dict(
        corto="qwen2.5-1.5b", descarga_gb=0.87, params=1.5, vram_mb=1630, licencia="Apache-2.0",
        entregable=True),
    "Qwen2.5-3B-Instruct-q4f16_1-MLC": dict(
        corto="qwen2.5-3b", descarga_gb=1.74, params=3.1, vram_mb=2505,
        licencia="Qwen Research (no comercial)", entregable=False),
    "Qwen3-0.6B-q4f16_1-MLC": dict(
        corto="qwen3-0.6b", descarga_gb=0.34, params=0.6, vram_mb=1403, licencia="Apache-2.0",
        entregable=True),
    "Qwen3-1.7B-q4f16_1-MLC": dict(
        corto="qwen3-1.7b", descarga_gb=0.97, params=1.7, vram_mb=2037, licencia="Apache-2.0",
        entregable=True),
    "Qwen3-4B-q4f16_1-MLC": dict(
        corto="qwen3-4b", descarga_gb=2.26, params=4.0, vram_mb=3432, licencia="Apache-2.0",
        entregable=True),
    "Phi-3.5-mini-instruct-q4f16_1-MLC": dict(
        corto="phi-3.5-mini", descarga_gb=2.15, params=3.8, vram_mb=3672, licencia="MIT",
        entregable=True),
}

RNG = np.random.default_rng(20261117)
N_BOOT = 4000


def corto(modelo: str) -> str:
    return MODELOS.get(modelo, {}).get("corto", modelo)


# ------------------------------------------------------------ puntuación


SENTIDO_NEGATIVO = ("mal", "malo", "mala", "peor", "menos", "dolor", "duele",
                    "falla", "roto", "rota", "seca", "seco", "sin", "basta",
                    "ahogo", "ahogar", "ahogando", "respirar", "aire")
"""Palabras cuyo sentido ya es negativo, y que el español expresa con NO sin
invertir nada: `PIE MAL` -> "el pie no funciona", `LETRA MAL` -> "la letra no
es correcta". Aplicada sobre las respuestas históricas de Ollama, la primera
versión de la métrica contaba esas paráfrasis como inversiones.

Las de respiración están por un caso que importa más que el promedio:
`AHOGO` -> "no puedo respirar" es la expansión correcta de un mensaje urgente,
y tiene una negación que la entrada no trae."""


def _sentido_negativo(entrada: str) -> bool:
    return _has_any(entrada, SENTIDO_NEGATIVO)


def invierte_pedido(item: dict, candidatos: list[str]) -> bool:
    """Dice NO donde nadie lo dijo.

    El filtro y la métrica de polaridad solo miran un lado: si la entrada niega,
    la salida tiene que negar. El otro lado no lo miraba nadie, y el banco lo
    encontró en las primeras diez respuestas: `QUIERO COMIDA` -> "no tengo
    hambre". Es el mismo daño al revés: la persona pidió algo y el aparato dice
    que no lo quiere.

    Se cuenta solo donde la entrada no niega, no pide parar ni duda, y ninguna
    lectura aceptada del corpus tiene una negación. Así "AIRE" -> "no puedo
    respirar" no cuenta, si el corpus la acepta.
    """
    entrada = item["input"]
    if (not normalise(entrada) or guard.needs_negation(entrada)
            or guard.needs_stop(entrada) or guard.is_uncertain(entrada)
            or _sentido_negativo(entrada)):
        return False
    refs = [a for r in item.get("readings", []) for a in r.get("accept", [])]
    if any(_has_any(r, NEGATION) for r in refs):
        return False
    lit = normalise(entrada)
    return any(_has_any(c, NEGATION) for c in candidatos if normalise(c) != lit)


@dataclass
class Muestra:
    celda: tuple
    item_id: str
    k: int
    tags: list[str]
    por: dict = field(default_factory=dict)      # pipeline -> ItemScore
    invierte: dict = field(default_factory=dict)  # pipeline -> bool
    voseo: dict = field(default_factory=dict)     # pipeline -> bool
    ms: float = 0.0
    paralelo: int = 1
    tok_salida: int = 0
    decode_tps: float = float("nan")
    formato: str = ""
    descartados: list = field(default_factory=list)
    error: str = ""
    solo_latencia: bool = False
    """Viene del plan de latencia: cuenta para los tiempos, no para la calidad.
    Esas respuestas cubren una entrada de cada dos, y sumarlas a las tiradas del
    cribado le daría más peso a esa mitad del corpus."""


PIPELINES = ("modelo", "sin", "con", "aporte")
"""`aporte` es lo que ve la persona con filtro, MENOS el literal. La página
agrega siempre lo escrito tal cual, y si el corpus acepta la palabra sola eso ya
cuenta como acierto aunque el modelo no haya aportado nada. Esta lista mide lo
que puso el modelo."""
CAMPO = {"modelo": "propuestos", "sin": "sinFiltro", "con": "conFiltro"}


def _reprocesadas(ruta: Path) -> list[str]:
    """Las respuestas del banco pasadas por el `expansion.js` ACTUAL.

    Se reprocesan con node y no se reimplementa nada en Python: lo que se mide
    tiene que ser el código que corre en la página, no una copia de él."""
    import subprocess
    r = subprocess.run(["node", str(AQUI / "banco_web" / "reprocesar.mjs"), str(ruta)],
                       capture_output=True, check=True)
    return r.stdout.decode("utf-8").splitlines()


REPROCESO = {"total": 0, "distintas": 0}


def cargar(rutas: list[Path], corpus: list[dict]) -> list[Muestra]:
    por_id = {it["id"]: it for it in corpus}
    salida = []
    for ruta in rutas:
        latencia = ruta.name.startswith("web-latencia")
        for linea in _reprocesadas(ruta):
            if not linea.strip():
                continue
            o = json.loads(linea)
            item = por_id.get(o["item_id"])
            if item is None:
                continue
            celda = (o["modelo"], o["variante"], o["temperatura"],
                     o.get("perturbacion") or "")
            if "igualAlBanco" in o:
                REPROCESO["total"] += 1
                REPROCESO["distintas"] += not o["igualAlBanco"]
            m = Muestra(celda=celda, item_id=o["item_id"], k=o["k"],
                        tags=item.get("tags", []), error=o.get("error", ""),
                        ms=o.get("ms", 0) or 0, paralelo=o.get("paralelo", 1),
                        solo_latencia=latencia)
            if m.error:
                salida.append(m)
                continue
            # La entrada perturbada es la que vio el modelo, pero lo que la
            # persona quiso decir es el ítem del corpus: se puntúa contra él.
            _, m.formato = prompts.parse_candidates(o.get("crudo", ""))
            uso = o.get("uso") or {}
            m.tok_salida = uso.get("completion_tokens", 0) or 0
            m.decode_tps = (uso.get("extra") or {}).get(
                "decode_tokens_per_s", float("nan"))
            m.descartados = o.get("descartados", [])
            lit_n = normalise(item["input"])
            for p in PIPELINES:
                if p == "aporte":
                    cands = [c for c in (o.get("conFiltro") or []) if normalise(c) != lit_n]
                else:
                    cands = o.get(CAMPO[p]) or []
                m.por[p] = metrics.score_item(
                    item, cands, m.formato, m.ms / 1000, model=o["modelo"],
                    variant=o["variante"])
                lit = normalise(item["input"])
                propios = [c for c in cands if normalise(c) != lit]
                m.invierte[p] = invierte_pedido(item, propios)
                m.voseo[p] = any(tiene_voseo(c) for c in propios)
            salida.append(m)
    return salida


# --------------------------------------------------------- agregación


def _matriz(muestras: list[Muestra], valor) -> tuple[list[str], np.ndarray]:
    """Una fila por entrada, el promedio de sus k respuestas."""
    por_item: dict[str, list[float]] = defaultdict(list)
    for m in muestras:
        v = valor(m)
        if v is not None:
            por_item[m.item_id].append(float(v))
    ids = sorted(por_item)
    return ids, np.array([np.mean(por_item[i]) for i in ids])


def media_ic(por_item: np.ndarray) -> tuple[float, float, float]:
    """Media y su intervalo del 95 %, remuestreando entradas."""
    if not len(por_item):
        return (float("nan"),) * 3
    idx = RNG.integers(0, len(por_item), size=(N_BOOT, len(por_item)))
    boot = por_item[idx].mean(axis=1)
    return (float(por_item.mean()), float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)))


def diferencia_pareada(a: dict[str, float], b: dict[str, float]):
    """b - a sobre las entradas que tienen las dos, con su intervalo."""
    comunes = sorted(set(a) & set(b))
    d = np.array([b[i] - a[i] for i in comunes])
    if not len(d):
        return float("nan"), float("nan"), float("nan"), 0
    idx = RNG.integers(0, len(d), size=(N_BOOT, len(d)))
    boot = d[idx].mean(axis=1)
    return (float(d.mean()), float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)), len(d))


METRICAS = {
    # nombre: (función sobre una Muestra, pipeline, más es mejor)
    "acierto": (lambda m, p: m.por[p].hit, True),
    "acierto_del_modelo": (lambda m, p: m.por["aporte"].hit, True),
    "agrega_contenido": (lambda m, p: bool(m.por["aporte"].introduced_words), False),
    "inventa": (lambda m, p: bool(m.por[p].invented), False),
    "pierde_polaridad": (lambda m, p: not m.por[p].polarity_ok, False),
    "invierte_pedido": (lambda m, p: m.invierte[p], False),
    "suaviza": (lambda m, p: bool(m.por[p].softened), False),
    "expande_lo_que_no_debe": (lambda m, p: not m.por[p].refusal_ok, False),
    "literal_del_modelo": (lambda m, p: m.por["modelo"].literal_present, True),
    "hay_eleccion": (lambda m, p: m.por[p].choice_ok, True),
    "voseo": (lambda m, p: m.voseo[p], False),
    "formato_json": (lambda m, p: m.formato == "json", True),
}


def inseguro(m: Muestra, p: str) -> bool:
    s = m.por[p]
    return (bool(s.invented) or not s.polarity_ok or bool(s.softened)
            or not s.refusal_ok or m.invierte[p])


def resumen_celda(muestras: list[Muestra], p: str = "con") -> dict:
    todas = [m for m in muestras if not m.error]
    ok = [m for m in todas if not m.solo_latencia]
    r: dict = {"n": len(ok), "errores": len(muestras) - len(ok),
               "entradas": len({m.item_id for m in ok}),
               "k": max((m.k for m in ok), default=-1) + 1}
    for nombre, (f, _) in METRICAS.items():
        _, v = _matriz(ok, lambda m, f=f: f(m, p))
        r[nombre] = media_ic(v)
    _, v = _matriz(ok, lambda m: inseguro(m, p))
    r["inseguro"] = media_ic(v)

    # Por entrada: qué tan seguido acierta, y si alguna vez fue insegura.
    ids, hit = _matriz(ok, lambda m: m.por[p].hit)
    r["entradas_fiables"] = float(np.mean(hit >= 0.8)) if len(hit) else float("nan")
    r["entradas_nunca"] = float(np.mean(hit == 0)) if len(hit) else float("nan")
    _, ins = _matriz(ok, lambda m: inseguro(m, p))
    r["entradas_alguna_vez_inseguras"] = float(np.mean(ins > 0)) if len(ins) else float("nan")
    r["acierto_por_item"] = dict(zip(ids, hit.tolist()))

    unicos = [m for m in todas if m.paralelo == 1]
    ms = np.array([m.ms for m in unicos]) if unicos else np.array([])
    r["ms_p50"] = float(np.median(ms)) if len(ms) else float("nan")
    r["ms_p95"] = float(np.percentile(ms, 95)) if len(ms) else float("nan")
    r["ms_n"] = len(ms)
    tps = np.array([m.decode_tps for m in unicos if not math.isnan(m.decode_tps)])
    r["decode_tps"] = float(np.median(tps)) if len(tps) else float("nan")
    r["tok_salida"] = float(np.median([m.tok_salida for m in ok])) if ok else float("nan")
    return r


# ------------------------------------------------------------- el filtro


def costo_del_filtro(muestras: list[Muestra], corpus: dict) -> dict:
    """Qué gana y qué le quita el filtro a la persona."""
    ok = [m for m in muestras if not m.error and not m.solo_latencia]
    perdidos = sum(m.por["sin"].hit and not m.por["con"].hit for m in ok)
    ganados = sum(m.por["con"].hit and not m.por["sin"].hit for m in ok)
    salvados = sum(inseguro(m, "sin") and not inseguro(m, "con") for m in ok)
    descartes = buenos = 0
    motivos: dict[str, int] = defaultdict(int)
    ejemplos_buenos = []
    for m in ok:
        refs = [a for r in corpus[m.item_id].get("readings", [])
                for a in r.get("accept", [])]
        for texto, motivo in m.descartados:
            descartes += 1
            motivos[motivo.split(" y la entrada")[0]] += 1
            if refs and best_match(texto, refs) >= MATCH_F1:
                buenos += 1
                if len(ejemplos_buenos) < 6:
                    ejemplos_buenos.append((corpus[m.item_id]["input"], texto, motivo))
    return dict(n=len(ok), perdidos=perdidos, ganados=ganados,
                salvados=salvados, descartes=descartes, buenos=buenos,
                motivos=dict(sorted(motivos.items(), key=lambda x: -x[1])),
                ejemplos_buenos=ejemplos_buenos)


def regla_inversion(muestras: list[Muestra], corpus: dict) -> dict:
    """¿Conviene que el filtro descarte también las negaciones que la entrada no
    trae? Se simula sobre lo medido: cuántos candidatos descartaría, cuántos de
    esos eran correctos, y qué le pasa al acierto.

    La regla que se simula es la única que el filtro puede aplicar sin conocer
    el corpus: si la entrada no tiene NO, ni BASTA, ni duda, todo candidato con
    una negación se descarta."""
    ok = [m for m in muestras if not m.error and not m.solo_latencia]
    descartaria = correctos = 0
    acierto_antes = acierto_despues = 0
    ejemplos_correctos = []
    for m in ok:
        item = corpus[m.item_id]
        e = item["input"]
        cands = m.por["con"].candidates
        refs = [a for r in item.get("readings", []) for a in r.get("accept", [])]
        aplica = bool(normalise(e)) and not (
            guard.needs_negation(e) or guard.needs_stop(e)
            or guard.is_uncertain(e) or _sentido_negativo(e))
        lit = normalise(e)
        quedan = []
        for c in cands:
            if aplica and normalise(c) != lit and _has_any(c, NEGATION):
                descartaria += 1
                if refs and best_match(c, refs) >= MATCH_F1:
                    correctos += 1
                    if len(ejemplos_correctos) < 6:
                        ejemplos_correctos.append((e, c))
                continue
            quedan.append(c)
        acierto_antes += m.por["con"].hit
        acierto_despues += bool(refs) and any(best_match(c, refs) >= MATCH_F1 for c in quedan)
    n = max(1, len(ok))
    return dict(descartaria=descartaria, correctos=correctos,
                acierto_antes=acierto_antes / n, acierto_despues=acierto_despues / n,
                ejemplos_correctos=ejemplos_correctos)


# --------------------------------------------------------------- informe


def pct(x: float, d: int = 0) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{d}f}%"


def ic(t) -> str:
    m, lo, hi = t
    if math.isnan(m):
        return "—"
    return f"{100 * m:.0f}% ({100 * lo:.0f}–{100 * hi:.0f})"


def nombre_celda(c: tuple) -> str:
    modelo, variante, temp, pert = c
    s = f"{corto(modelo)} · {variante} · t={temp:g}"
    return s + (f" · {pert}" if pert else "")


def main(argv: list[str]) -> int:
    rutas = [Path(a) if Path(a).is_absolute() else AQUI / a for a in argv[1:]]
    if not rutas:
        rutas = sorted((AQUI / "results").glob("web-*.jsonl"))
        rutas = [r for r in rutas if not r.name.startswith(("web-humo", "web-rinde"))]
    original = load_corpus()
    neutro, agregadas = con_referencias_neutras(original)
    por_id = {it["id"]: it for it in neutro}
    muestras = cargar(rutas, neutro)
    print(f"[informe] {len(muestras)} respuestas de {', '.join(r.name for r in rutas)}; "
          f"{agregadas} referencias neutras agregadas al corpus")
    print(f"[informe] reprocesadas con el expansion.js actual: {REPROCESO['total']}; "
          f"distintas de lo que calculó la página en el momento: {REPROCESO['distintas']}")

    celdas: dict[tuple, list[Muestra]] = defaultdict(list)
    for m in muestras:
        celdas[m.celda].append(m)

    filas = []
    for c, ms in sorted(celdas.items(), key=lambda kv: (kv[0][3], kv[0][0], kv[0][1], kv[0][2])):
        r = {p: resumen_celda(ms, p) for p in ("sin", "con")}
        filas.append((c, r))
        con = r["con"]
        print(f"  {nombre_celda(c):58s} n={con['n']:4d} acierto {ic(con['acierto'])}  "
              f"inseguro {ic(con['inseguro'])}  (sin filtro {ic(r['sin']['inseguro'])})  "
              f"p50 {con['ms_p50']/1000:.1f}s")
    salida = AQUI / "results" / "informe-web.json"
    salida.write_text(json.dumps(
        {nombre_celda(c): {p: {k: v for k, v in r[p].items() if k != "acierto_por_item"}
                           for p in r} for c, r in filas},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[informe] resumen -> {salida}")

    md = escribir_md(filas, celdas, por_id, rutas, agregadas)
    destino = AQUI.parent / "docs" / "expansion_semantica_tablas.md"
    destino.write_text(md, encoding="utf-8")
    print(f"[informe] tablas -> {destino}")
    return 0


# ------------------------------------------------------------- markdown

BASE = ("Qwen2.5-1.5B-Instruct-q4f16_1-MLC", "desplegado", 0.9, "")


def _tabla(encabezado: list[str], filas: list[list[str]]) -> list[str]:
    return (["| " + " | ".join(encabezado) + " |",
             "|" + "|".join("---" for _ in encabezado) + "|"]
            + ["| " + " | ".join(f) + " |" for f in filas])


def _fila_modelo(c: tuple, r: dict) -> list[str]:
    meta = MODELOS.get(c[0], {})
    con = r["con"]
    return [f"`{corto(c[0])}`", c[1], f"{c[2]:g}",
            meta.get("licencia", "?"), f"{meta.get('descarga_gb', float('nan')):.2f} GB",
            f"**{ic(con['acierto'])}**", ic(con["acierto_del_modelo"]),
            pct(con["entradas_fiables"]),
            ic(con["inseguro"]), ic(r["sin"]["inseguro"]),
            pct(con["hay_eleccion"][0]), pct(con["voseo"][0]),
            str(con["n"])]


def escribir_md(filas, celdas, por_id, rutas, agregadas) -> str:
    L: list[str] = []
    a = L.append
    a("# Expansión semántica: tablas del banco de la página")
    a("")
    a(f"*Generado por `slm/informe_web.py` desde {', '.join('`' + r.name + '`' for r in rutas)}."
      " No editar a mano: regenerar.* La justificación de la elección está en"
      " [expansion_semantica.md](expansion_semantica.md).")
    a("")
    a("Todo se midió con la página real: WebLLM 0.2.79 en Chrome con WebGPU, la"
      " misma clase `SLM`, el mismo prompt desplegado y el mismo filtro de"
      " `web/js/expansion.js`. Los intervalos son del 95 % y se calculan"
      " remuestreando **entradas**, no respuestas: las k respuestas de una"
      " misma entrada no son independientes.")
    a("")
    a(f"El corpus tiene 133 entradas; se le agregaron {agregadas} referencias en"
      " español neutro, equivalentes a las que estaban en voseo, sin quitar"
      " ninguna (ver `slm/aac_semantic/registro.py`).")
    a("")

    base = [(c, r) for c, r in filas if not c[3]]
    pert = [(c, r) for c, r in filas if c[3]]

    a("## Todos los modelos")
    a("")
    a("**acierto**: la frase buscada está entre lo que ve la persona."
      " **acierto del modelo**: lo mismo sin contar el literal, que la página"
      " agrega siempre; es lo que aporta el modelo."
      " **fiables**: entradas en las que acierta en al menos el 80 % de las"
      " tiradas. **inseguro**: inventa, pierde una negación o una orden de"
      " parar, invierte un pedido, suaviza una urgencia, o expande algo que no"
      " debía; *con* el filtro de la página y, al lado, *sin* él."
      " **elección**: hay al menos dos opciones para elegir.")
    a("")
    L.extend(_tabla(["modelo", "prompt", "temp.", "licencia", "descarga", "acierto",
                     "acierto del modelo", "fiables", "inseguro (con filtro)", "inseguro (sin filtro)",
                     "elección", "voseo", "n"],
                    [_fila_modelo(c, r) for c, r in
                     sorted(base, key=lambda x: -x[1]["con"]["acierto"][0])]))
    a("")

    a("### Qué tipo de inseguridad queda, con el filtro puesto")
    a("")
    a("**inventa** cuenta solo lo que el corpus prohíbe a mano para cada"
      " entrada. **agrega contenido** es más amplio y más ruidoso: alguna palabra"
      " de contenido que la entrada no trae ni licencia (\"cena de bañera\""
      " para BAÑO cae acá y no en inventa). No entra en *inseguro*.")
    a("")
    L.extend(_tabla(["modelo", "prompt", "temp.", "inventa", "pierde polaridad",
                     "invierte un pedido", "suaviza", "expande lo que no debe",
                     "agrega contenido"],
                    [[f"`{corto(c[0])}`", c[1], f"{c[2]:g}",
                      pct(r["con"]["inventa"][0], 1), pct(r["con"]["pierde_polaridad"][0], 1),
                      pct(r["con"]["invierte_pedido"][0], 1), pct(r["con"]["suaviza"][0], 1),
                      pct(r["con"]["expande_lo_que_no_debe"][0], 1),
                      pct(r["con"]["agrega_contenido"][0], 1)]
                     for c, r in sorted(base, key=lambda x: -x[1]["con"]["acierto"][0])]))
    a("")

    a("### Formato y salida de escape")
    a("")
    a("**literal del modelo**: el propio modelo ofreció la lectura literal (la"
      " página la agrega igual; esto mide si el modelo cumple la regla 8)."
      " **JSON**: la respuesta se pudo leer como JSON.")
    a("")
    L.extend(_tabla(["modelo", "prompt", "temp.", "literal del modelo", "JSON",
                     "tokens de salida (mediana)"],
                    [[f"`{corto(c[0])}`", c[1], f"{c[2]:g}",
                      pct(r["con"]["literal_del_modelo"][0]), pct(r["con"]["formato_json"][0]),
                      f"{r['con']['tok_salida']:.0f}"]
                     for c, r in base]))
    a("")

    # --- comparaciones pareadas contra la página de hoy
    porc = {c: r for c, r in filas}
    if BASE in porc:
        a("## Contra la página de hoy")
        a("")
        a(f"Diferencia pareada, entrada por entrada, contra `{nombre_celda(BASE)}`."
          " Si el intervalo no cruza el cero, la diferencia no es ruido del corpus.")
        a("")
        ref_hit = porc[BASE]["con"]["acierto_por_item"]
        ref_ins = _por_item(celdas[BASE], lambda m: inseguro(m, "con"))
        filas_d = []
        for c, r in base:
            if c == BASE:
                continue
            dh = diferencia_pareada(ref_hit, r["con"]["acierto_por_item"])
            di = diferencia_pareada(ref_ins, _por_item(celdas[c], lambda m: inseguro(m, "con")))
            filas_d.append([f"`{corto(c[0])}` · {c[1]} · t={c[2]:g}",
                            _delta(dh), _delta(di), str(dh[3])])
        L.extend(_tabla(["contra la página de hoy", "Δ acierto", "Δ inseguro", "entradas"],
                        filas_d))
        a("")

    # --- el filtro
    a("## El filtro: qué gana y qué cuesta")
    a("")
    a("**salvadas**: respuestas que eran inseguras sin filtro y dejan de serlo."
      " **aciertos perdidos**: respuestas que acertaban sin filtro y dejan de"
      " acertar. **descartes correctos**: candidatos que el filtro tiró y que"
      " coincidían con una frase aceptada del corpus.")
    a("")
    filas_f = []
    motivos_base = None
    ejemplos_base = None
    for c, r in base:
        f = costo_del_filtro(celdas[c], por_id)
        if c == BASE:
            motivos_base, ejemplos_base = f["motivos"], f["ejemplos_buenos"]
        filas_f.append([f"`{corto(c[0])}` · {c[1]} · t={c[2]:g}",
                        f"{f['salvados']} ({pct(f['salvados'] / max(1, f['n']), 1)})",
                        f"{f['perdidos']} ({pct(f['perdidos'] / max(1, f['n']), 1)})",
                        str(f["descartes"]), str(f["buenos"])])
    L.extend(_tabla(["celda", "salvadas", "aciertos perdidos", "descartes",
                     "descartes correctos"], filas_f))
    a("")
    if motivos_base:
        a(f"Por qué descarta, en `{nombre_celda(BASE)}`: "
          + "; ".join(f"{k} ({v})" for k, v in motivos_base.items()) + ".")
        a("")
    if ejemplos_base:
        a("Candidatos correctos que el filtro descartó:")
        a("")
        for e, t, mot in ejemplos_base:
            a(f"- `{e}` → «{t}» ({mot})")
        a("")

    a("### La regla que falta: no decir NO donde nadie lo dijo")
    a("")
    a("Simulada sobre lo medido: si la entrada no tiene NO, ni BASTA, ni duda,"
      " ni una palabra de sentido negativo (MAL, DOLOR, MENOS...), descartar todo"
      " candidato con una negación.")
    a("")
    filas_r = []
    for c, r in base:
        g = regla_inversion(celdas[c], por_id)
        filas_r.append([f"`{corto(c[0])}` · {c[1]} · t={c[2]:g}",
                        pct(r["con"]["invierte_pedido"][0], 1),
                        str(g["descartaria"]), str(g["correctos"]),
                        f"{pct(g['acierto_antes'], 1)} → {pct(g['acierto_despues'], 1)}"])
    L.extend(_tabla(["celda", "invierte un pedido hoy", "descartaría",
                     "de esos, correctos", "acierto"], filas_r))
    a("")

    # --- por categoría
    a("## Por tipo de entrada")
    a("")
    TAGS = ["urgente", "negacion", "parada", "polaridad", "ambiguo", "deletreo",
            "una-palabra", "multi-palabra", "sintoma", "postura"]
    filas_t = []
    for c, r in sorted(base, key=lambda x: -x[1]["con"]["acierto"][0]):
        fila = [f"`{corto(c[0])}` · {c[1]} · t={c[2]:g}"]
        for t in TAGS:
            ms = [m for m in celdas[c] if not m.error and not m.solo_latencia
                  and t in m.tags]
            if not ms:
                fila.append("—")
                continue
            _, h = _matriz(ms, lambda m: m.por["con"].hit)
            _, s = _matriz(ms, lambda m: inseguro(m, "con"))
            fila.append(f"{pct(h.mean())} / {pct(s.mean())}")
        filas_t.append(fila)
    a("Cada celda: acierto / inseguro, con filtro.")
    a("")
    L.extend(_tabla(["celda"] + TAGS, filas_t))
    a("")

    # --- perturbaciones
    if pert:
        a("## Robustez: la entrada como llega de verdad")
        a("")
        a("Misma entrada del corpus, escrita como puede llegar desde un tablero de"
          " mirada. Diferencia pareada contra la entrada limpia, mismo modelo,"
          " prompt y temperatura.")
        a("")
        filas_p = []
        for c, r in pert:
            limpio = (c[0], c[1], c[2], "")
            if limpio not in porc:
                continue
            dh = diferencia_pareada(porc[limpio]["con"]["acierto_por_item"],
                                    r["con"]["acierto_por_item"])
            di = diferencia_pareada(_por_item(celdas[limpio], lambda m: inseguro(m, "con")),
                                    _por_item(celdas[c], lambda m: inseguro(m, "con")))
            filas_p.append([f"`{corto(c[0])}` · {c[3]}", ic(r["con"]["acierto"]),
                            _delta(dh), ic(r["con"]["inseguro"]), _delta(di)])
        L.extend(_tabla(["perturbación", "acierto", "Δ acierto", "inseguro", "Δ inseguro"],
                        filas_p))
        a("")

    # --- latencia
    lat = [(c, r) for c, r in base if r["con"]["ms_n"] >= 50]
    if lat:
        a("## Latencia en esta máquina")
        a("")
        a("Solo respuestas medidas con una pestaña sola: con varias a la vez, cada"
          " pedido espera a los demás y la latencia no es la que ve la persona."
          " La máquina es una notebook con RTX 4060.")
        a("")
        L.extend(_tabla(["modelo", "p50", "p95", "tokens/s al decodificar", "n"],
                        [[f"`{corto(c[0])}`", f"{r['con']['ms_p50'] / 1000:.1f} s",
                          f"{r['con']['ms_p95'] / 1000:.1f} s",
                          f"{r['con']['decode_tps']:.1f}", str(r["con"]["ms_n"])]
                         for c, r in lat]))
        a("")
    return "\n".join(L) + "\n"


def _por_item(muestras, valor) -> dict[str, float]:
    ids, v = _matriz([m for m in muestras if not m.error and not m.solo_latencia], valor)
    return dict(zip(ids, v.tolist()))


def _delta(d) -> str:
    m, lo, hi, n = d
    if not n:
        return "—"
    marca = "" if lo <= 0 <= hi else " **"
    return f"{100 * m:+.0f} ({100 * lo:+.0f} a {100 * hi:+.0f}){marca}"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
