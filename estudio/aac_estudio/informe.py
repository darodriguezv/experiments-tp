"""Juntar los bloques de todos los participantes y decir qué se puede afirmar.

## El análisis correcto para este diseño

Cada persona hace las tres condiciones, así que las tres medidas de una misma
persona están emparejadas. Compararlas como si fueran grupos independientes
tiraría a la basura justamente lo que el diseño compró: que la variabilidad
entre personas -que en control por mirada es enorme- se cancele dentro de cada
participante.

Entonces el contraste es **intra-sujeto**: para cada persona, su ahorro contra
su propia línea de base. Eso además hace que el resultado no dependa de que
todos sean igual de diestros, que es lo que no se puede garantizar.

Con n=10 y una diferencia emparejada, lo honesto es reportar la mediana del
ahorro y el rango, no una media con desvío: diez valores no sostienen una
distribución, y un solo participante muy bueno o muy malo mueve la media.

## Lo que este módulo NO hace

No corre pruebas de significancia. Con diez participantes y tres condiciones,
cualquier p que salga de acá se va a leer como más de lo que es. Si la tesis
necesita un contraste formal, el dato está guardado por participante y por
tarea para que lo haga quien corresponda, con el test emparejado que
corresponda (Wilcoxon, por el n). Fabricar un p-valor acá sería darle autoridad
estadística a un piloto.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import diseno, medidas, protocolo, sesion

ADJUDICACIONES = sesion.DATOS / "adjudicaciones.json"


def _adjudicaciones() -> dict:
    if not ADJUDICACIONES.exists():
        return {}
    try:
        return json.loads(ADJUDICACIONES.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def adjudicar(participante: int, condicion: str, tarea: str, veredicto: str,
              quien: str = "investigador") -> None:
    """Resuelve a mano un enunciado que cayó en la zona gris.

    Queda registrado quién y cuándo. Un juicio humano sin firma es
    indistinguible de un número que alguien cambió.
    """
    import time

    if veredicto not in ("acierto", "fallo"):
        raise ValueError(f"veredicto {veredicto!r}: solo acierto o fallo")
    datos = _adjudicaciones()
    datos[f"{participante}|{condicion}|{tarea}"] = {
        "veredicto": veredicto, "quien": quien, "cuando": time.time()}
    ADJUDICACIONES.parent.mkdir(parents=True, exist_ok=True)
    ADJUDICACIONES.write_text(
        json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")


def bloques() -> list[medidas.Bloque]:
    """Todos los bloques medidos, con las adjudicaciones ya aplicadas."""
    estado = sesion.EstadoEstudio.cargar()
    juicios = _adjudicaciones()
    salida = []

    for clave, p in sorted(estado.participantes.items(), key=lambda kv: int(kv[0])):
        num = int(clave)
        asig = diseno.asignar(num)
        for _, cond, conj in asig.bloques():
            guardado = p["bloques"].get(cond)
            if not guardado or not Path(guardado["csv"]).exists():
                continue
            tareas = sesion.tareas_de(num, cond)
            bloque = medidas.extraer(
                guardado["csv"], {t.id: t.intencion for t in tareas},
                participante=num, condicion=cond, conjunto=conj)
            for e in bloque.enunciados:
                fallo = juicios.get(f"{num}|{cond}|{e.tarea_id}")
                if fallo:
                    e.veredicto = fallo["veredicto"]
                    e.resuelto_por = fallo["quien"]
            salida.append(bloque)
    return salida


def pendientes(todos: list[medidas.Bloque] | None = None) -> list[dict]:
    """Los enunciados que esperan juicio humano."""
    salida = []
    for b in (todos if todos is not None else bloques()):
        for e in b.enunciados:
            if e.veredicto == "pendiente":
                intencion = next(
                    (t.intencion for t in sesion.tareas_de(b.participante,
                                                           b.condicion)
                     if t.id == e.tarea_id), "")
                salida.append({
                    "participante": b.participante, "condicion": b.condicion,
                    "tarea": e.tarea_id, "dicho": e.dicho,
                    "intencion": intencion, "f1": round(e.f1, 2)})
    return salida


def por_participante(todos: list[medidas.Bloque] | None = None) -> list[dict]:
    """El ahorro de cada persona contra SU PROPIA línea de base."""
    todos = bloques() if todos is None else todos
    por_p: dict[int, dict[str, medidas.Bloque]] = {}
    for b in todos:
        por_p.setdefault(b.participante, {})[b.condicion] = b

    filas = []
    for num in sorted(por_p):
        base = por_p[num].get("letra")
        fila = {"participante": num,
                "completo": len(por_p[num]) == len(diseno.CONDICIONES)}
        for cond in diseno.CONDICIONES:
            b = por_p[num].get(cond)
            fila[cond] = b.resumen() if b else {"n": 0}
        if base and base.resumen().get("n"):
            for cond in ("sin_contexto", "con_contexto"):
                otra = por_p[num].get(cond)
                fila[f"ahorro_{cond}"] = (medidas.ahorro(base, otra)
                                          if otra else {})
        filas.append(fila)
    return filas


def _mediana(valores: list[float]) -> float:
    import math

    import numpy as np

    limpios = [v for v in valores if v is not None and not math.isnan(v)]
    return float(np.median(limpios)) if limpios else float("nan")


def agregado(filas: list[dict] | None = None) -> dict:
    """La mediana del ahorro intra-sujeto, con el rango."""
    import math

    filas = por_participante() if filas is None else filas
    salida = {"n_participantes": len(filas),
              "n_completos": sum(1 for f in filas if f["completo"])}
    for cond in ("sin_contexto", "con_contexto"):
        perm = [f.get(f"ahorro_{cond}", {}).get("ahorro_permanencias")
                for f in filas]
        seg = [f.get(f"ahorro_{cond}", {}).get("ahorro_segundos")
               for f in filas]
        acc = [f.get(cond, {}).get("tasa_acierto") for f in filas]
        limpios = [v for v in perm if v is not None and not math.isnan(v)]
        salida[cond] = {
            "n": len(limpios),
            "ahorro_permanencias_p50": _mediana(perm),
            "ahorro_permanencias_min": min(limpios) if limpios else float("nan"),
            "ahorro_permanencias_max": max(limpios) if limpios else float("nan"),
            "ahorro_segundos_p50": _mediana(seg),
            "acierto_p50": _mediana(acc),
        }
    return salida


def para_web() -> dict:
    todos = bloques()
    filas = por_participante(todos)
    return {
        "participantes": filas,
        "agregado": agregado(filas),
        "pendientes": pendientes(todos),
        "nombres": diseno.NOMBRES,
        "caso": protocolo.CASO.nombre,
    }


def _pct(v: float) -> str:
    import math

    return "   -" if v is None or math.isnan(v) else f"{v:+.0%}"


def texto() -> str:
    todos = bloques()
    filas = por_participante(todos)
    ag = agregado(filas)
    pend = pendientes(todos)

    out = [f"Estudio de las tres condiciones - caso {protocolo.CASO.nombre}", ""]
    out.append(f"  {ag['n_participantes']} participante(s), "
               f"{ag['n_completos']} con las tres condiciones hechas.")
    out.append("")

    if not filas:
        out.append("  Todavía no hay ningún bloque medido.")
        out.append("  Corré `python main.py servir` y dá de alta a alguien.")
        return "\n".join(out) + "\n"

    out.append("  Ahorro de cada persona contra SU PROPIO deletreo")
    out.append(f"    {'part':>5}{'permanencias':>14}{'segundos':>11}"
               f"{'acierto':>9}   condición")
    out.append("    " + "-" * 60)
    for f in filas:
        for cond in ("sin_contexto", "con_contexto"):
            a = f.get(f"ahorro_{cond}", {})
            r = f.get(cond, {})
            if not a and not r.get("n"):
                continue
            acc = r.get("tasa_acierto")
            out.append(
                f"    {f['participante']:>5}"
                f"{_pct(a.get('ahorro_permanencias')):>14}"
                f"{_pct(a.get('ahorro_segundos')):>11}"
                f"{('   -' if acc is None else f'{acc:.0%}'):>9}   "
                f"{diseno.NOMBRES[cond]}")

    out.append("")
    out.append("  Mediana del ahorro (intra-sujeto)")
    for cond in ("sin_contexto", "con_contexto"):
        a = ag[cond]
        out.append(f"    {diseno.NOMBRES[cond]:<26} "
                   f"permanencias {_pct(a['ahorro_permanencias_p50'])}  "
                   f"(rango {_pct(a['ahorro_permanencias_min'])} a "
                   f"{_pct(a['ahorro_permanencias_max'])}, n={a['n']})")

    if pend:
        out.append("")
        out.append(f"  {len(pend)} enunciado(s) esperando juicio humano:")
        for p in pend[:8]:
            out.append(f"    P{p['participante']} {p['tarea']}  "
                       f"dijo {p['dicho']!r} para {p['intencion']!r} "
                       f"(F1 {p['f1']})")
        out.append("    Se resuelven en el conductor, y queda registrado quién.")

    out.append("")
    out.append("  Sin pruebas de significancia a propósito: con diez "
               "participantes cualquier p")
    out.append("  se leería como más de lo que es. El dato está por "
               "participante y por tarea")
    out.append("  para que el contraste emparejado lo haga quien corresponda.")
    return "\n".join(out) + "\n"
