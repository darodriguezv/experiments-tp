"""Chequeos del instrumento del estudio, sin cámara ni modelo.

Lo que se verifica acá no es que el código corra: es que el **diseño** no se
pueda romper sin que algo grite. Las tres formas de arruinar este estudio en
silencio son dejar que todos hagan las condiciones en el mismo orden, dejar que
un conjunto de frases sea más fácil que otro, y contar como acierto algo que no
comunicó lo que se pidió. Las tres tienen un chequeo.
"""

from __future__ import annotations

import csv
import tempfile
import time
import traceback
from pathlib import Path

from . import diseno, medidas, protocolo


def _check(nombre: str, fn) -> bool:
    t0 = time.perf_counter()
    try:
        detalle = fn()
    except Exception as exc:  # noqa: BLE001
        print(f"  FALLA {nombre}: {type(exc).__name__}: {exc}")
        if not isinstance(exc, AssertionError):
            print("".join("         " + l for l in
                          traceback.format_exc(limit=2).splitlines(True)))
        return False
    ms = (time.perf_counter() - t0) * 1000
    print(f"  ok    {nombre} - {detalle}   [{ms:.0f} ms]")
    return True


# ----------------------------------------------------------------- diseño

def _orden_no_es_siempre_el_mismo() -> str:
    """La forma más barata de arruinar el estudio es que todos hagan 1-2-3."""
    secuencias = {diseno.asignar(p).secuencia for p in range(1, 7)}
    assert len(secuencias) == 6, (
        f"solo {len(secuencias)} órdenes distintos en seis participantes")
    narrativo = tuple(diseno.CONDICIONES)
    cuantos = sum(1 for p in range(1, 13)
                  if diseno.asignar(p).secuencia == narrativo)
    assert cuantos <= 2, f"{cuantos}/12 hacen el orden narrativo"
    return "seis órdenes distintos en seis participantes"


def _cada_condicion_cae_en_cada_posicion() -> str:
    b = diseno.balance(12)
    assert b["exacto"], "12 debería ser exacto"
    assert b["desvio_posicion"] == 0, b["posicion"]
    assert b["desvio_arrastre"] == 0, b["arrastre"]
    # Y con 10 NO es exacto: eso tiene que reportarse, no esconderse.
    b10 = diseno.balance(10)
    assert not b10["exacto"] and b10["n_exacto_mas_cercano"] == 12
    texto = diseno.informe_balance(10)
    assert "DESBALANCEADO" in texto and "12 participantes" in texto
    return "12 equilibra exacto; 10 se declara desbalanceado y propone 12"


def _la_asignacion_es_reproducible() -> str:
    """Sin azar: el número de participante ES la asignación."""
    a = [diseno.asignar(p) for p in range(1, 11)]
    b = [diseno.asignar(p) for p in range(1, 11)]
    assert a == b, "dos llamadas dieron asignaciones distintas"
    assert diseno.asignar(7).secuencia == diseno.asignar(1).secuencia
    return "determinista: el numero de participante ES la asignacion"


def _las_frases_no_se_pegan_a_una_condicion() -> str:
    """Cada condición tiene que ver los tres conjuntos por igual.

    La primera versión rotaba los conjuntos por POSICIÓN en la secuencia, y
    medido daba que `letra` recibía F0 en 8 de cada 12 casos mientras que
    `sin_contexto` no lo recibía nunca. Los conjuntos quedaban pegados a las
    condiciones -justo el confundido que rotarlos venía a evitar- y la
    diferencia entre condiciones habría incluido la diferencia entre conjuntos.
    """
    from collections import Counter

    for n in (6, 12):
        pares = Counter()
        for p in range(1, n + 1):
            a = diseno.asignar(p)
            for cond, conj in zip(a.secuencia, a.conjuntos):
                pares[(cond, conj)] += 1
        faltan = [(c, f) for c in diseno.CONDICIONES
                  for f in diseno.CONJUNTOS_FRASES if (c, f) not in pares]
        assert not faltan, f"n={n}: estas parejas nunca ocurren: {faltan}"
        esperado = n / len(diseno.CONJUNTOS_FRASES)
        peor = max(abs(v - esperado) for v in pares.values())
        assert peor == 0, (
            f"n={n}: reparto desparejo (desvio {peor:.0f}): {dict(pares)}")

    # Y nadie repite conjunto dentro de su propia sesión: si lo repitiera,
    # llegaría a la segunda condición sabiendo dónde está cada letra.
    for p in range(1, 13):
        conj = diseno.asignar(p).conjuntos
        assert len(set(conj)) == len(conj), f"P{p} repite conjunto: {conj}"
    return "cada condicion ve los tres conjuntos por igual, sin repetir"


def _los_conjuntos_de_frases_estan_emparejados() -> str:
    quejas = protocolo.verificar_emparejamiento()
    assert not quejas, "; ".join(quejas)
    largos = [sum(protocolo.costo_deletreo(t) for t in c)
              for c in protocolo.TAREAS]
    return f"{len(protocolo.TAREAS)} conjuntos de {largos} caracteres"


def _las_pistas_no_dictan_la_frase() -> str:
    """Si la pista contiene la frase, el estudio mide transcripción.

    Y además la condición con modelo pierde todo sentido, porque no quedaría
    nada que interpretar: el SLM existe para adivinar la intención detrás de un
    telegrama, no para copiar un dictado.
    """
    for conj in protocolo.TAREAS:
        for t in conj:
            pista = protocolo._sin_tildes(t.pista.lower())
            objetivo = protocolo._sin_tildes(t.intencion.lower())
            assert objetivo not in pista, (
                f"la pista de {t.id} contiene la frase objetivo: {t.pista!r}")
            # Tampoco el verbo + objeto completos, que es dictar con otro orden.
            contenido = [p for p in objetivo.split() if len(p) > 3]
            comunes = sum(1 for p in contenido if p in pista)
            assert comunes < len(contenido), (
                f"la pista de {t.id} repite todas las palabras de contenido")
    return f"{sum(len(c) for c in protocolo.TAREAS)} pistas, ninguna dicta"


def _las_condiciones_se_diferencian() -> str:
    cfgs = {c: protocolo.CONFIG_CONDICION[c] for c in diseno.CONDICIONES}
    assert not cfgs["letra"]["expandir"], "la línea de base usa el modelo"
    assert cfgs["letra"]["pagina_inicial"] == "abc"
    assert cfgs["sin_contexto"]["expandir"]
    assert not cfgs["sin_contexto"]["contexto"]
    assert cfgs["con_contexto"]["contexto"], "la condición con contexto no lo pasa"
    # sin_contexto y con_contexto tienen que diferir SOLO en el contexto y la
    # predicción; si difirieran también en el modelo o el prompt, la
    # comparación no mediría el contexto.
    difs = {k for k in cfgs["sin_contexto"]
            if cfgs["sin_contexto"][k] != cfgs["con_contexto"][k]}
    assert difs == {"contexto", "predecir"}, (
        f"difieren en {difs}, no solo en el contexto")
    return "las tres difieren en lo que deben y en nada más"


# ---------------------------------------------------------------- medidas

def _csv_falso(filas: list[tuple[float, str]]) -> Path:
    tmp = Path(tempfile.mkdtemp()) / "aac-session-20260101-120000.csv"
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["t_since_start_s", "event", "label", "gaze_x", "gaze_y"])
        for t, label in filas:
            w.writerow([f"{t:.3f}", "select", label, 0, 0])
    return tmp


def _extrae_las_medidas() -> str:
    path = _csv_falso([
        (0.0, "TAREA:A1"),
        (1.0, "QUIERO"), (2.0, "AGUA"), (2.5, "BORRAR"), (3.0, "AGUA"),
        (4.0, "HABLAR"), (5.0, "DICHO:quiero agua"),
        (5.1, "TAREA:A2"),
        (6.0, "ABC"), (7.0, "M"), (8.0, "HABLAR"), (9.0, "DICHO:tengo sueño"),
        (9.1, "BLOQUE:fin"),
    ])
    b = medidas.extraer(path, {"A1": "quiero agua", "A2": "llama a marta"})
    assert len(b.enunciados) == 2, b.enunciados
    e1, e2 = b.enunciados
    assert abs(e1.segundos - 5.0) < 1e-6, e1.segundos
    # HABLAR cuenta como permanencia: le cuesta a la persona
    # exactamente lo mismo que cualquier otra celda.
    assert e1.permanencias == 5, e1.permanencias
    assert e1.correcciones == 1, e1.correcciones
    assert e1.navegacion == 0
    assert e1.veredicto == "acierto", (e1.f1, e1.veredicto)
    # La navegación se cuenta aparte del mensaje: es el costo de la ESTRUCTURA
    # del tablero y en deletreo es cero por definición.
    assert e2.navegacion == 1, e2.navegacion
    assert e2.veredicto == "fallo", (e2.f1, e2.veredicto)
    assert not b.incompleto
    return "tiempos, permanencias, correcciones y navegación por tarea"


def _un_bloque_cortado_se_marca() -> str:
    """Abandonar es un dato sobre la condición, no un archivo roto."""
    path = _csv_falso([
        (0.0, "TAREA:A1"), (1.0, "AGUA"), (2.0, "HABLAR"),
        (3.0, "DICHO:quiero agua"), (3.1, "TAREA:A2"),
    ])
    b = medidas.extraer(path, {"A1": "quiero agua", "A2": "llama a marta"})
    assert b.incompleto, "no marcó el bloque cortado"
    assert len(b.enunciados) == 1, "descartó lo que sí se había medido"
    return "se marca incompleto y se conserva lo medido"


def _la_zona_gris_queda_pendiente() -> str:
    """Una paráfrasis dudosa no se resuelve con una fórmula."""
    casos = [("quiero agua", "quiero agua", "acierto"),
             ("tengo sed", "quiero agua", "fallo"),
             ("quiero un poco de agua fresca", "quiero agua", "pendiente")]
    vistos = []
    for dicho, intencion, esperado in casos:
        path = _csv_falso([(0.0, "TAREA:X"), (1.0, "AGUA"),
                           (2.0, f"DICHO:{dicho}"), (2.1, "BLOQUE:fin")])
        e = medidas.extraer(path, {"X": intencion}).enunciados[0]
        assert e.veredicto == esperado, (dicho, e.f1, e.veredicto, esperado)
        vistos.append(e.veredicto)
    assert medidas.UMBRAL_BAJO < medidas.UMBRAL_ALTO
    return f"acierto/fallo/pendiente según el F1: {vistos}"


def _el_ahorro_ignora_los_fallos() -> str:
    """Rendirse es rápido: si contara, la peor condición ganaría en tiempo."""
    def _bloque(cond, filas):
        p = _csv_falso(filas)
        return medidas.extraer(p, {"A1": "quiero agua", "A2": "quiero agua"},
                               condicion=cond)

    base = _bloque("letra", [
        (0.0, "TAREA:A1")] + [(i * 0.5, "A") for i in range(1, 21)]
        + [(11.0, "DICHO:quiero agua"), (11.1, "BLOQUE:fin")])
    # La otra: un acierto caro y un fallo rapidísimo. El fallo NO debe
    # abaratar la mediana.
    otra = _bloque("sin_contexto", [
        (0.0, "TAREA:A1"), (1.0, "QUIERO"), (2.0, "AGUA"),
        (6.0, "DICHO:quiero agua"),
        (6.1, "TAREA:A2"), (6.2, "NO"), (6.3, "DICHO:no"),
        (6.4, "BLOQUE:fin")])
    a = medidas.ahorro(base, otra)
    assert a["permanencias_otra"] == 2, a
    assert a["ahorro_permanencias"] > 0.8, a
    assert a["acierto_otra"] == 0.5, a
    return (f"ahorro {a['ahorro_permanencias']:.0%} sobre los acertados, "
            f"con {a['acierto_otra']:.0%} de acierto declarado aparte")


def _el_estudio_no_guarda_identidad() -> str:
    """Si algún día se agrega un nombre, esto tiene que fallar."""
    from . import sesion

    campos = set()
    estado = sesion.EstadoEstudio()
    estado.participantes["1"] = {"alta": 0.0, "notas": "", "bloques": {}}
    campos |= set(estado.participantes["1"])
    prohibidos = {"nombre", "apellido", "email", "correo", "telefono",
                  "dni", "documento", "direccion"}
    assert not (campos & prohibidos), f"guarda identidad: {campos & prohibidos}"
    return f"campos guardados: {sorted(campos)}"


def _el_tablero_acepta_el_guion() -> str:
    """La firma que el estudio necesita del tablero sigue existiendo."""
    import inspect
    import sys as _sys

    raiz = Path(__file__).resolve().parents[2]
    for sub in ("slm", "Eyetrax"):
        if str(raiz / sub) not in _sys.path:
            _sys.path.insert(0, str(raiz / sub))
    from aac_semantic import gaze_board

    firma = inspect.signature(gaze_board.run).parameters
    for p in ("guion", "expandir", "contexto", "pagina_inicial"):
        assert p in firma, f"gaze_board.run perdió el parámetro {p!r}"

    g = gaze_board._Guion([("A1", "Tiene sed."), ("A2", "Tiene frío.")])
    assert "1/2" in g.banda()
    g.avanzar()
    g.avanzar()
    assert g.terminado and g.banda() == ""
    return "run() acepta guion/expandir/contexto/pagina_inicial"


def _la_api_sirve_json_valido() -> str:
    """NaN no es JSON y `json.dumps` lo escribe igual.

    `json.loads` de Python lo ACEPTA, así que probarlo con Python no detecta
    nada: hay que verificar como lo hace `JSON.parse` del navegador, que lanza
    una excepción y deja la página en blanco. Y los NaN acá son normales -un
    participante sin línea de base todavía no tiene ahorro calculable- así que
    aparecerían desde la primera sesión.
    """
    import json

    from . import servidor

    sucio = {"ahorro": float("nan"), "lista": [1.0, float("inf")],
             "anidado": {"x": float("nan")}, "texto": "ok"}
    texto = json.dumps(servidor._sin_nan(sucio), allow_nan=False)
    assert "NaN" not in texto and "Infinity" not in texto, texto

    def _explota(c):
        raise AssertionError(f"constante no-JSON en la respuesta: {c}")

    vuelta = json.loads(texto, parse_constant=_explota)
    assert vuelta["ahorro"] is None and vuelta["anidado"]["x"] is None
    assert vuelta["lista"] == [1.0, None] and vuelta["texto"] == "ok"
    return "NaN e Infinity salen como null, en todo el árbol"


CHECKS = (
    ("el orden no es siempre el mismo", _orden_no_es_siempre_el_mismo),
    ("cada condición cae en cada posición", _cada_condicion_cae_en_cada_posicion),
    ("la asignación es reproducible", _la_asignacion_es_reproducible),
    ("las frases no se pegan a una condición",
     _las_frases_no_se_pegan_a_una_condicion),
    ("los conjuntos de frases están emparejados",
     _los_conjuntos_de_frases_estan_emparejados),
    ("las pistas no dictan la frase", _las_pistas_no_dictan_la_frase),
    ("las condiciones difieren solo en lo que deben",
     _las_condiciones_se_diferencian),
    ("extrae las medidas del CSV", _extrae_las_medidas),
    ("un bloque cortado se marca, no se tira", _un_bloque_cortado_se_marca),
    ("la zona gris espera juicio humano", _la_zona_gris_queda_pendiente),
    ("el ahorro ignora los fallos", _el_ahorro_ignora_los_fallos),
    ("el estudio no guarda identidad", _el_estudio_no_guarda_identidad),
    ("el tablero acepta el guion", _el_tablero_acepta_el_guion),
    ("la API sirve JSON válido", _la_api_sirve_json_valido),
)


def run() -> int:
    print("chequeos del estudio (sin cámara ni modelo):\n")
    pasaron = sum(_check(n, f) for n, f in CHECKS)
    print(f"\n{pasaron}/{len(CHECKS)} pasaron")
    print("\n  Un fallo acá es un fallo del DISEÑO, no de un participante: "
          "se mide contra\n  sesiones sintéticas con la respuesta conocida.")
    return 0 if pasaron == len(CHECKS) else 1
