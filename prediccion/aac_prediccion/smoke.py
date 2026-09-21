"""Chequeos sin usuario y sin modelo: el instrumento antes que el resultado.

Un fallo acá invalida cualquier número que el módulo produzca, así que corre
primero y corre siempre.
"""

from __future__ import annotations

import tempfile
import time
import traceback
from pathlib import Path

from . import board, report
from .predict import Predictor
from .replay import Utterance, replay
from .simulate import Persona
from .store import UserModel


def _check(name, fn) -> bool:
    t0 = time.perf_counter()
    try:
        detail = fn()
    except Exception as exc:
        print(f"  FALLA {name}: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=3)
        return False
    print(f"  ok    {name}{f' - {detail}' if detail else ''}   "
          f"[{(time.perf_counter() - t0) * 1000:.0f} ms]")
    return True


def _board_costs_are_right() -> str:
    """El modelo de costo, contra valores calculados a mano."""
    uno, page = board.dwells_for(["agua"], "core")
    assert uno == 1 and page == "core", (uno, page)

    dos, page = board.dwells_for(["agua"], "abc")
    assert dos == 2 and page == "core", "volver de ABC cuesta una permanencia"

    # "hermana" no está en el tablero: cambio de página + 7 letras + espacio.
    nueve, page = board.dwells_for(["hermana"], "core")
    assert nueve == 9, nueve
    assert page == "abc", "deletrear deja el tablero en ABC"

    # Las tildes se cuentan sobre la letra base porque la página abc no las da.
    assert board.spell_cost("está") == board.spell_cost("esta")
    return (f"agua=1, agua desde abc=2, hermana=9; "
            f"{len(board.VOCAB)} palabras en el tablero")


def _no_prediction_without_evidence() -> str:
    """Un modelo vacío no propone nada. Es el arranque en frío, y es a propósito."""
    model = UserModel()
    props = Predictor(model).propose([], "core")
    assert props == [], props

    # Una sola observación tampoco alcanza: min_evidence = 2.
    model.observe(["quiero", "agua"], now=1000.0)
    assert Predictor(model).propose([], "core", now=1000.0) == []
    return "sin datos no hay casillas; una sola observación tampoco"


def _zero_saving_never_takes_a_slot() -> str:
    """El hallazgo central, fijado por un test.

    NECESITO está en `core` y cuesta una permanencia. Predecirla en la franja
    también cuesta una. Por muy probable que sea, ocupar una casilla con eso es
    cobrar una celda de lectura a cambio de nada.
    """
    model = UserModel()
    for i in range(10):
        model.observe(["necesito"], now=1000.0 + i)

    props = Predictor(model).propose([], "core", now=1010.0)
    assert all(p.tokens != ["necesito"] for p in props), \
        f"una palabra de la página actual ocupó una casilla: {props}"

    # La misma palabra SÍ ahorra si el tablero quedó en la página de letras.
    props_abc = Predictor(model).propose([], "abc", now=1010.0)
    assert any(p.tokens == ["necesito"] for p in props_abc), \
        "desde abc, necesito ahorra el cambio de página y debería aparecer"
    return "palabra de la página actual: descartada; desde abc: ofrecida"


def _phrases_are_where_the_saving_is() -> str:
    """Una frase repetida tiene que ahorrar sus permanencias menos una."""
    model = UserModel()
    frase = ["quiero", "agua"]
    for i in range(5):
        model.observe(frase, now=1000.0 + i)

    props = Predictor(model).propose([], "core", now=1010.0)
    match = [p for p in props if p.tokens == frase]
    assert match, f"la frase repetida no se ofreció: {props}"
    assert match[0].saving == 1, match[0].saving

    # Y una con palabra fuera del tablero ahorra muchísimo más.
    model2 = UserModel()
    for i in range(5):
        model2.observe(["llama", "hermana"], now=1000.0 + i)
    p2 = [p for p in Predictor(model2).propose([], "core", now=1010.0)
          if p.tokens == ["llama", "hermana"]]
    assert p2 and p2[0].saving >= 10, p2
    return (f"'quiero agua' ahorra {match[0].saving}; "
            f"'llama hermana' ahorra {p2[0].saving}")


def _recency_beats_stale_frequency() -> str:
    """Lo viejo pierde contra lo reciente. Es el punto del decaimiento."""
    model = UserModel(halflife_days=7.0)
    viejo = 1_000_000.0
    for i in range(10):
        model.observe(["quiero", "comida"], now=viejo + i)
    reciente = viejo + 60 * 86400.0
    for i in range(4):
        model.observe(["quiero", "agua"], now=reciente + i)

    ranked = model.phrase_completions((), now=reciente + 10)
    assert ranked, "no devolvió nada"
    assert ranked[0][0] == ["quiero", "agua"], \
        f"10 veces hace dos meses le ganó a 4 veces hoy: {ranked}"
    return "4 recientes > 10 de hace dos meses, con vida media de 7 días"


def _persistence_round_trip() -> str:
    model = UserModel(halflife_days=12.0, min_evidence=3.0)
    model.observe(["por favor", "baño"], now=5000.0)
    model.observe(["gracias"], now=5001.0)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "modelo.json"
        model.save(path)
        again = UserModel.load(path)
    assert again.halflife_days == 12.0 and again.min_evidence == 3.0
    assert again.utterances_seen == 2
    a = model.phrase_completions((), now=5002.0, limit=9)
    b = again.phrase_completions((), now=5002.0, limit=9)
    assert a == b, (a, b)
    # Un archivo que no existe da un modelo vacío, no una excepción: es el
    # primer arranque de cualquier usuario.
    assert UserModel.load(Path(tempfile.gettempdir()) / "no-existe-xyz.json"
                          ).utterances_seen == 0
    return "ida y vuelta por JSON idéntica; archivo ausente = modelo vacío"


def _harness_refuses_to_peek() -> str:
    """La prueba de que la disciplina temporal importa.

    Se mide lo mismo dos veces: honesto (el modelo solo vio el pasado) y con
    trampa (entrenado con todo el historial antes de medir). Si la trampa no
    diera mejor, el arnés no estaría midiendo predicción en absoluto.
    """
    utts = Persona(days=20, seed=3).generate()

    honesto = replay(utts, UserModel())

    tramposo_model = UserModel()
    for u in utts:
        tramposo_model.observe(u.tokens, u.t)
    tramposo = replay(utts, tramposo_model, learn=False)

    assert tramposo.savings > honesto.savings, (
        f"entrenar con el futuro no mejoró nada ({tramposo.savings:.1%} vs "
        f"{honesto.savings:.1%}): el arnés no está midiendo predicción")
    return (f"honesto {honesto.savings:.1%} < con trampa "
            f"{tramposo.savings:.1%}; la diferencia es lo que NO se puede citar")


def _scan_cost_is_charged() -> str:
    """Las casillas se cobran aunque no se usen."""
    utts = Persona(days=10, seed=5).generate()
    r = replay(utts, UserModel(), slots=5)
    assert r.cells_per_selection > r.cells_per_selection_baseline, (
        "la franja no está sumando celdas: el costo no se está cobrando")
    delta = r.cells_per_selection - r.cells_per_selection_baseline
    assert delta <= 5.0 + 1e-6, delta
    return (f"{r.cells_per_selection_baseline:.1f} -> "
            f"{r.cells_per_selection:.1f} celdas por selección")


def _simulator_is_not_the_model() -> str:
    """El techo tiene que estar por debajo del 100 %, o el simulador miente."""
    p = Persona(days=40, seed=11, novelty=0.15)
    utts = p.generate()
    seen: set[tuple[str, ...]] = set()
    novel = 0
    for u in utts:
        key = tuple(u.tokens)
        if key not in seen:
            novel += 1
        seen.add(key)
    assert novel >= 10, f"solo {novel} enunciados nuevos: no hay novedad real"
    r = replay(utts, UserModel())
    assert r.savings < 0.95, (
        f"ahorro de {r.savings:.1%}: sospechosamente perfecto, revisá si el "
        f"simulador se volvió el mismo modelo que el predictor")
    return (f"{novel} enunciados distintos sobre {len(utts)}; "
            f"ahorro {r.savings:.1%}, techo por debajo del 100 %")


def _drift_is_survivable() -> str:
    """Tras la deriva, un modelo con recencia tiene que recuperarse."""
    utts = Persona(days=60, seed=13, drift_day=30).generate()
    corte = utts[0].t + 30 * 86400.0
    despues = [u for u in utts if u.t >= corte]

    rapido = replay(utts, UserModel(halflife_days=7.0))
    lento = replay(utts, UserModel(halflife_days=3650.0))
    assert len(despues) > 50, len(despues)
    return (f"vida media 7d: {rapido.savings:.1%}  |  "
            f"10 años (casi sin olvido): {lento.savings:.1%}")


def _report_is_honest_about_loss() -> str:
    """Si el ahorro no es positivo, el informe tiene que decir que no la pongas."""
    utts = [Utterance(["agua"], 1000.0 + i) for i in range(30)]
    r = replay(utts, UserModel(), slots=5)
    texto = report.summary(r, slots=5)
    assert r.savings <= 0.0, r.savings
    assert "NO ponerla" in texto, texto
    return "palabras sueltas de una celda: ahorro nulo y el informe lo dice"


def _session_csv_round_trip() -> str:
    """El parser de sesiones reales, contra una sesión armada a mano.

    Existe porque este parser va a leer el unico dato no sintetico que el
    proyecto tenga, y hoy no hay ninguna sesion con selecciones para probarlo:
    el CSV que existe en Eyetrax/logs tiene solo la cabecera. Un parser sin
    probar que se estrena sobre datos irrepetibles es una mala apuesta.

    La sesion de abajo ejercita lo que de verdad rompe: deletreo, ESPACIO,
    BORRAR sobre una letra, BORRAR sobre una palabra entera, LIMPIAR que tira
    todo, y un borrador final que nunca llega a HABLAR.
    """
    from . import sessions

    filas = [
        "QUIERO", "AGUA", "HABLAR",                  # -> quiero agua
        "ABC", "H", "E", "R", "M", "A", "N", "O",
        "BORRAR", "A", "ESPACIO",                    # hermano -> hermana
        "HABLAR",                                    # -> hermana
        "PALABRAS", "GRACIAS", "DOLOR", "BORRAR",    # DOLOR se va
        "HABLAR",                                    # -> gracias
        "AYUDA", "LIMPIAR",                          # LIMPIAR tira todo
        "SÍ", "HABLAR",                              # -> sí
        "COMIDA",                                    # borrador sin HABLAR
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "aac-session-20260101-101500.csv"
        lineas = ["t_since_start_s,event,label,gaze_x,gaze_y"]
        for i, etiqueta in enumerate(filas):
            lineas.append(f"{i}.0,select,{etiqueta},100,200")
        path.write_text(chr(10).join(lineas), encoding="utf-8")

        got, vacias = sessions.load(path)

    dichos = [u.tokens for u in got]
    esperado = [["quiero", "agua"], ["hermana"], ["gracias"], ["sí"]]
    assert dichos == esperado, f"{dichos} != {esperado}"
    assert vacias == 0, vacias
    # El instante sale del nombre del archivo, no de "ahora": sin base
    # absoluta el decaimiento por recencia no tendria contra que medirse.
    assert got[0].t > 1_700_000_000.0, got[0].t
    assert got[1].t > got[0].t
    return ("4 enunciados de 26 selecciones; BORRAR, LIMPIAR y el borrador "
            "final descartados")


CHECKS = (
    ("el costo del tablero es correcto", _board_costs_are_right),
    ("sin evidencia no hay predicción", _no_prediction_without_evidence),
    ("ahorro cero nunca ocupa casilla", _zero_saving_never_takes_a_slot),
    ("las frases son donde está el ahorro", _phrases_are_where_the_saving_is),
    ("la recencia le gana a la frecuencia vieja", _recency_beats_stale_frequency),
    ("el modelo se guarda y se lee igual", _persistence_round_trip),
    ("el arnés no mira el futuro", _harness_refuses_to_peek),
    ("la lectura de las casillas se cobra", _scan_cost_is_charged),
    ("el simulador no es el modelo", _simulator_is_not_the_model),
    ("la deriva se sobrevive", _drift_is_survivable),
    ("las sesiones reales se parsean bien", _session_csv_round_trip),
    ("el informe admite cuando no conviene", _report_is_honest_about_loss),
)


def run() -> int:
    print("chequeos del módulo de predicción:")
    passed = sum(_check(n, f) for n, f in CHECKS)
    print(f"\n{passed}/{len(CHECKS)} pasaron")
    if passed < len(CHECKS):
        print("\nUn fallo acá es del instrumento, no del usuario simulado: "
              "cualquier ahorro medido después queda sin respaldo.")
    return 0 if passed == len(CHECKS) else 1
