"""Chequeos de la tabla que decide y de la medición de costo.

Lo que se verifica no es el formato de la tabla sino **el criterio**: que la
seguridad sea eliminatoria, que la licencia sea eliminatoria para distribuir, y
que el desempate sea por tamaño. Si alguna de esas tres se invierte por un
descuido, la tabla sigue imprimiéndose igual de linda y recomendando el modelo
equivocado, que es la clase de error que nadie nota hasta que está en la tesis.
"""

from __future__ import annotations

from . import decision, footprint, metrics


def _scores(model: str, variant: str, *, n: int = 100, recall: float = 0.9,
            inventos: int = 0) -> list[metrics.ItemScore]:
    """Un bloque de puntajes sintéticos con el recall pedido.

    Todo lo demás queda perfecto a propósito: así, cuando una fila no pasa la
    puerta, se sabe exactamente cuál de las condiciones la volteó.
    """
    out = []
    for i in range(n):
        out.append(metrics.ItemScore(
            item_id=f"i{i}", variant=variant, model=model,
            candidates=["uno", "dos", "tres"], fmt="json",
            latency_s=0.5, model_s=0.5,
            hit=i < int(round(recall * n)),
            literal_present=True,
            invented=["inventado"] if i < inventos else [],
            diversity=0.5,
        ))
    return out


def gate_is_eliminatory() -> str:
    """Un modelo que inventa no gana aunque tenga el mejor recall."""
    scores = (_scores("qwen2.5:7b", "v6-glosario", recall=0.99, inventos=5)
              + _scores("qwen2.5:1.5b", "v6-glosario", recall=0.85))
    texto = decision.tabla_decision(scores)
    assert "qwen2.5:1.5b" in texto.split("Recomendación")[1], (
        "la recomendación no nombra al modelo seguro")
    assert "qwen2.5:7b" not in texto.split("Recomendación")[1].split("\n")[2], (
        "recomienda el modelo que inventa")
    return "el que inventa pierde contra uno 14 puntos peor de recall"


def license_blocks_distribution() -> str:
    """qwen2.5:3b es Qwen Research: apto para medir, no para entregar."""
    assert "qwen2.5:3b" in decision.CANDIDATE_MODELS
    lic = decision.CANDIDATE_MODELS["qwen2.5:3b"]["license"]
    assert lic not in decision.LICENCIAS_DISTRIBUIBLES, (
        f"qwen2.5:3b figura como distribuible con licencia {lic!r}")

    # Mismo recall, misma seguridad: lo único que los separa es la licencia.
    scores = (_scores("qwen2.5:3b", "v6-glosario", recall=0.9)
              + _scores("qwen2.5:1.5b", "v6-glosario", recall=0.9))
    reco = decision.tabla_decision(scores).split("Recomendación")[1]
    assert "qwen2.5:1.5b" in reco, "la licencia no desempató"
    return f"qwen2.5:3b ({lic}) queda fuera; gana el Apache-2.0 con igual recall"


def smallest_wins_above_the_gate() -> str:
    """Entre los que pasan, gana el más chico, no el de mejor recall."""
    scores = (_scores("qwen2.5:7b", "v6-glosario", recall=0.95)
              + _scores("qwen2.5:1.5b", "v6-glosario", recall=0.85))
    reco = decision.tabla_decision(scores).split("Recomendación")[1]
    assert "qwen2.5:1.5b" in reco.split("\n")[2], (
        "no desempató por tamaño: recomendó el grande")
    # Y tiene que DECIR lo que se está dejando sobre la mesa, no esconderlo.
    assert "qwen2.5:7b" in reco and "+10%" in reco, (
        "no declara cuánto recall se resigna al elegir el chico")
    return "gana el de 1.0 GB sobre el de 4.7 GB, y declara los 10 puntos que cede"


def nobody_passes_is_reported() -> str:
    """Cuando nadie pasa la puerta, se dice; no se corre el umbral."""
    scores = _scores("qwen2.5:3b", "v6-glosario", recall=0.6)
    texto = decision.tabla_decision(scores)
    assert "NINGUNA configuración pasa" in texto
    assert "20%" in texto, "no dice cuánto falta para la puerta"
    assert "no el umbral" in texto, "no aclara qué hay que mover"
    return "declara que nadie pasa y cuánto falta, sin bajar la puerta"


def family_needs_both_arms() -> str:
    """Una familia sin su par no se compara: se declara faltante."""
    solo = _scores("qwen2.5:3b", "v6-glosario", recall=0.9)
    texto = decision.comparacion_familias(solo)
    assert "falta llama3.2:3b" in texto, "no avisa qué brazo falta"
    assert "Ninguna pareja completa" in texto

    par = solo + _scores("llama3.2:3b", "v6-glosario", recall=0.7)
    texto2 = decision.comparacion_familias(par)
    assert "qwen2.5:3b vs llama3.2:3b" in texto2
    assert "-20%" in texto2, "no reporta el delta entre familias"
    return "sin el par declara el faltante; con el par reporta delta y veredicto"


def scale_curve_calls_flat_flat() -> str:
    """Una curva plana se reporta como resultado, no como falta de resultado."""
    plana = (_scores("qwen2.5:0.5b", "v6", recall=0.70)
             + _scores("qwen2.5:3b", "v6", recall=0.72))
    assert "PLANA" in decision.curva_de_escala(plana), (
        "no reconoce una curva plana como resultado")

    empinada = (_scores("qwen2.5:0.5b", "v6", recall=0.50)
                + _scores("qwen2.5:3b", "v6", recall=0.85))
    assert "SÍ compra recall" in decision.curva_de_escala(empinada)
    return "plana -> 'la restricción sale gratis'; empinada -> 'tiene costo'"


def declared_sizes_are_marked() -> str:
    """Un tamaño copiado del fabricante no se imprime como medición."""
    scores = _scores("qwen2.5:1.5b", "v6-glosario", recall=0.9)
    sin = decision.tabla_decision(scores)
    assert "~" in sin and "DECLARADO" in sin, (
        "pasa un tamaño declarado por medido")

    fp = footprint.Footprint(model="qwen2.5:1.5b", disk_gb=0.98, ram_gb=1.4,
                             reload_s=2.1, quantization="Q4_K_M")
    con = decision.tabla_decision(scores, [fp])
    fila = [l for l in con.splitlines() if "qwen2.5:1.5b" in l][0]
    assert "1.0G~" not in fila and "1.4G" in fila, (
        f"no usó la medición real: {fila!r}")
    return "sin medición marca ~; con medición usa el disco y la RAM reales"


def footprint_survives_a_dead_ollama() -> str:
    """Sin Ollama la medición falla ordenada, no revienta la corrida."""
    fp = footprint.measure("qwen2.5:1.5b", url="http://127.0.0.1:1",
                           repeticiones=1)
    assert not fp.ok and fp.error, "no reportó el fallo"
    texto = footprint.table([fp])
    assert "No hay ninguna medición válida" in texto
    # Y sobre todo: tiene que decir CUÁL falló y POR QUÉ, incluso cuando no
    # sobrevive ninguna fila. Si se pierden los motivos, el comando devuelve
    # una tabla vacía y nadie sabe que Ollama estaba apagado.
    assert "sin medir" in texto and "qwen2.5:1.5b" in texto, (
        f"se perdieron los fallos cuando no hay ninguna medición:\n{texto}")
    return f"falla declarada sin excepción: {fp.error[:40]}"


def cold_start_limit_is_declared() -> str:
    """La recarga medida es tibia, y la tabla tiene que decirlo."""
    fp = footprint.Footprint(model="x", disk_gb=1.0, reload_s=4.3)
    texto = footprint.table([fp])
    assert "TIBIA" in texto, "no declara que la recarga no es en frío"
    assert "39 s" in texto, "no da la magnitud de la diferencia"
    return "la tabla declara que no mide el arranque en frío real"


CHECKS = (
    ("la seguridad es eliminatoria", gate_is_eliminatory),
    ("la licencia bloquea la distribución", license_blocks_distribution),
    ("arriba de la puerta gana el más chico", smallest_wins_above_the_gate),
    ("cuando nadie pasa, lo dice", nobody_passes_is_reported),
    ("comparar familias exige los dos brazos", family_needs_both_arms),
    ("una curva plana es un resultado", scale_curve_calls_flat_flat),
    ("los tamaños declarados se marcan", declared_sizes_are_marked),
    ("footprint sobrevive sin Ollama", footprint_survives_a_dead_ollama),
    ("el límite del arranque en frío se declara", cold_start_limit_is_declared),
)
