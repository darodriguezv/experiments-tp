"""Las tablas. Siempre con el costo al lado del ahorro."""

from __future__ import annotations

from .replay import Result


def summary(result: Result, slots: int) -> str:
    base_u, pred_u = result.dwells_per_utterance
    lines = [
        "=" * 70,
        "  PREDICCIÓN PERSONALIZADA - resultados",
        "=" * 70,
        "",
        f"  enunciados                {result.utterances}",
        f"  permanencias sin predecir {result.baseline_dwells}",
        f"  permanencias prediciendo  {result.predicted_dwells}",
        f"  AHORRO                    {result.savings:.1%}",
        "",
        f"  por enunciado             {base_u:.1f} -> {pred_u:.1f} permanencias",
        f"  aciertos                  {result.hit_rate:.1%} de las selecciones",
        "",
        "  lo que cuesta:",
        f"  celdas por selección      {result.cells_per_selection_baseline:.1f} "
        f"-> {result.cells_per_selection:.1f}  (+{slots} casillas fijas)",
    ]
    if result.savings <= 0:
        lines += [
            "",
            "  El ahorro no es positivo: la franja está cobrando lectura sin "
            "devolver",
            "  permanencias. Con estos datos, NO ponerla es la mejor opción.",
        ]
    return "\n".join(lines)


def sweep_table(rows: list[tuple[str, Result]], slots: int) -> str:
    head = (f"{'configuración':<26}{'ahorro':>9}{'aciertos':>10}"
            f"{'perm/enunc':>12}{'celdas/sel':>12}")
    out = [head, "-" * len(head)]
    for label, r in rows:
        _, pred_u = r.dwells_per_utterance
        out.append(f"{label:<26}{r.savings:>8.1%}{r.hit_rate:>10.1%}"
                   f"{pred_u:>12.1f}{r.cells_per_selection:>12.1f}")
    return "\n".join(out)
