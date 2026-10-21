"""Regenera `web/js/prompt.js` desde la fuente de verdad en Python.

    python web/generar_prompt.py [variante]

El prompt que corre en el navegador tiene que ser **el mismo texto** que se
midió en la grilla. Copiarlo a mano garantiza que en algún momento diverjan y
que la página quede corriendo algo que ninguna tabla de la tesis respalda, sin
que nadie se entere.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "slm"))

from aac_semantic import prompts  # noqa: E402

CABECERA = """/* El prompt. Generado desde `slm/aac_semantic/prompts.py`, variante {v}.
 *
 * NO se edita a mano. Es el mismo texto que se midió sobre las 133 entradas
 * del corpus; reescribirlo acá haría que la página corra algo que ninguna
 * tabla de la tesis respalda.
 *
 * Para cambiarlo: tocar prompts.py, correr la grilla de nuevo, y regenerar
 * este archivo con `python web/generar_prompt.py`.
 */

export const PROMPT_SISTEMA = """


def main(argv: list[str]) -> int:
    nombre = argv[1] if len(argv) > 1 else "v4-literal"
    if nombre not in prompts.VARIANTS:
        print(f"{nombre!r} no existe. Hay: {', '.join(prompts.VARIANTS)}")
        return 1
    v = prompts.VARIANTS[nombre]

    # El marcador se reemplaza DESPUÉS de serializar, para que json.dumps
    # escape el texto y la concatenación de JavaScript quede afuera de las
    # comillas en vez de adentro.
    usuario = json.dumps(v.user("@@"), ensure_ascii=False)
    usuario = usuario.replace("@@", '" + telegrama + "')

    destino = RAIZ / "web" / "js" / "prompt.js"
    partes = [
        CABECERA.format(v=nombre),
        json.dumps(v.system, ensure_ascii=False),
        ";\n\nexport function construirUsuario(telegrama) {\n  return ",
        usuario,
        ";\n}\n",
    ]
    destino.write_text("".join(partes), encoding="utf-8")
    print(f"[prompt] {nombre} -> {destino} ({destino.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
