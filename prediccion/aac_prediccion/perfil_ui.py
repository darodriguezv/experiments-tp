"""El onboarding del perfil, por teclado. Opcional y salteable en cada paso.

Por teclado y no por mirada a propósito: esto lo completa quien acompaña a la
persona, una vez, antes de empezar. Hacerlo con permanencia costaría cientos de
selecciones para escribir cuatro nombres, que es exactamente el problema que el
dispositivo intenta resolver.

Toda pregunta se saltea con ENTER vacío. Un perfil a medias es útil -los
nombres solos ya ahorran cinco permanencias cada uno- y obligar a completarlo
entero haría que se complete mal.
"""

from __future__ import annotations

from pathlib import Path

from .perfil import Perfil, sembrar, sugerir_terminos
from .store import UserModel


def _preguntar(texto: str, ayuda: str = "") -> str:
    if ayuda:
        print(f"  ({ayuda})")
    try:
        return input(f"  {texto}: ").strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def _lista(texto: str, ayuda: str = "") -> list[str]:
    bruto = _preguntar(texto, ayuda)
    return [p.strip() for p in bruto.split(",") if p.strip()]


def entrevistar(perfil: Perfil | None = None) -> Perfil:
    """Las preguntas. Devuelve el perfil con lo que se haya contestado."""
    p = perfil or Perfil()
    print()
    print("=" * 68)
    print("  PERFIL DE LA PERSONA  (todo opcional, ENTER saltea)")
    print("=" * 68)
    print("  Esto sirve para que el tablero no arranque en blanco. Nada de")
    print("  esto sale del equipo, y se borra con `main.py perfil --borrar`.")
    print()

    p.nombre = _preguntar("Nombre", "solo para mostrarlo; no se usa en el modelo") or p.nombre
    edad = _preguntar("Edad")
    if edad.isdigit():
        p.edad = int(edad)
    p.sexo = _preguntar("Sexo o género") or p.sexo
    p.region = _preguntar("De dónde es", "define el trato: «quedate» o «quédate»") or p.region
    p.profesion = _preguntar("A qué se dedicaba") or p.profesion

    print()
    print("  Lo que sigue es lo que MÁS permanencias ahorra: cada nombre")
    print("  deletreado cuesta cinco o más, y como frase sembrada cuesta una.")
    print()
    p.personas = _lista("Personas cercanas", "separadas por coma: Ana, Marta, el doctor Ruiz") or p.personas
    p.intereses = _lista("Cosas que le gustan", "fútbol, tango, jardinería") or p.intereses
    p.objetos = _lista("Objetos de su entorno", "la radio, los anteojos, el perro") or p.objetos
    p.notas = _preguntar("Algo más que ayude a nombrar su mundo") or p.notas
    return p


def revisar_sugerencias(p: Perfil, backend) -> Perfil:
    """Le pide términos al modelo y los acepta de a uno.

    De a uno y no en bloque: el modelo acierta la mayoría y se equivoca en
    algunos, y aceptar en bloque mete los errores junto con los aciertos. Como
    cada término aceptado se va a ofrecer después como si la persona lo hubiera
    dicho, el repaso vale los veinte segundos que cuesta.
    """
    print()
    print("  Pensando qué más podría necesitar nombrar...")
    try:
        sugeridos = sugerir_terminos(p, backend)
    except Exception as exc:
        print(f"  (no se pudo consultar al modelo: {exc})")
        return p
    ya = set(p.terminos())
    nuevos = [t for t in sugeridos if t not in ya]
    if not nuevos:
        print("  Sin sugerencias nuevas.")
        return p

    print(f"  El modelo sugiere {len(nuevos)}. ENTER acepta, 'n' descarta, "
          f"'basta' corta.")
    aceptados = []
    for t in nuevos:
        r = _preguntar(f"¿Agregar «{t}»? [S/n]").lower()
        if r in ("basta", "q"):
            break
        if r not in ("n", "no"):
            aceptados.append(t)
    p.objetos = list(p.objetos) + aceptados
    print(f"  {len(aceptados)} término(s) agregado(s).")
    return p


def run(perfil_path: str, modelo_path: str, con_modelo: bool = True) -> int:
    """El onboarding completo. Devuelve código de salida."""
    pp, mp = Path(perfil_path), Path(modelo_path)
    p = Perfil.load(pp)
    if not p.vacio:
        print(f"[perfil] ya existe en {pp}; se edita sobre lo que hay "
              f"(ENTER conserva cada valor).")

    p = entrevistar(p)

    if con_modelo:
        backend = _backend()
        if backend is not None:
            p = revisar_sugerencias(p, backend)

    if p.vacio:
        print("[perfil] no se completó nada; no se guarda.")
        return 0

    p.save(pp)
    print(f"\n[perfil] guardado -> {pp}")

    modelo = UserModel.load(mp)
    n = sembrar(modelo, p)
    modelo.save(mp)
    print(f"[perfil] {n} frase(s) sembrada(s) en {mp}")
    print(f"[perfil] términos: {', '.join(p.terminos()) or '(ninguno)'}")
    print()
    print("  Ojo: lo sembrado entra con el peso MÍNIMO. Una sola vez que la")
    print("  persona diga algo de verdad, eso le gana al perfil - que es el")
    print("  orden correcto.")
    return 0


def _backend():
    """El SLM, si está. Sin él el onboarding funciona igual, con menos ayuda."""
    import sys

    raiz = Path(__file__).resolve().parents[2]
    if str(raiz / "slm") not in sys.path:
        sys.path.insert(0, str(raiz / "slm"))
    try:
        from aac_semantic import backends

        b = backends.OllamaBackend("qwen2.5:3b", temperature=0.3, seed=None)
        if not b.available():
            print("  (Ollama no responde; se saltean las sugerencias)")
            return None
        return b
    except Exception as exc:
        print(f"  (sin sugerencias del modelo: {exc})")
        return None
