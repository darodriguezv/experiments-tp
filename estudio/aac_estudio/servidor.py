"""El conductor del estudio, en el navegador.

## Por qué el navegador y no OpenCV

El tablero sigue siendo nativo: necesita pantalla completa, coordenadas exactas
en píxeles de pantalla y un lazo de cámara a ~22 fps. Meterlo en una página
agregaría latencia en el único lazo que hoy funciona bien, y el viewport del
navegador no es la pantalla, así que habría que recalibrar el mapeo.

Pero el **conductor** no tiene ninguno de esos requisitos y sí tiene los que el
navegador resuelve bien: formularios, texto largo y cómodo de leer, listas,
estado que se revisa de un vistazo. Dibujar un formulario de alta con
`cv2.putText` es pelear contra la herramienta.

Y hay una razón del estudio, no de comodidad: el guion del roleplay lo lee el
participante, no el investigador. Tiene que verse como algo hecho para una
persona -párrafos, tipografía legible, sin jerga- y no como la consola de un
experimento. Alguien que lee su personaje en una terminal con fondo negro ya
empezó la sesión intimidado, y eso entra en lo que medimos.

## Sin dependencias

`http.server` de la biblioteca estándar. Es suficiente para una app que
atiende a una persona en localhost, y evita agregarle un framework web a un
proyecto que además tiene que correr en un equipo dedicado.

## El tablero corre en un hilo, no en un proceso

Abrir el tablero en el proceso del servidor mantiene una sola calibración y un
solo acceso a la cámara. El hilo es necesario igual porque OpenCV bloquea, y
mientras tanto la página consulta el estado.
"""

from __future__ import annotations

import json
import threading
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from . import diseno, informe, protocolo, sesion

WEB = Path(__file__).resolve().parent / "web"


class _Corrida:
    """El bloque que está corriendo ahora, si hay alguno.

    Uno solo a la vez, y el servidor lo rechaza explícitamente en vez de
    dejarlo pasar: dos tableros compitiendo por la misma cámara no fallan de
    forma obvia -uno recibe cuadros negros- y la sesión se perdería entera sin
    que nadie entienda por qué.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.activa: dict | None = None
        self.ultimo: dict | None = None

    def arrancar(self, participante: int, condicion: str, monitor: str,
                 camera: int, model: str | None) -> tuple[bool, str]:
        with self.lock:
            if self.activa is not None:
                return False, ("Ya hay un bloque abierto. Cerrá la ventana del "
                               "tablero antes de empezar otro.")
            self.activa = {"participante": participante,
                           "condicion": condicion}

        def _correr():
            try:
                res = sesion.correr_bloque(
                    participante, condicion, monitor=monitor, camera=camera,
                    model=model)
            except Exception as exc:  # noqa: BLE001 - se reporta a la página
                res = {"error": f"{type(exc).__name__}: {exc}"}
            with self.lock:
                self.ultimo = dict(res, participante=participante,
                                   condicion=condicion)
                self.activa = None

        threading.Thread(target=_correr, daemon=True).start()
        return True, ""

    def estado(self) -> dict:
        with self.lock:
            return {"corriendo": self.activa, "ultimo": self.ultimo}


CORRIDA = _Corrida()


def _sin_nan(obj):
    """NaN -> null, en todo el árbol.

    `json.dumps` de Python escribe `NaN` tal cual, que **no es JSON válido**:
    `JSON.parse` del navegador lanza una excepción y la página se queda en
    blanco. Y los NaN acá son normales, no excepcionales -un participante sin
    línea de base todavía no tiene ahorro calculable-, así que esto se dispara
    desde la primera sesión. `allow_nan=False` en el volcado deja que falle
    ruidosamente si algún día se cuela uno por otro camino.
    """
    import math

    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {k: _sin_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sin_nan(v) for v in obj]
    return obj


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(WEB), **kw)

    def log_message(self, *a):
        """Silencio: una línea por pedido tapa los mensajes del tablero, que
        son los que el investigador necesita ver en la consola."""

    # ------------------------------------------------------------------ API

    def do_GET(self):  # noqa: N802 - lo define la clase base
        if self.path.startswith("/api/"):
            return self._api_get()
        return super().do_GET()

    def do_POST(self):  # noqa: N802
        if not self.path.startswith("/api/"):
            return self._json({"error": "no existe"}, 404)
        largo = int(self.headers.get("Content-Length", 0))
        try:
            cuerpo = json.loads(self.rfile.read(largo) or b"{}")
        except json.JSONDecodeError:
            return self._json({"error": "JSON inválido"}, 400)
        return self._api_post(cuerpo)

    def _api_get(self):
        ruta = self.path.split("?")[0]
        estado = sesion.EstadoEstudio.cargar()

        if ruta == "/api/estudio":
            n = len(estado.participantes)
            return self._json({
                "participantes": sorted(int(k) for k in estado.participantes),
                "caso": {"nombre": protocolo.CASO.nombre,
                         "resumen": protocolo.CASO.resumen,
                         "guion": protocolo.CASO.guion},
                "condiciones": [{"clave": c, "nombre": diseno.NOMBRES[c]}
                                for c in diseno.CONDICIONES],
                "balance": diseno.informe_balance(max(n, 1)),
                "objetivo": 12,
                "n": n,
            })

        if ruta.startswith("/api/participante/"):
            try:
                p = int(ruta.rsplit("/", 1)[1])
            except ValueError:
                return self._json({"error": "número inválido"}, 400)
            ficha = estado.ficha(p)
            if not ficha:
                return self._json({"error": "no existe"}, 404)
            if ficha["siguiente"]:
                cond = ficha["siguiente"]["condicion"]
                ficha["siguiente"]["tareas"] = [
                    {"id": t.id, "pista": t.pista}
                    for t in sesion.tareas_de(p, cond)]
            return self._json(ficha)

        if ruta == "/api/corrida":
            return self._json(CORRIDA.estado())

        if ruta == "/api/resultados":
            return self._json(informe.para_web())

        return self._json({"error": "no existe"}, 404)

    def _api_post(self, cuerpo: dict):
        ruta = self.path.split("?")[0]

        if ruta == "/api/participante":
            n = sesion.EstadoEstudio.cargar().alta(cuerpo.get("notas", ""))
            return self._json({"participante": n})

        if ruta == "/api/lanzar":
            try:
                p = int(cuerpo["participante"])
                cond = str(cuerpo["condicion"])
            except (KeyError, ValueError, TypeError):
                return self._json({"error": "faltan datos"}, 400)
            if cond not in diseno.CONDICIONES:
                return self._json({"error": f"condición {cond!r}"}, 400)

            # El orden lo manda el contrabalanceo, no la página. Si se pudiera
            # elegir el bloque desde la UI, el diseño se rompería sin que nadie
            # se entere -que es exactamente la clase de error que no deja
            # rastro en los datos.
            ficha = sesion.EstadoEstudio.cargar().ficha(p)
            siguiente = ficha.get("siguiente")
            if not siguiente:
                return self._json({"error": "ese participante ya terminó"}, 409)
            if siguiente["condicion"] != cond:
                return self._json({
                    "error": f"le toca {diseno.NOMBRES[siguiente['condicion']]}, "
                             f"no {diseno.NOMBRES[cond]}. El orden lo fija el "
                             f"contrabalanceo."}, 409)

            ok, msg = CORRIDA.arrancar(
                p, cond, str(cuerpo.get("monitor", "ask")),
                int(cuerpo.get("camera", 0)), cuerpo.get("model") or None)
            return self._json({"ok": ok, "error": msg}, 200 if ok else 409)

        if ruta == "/api/adjudicar":
            try:
                informe.adjudicar(
                    int(cuerpo["participante"]), str(cuerpo["condicion"]),
                    str(cuerpo["tarea"]), str(cuerpo["veredicto"]),
                    str(cuerpo.get("quien", "investigador")))
            except (KeyError, ValueError) as exc:
                return self._json({"error": str(exc)}, 400)
            return self._json({"ok": True})

        return self._json({"error": "no existe"}, 404)

    def _json(self, data: dict, codigo: int = 200):
        cuerpo = json.dumps(_sin_nan(data), ensure_ascii=False,
                            allow_nan=False).encode("utf-8")
        self.send_response(codigo)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(cuerpo)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(cuerpo)


def servir(puerto: int = 8765, abrir: bool = True) -> int:
    # 127.0.0.1 y no 0.0.0.0: el conductor muestra el guion y los resultados de
    # un estudio con personas. No tiene por qué estar disponible en la red de
    # la facultad porque alguien dejó el servidor abierto.
    servidor = HTTPServer(("127.0.0.1", puerto), Handler)
    url = f"http://127.0.0.1:{puerto}/"
    print(f"[estudio] conductor en {url}")
    print("[estudio] Ctrl-C para cerrar.\n")
    if abrir:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\n[estudio] cerrado.")
    finally:
        servidor.server_close()
    return 0
