/* WebGazer.js como estimador alternativo.
 *
 * El estimador propio sigue dando un desvío grande que cuatro rondas de
 * arreglos no cerraron. WebGazer hace lo mismo con una década de uso real
 * encima, así que vale medirlo en el mismo banco: misma calibración de nueve
 * puntos, mismo suavizado, misma corrección de deriva, misma prueba de
 * precisión. Si WebGazer anda y el propio no, el problema es el estimador; si
 * fallan los dos igual, el problema es el banco.
 *
 * ## Qué mira WebGazer, que es otra cosa
 *
 * El estimador propio toma 161 puntos faciales, los lleva a un marco local
 * anclado a la cara y regresiona sobre esa geometría: 486 rasgos. WebGazer
 * recorta los dos parches de los ojos, los reduce a 10x6 píxeles en grises con
 * el histograma ecualizado y regresiona sobre esos 120 valores. Es decir mira
 * la *imagen* del ojo, no la *forma* de la cara.
 *
 * Eso tiene una consecuencia que conviene anticipar: al ser píxeles crudos, es
 * más sensible a que la cabeza se mueva, porque el parche cambia de contenido.
 * A cambio no depende de que los puntos del iris salgan bien.
 *
 * ## Las cinco cosas que hay que corregirle a la configuración por defecto
 *
 * 1. **`faceMeshSolutionPath` apunta a `./mediapipe/face_mesh`**, relativo a
 *    la página. Sirviendo desde GitHub Pages eso es una ruta que no existe y
 *    el detector no carga nunca. Hay que apuntarlo al CDN.
 *
 * 2. **`saveDataAcrossSessions` viene en `true`**: WebGazer guarda lo
 *    calibrado en localforage y lo vuelve a cargar al abrir la página. Eso
 *    contradice de frente la regla del aparato -calibrar siempre, sin
 *    excepción- y además haría que la primera medición del día arrastre la
 *    postura de ayer. Va en `false`, y además se borra al empezar.
 *
 * 3. **`applyKalmanFilter` viene en `true`.** WebGazer trae su propio Kalman
 *    adentro de la regresión. Si se deja, la comparación mide dos estimadores
 *    con dos filtros distintos y la diferencia no es atribuible a ninguno. Se
 *    apaga y queda el suavizado compartido de `suavizado.js`.
 *
 * 4. **`init()` engancha los escuchas de mouse**, así que cada clic y cada
 *    movimiento del cursor entra como dato de entrenamiento. En un tablero por
 *    mirada eso es ruido puro. Se desenganchan apenas arranca.
 *
 * 5. **Muestra su propio video, su malla facial, su recuadro y su punto rojo.**
 *    Todo eso se apaga: la interfaz ya tiene su óvalo y su cursor.
 *
 * ## El límite de 50 muestras, que es lo que decide el diseño
 *
 * `util_regression.InitRegression` crea los arreglos de entrenamiento como
 * `DataWindow(50)`: un buffer circular de cincuenta posiciones. Pasada la
 * cincuenta, cada muestra nueva **pisa la más vieja**.
 *
 * El estimador propio graba 30 muestras por punto, 270 en total. Con WebGazer
 * esas 270 dejarían vivas solo las últimas 50, o sea el último punto y medio
 * de los nueve: un modelo entrenado contra la esquina inferior derecha y nada
 * más. Habría colapsado, y el síntoma habría sido idéntico al que estamos
 * persiguiendo -sesgo enorme, temblor chico-, con lo cual la comparación
 * habría dado "los dos fallan igual" por un motivo que no tiene nada que ver
 * con el estimador.
 *
 * Por eso acá son **5 muestras por punto**: 45 de las 50, y el margen que
 * queda absorbe algún clic suelto. No es una limitación que estemos
 * esquivando, es el tamaño para el que WebGazer está hecho: su propio ejemplo
 * de calibración usa nueve puntos y cinco clics en cada uno.
 *
 * Para que las cinco no salgan todas de la misma décima de segundo se las
 * separa 260 ms, que reparte las cinco a lo largo de la ventana de grabado.
 *
 * ## Licencia
 *
 * WebGazer es **GPL-3.0-or-later**; el encabezado del paquete agrega que
 * empresas valuadas en menos de un millón de dólares pueden usarlo bajo
 * LGPLv3. Para una tesis y para medir no cambia nada. Para distribuir el
 * aparato sí: la GPL es contagiosa y obligaría a liberar todo el sistema bajo
 * la misma licencia. Es una decisión a tomar recién si WebGazer gana la
 * medición, no antes.
 */

import { FalloEstimador } from "./estimador.js";
import { recorridoDe } from "./modelo.js";

const WG = "https://cdn.jsdelivr.net/npm/webgazer@3.5.3";

/* Los mismos índices de párpado que usa `rasgos.js`, para que el umbral de
 * parpadeo signifique lo mismo en los dos estimadores. Son todos menores a
 * 468, así que existen también en la malla sin iris que usa WebGazer. */
const OJO_IZQ = { interno: 133, externo: 33, arriba: 159, abajo: 145 };
const OJO_DER = { interno: 362, externo: 263, arriba: 386, abajo: 374 };

export class EstimadorWebGazer {
  constructor() {
    this.nombre = "webgazer";
    this.etiqueta = "WebGazer.js 3.5.3 (parches de ojo + cresta)";
    this.muestrasPorPunto = 5;      // 9 x 5 = 45 de las 50 que entran
    this.msEntreMuestras = 260;
    this.wg = null;
    this.video = null;
    this.resolucion = "";
    this.nPuntosCara = 0;
    this._ultima = null;            // última predicción del escucha
    this._grupos = [];              // a qué punto corresponde cada muestra
  }

  async iniciar() {
    await cargarScript(WG + "/dist/webgazer.js");
    const wg = window.webgazer;
    if (!wg) {
      throw new FalloEstimador(
        "No se pudo cargar WebGazer.",
        "El script se bajó pero no dejó la biblioteca en window.webgazer.");
    }
    this.wg = wg;

    // Todo esto va ANTES de begin(): `init()` lee los parámetros al construir
    // sus elementos, así que después ya es tarde para algunos.
    wg.params.faceMeshSolutionPath = WG + "/dist/mediapipe/face_mesh";
    wg.params.camConstraints = {
      video: { width: { ideal: 1280 }, height: { ideal: 720 },
               facingMode: "user" },
    };
    wg.params.saveDataAcrossSessions = false;
    wg.params.applyKalmanFilter = false;
    // `showVideo` queda en TRUE a propósito. Con `false`, WebGazer le pone
    // `display: none` al <video>, y su propio código trata a Safari y Firefox
    // como excepción -ahí usa opacidad- justamente porque un elemento sin caja
    // deja de entregar cuadros. Se deja que lo muestre y `estilo.css` lo saca
    // de pantalla conservando sus dimensiones.
    wg.params.showVideo = true;
    wg.params.showVideoPreview = true;
    wg.params.showFaceOverlay = false;
    wg.params.showFaceFeedbackBox = false;
    wg.params.showGazeDot = false;
    wg.params.storingPoints = false;
    wg.setRegression("ridge");
    wg.setTracker("TFFacemesh");

    /* `begin()` no resuelve hasta que el <video> dispara `loadeddata` y
     * WebGazer termina de armar su vista previa. Si ese evento no llega -pasa
     * cuando el navegador no entrega cuadros- la promesa queda colgada para
     * siempre y la interfaz se quedaría en "Esperando la cámara" sin decir
     * nada. Treinta segundos, contando que la primera vez hay que bajar 8 MB
     * de WebAssembly. */
    try {
      await conTiempo(wg.begin(), 30000);
    } catch (e) {
      throw e instanceof FalloEstimador ? e : new FalloEstimador(
        "No se pudo usar la cámara con WebGazer.",
        "Hace falta permiso de cámara y una conexión segura (https). "
        + String(e?.message || e));
    }

    // Sin esto, cada clic y cada movimiento del mouse entra como dato de
    // entrenamiento. En un tablero por mirada eso es ruido puro.
    wg.removeMouseEventListeners();
    // La malla y el punto rojo, apagados. El contenedor del video NO se oculta
    // acá: `estilo.css` lo deja con sus dimensiones reales y lo saca de
    // pantalla, porque un <video> sin caja deja de entregar cuadros.
    wg.showPredictionPoints(false);

    wg.setGazeListener((d) => {
      if (d && Number.isFinite(d.x) && Number.isFinite(d.y)) {
        this._ultima = [d.x, d.y];
      }
    });

    this.video = document.getElementById(wg.params.videoElementId);
    this.stream = this.video?.srcObject ?? null;
    await new Promise((r) => {
      if (this.video?.videoWidth) return r();
      this.video?.addEventListener("loadedmetadata", r, { once: true });
      setTimeout(r, 3000);
    });
    this.resolucion = this.video
      ? `${this.video.videoWidth}x${this.video.videoHeight}` : "?";
    await this.limpiarDatos();
  }

  leer(_t) {
    const pos = this.wg?.getTracker?.()?.getPositions?.();
    const v = this.video;
    if (!pos || !pos.length || !v?.videoWidth) return null;
    this.nPuntosCara = pos.length;

    // Los puntos vienen en píxeles del cuadro. Se normalizan dividiendo x por
    // el ancho e y por el alto, que es la misma convención en la que están
    // escritos los umbrales de colocación y el de parpadeo.
    const W = v.videoWidth, H = v.videoHeight;
    const p = (i) => [pos[i][0] / W, pos[i][1] / H];
    const izq = p(33), der = p(263);
    const centro = [(izq[0] + der[0]) / 2, (izq[1] + der[1]) / 2];
    const distOjos = Math.hypot(der[0] - izq[0], der[1] - izq[1]);

    return {
      centro,
      distOjos,
      ear: (razonOjo(p, OJO_IZQ) + razonOjo(p, OJO_DER)) / 2,
      rasgos: null,
      cruda: this._ultima,
    };
  }

  /** Registra que la mirada está, ahora, en `objetivo`.
   *
   * WebGazer no recibe los rasgos: usa los últimos que calculó su propio lazo
   * (`latestEyeFeatures`). Si todavía no calculó ninguno -la cara entró recién
   * al cuadro- `recordScreenPosition` no hace nada y lo hace en silencio, así
   * que acá se confirma mirando el contador real del regresor.
   */
  grabar(_datos, objetivo, grupo) {
    const reg = this.wg?.getRegression?.()?.[0];
    if (!reg) return false;
    const antes = reg.eyeFeaturesClicks.length;
    this.wg.recordScreenPosition(objetivo[0], objetivo[1], "click");
    const grabo = reg.eyeFeaturesClicks.length > antes;
    if (grabo) this._grupos.push(grupo);
    return grabo;
  }

  async limpiarDatos() {
    this._grupos = [];
    this._ultima = null;
    try { await this.wg?.clearData?.(); } catch (_) {}
  }

  /** WebGazer entrena en línea: no hay nada que ajustar al cerrar.
   *
   * Lo que sí se hace acá es el mismo diagnóstico de colapso que el estimador
   * propio, para que la puerta de calidad sirva para los dos. Se le pide al
   * regresor que prediga sobre la primera muestra guardada de cada punto y se
   * compara cuánto se mueven las predicciones contra cuánto se mueven los
   * objetivos. Cerca de 1 sigue la mirada; cerca de 0 colapsó.
   *
   * Pedirle 9 predicciones no es gratis -WebGazer reajusta la cresta entera en
   * cada `predict()`- pero es una sola vez y son 9.
   */
  async ajustar() {
    const reg = this.wg?.getRegression?.()?.[0];
    const guardadas = reg?.dataClicks?.data ?? [];
    if (guardadas.length < 18) {
      throw new FalloEstimador(
        "La calibración no juntó suficientes muestras.",
        `WebGazer registró ${guardadas.length} de las `
        + `${this.muestrasPorPunto * 9} esperadas. Suele ser falta de luz o `
        + `la cara fuera de cuadro.`);
    }

    // Primera muestra de cada punto. `_grupos` lleva el mismo orden que
    // `dataClicks` porque solo se empuja cuando el contador del regresor
    // confirmó la grabación.
    const pred = [], obj = [];
    const vistos = new Set();
    for (let i = 0; i < Math.min(this._grupos.length, guardadas.length); i++) {
      const g = this._grupos[i];
      if (vistos.has(g)) continue;
      vistos.add(g);
      const q = reg.predict(guardadas[i].eyes);
      if (!q || !Number.isFinite(q.x) || !Number.isFinite(q.y)) continue;
      pred.push([q.x, q.y]);
      obj.push(guardadas[i].screenPos);
    }

    const recorrido = recorridoDe(pred, obj);
    return {
      recorrido,
      errorCV: NaN,
      alfa: reg.ridgeParameter,
      nota: `${guardadas.length} muestras, cresta ${reg.ridgeParameter}`
          + " (WebGazer entrena en línea; no hay validación cruzada)",
    };
  }

  detener() {
    try { this.wg?.end?.(); } catch (_) {}
  }
}

/** Una promesa con plazo. Sin esto, un `begin()` que nunca resuelve deja la
 *  interfaz esperando para siempre y sin ningún mensaje. */
function conTiempo(promesa, ms) {
  return Promise.race([
    promesa,
    new Promise((_, falla) => setTimeout(() => falla(new FalloEstimador(
      "WebGazer no terminó de arrancar.",
      `Pasaron ${Math.round(ms / 1000)} s sin que la cámara entregara `
      + `cuadros. Pruebe recargando, o use el estimador propio.`)), ms)),
  ]);
}

function razonOjo(p, o) {
  const a = p(o.arriba), b = p(o.abajo);
  const i = p(o.interno), e = p(o.externo);
  const alto = Math.hypot(a[0] - b[0], a[1] - b[1]);
  const ancho = Math.hypot(e[0] - i[0], e[1] - i[1]);
  return ancho > 1e-9 ? alto / ancho : 0;
}

/* WebGazer se publica como un bundle UMD de 1,9 MB que declara `var webgazer`
 * en el tope del archivo, así que cargado con un <script> clásico queda en
 * `window`. No se importa como módulo ES a propósito: el campo `module` del
 * paquete apunta al fuente sin empaquetar, que arrastra TensorFlow.js entero
 * por el transformador del CDN. */
function cargarScript(url) {
  return new Promise((listo, falla) => {
    if (document.querySelector(`script[src="${url}"]`)) return listo();
    const s = document.createElement("script");
    s.src = url;
    s.async = true;
    s.onload = () => listo();
    s.onerror = () => falla(new FalloEstimador(
      "No se pudo descargar WebGazer.",
      `Falló la descarga de ${url}. Revise la conexión.`));
    document.head.appendChild(s);
  });
}
