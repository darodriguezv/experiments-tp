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
import { elegirCresta, elegirCrestaCentrada, ajustarCentrado,
         predecirCentrado } from "./cresta.js";

/* Dieciséis lugares en vez de nueve, en serpentina para que cada salto sea
 * corto. Con nueve, el modelo no tenía con qué orientarse entre punto y
 * punto: el desvío medido cambiaba de signo entre vecinos. No coinciden con
 * los de la prueba de precisión (0.25/0.5/0.75) ni con los de la corrección
 * (0.2/0.5/0.8), así que la prueba sigue midiendo lugares no vistos. */
const F4 = [0.1, 0.37, 0.63, 0.9];
export const PUNTOS_16 = F4.flatMap((fy, fila) =>
  (fila % 2 ? [...F4].reverse() : F4).map((fx) => [fx, fy]));

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
    // Regresión propia: ver `_tomarRegresion`. Hasta que no se confirme que
    // WebGazer expone lo necesario, todo queda como de fábrica.
    this._propia = false;
    this.puntos = null;             // null = los nueve de siempre
    this._X = []; this._yx = []; this._yy = [];
    this._modelo = null;
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

    this._tomarRegresion();

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
    if (this._propia) {
      const n = this._X.length;
      this.wg.recordScreenPosition(objetivo[0], objetivo[1], "click");
      if (this._X.length <= n) return false;
      this._grupos.push(grupo);
      return true;
    }
    const antes = reg.eyeFeaturesClicks.length;
    this.wg.recordScreenPosition(objetivo[0], objetivo[1], "click");
    const grabo = reg.eyeFeaturesClicks.length > antes;
    if (grabo) this._grupos.push(grupo);
    return grabo;
  }

  async limpiarDatos() {
    this._grupos = [];
    this._ultima = null;
    this._X = []; this._yx = []; this._yy = [];
    this._modelo = null;
    try { await this.wg?.clearData?.(); } catch (_) {}
    // `clearData` reinicia el regresor. Si al hacerlo lo cambiara por otro
    // objeto, los reemplazos se perderían en silencio y la calibración
    // siguiente no grabaría nada: se vuelven a poner siempre.
    if (this._propia) this._tomarRegresion();
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
    if (this._propia) return this._ajustarPropia();
    const reg = this.wg?.getRegression?.()?.[0];
    const guardadas = reg?.dataClicks?.data ?? [];
    if (guardadas.length < 18) {
      throw new FalloEstimador(
        "La calibración no juntó suficientes muestras.",
        `WebGazer registró ${guardadas.length} de las `
        + `${this.muestrasPorPunto * 9} esperadas. Suele ser falta de luz o `
        + `la cara fuera de cuadro.`);
    }

    /* Elegir la cresta ANTES de medir el recorrido: si se midiera primero, el
     * diagnóstico describiría un modelo que ya no es el que se va a usar.
     *
     * `_grupos` tiene que venir alineado con los rasgos guardados. Lo está
     * mientras no se desborde la ventana de 50, que es justo lo que la prueba
     * `estimador.test.mjs` fija. Si por lo que fuera no coincidieran, se
     * prefiere no validar antes que validar contra etiquetas corridas.
     */
    const X = reg.eyeFeaturesClicks?.data ?? [];
    const deFabrica = reg.ridgeParameter;
    let cv = null;
    if (X.length === this._grupos.length) {
      await new Promise((r) => setTimeout(r, 0));   // que pinte el cuadro
      cv = elegirCresta(X,
        reg.screenXClicksArray.data.map((v) => v[0]),
        reg.screenYClicksArray.data.map((v) => v[0]),
        this._grupos);
    }
    if (cv) {
      reg.ridgeParameter = cv.cresta;
      console.log("[cresta]", cv.tabla.map((f) =>
        `${f.relativo}: ${f.error.toFixed(0)} px`).join("  "));
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
      errorCV: cv ? cv.error : NaN,
      alfa: reg.ridgeParameter,
      nota: cv
        ? `${guardadas.length} muestras, cresta ${cv.cresta.toPrecision(3)} `
          + `elegida dejando un punto afuera (error ${cv.error.toFixed(0)} px; `
          + `sin regularizar, como viene de fábrica, ${cv.mejoraSobre.toFixed(0)} px)`
        : `${guardadas.length} muestras, cresta ${deFabrica} de fábrica `
          + `(no se pudo validar)`,
    };
  }

  /** Reemplaza la regresión de WebGazer por la de este proyecto, y se queda
   *  con lo que WebGazer hace bien: seguir la cara y recortar los ojos.
   *
   * ## Por qué
   *
   * Medido sin el artefacto del filtro, WebGazer erra 105 px y el error no es
   * parejo: en una misma fila pasa de +98 px a -155 px entre puntos vecinos.
   * Su regresión tiene tres límites que desde afuera no se pueden mover:
   * guarda 50 muestras como máximo, no tiene intercepto, y se reajusta entera
   * en cada cuadro. Elegirle la cresta por validación cruzada no cambió nada
   * (109 px contra 109 px): con 45 muestras en nueve lugares no hay de dónde.
   *
   * ## Cómo
   *
   * El lazo de WebGazer llama a `addData` para entrenar y a `predict` para
   * predecir, sobre este mismo objeto. Se reemplazan las dos:
   *
   * - `addData` guarda los rasgos -los mismos 120 valores que usa WebGazer,
   *   calculados con su propia función- en arreglos sin tope;
   * - `predict` aplica un modelo ya ajustado: un producto y una suma por
   *   cuadro, en vez de resolver un sistema de 120x120 cada vez.
   *
   * Si WebGazer no expone lo necesario, no se toca nada y queda su regresión
   * con sus 45 muestras: `this._propia` dice cuál de las dos corre, y el
   * informe lo muestra.
   *
   * @returns {boolean} si quedó la regresión propia
   */
  _tomarRegresion() {
    const reg = this.wg?.getRegression?.()?.[0];
    const util = this.wg?.util;
    if (!reg || typeof util?.getEyeFeats !== "function") return false;

    const rasgosDe = (eyes) => {
      try {
        const f = util.getEyeFeats(eyes);
        return f && f.length ? f : null;
      } catch (_) { return null; }
    };
    reg.addData = (eyes, pos, tipo) => {
      if (!eyes || tipo !== "click") return;
      const f = rasgosDe(eyes);
      if (!f) return;
      this._X.push(Float64Array.from(f));
      this._yx.push(pos[0]);
      this._yy.push(pos[1]);
    };
    reg.predict = (eyes) => {
      const m = this._modelo;
      if (!eyes || !m) return null;
      const f = rasgosDe(eyes);
      if (!f || f.length !== m.cx.length) return null;
      const [x, y] = predecirCentrado(m, f);
      return { x, y };
    };

    this._propia = true;
    this.puntos = PUNTOS_16;
    this.muestrasPorPunto = 12;
    this.msEntreMuestras = 110;
    this.etiqueta = "WebGazer.js 3.5.3 (parches de ojo) + regresión propia";
    return true;
  }

  /** Cierra la calibración cuando la regresión es la propia. */
  async _ajustarPropia() {
    const n = this._X.length;
    const lugares = new Set(this._grupos).size;
    if (n !== this._grupos.length || lugares < 6 || n < lugares * 3) {
      throw new FalloEstimador(
        "La calibración no juntó suficientes muestras.",
        `Quedaron ${n} muestras en ${lugares} puntos de `
        + `${this.puntos.length}. Suele ser falta de luz o la cara fuera de `
        + `cuadro.`);
    }
    await new Promise((r) => setTimeout(r, 0));   // que pinte el cuadro

    const cv = elegirCrestaCentrada(this._X, this._yx, this._yy, this._grupos);
    const modelo = cv
      ? ajustarCentrado(this._X, this._yx, this._yy, cv.cresta) : null;
    if (!modelo) {
      throw new FalloEstimador(
        "No se pudo ajustar el modelo de mirada.",
        "La regresión no encontró solución con las muestras grabadas.");
    }
    this._modelo = modelo;
    console.log("[cresta]", cv.tabla.map((f) =>
      `${f.relativo}: ${f.error.toFixed(0)} px`).join("  "));

    // El mismo diagnóstico de colapso que el otro estimador: cuánto se mueven
    // las predicciones contra cuánto se mueven los objetivos.
    const pred = [], obj = [];
    const vistos = new Set();
    for (let i = 0; i < n; i++) {
      const g = this._grupos[i];
      if (vistos.has(g)) continue;
      vistos.add(g);
      pred.push(predecirCentrado(modelo, this._X[i]));
      obj.push([this._yx[i], this._yy[i]]);
    }
    return {
      recorrido: recorridoDe(pred, obj),
      errorCV: cv.error,
      alfa: cv.cresta,
      nota: `${n} muestras en ${lugares} puntos, regresión propia con `
          + `intercepto, cresta relativa ${cv.relativo} elegida dejando un `
          + `punto afuera (error ${cv.error.toFixed(0)} px en puntos no vistos)`
          + (cv.enBorde ? "; el óptimo cayó en el borde de la escalera" : ""),
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
