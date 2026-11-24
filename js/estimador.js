/* Dos estimadores de mirada detrás de una sola interfaz.
 *
 * Hasta acá el estimador estaba cableado dentro de `app.js`: la cámara, el
 * detector facial, los rasgos y la regresión aparecían mezclados con las fases
 * de la interfaz. Eso impedía la única medición que importa ahora, que es
 * comparar el estimador propio contra WebGazer **con todo lo demás idéntico**:
 * la misma calibración de nueve puntos, el mismo suavizado, la misma
 * corrección de deriva y la misma prueba de precisión sobre los mismos nueve
 * puntos de control.
 *
 * Si cambiaran dos cosas a la vez -el estimador y el filtro, por ejemplo- la
 * diferencia medida no sería atribuible a ninguna de las dos.
 *
 * ## La interfaz
 *
 * ```
 * async iniciar()            arranca cámara y detector; tira FalloEstimador
 * leer(t) -> datos | null    un cuadro: null si no hay cara
 * grabar(datos, objetivo, g) registra "ahora mira este punto"; -> bool
 * async ajustar()            cierra la calibración -> diagnóstico
 * limpiarDatos()             borra lo calibrado
 * ```
 *
 * `datos` es siempre:
 *
 * ```
 * { centro: [fx, fy],   // punto medio entre ojos, normalizado al cuadro
 *   distOjos: number,   // separación de los ojos, normalizada
 *   ear: number,        // apertura del párpado; cae al parpadear
 *   cruda: [x, y]|null, // predicción en píxeles, SIN suavizar ni corregir
 *   rasgos: ...|null }  // solo el estimador propio lo usa
 * ```
 *
 * El diagnóstico que devuelve `ajustar()` también es común, y es lo que mira
 * la puerta de calidad:
 *
 * ```
 * { recorrido, errorCV, alfa, nota }
 * ```
 */

import { extraer } from "./rasgos.js";
import { ModeloMirada, entrenarConCV } from "./modelo.js";

const MP = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";
const MODELO_CARA =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/" +
  "face_landmarker/float16/1/face_landmarker.task";

/** Un fallo con un mensaje para mostrar y un detalle técnico aparte. */
export class FalloEstimador extends Error {
  constructor(msg, detalle = "") {
    super(msg);
    this.detalle = detalle;
  }
}

/* --------------------------------------------------------------- propio */

export class EstimadorPropio {
  constructor() {
    this.nombre = "propio";
    this.etiqueta = "Propio — MediaPipe + regresión de cresta";
    /* Treinta muestras por punto. Son casi idénticas entre sí dentro de una
     * fijación, pero la mediana de treinta aguanta un cuadro malo y el costo
     * es cero: el ajuste se hace en la forma dual, sobre una matriz de
     * 270x270 y no de 486x486. */
    this.muestrasPorPunto = 30;
    this.msEntreMuestras = 0;
    this.video = null;
    this.landmarker = null;
    this.modelo = null;
    this.resolucion = "";
    this.nPuntosCara = 0;
    this._ultimoTs = -1;
    this._cal = { muestras: [], objetivos: [], grupos: [] };
  }

  async iniciar() {
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 },
                 facingMode: "user" },
        audio: false,
      });
    } catch (e) {
      throw new FalloEstimador(
        "No se pudo usar la cámara.",
        "Hace falta permiso de cámara y una conexión segura (https). " +
        String(e?.message || e));
    }
    this.stream = stream;
    this.video = document.querySelector("#video");
    this.video.srcObject = stream;
    await this.video.play();

    // Si el elemento no tiene dimensiones reales, el detector recibe una
    // imagen inservible y nada de lo que sigue significa algo. Vale la pena
    // comprobarlo y decirlo, en vez de dejar que falle silenciosamente.
    await new Promise((r) => {
      if (this.video.videoWidth) return r();
      this.video.addEventListener("loadedmetadata", r, { once: true });
      setTimeout(r, 3000);
    });
    this.resolucion = `${this.video.videoWidth}x${this.video.videoHeight}`;
    if (this.video.videoWidth < 320) {
      throw new FalloEstimador(
        "La cámara entrega una imagen demasiado chica.",
        `${this.resolucion}. Se necesita al menos 320 px de ancho para `
        + `ubicar los puntos del iris.`);
    }

    try {
      const vision = await import(MP + "/+esm");
      const archivos = await vision.FilesetResolver.forVisionTasks(MP + "/wasm");
      this.landmarker = await vision.FaceLandmarker.createFromOptions(archivos, {
        baseOptions: { modelAssetPath: MODELO_CARA, delegate: "GPU" },
        // VIDEO y no IMAGE: el modo video mantiene estado entre cuadros y da
        // puntos mucho más estables, que es de donde sale la mitad del temblor.
        runningMode: "VIDEO",
        numFaces: 1,
        outputFaceBlendshapes: false,
        outputFacialTransformationMatrixes: false,
      });
    } catch (e) {
      throw new FalloEstimador("No se pudo cargar el detector facial.",
                               String(e?.message || e));
    }
  }

  leer(_t) {
    const v = this.video;
    if (!v || v.readyState < 2) return null;
    let lm = null;
    try {
      // MediaPipe exige marcas de tiempo estrictamente crecientes; dos cuadros
      // en el mismo milisegundo lo hacen tirar excepción y cortar el lazo.
      let ts = Math.round(performance.now());
      if (ts <= this._ultimoTs) ts = this._ultimoTs + 1;
      this._ultimoTs = ts;
      lm = this.landmarker.detectForVideo(v, ts)?.faceLandmarks?.[0] ?? null;
    } catch (_) { return null; }   // un cuadro perdido no corta nada
    if (!lm) return null;
    this.nPuntosCara = lm.length;

    const d = extraer(lm);
    if (!d) return null;
    const listo = this.modelo?.listo;
    return {
      centro: d.centro,
      distOjos: d.distOjos,
      ear: d.ear,
      rasgos: d.rasgos,
      cruda: listo ? this.modelo.predecir(d.rasgos) : null,
    };
  }

  grabar(datos, objetivo, grupo) {
    if (!datos?.rasgos) return false;
    this._cal.muestras.push(datos.rasgos);
    this._cal.objetivos.push(objetivo);
    this._cal.grupos.push(grupo);
    return true;
  }

  limpiarDatos() {
    this._cal = { muestras: [], objetivos: [], grupos: [] };
    this.modelo = null;
  }

  async ajustar() {
    const { muestras, objetivos, grupos } = this._cal;
    if (muestras.length < 36) {
      throw new FalloEstimador(
        "La calibración no juntó suficientes muestras.",
        `Se grabaron ${muestras.length}. Suele ser falta de luz o la cara `
        + `fuera de cuadro. Pruebe de nuevo con la cara bien iluminada.`);
    }
    try {
      // Alpha lo elige la propia calibración, por validación cruzada dejando
      // fuera un punto entero. El valor fijo que había antes (5000) colapsaba
      // el modelo: predecía casi el centro de la pantalla para cualquier
      // entrada, con 354 px de sesgo y 40 px de temblor.
      this.modelo = entrenarConCV(muestras, objetivos, grupos);
    } catch (e) {
      throw new FalloEstimador("No se pudo ajustar el modelo.",
                               String(e?.message || e));
    }
    return {
      recorrido: this.modelo.recorrido,
      errorCV: this.modelo.errorCV,
      alfa: this.modelo.alfaElegido,
      nota: `alpha ${this.modelo.alfaElegido}, `
          + `LOO ${Math.round(this.modelo.errorCV)} px`,
    };
  }

  /** Error de cada punto de calibración, dejándolo fuera del ajuste.
   *
   * Casi siempre hay uno o dos puntos que arruinan el ajuste -un parpadeo
   * largo, haber mirado fuera de la pantalla- y saber cuál permite repetir
   * prestando atención ahí en vez de repetir a ciegas.
   */
  erroresPorPunto() {
    const c = this._cal;
    if (!this.modelo) return [];
    const puntos = [...new Set(c.grupos)];
    return puntos.map((fuera) => {
      const dentro = [], afuera = [];
      c.grupos.forEach((g, i) => (g === fuera ? afuera : dentro).push(i));
      if (dentro.length < 4 || !afuera.length) return NaN;
      try {
        const m = new ModeloMirada().entrenar(
          dentro.map((i) => c.muestras[i]), dentro.map((i) => c.objetivos[i]),
          this.modelo.alfaElegido);
        let s = 0;
        for (const i of afuera) {
          const [px, py] = m.predecir(c.muestras[i]);
          s += Math.hypot(px - c.objetivos[i][0], py - c.objetivos[i][1]);
        }
        return s / afuera.length;
      } catch (_) {
        return NaN;
      }
    });
  }

  detener() {
    try { this.stream?.getTracks().forEach((t) => t.stop()); } catch (_) {}
  }
}

/* ------------------------------------------------------------ selección */

/** Devuelve el estimador pedido. Carga el módulo de WebGazer solo si hace falta. */
export async function crearEstimador(nombre) {
  if (nombre === "webgazer") {
    const m = await import("./webgazer_est.js");
    return new m.EstimadorWebGazer();
  }
  return new EstimadorPropio();
}

export const ESTIMADORES = [
  { clave: "propio", nombre: "Estimador propio",
    detalle: "MediaPipe con 478 puntos faciales y regresión de cresta sobre "
           + "486 rasgos. Es el que se midió del lado de Python." },
  { clave: "webgazer", nombre: "WebGazer.js",
    detalle: "Biblioteca de la Universidad Brown, usada en cientos de estudios. "
           + "Mira los parches de los ojos en vez de la geometría facial. "
           + "Descarga unos 8 MB la primera vez." },
];
