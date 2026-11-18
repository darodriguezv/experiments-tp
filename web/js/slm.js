/* El modelo de lenguaje, corriendo en la máquina de quien abre la página.
 *
 * ## Por qué en el navegador y no en un servidor
 *
 * Mandar lo que la persona quiere decir a un servidor contradice el argumento
 * central del aparato: que funciona sin internet y sin que nada salga del
 * cuarto. El contenido es médico y lo dice alguien que no puede volver a
 * consentir con facilidad. WebLLM corre el modelo con WebGPU en la propia
 * máquina, así que la página se hostea pero la inferencia no.
 *
 * ## Los tres estados posibles, y ninguno miente
 *
 * 1. **WebGPU disponible y modelo cargado**: inferencia de verdad, en vivo.
 * 2. **WebGPU disponible pero el modelo no se bajó todavía**: la persona
 *    decide si quiere bajar ~1 GB. Hasta entonces, modo grabado.
 * 3. **Sin WebGPU** (Safari, Firefox, equipos viejos): modo grabado y se dice.
 *
 * El modo grabado usa las salidas REALES de `qwen2.5:7b` sobre las 133
 * entradas del corpus, guardadas de una corrida medida. No son ejemplos
 * inventados para la demostración, y la interfaz marca cuál de los dos modos
 * está activo en vez de dejarlo ambiguo.
 */

import { PROMPT_SISTEMA, construirUsuario } from "./prompt.js";
import { procesar, procesarSinFiltro, filtrar, conLiteral }
  from "./expansion.js";

/* La versión de WebLLM. Exportada para que el banco de pruebas cargue la misma
 * y no otra: dos versiones pueden muestrear distinto con los mismos pesos. */
export const WEBLLM = "https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.79/+esm";

/* La temperatura con la que corre la página. Está acá y no escrita en la
 * llamada para que el banco de pruebas mida ESTE número. */
export const TEMPERATURA = 0.9;
export const MAX_TOKENS = 160;

export const MODELO_WEB = "Qwen2.5-1.5B-Instruct-q4f16_1-MLC";
/* El 1.5b y no el 7b: el 7b en el navegador son ~4.5 GB de descarga y bastante
 * más VRAM de la que tiene una notebook común. Sobre el corpus el 1.5b saca
 * 62% de recall contra 77% del 7b, así que esta página muestra una versión
 * MÁS DÉBIL que la medida. Se dice en pantalla, no se disimula. */

const DESCARGA_GB = {
  "Qwen2.5-0.5B-Instruct-q4f16_1-MLC": 0.28,
  "Qwen2.5-1.5B-Instruct-q4f16_1-MLC": 0.87,
  "Qwen3-0.6B-q4f16_1-MLC": 0.34,
  "Qwen3-1.7B-q4f16_1-MLC": 0.97,
  "Qwen3-4B-q4f16_1-MLC": 2.26,
};

export class SLM {
  /**
   * @param {object} grabados  salidas reales guardadas, para el modo grabado
   * @param {object} [opciones]  solo para el banco de pruebas: la página usa
   *   los valores por defecto, que son los de arriba. Se pueden cambiar acá
   *   para medir alternativas con ESTA MISMA clase, en vez de con una copia.
   */
  constructor(grabados, opciones = {}) {
    this.grabados = grabados || {};
    this.modelo = opciones.modelo ?? MODELO_WEB;
    this.temperatura = opciones.temperatura ?? TEMPERATURA;
    this.sistema = opciones.sistema ?? PROMPT_SISTEMA;
    this.sinFiltro = !!opciones.sinFiltro;
    /** Quién usa el tablero, para el prompt. Vacío salvo en la condición
     *  contextualizada: la diferencia entre llenarlo y no llenarlo ES la
     *  variable independiente de esa comparación. */
    this.contexto = "";
    this.motor = null;
    this.cargando = false;
    this.progreso = 0;
    this.error = "";
  }

  get soportado() {
    return typeof navigator !== "undefined" && "gpu" in navigator;
  }

  get modo() {
    if (this.motor) return "vivo";
    if (this.cargando) return "cargando";
    return "grabado";
  }

  /** Cuánto pesa bajar el modelo, en GB, para decírselo a la persona antes.
   *  Suma de los pesos publicados en huggingface.co/mlc-ai. */
  get descargaGB() { return DESCARGA_GB[this.modelo] ?? null; }

  /** ¿El modelo ya está guardado en este navegador? Si lo está, cargarlo no
   *  baja nada y se puede hacer sin preguntar. */
  async enCache() {
    if (!this.soportado) return false;
    try {
      const webllm = await import(WEBLLM);
      return await webllm.hasModelInCache(this.modelo);
    } catch (_) {
      return false;
    }
  }

  /** Baja y arranca el modelo. Devuelve true si quedó listo. */
  async cargar(alProgresar) {
    if (this.motor || this.cargando) return !!this.motor;
    if (!this.soportado) {
      this.error = "Este navegador no tiene WebGPU.";
      return false;
    }
    this.cargando = true;
    try {
      const webllm = await import(WEBLLM);
      this.motor = await webllm.CreateMLCEngine(this.modelo, {
        initProgressCallback: (p) => {
          this.progreso = p.progress ?? 0;
          alProgresar?.(p.text ?? "", this.progreso);
        },
      });
      return true;
    } catch (e) {
      this.error = String(e?.message || e);
      this.motor = null;
      return false;
    } finally {
      this.cargando = false;
    }
  }

  /** Libera la GPU. La página no lo necesita; el banco sí, entre modelos. */
  async descargar() {
    try { await this.motor?.unload(); } catch (_) { /* ya no estaba */ }
    this.motor = null;
  }

  /**
   * @param {string} telegrama  lo que se armó en el tablero
   * @returns {Promise<{candidatos: string[], modo: string, ms: number,
   *   crudo?: string, descartados?: Array<[string,string]>, uso?: object}>}
   *   `crudo`, `descartados` y `uso` solo vienen en modo vivo: son para
   *   el banco de pruebas, la página no los mira.
   */
  async expandir(telegrama) {
    const t = telegrama.trim();
    if (!t) return { candidatos: [], modo: this.modo, ms: 0 };

    if (this.motor) {
      const t0 = performance.now();
      try {
        // El contexto va al final del system, nunca al user: el user es la
        // entrada telegráfica y nada más. Mezclarlos haría que el modelo trate
        // los datos de la persona como parte de lo que quiso decir, que es el
        // error que el filtro de invención después castiga.
        const system = this.contexto
          ? [this.sistema, "", "Sobre quien usa el tablero:",
             this.contexto].join("\n")
          : this.sistema;
        const r = await this.motor.chat.completions.create({
          messages: [
            { role: "system", content: system },
            { role: "user", content: construirUsuario(t) },
          ],
          temperature: this.temperatura,
          max_tokens: MAX_TOKENS,
          // Qwen3 razona antes de contestar si no se le dice que no: cientos
          // de tokens de más por pedido. Los demás modelos lo ignoran.
          extra_body: { enable_thinking: false },
        });
        const texto = r.choices?.[0]?.message?.content ?? "";
        // Parsear, FILTRAR y garantizar el literal: ver expansion.js. Antes
        // de este cambio la página saltaba el filtro.
        const p = this.sinFiltro ? procesarSinFiltro(t, texto) : procesar(t, texto);
        return {
          candidatos: p.candidatos,
          modo: "vivo",
          ms: Math.round(performance.now() - t0),
          crudo: texto,
          descartados: p.descartados,
          uso: r.usage ?? null,
        };
      } catch (e) {
        this.error = String(e?.message || e);
        // Cae al modo grabado en vez de dejar a la persona sin candidatos.
      }
    }

    const g = this.grabados[t.toUpperCase()];
    return {
      // Las salidas grabadas también pasan por el filtro: son respuestas
      // reales de un modelo y pueden traer los mismos vuelcos.
      candidatos: conLiteral(g ? filtrar(t, g.c).candidatos : [], t),
      modo: g ? "grabado" : "literal",
      ms: g ? g.ms : 0,
    };
  }
}
