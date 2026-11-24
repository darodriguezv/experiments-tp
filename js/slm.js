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
 * llamada para que el banco de pruebas mida ESTE número.
 *
 * 0.6, medido con Qwen3-4B (docs/expansion_semantica.md): acierto y seguridad
 * iguales a 0.3 y a 0.9, pero cubre mejor las lecturas de una entrada ambigua
 * que 0.9 (+6 puntos) y repite más las mismas opciones para la misma entrada
 * (constancia 0.79 contra 0.68), que es lo que permite aprender dónde aparece
 * lo que uno busca. La 0.9 anterior nunca se había justificado. */
export const TEMPERATURA = 0.6;
export const MAX_TOKENS = 160;

export const MODELO_WEB = "Qwen3-4B-q4f16_1-MLC";
/* Elegido con el banco de la página (docs/expansion_semantica.md), con un
 * criterio fijado antes de ver los resultados: entregable (Apache-2.0), apto
 * para poca memoria según WebLLM, no menos seguro que el anterior, y el de
 * mayor acierto sin contar el literal. Contra Qwen2.5-1.5B, el de antes, con el
 * mismo prompt: +28 puntos de acierto del modelo y la mitad de respuestas
 * inseguras. Lo que cuesta: 2.26 GB de descarga y unos 4.7 s por expansión en
 * una notebook con RTX 4060, contra unos 3 s. */

/* Dos modelos, según el equipo.
 *
 * Qwen3-4B es el elegido por el banco, que corrió sobre una GPU dedicada. En la
 * gráfica integrada de la misma notebook, Chrome procesa el prompt a 13
 * tokens/s con Qwen3-4B: unos 55 s por expansión, la mayoría solo en leer las
 * ~500 fichas del prompt. Qwen3-1.7B tarda unos 20 s ahí, con 68 % de acierto
 * contra 76 % y la misma seguridad. Se recomienda uno u otro según la GPU que
 * WebGPU entrega, y la persona puede cambiarlo. */
export const MODELOS_WEB = {
  completo: { id: "Qwen3-4B-q4f16_1-MLC", nombre: "Completo (Qwen3-4B)",
              detalle: "Las mejores frases. Necesita una GPU dedicada: con "
                     + "gráficos integrados tarda cerca de un minuto por frase." },
  liviano: { id: "Qwen3-1.7B-q4f16_1-MLC", nombre: "Liviano (Qwen3-1.7B)",
             detalle: "Algo menos acertado e igual de seguro. Más rápido en "
                    + "equipos sin GPU dedicada." },
};

/** Qué modelo conviene en este equipo. Con una GPU dedicada de NVIDIA o con un
 *  chip de Apple, el completo; con cualquier otra cosa (casi siempre gráficos
 *  integrados), el liviano. Es una heurística sobre el fabricante que informa
 *  WebGPU, no una medición. */
export async function modeloRecomendado() {
  try {
    const a = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
    const v = String(a?.info?.vendor ?? "").toLowerCase();
    return v === "nvidia" || v === "apple" ? "completo" : "liviano";
  } catch (_) {
    return "liviano";
  }
}

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
      // El primer pedido real compila los shaders de WebGPU, y medido en la
      // página con Qwen3-4B tardó 53 s. Se hace acá, antes de decir que el
      // modelo está listo, para que no lo pague la persona en su primer
      // HABLAR. Tiene que ser con el prompt de verdad: WebLLM compila según el
      // largo de lo que procesa, y un calentamiento con "hola" -que se probó-
      // dejaba el primer HABLAR igual de lento.
      alProgresar?.("Preparando el modelo", 1);
      await this.motor.chat.completions.create({
        messages: [
          { role: "system", content: this.sistema },
          { role: "user", content: construirUsuario("AGUA") },
        ],
        temperature: this.temperatura,
        max_tokens: 8,
        extra_body: { enable_thinking: false },
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
