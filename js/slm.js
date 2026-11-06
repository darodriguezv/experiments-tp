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

const MODELO_WEB = "Qwen2.5-1.5B-Instruct-q4f16_1-MLC";
/* El 1.5b y no el 7b: el 7b en el navegador son ~4.5 GB de descarga y bastante
 * más VRAM de la que tiene una notebook común. Sobre el corpus el 1.5b saca
 * 62% de recall contra 77% del 7b, así que esta página muestra una versión
 * MÁS DÉBIL que la medida. Se dice en pantalla, no se disimula. */

export class SLM {
  constructor(grabados) {
    this.grabados = grabados || {};
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

  /** Baja y arranca el modelo. Devuelve true si quedó listo. */
  async cargar(alProgresar) {
    if (this.motor || this.cargando) return !!this.motor;
    if (!this.soportado) {
      this.error = "Este navegador no tiene WebGPU.";
      return false;
    }
    this.cargando = true;
    try {
      const webllm = await import(
        "https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.79/+esm");
      this.motor = await webllm.CreateMLCEngine(MODELO_WEB, {
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

  /**
   * @param {string} telegrama  lo que se armó en el tablero
   * @returns {Promise<{candidatos: string[], modo: string, ms: number}>}
   */
  async expandir(telegrama) {
    const t = telegrama.trim();
    if (!t) return { candidatos: [], modo: this.modo, ms: 0 };

    if (this.motor) {
      const t0 = performance.now();
      try {
        const r = await this.motor.chat.completions.create({
          messages: [
            { role: "system", content: PROMPT_SISTEMA },
            { role: "user", content: construirUsuario(t) },
          ],
          temperature: 0.9,
          max_tokens: 160,
        });
        const texto = r.choices?.[0]?.message?.content ?? "";
        return {
          candidatos: conLiteral(parsear(texto), t),
          modo: "vivo",
          ms: Math.round(performance.now() - t0),
        };
      } catch (e) {
        this.error = String(e?.message || e);
        // Cae al modo grabado en vez de dejar a la persona sin candidatos.
      }
    }

    const g = this.grabados[t.toUpperCase()];
    return {
      candidatos: conLiteral(g ? g.c.slice() : [], t),
      modo: g ? "grabado" : "literal",
      ms: g ? g.ms : 0,
    };
  }
}

/* La salida de escape del diseño: el literal SIEMPRE disponible, lo haya
 * propuesto el modelo o no. Si las tres propuestas están mal, la persona tiene
 * que poder decir exactamente lo que escribió - el tablero pelado ya podía, y
 * este módulo no debería quitárselo. */
function conLiteral(lista, telegrama) {
  const lit = telegrama.toLowerCase();
  const fuera = lista.filter((c) => c && c.toLowerCase() !== lit).slice(0, 3);
  return [...fuera, lit];
}

/* Tolera las formas en que un modelo chico responde: JSON con "candidatos",
 * un array pelado, o líneas con viñetas. Un modelo que acierta pero contesta
 * en prosa es un fallo de formato, no de la tarea, y tratarlos igual
 * confundiría las dos cosas. */
function parsear(texto) {
  const t = (texto || "").trim();
  if (!t) return [];
  try {
    const bloque = t.match(/\{[\s\S]*\}|\[[\s\S]*\]/);
    if (bloque) {
      const j = JSON.parse(bloque[0]);
      const arr = Array.isArray(j) ? j
        : (j.candidatos || j.candidates || j.opciones || []);
      if (Array.isArray(arr) && arr.length) {
        return arr.map((s) => String(s).trim()).filter(Boolean).slice(0, 3);
      }
    }
  } catch (_) { /* no era JSON: se sigue por líneas */ }

  return t.split(/\r?\n/)
    .map((l) => l.replace(/^[\s\-*•\d.)"']+/, "").replace(/["',]+$/, "").trim())
    .filter((l) => l && l.length < 90)
    .slice(0, 3);
}
