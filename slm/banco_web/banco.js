/* El lado del navegador del banco de pruebas.
 *
 * Usa la clase `SLM` de la página, no una copia: mismo WebLLM, mismos pesos
 * cuantizados, misma temperatura, mismo prompt desplegado, mismo filtro. Lo
 * único que el banco cambia, cuando la corrida lo pide, es el modelo, la
 * temperatura o el texto del prompt, y lo hace con las opciones que la propia
 * clase expone para eso.
 *
 * De cada respuesta devuelve TODO: el texto crudo del modelo, lo que se parseó,
 * lo que vería la persona sin filtro (la página hasta ahora) y con filtro (la
 * página desde ahora). Así las métricas se pueden recalcular sin volver a
 * correr nada.
 */
import { SLM, MODELO_WEB, TEMPERATURA, MAX_TOKENS, WEBLLM } from "/web/js/slm.js";
import { PROMPT_SISTEMA } from "/web/js/prompt.js";
import { parsear, procesar, procesarSinFiltro } from "/web/js/expansion.js";

let slm = null;

window.banco = {
  desplegado: { modelo: MODELO_WEB, temperatura: TEMPERATURA,
                maxTokens: MAX_TOKENS, webllm: WEBLLM,
                prompt: PROMPT_SISTEMA },

  async cargar(modelo) {
    if (slm) await slm.descargar();
    slm = new SLM({}, { modelo });
    const t0 = performance.now();
    const listo = await slm.cargar((texto) => { window.__progreso = texto; });
    return { listo, error: slm.error, ms: Math.round(performance.now() - t0) };
  },

  /** @param {string|null} sistema  null = el prompt desplegado */
  async expandir(entrada, { temperatura, sistema }) {
    slm.temperatura = temperatura;
    slm.sistema = sistema ?? PROMPT_SISTEMA;
    slm.error = "";
    const r = await slm.expandir(entrada);
    if (r.modo !== "vivo") {
      return { error: slm.error || `modo ${r.modo}`, modo: r.modo };
    }
    const crudo = r.crudo ?? "";
    return {
      crudo,
      // No se llama `modelo`: ese es el nombre del modelo en el registro que
      // guarda correr.mjs, y una primera versión lo pisaba con esta lista.
      propuestos: parsear(crudo),
      sinFiltro: procesarSinFiltro(entrada, crudo).candidatos,
      conFiltro: procesar(entrada, crudo).candidatos,
      paginaDevolvio: r.candidatos,
      descartados: r.descartados ?? [],
      ms: r.ms,
      uso: r.uso,
    };
  },
};
window.__listo = true;
