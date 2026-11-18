/* Qué se corre. Cada celda: un modelo, una variante de prompt, una
 * temperatura, y cuántas veces se repite el corpus entero (k).
 *
 * Se repite porque la página muestrea a temperatura 0.9: la misma entrada da
 * candidatos distintos cada vez. Una sola pasada mide una tirada; lo que le
 * importa a la persona es qué tan seguido acierta, y qué tan seguido se
 * equivoca de una manera peligrosa.
 *
 * Los modelos son los que WebLLM 0.2.79 trae compilados, que es la versión que
 * usa la página: si un modelo no está ahí, la página no puede usarlo.
 */

const Q25_05 = "Qwen2.5-0.5B-Instruct-q4f16_1-MLC";
const Q25_15 = "Qwen2.5-1.5B-Instruct-q4f16_1-MLC";
const Q25_3 = "Qwen2.5-3B-Instruct-q4f16_1-MLC";
const Q3_06 = "Qwen3-0.6B-q4f16_1-MLC";
const Q3_17 = "Qwen3-1.7B-q4f16_1-MLC";
const Q3_4 = "Qwen3-4B-q4f16_1-MLC";
const PHI = "Phi-3.5-mini-instruct-q4f16_1-MLC";

const celda = (modelo, variante, temperatura, k, extra = {}) =>
  ({ modelo, variante, temperatura, k, ...extra });

export const PLANES = {
  /** Diez entradas, una vez: que el banco ande. */
  humo: [celda(Q25_15, "desplegado", 0.9, 1, { items: 10 })],

  /** Para medir cuánto rinde correr varias pestañas a la vez. No se analiza. */
  rinde: [celda(Q25_15, "v12-neutro", 0.9, 1, { items: 36 })],

  /** Todos los candidatos que la página puede cargar.
   *
   * Primero lo que decide: la página de hoy; el mismo modelo con el prompt en
   * español neutro, que separa el efecto del prompt del efecto del modelo; y
   * los dos Qwen3 entregables que entran en una notebook. Cinco pasadas del
   * corpus cada uno. Después los de contexto -el MIT, el techo no entregable y
   * los dos pisos-, con tres pasadas: alcanzan para ubicarlos, y cada pasada de
   * un modelo de 4B cuesta cerca de una hora en esta máquina. */
  cribado: [
    celda(Q25_15, "desplegado", 0.9, 5),
    celda(Q3_17, "v12-neutro", 0.9, 5),
    celda(Q25_15, "v12-neutro", 0.9, 5),
    celda(Q3_4, "v12-neutro", 0.9, 5),
    celda(PHI, "v12-neutro", 0.9, 3),
    celda(Q25_3, "v12-neutro", 0.9, 3),
    celda(Q3_06, "v12-neutro", 0.9, 3),
    celda(Q25_05, "v12-neutro", 0.9, 3),
  ],
};

/** La latencia que vería la persona: UN navegador, un pedido a la vez, una
 * entrada de cada dos. Correr con NAVEGADORES=1. En el cribado los pedidos
 * comparten la GPU y la latencia de cada uno no es la de la página. */
PLANES.latencia = [Q25_15, Q3_17, Q3_4, PHI, Q3_06, Q25_05, Q25_3].map(
  (m) => celda(m, "v12-neutro", 0.9, 1, { paso: 2 }));

/** Segunda fase, sobre los finalistas que dejó el cribado. Se eligen por
 * variable de entorno para no tocar este archivo después de ver resultados:
 *
 *   FINALISTAS=Q3_17,Q25_15 node correr.mjs temperatura
 *   FINALISTAS=Q3_17 node correr.mjs robustez
 *
 * - temperatura: 0.3 y 0.6 contra la 0.9 de la página, que nunca se justificó;
 * - robustez: la entrada como llega de un tablero de mirada. Dos pasadas por
 *   perturbación, contra las del cribado con la entrada limpia. */
const FINALISTAS = (process.env.FINALISTAS ?? "").split(",").filter(Boolean)
  .map((k) => ({ Q25_05, Q25_15, Q25_3, Q3_06, Q3_17, Q3_4, PHI })[k])
  .filter(Boolean);
PLANES.temperatura = FINALISTAS.flatMap((m) => [0.3, 0.6].map(
  (t) => celda(m, "v12-neutro", t, 3)));
PLANES.robustez = FINALISTAS.map((m) => celda(m, "v12-neutro", 0.9, 2, {
  perturbaciones: ["minusculas", "sin_acentos", "espacios", "palabra_doble",
                   "letra_vecina"] }));

export const MODELOS = { Q25_05, Q25_15, Q25_3, Q3_06, Q3_17, Q3_4, PHI };
