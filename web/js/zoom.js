/* Selección en dos tiempos: acercar donde cae la mirada, después elegir.
 *
 * ## El problema que resuelve
 *
 * Medido con el error real de la mirada, una celda de 384x185 px -la del
 * tablero de 5x5- se acierta el **60%** de las veces. Contando que cada fallo
 * cuesta un BORRAR y un reintento, son 2.79 segundos por palabra acertada, y
 * para una frase de diez selecciones queda 0.6% de probabilidad de escribirla
 * sin un error.
 *
 * ## Por qué acercar funciona, y por qué es barato
 *
 * El error de la mirada es aproximadamente constante **en píxeles de
 * pantalla**. Si la celda crece k veces, el error pasa a ser una fracción k
 * veces menor de la celda. Medido sobre las sesiones reales:
 *
 *     sin acercar  384x185   60%
 *     2.0x         768x369   89%
 *     2.5x         960x462   94%
 *     3.0x        1152x554   97%
 *
 * Y lo que lo vuelve barato es que **acercar no puede fallar**. El zoom se
 * centra en donde cayó la mirada, no elige una región de una lista: si la
 * estimación está 100 px corrida, el zoom queda 100 px corrido y la celda
 * buscada sigue en pantalla, más grande. Una SELECCIÓN equivocada cuesta
 * borrar y rehacer; un acercamiento equivocado solo cuesta volver a mirar.
 *
 * Por eso el tiempo de permanencia para acercar es corto y el de elegir no.
 *
 * ## Un salto, no varios
 *
 * Pasar de 5x5 a 4x4 y después a 3x3 suena más gradual, pero cada paso cuesta
 * su propia permanencia y 5x5 -> 4x4 es apenas 1.25x, que deja el acierto
 * cerca del 70%. Un salto de 2.5x llega a 94% de una. `PASOS` queda
 * configurable para poder medirlo, pero el valor por defecto es uno.
 */

export const FACTOR = 2.5;
export const PASOS = 1;
export const DWELL_ACERCAR_MS = 450;
/* Corto a propósito: equivocarse acercando no cuesta casi nada. Ver arriba. */

export const MS_SIN_ELEGIR = 4000;
/* Si en el estado acercado no se elige nada, se vuelve solo. Sin esto, alguien
 * que se acercó a la zona equivocada queda atrapado mirando celdas que no
 * quiere, y la única salida sería adivinar cuál es la de volver. */

export class Zoom {
  constructor(opciones = {}) {
    this.factor = opciones.factor ?? FACTOR;
    this.pasos = opciones.pasos ?? PASOS;
    this.nivel = 0;
    this.origen = null;      // [x, y] en píxeles de viewport
    this.desde = 0;
  }

  get acercado() { return this.nivel > 0; }

  /** La escala acumulada que hay que aplicar al contenedor. */
  get escala() { return Math.pow(this.factor, this.nivel); }

  /**
   * Acerca centrado en un punto. El origen de la transformación ES ese punto,
   * así que lo que la persona está mirando no se mueve: crece alrededor.
   * Eso es lo que mantiene la conexión espacial; un salto que desplaza el
   * contenido obliga a buscar de nuevo dónde quedó todo.
   */
  acercar(x, y, t) {
    if (this.nivel >= this.pasos) return false;
    // En niveles sucesivos el origen se compone con el anterior, porque el
    // punto que llega ya está en coordenadas de la vista acercada.
    this.origen = this.nivel === 0 ? [x, y] : this.origen;
    this.nivel += 1;
    this.desde = t;
    return true;
  }

  alejar() {
    this.nivel = 0;
    this.origen = null;
    this.desde = 0;
  }

  /** ¿Pasó demasiado tiempo acercado sin elegir nada? */
  venció(t) {
    return this.acercado && this.desde && (t - this.desde) > MS_SIN_ELEGIR;
  }

  /** Lo que hay que escribir en el style del contenedor. */
  estilo() {
    if (!this.acercado || !this.origen) {
      return { transformOrigin: "50% 50%", transform: "none" };
    }
    return {
      transformOrigin: `${this.origen[0]}px ${this.origen[1]}px`,
      transform: `scale(${this.escala})`,
    };
  }
}

/**
 * Tasa de acierto esperada para una celda, dado el error medido.
 * Se usa para que la interfaz pueda decir cuánto está ganando el zoom en vez
 * de que haya que creerlo.
 * @param {number[]} dxs errores horizontales absolutos, en px
 * @param {number[]} dys errores verticales absolutos, en px
 */
export function aciertoEsperado(dxs, dys, ancho, alto) {
  if (!dxs.length) return NaN;
  let dentro = 0;
  for (let i = 0; i < dxs.length; i++) {
    if (dxs[i] < ancho / 2 && dys[i] < alto / 2) dentro += 1;
  }
  return dentro / dxs.length;
}
