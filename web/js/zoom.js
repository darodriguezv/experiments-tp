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
export const DWELL_ACERCAR_MS = 800;
/* Antes eran 450 ms y el objetivo de esta permanencia era la grilla ENTERA, o
 * sea siempre el mismo. Como nunca cambiaba, nunca se reiniciaba: la cuenta
 * arrancaba cuando aparecía el tablero y disparaba 450 ms después, mirara la
 * persona donde mirara. Se veía como que acercaba solo, y acercaba a donde
 * estuviera la mirada en ese instante y no a donde la persona quería ir.
 *
 * Ahora el objetivo es la ZONA, así que mover la mirada reinicia la cuenta y
 * acercar exige haberse quedado quieto en algún lado. Con eso el número puede
 * ser más largo sin que moleste: lo que se pide es que la mirada se asiente,
 * no que espere. */

/* Al entrar al tablero la mirada viene del último punto de la prueba y puede
 * estar parada en cualquier zona. Sin esta pausa, el acercamiento dispara
 * antes de que la persona haya mirado el tablero. */
export const MS_ORIENTARSE = 600;

/* Cuántas zonas por lado tiene la rejilla gruesa que decide "la mirada se
 * asentó acá". Tres: con 90 px de error medido, una zona de medio millar de
 * píxeles se sostiene sin esfuerzo, y 3x3 alcanza para distinguir a qué parte
 * del tablero se quiere ir. Más finas y volvería a hacer falta la precisión
 * que el acercamiento existe para no necesitar. */
export const ZONAS = 3;

/** En qué zona gruesa del tablero cae un punto, o null si cae afuera. */
export function zonaEn(x, y, marco) {
  if (!marco || !(marco.width > 0) || !(marco.height > 0)) return null;
  if (x < marco.left || x >= marco.right) return null;
  if (y < marco.top || y >= marco.bottom) return null;
  const c = Math.min(ZONAS - 1,
    Math.floor(((x - marco.left) / marco.width) * ZONAS));
  const f = Math.min(ZONAS - 1,
    Math.floor(((y - marco.top) / marco.height) * ZONAS));
  return `z:${f}:${c}`;
}

/** Los límites de una zona. La permanencia los necesita para su margen de
 *  pegado: sin eso, el temblor en el borde entre dos zonas reinicia la cuenta
 *  una y otra vez y no se acerca nunca. */
export function cajaZona(id, marco) {
  // Se parte en vez de usar expresion regular: este archivo viaja por
  // heredocs y las barras invertidas se han perdido en el camino mas de una
  // vez, dejando un patron que no casa con nada y una funcion que devuelve
  // null siempre. Partir no tiene nada que escapar.
  const p = String(id || "").split(":");
  if (!marco || p.length !== 3 || p[0] !== "z") return null;
  const fi = Number(p[1]), co = Number(p[2]);
  if (!Number.isInteger(fi) || !Number.isInteger(co)) return null;
  if (fi < 0 || fi >= ZONAS || co < 0 || co >= ZONAS) return null;
  const w = marco.width / ZONAS, h = marco.height / ZONAS;
  const left = marco.left + co * w;
  const top = marco.top + fi * h;
  return { left, top, right: left + w, bottom: top + h, width: w, height: h };
}

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

  /** Lo que hay que escribir en el style del contenedor.
   *
   * `marco` es el rectángulo del contenedor SIN transformar. Hace falta
   * porque `transform-origin` en píxeles se mide desde el borde del propio
   * elemento y no desde el viewport, y la grilla no arranca en 0,0: tiene el
   * panel del mensaje encima. Pasarle coordenadas de viewport acercaba
   * centrado bastante más arriba de donde la persona estaba mirando.
   */
  estilo(marco) {
    if (!this.acercado || !this.origen) {
      return { transformOrigin: "50% 50%", transform: "none" };
    }
    const ox = this.origen[0] - (marco ? marco.left : 0);
    const oy = this.origen[1] - (marco ? marco.top : 0);
    return {
      transformOrigin: `${ox}px ${oy}px`,
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
