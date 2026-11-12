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
export const DWELL_ACERCAR_MS = 1000;
/* Cuánto tiene que durar una FIJACIÓN para que se acerque.
 *
 * Esta permanencia tuvo dos versiones que acercaban antes de tiempo, y las dos
 * por lo mismo: lo que contaban no era "la mirada se asentó".
 *
 * - La primera tenía por objetivo la grilla entera. Nunca cambiaba, así que
 *   nunca se reiniciaba: acercaba a los 450 ms de aparecer el tablero.
 * - La segunda contaba el tiempo dentro de una zona de un tercio de pantalla.
 *   Pero recorrer el tablero leyendo celdas pasa casi todo dentro de una
 *   zona, así que acercaba mientras la persona todavía estaba buscando.
 *
 * Lo que distingue buscar de haber encontrado no es dónde está la mirada sino
 * si SE MUEVE: leyendo, salta de celda en celda; decidida, se queda. Eso es
 * una fijación, y se detecta por dispersión: ver DISPERSION_FIJACION. */
export const DISPERSION_FIJACION = 180;
/* El lado máximo, en píxeles, del rectángulo que encierra todo lo que recorrió
 * la mirada durante la fijación. Es el criterio clásico de dispersión.
 *
 * Se mide el rectángulo entero y no la distancia a un promedio que se va
 * actualizando. La diferencia se midió: una mirada que alterna entre dos
 * celdas deja el promedio en el medio y cada muestra a poco más de 100 px de
 * él, así que con un radio alrededor del promedio eso pasaba por estar quieto.
 * El rectángulo no se deja engañar: mide de punta a punta.
 *
 * 180 px: una fijación real, con el cursor más tembloroso que se simuló,
 * recorre unos 150; saltar a la celda de al lado son 307. */

export const DERIVA_FIJACION = 60;
/* Cuánto puede correrse el centro entre la primera y la segunda mitad de la
 * fijación. El rectángulo solo no alcanza: con señal ruidosa el filtro
 * convierte un salto corto en un deslizamiento lento, y la mirada que pasó
 * de una celda a la de 240 px más allá entraba entera en 180 px. Se veía
 * como quieta y acercaba en el medio de las dos. Una fijación de verdad no
 * tiene dirección: su primera mitad y su segunda mitad caen en el mismo
 * lugar. */

/* Al entrar al tablero la mirada viene del último punto de la prueba. */
export const MS_ORIENTARSE = 600;

/* Después de alejar no se vuelve a acercar enseguida: sin esto, quien se queda
 * mirando el mismo lugar vuelve a quedar acercado y siente que no puede
 * salir. */
export const MS_PAUSA_TRAS_ALEJAR = 700;

/* Cerrar los ojos este tiempo cancela: aleja y reinicia lo que se estuviera
 * acumulando. Un parpadeo espontáneo dura entre 100 y 300 ms, así que con 400
 * no se dispara solo, y sostenerlo no cuesta esfuerzo. */
export const MS_PARPADEO_LARGO = 400;

/* Mirar fuera del tablero este tiempo, estando acercado, también aleja. */
export const MS_MIRAR_AFUERA = 600;

/** Detecta que la mirada se quedó quieta, por dispersión. */
export class Fijacion {
  constructor({ ms = DWELL_ACERCAR_MS, dispersion = DISPERSION_FIJACION,
                deriva = DERIVA_FIJACION, graciaMs = 300 } = {}) {
    this.ms = ms; this.dispersion = dispersion; this.graciaMs = graciaMs;
    this.deriva = deriva;
    this.bloqueoHasta = 0;
    this.evitar = null;
    this.ultimoVisto = 0;
    this.reiniciar();
  }

  reiniciar() {
    this.caja = null; this.desde = 0;
    this.inicio = [0, 0, 0]; this.final = [0, 0, 0];
  }

  empezarEn(x, y, t) {
    this.caja = [x, x, y, y]; this.desde = t;
    this.inicio = [0, 0, 0]; this.final = [0, 0, 0];
  }

  bloquear(ms, t) { this.bloqueoHasta = Math.max(this.bloqueoHasta, t + ms); }

  /** No cuenta nada hasta que la mirada se haya ido de `punto`. Es para
   *  cuando se alejó por tiempo: la persona estuvo mirando ahí sin elegir. */
  exigirMovimiento(punto) { this.evitar = punto ? [punto[0], punto[1]] : null; }

  /**
   * @param {{x:number,y:number,ok:boolean}} muestra
   * @param {boolean} enTablero  si ese punto cae sobre el tablero
   * @param {number} t  milisegundos
   * @returns {{avance:number, disparo:[number,number]|null}}
   */
  paso(muestra, enTablero, t) {
    const nada = { avance: 0, disparo: null };
    if (!muestra.ok) {
      // Un parpadeo corto no rompe la fijación; perder la mirada, sí.
      if (t - this.ultimoVisto > this.graciaMs) this.reiniciar();
      return { avance: this.avance(t), disparo: null };
    }
    this.ultimoVisto = t;
    if (t < this.bloqueoHasta || !enTablero) { this.reiniciar(); return nada; }

    const { x, y } = muestra;
    if (this.evitar) {
      const lejos = Math.hypot(x - this.evitar[0], y - this.evitar[1])
                  > this.dispersion;
      if (!lejos) { this.reiniciar(); return nada; }
      this.evitar = null;
    }

    const c = this.caja;
    if (c) {
      c[0] = Math.min(c[0], x); c[1] = Math.max(c[1], x);
      c[2] = Math.min(c[2], y); c[3] = Math.max(c[3], y);
    }
    if (!c || c[1] - c[0] > this.dispersion || c[3] - c[2] > this.dispersion) {
      // Se movió: empieza otra fijación acá.
      this.empezarEn(x, y, t);
      return nada;
    }

    // Dónde acercar: el promedio de la SEGUNDA mitad de la fijación. La
    // primera mitad es la más ruidosa -el filtro recién llegaba- y una sola
    // muestra, la última, tiembla demasiado para centrar con ella.
    const mitad = t - this.desde >= this.ms / 2 ? this.final : this.inicio;
    mitad[0] += x; mitad[1] += y; mitad[2] += 1;

    if (t - this.desde >= this.ms) {
      const a = this.inicio, f = this.final;
      const fin = f[2] ? [f[0] / f[2], f[1] / f[2]] : [x, y];
      const ini = a[2] ? [a[0] / a[2], a[1] / a[2]] : fin;
      if (Math.hypot(fin[0] - ini[0], fin[1] - ini[1]) > this.deriva) {
        // Venía deslizándose: no estaba quieta. Se sigue contando desde acá.
        this.empezarEn(x, y, t);
        return nada;
      }
      this.reiniciar();
      return { avance: 0, disparo: fin };
    }
    return { avance: this.avance(t), disparo: null };
  }

  avance(t) {
    return this.caja ? Math.min(1, (t - this.desde) / this.ms) : 0;
  }
}

/** Avisa UNA vez cuando los ojos llevan cerrados más de lo que dura un
 *  parpadeo espontáneo.
 *
 * "Cerrado" se decide contra la apertura normal de ESTA persona con ESTA
 * cámara, no contra un número fijo. El umbral fijo que usa el resto del
 * sistema es 0.14, y un ojo cerrado mide entre 0.09 y 0.18 según la cara, la
 * distancia y la proporción del cuadro: justo en el borde. Con un gesto del
 * que depende poder salir, "a veces no lo detecta" no sirve.
 *
 * La apertura normal se aprende sola: es un promedio lento de lo que se ve
 * mientras el ojo está abierto. Cerrado es caer por debajo de una fracción de
 * eso.
 */
export class ParpadeoLargo {
  constructor({ ms = MS_PARPADEO_LARGO, fraccion = 0.6, piso = 0.14 } = {}) {
    this.ms = ms; this.fraccion = fraccion; this.piso = piso;
    this.base = null;        // apertura normal aprendida
    this.desde = null; this.avisado = false;
    this.cerrado = false;
  }

  /**
   * @param {number} ear       apertura del párpado en este cuadro
   * @param {boolean} hayCara  el detector ve la cara. Sin cara no se puede
   *                           saber, y perderla no es parpadear.
   * @returns {boolean} true en el cuadro en que el cierre pasa a ser largo
   */
  paso(ear, hayCara, t) {
    if (!hayCara || !Number.isFinite(ear)) {
      this.desde = null; this.avisado = false; this.cerrado = false;
      return false;
    }
    if (this.base === null) this.base = ear;
    const umbral = Math.max(this.piso, this.fraccion * this.base);
    this.cerrado = ear < umbral;

    if (!this.cerrado) {
      // Solo se aprende con el ojo abierto, y despacio: un parpadeo no puede
      // bajar la referencia contra la que se mide el siguiente.
      this.base += 0.02 * (ear - this.base);
      this.desde = null; this.avisado = false;
      return false;
    }
    if (this.desde === null) { this.desde = t; return false; }
    if (!this.avisado && t - this.desde >= this.ms) {
      this.avisado = true;
      return true;
    }
    return false;
  }
}

export const MS_SIN_ELEGIR = 5000;
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
