/* Selección por permanencia: sostener la mirada sobre una celda la dispara.
 *
 * Port del `DwellSelector` nativo, con los mismos cuatro parámetros. Cada uno
 * existe por un modo de fallar concreto:
 *
 * - **dwell_ms (1200)**: cuánto hay que sostener. Corto se misclickea y la
 *   frase se arruina; largo cansa. 1200 salió de subirlo después de que
 *   misclickear caracteres rompiera enunciados enteros.
 * - **margen de pegado (0.35)**: una vez elegida una celda, hay que salirse
 *   35% más allá de su borde para perderla. Sin esto, el temblor en el límite
 *   entre dos celdas reinicia la cuenta una y otra vez y no se selecciona
 *   nada.
 * - **refractario (600 ms)**: después de disparar, no se acepta nada. Evita
 *   que la misma fijación dispare dos veces.
 * - **gracia de parpadeo (250 ms)**: perder la cara por un parpadeo no
 *   reinicia la permanencia. Un parpadeo dura 100-400 ms y sin esto es
 *   imposible completar una selección.
 */

export const DWELL_MS = 1200;
export const REFRACTARIO_MS = 600;
export const GRACIA_PARPADEO_MS = 250;
export const MARGEN_PEGADO = 0.35;

export class Permanencia {
  constructor(opciones = {}) {
    this.dwellMs = opciones.dwellMs ?? DWELL_MS;
    this.refractarioMs = opciones.refractarioMs ?? REFRACTARIO_MS;
    this.graciaMs = opciones.graciaMs ?? GRACIA_PARPADEO_MS;
    this.margen = opciones.margen ?? MARGEN_PEGADO;
    this.objetivo = null;
    this.desde = 0;
    this.bloqueoHasta = 0;
    this.ultimoVisto = 0;
  }

  bloquear(ms) { this.bloqueoHasta = Math.max(this.bloqueoHasta, now() + ms); }

  reiniciar() { this.objetivo = null; this.desde = 0; }

  /**
   * @param {{x:number,y:number,ok:boolean}} muestra  mirada suavizada
   * @param {(x:number,y:number)=>string|null} golpear  qué celda hay en (x,y)
   * @param {(id:string)=>DOMRect|null} caja  los límites de una celda
   * @returns {{avance:number, disparo:string|null, objetivo:string|null}}
   */
  paso(muestra, golpear, caja) {
    const t = now();

    if (!muestra.ok) {
      // Sin cara: si fue hace poco, se asume parpadeo y se mantiene todo.
      if (t - this.ultimoVisto > this.graciaMs) this.reiniciar();
      return this.estado(t);
    }
    this.ultimoVisto = t;

    if (t < this.bloqueoHasta) { this.reiniciar(); return this.estado(t); }

    let id = golpear(muestra.x, muestra.y);

    // Pegado: si ya había un objetivo y la mirada se fue apenas afuera, se
    // conserva. Sin esto, dos celdas vecinas se turnan y ninguna llega.
    if (this.objetivo && id !== this.objetivo) {
      const r = caja(this.objetivo);
      if (r && dentroConMargen(muestra.x, muestra.y, r, this.margen)) {
        id = this.objetivo;
      }
    }

    if (!id) { this.reiniciar(); return this.estado(t); }
    if (id !== this.objetivo) { this.objetivo = id; this.desde = t; }

    if (t - this.desde >= this.dwellMs) {
      const disparo = this.objetivo;
      this.reiniciar();
      this.bloqueoHasta = t + this.refractarioMs;
      return { avance: 0, disparo, objetivo: null };
    }
    return this.estado(t);
  }

  estado(t) {
    const avance = this.objetivo
      ? Math.min(1, (t - this.desde) / this.dwellMs) : 0;
    return { avance, disparo: null, objetivo: this.objetivo };
  }
}

function dentroConMargen(x, y, r, margen) {
  const mx = r.width * margen, my = r.height * margen;
  return x >= r.left - mx && x <= r.right + mx
      && y >= r.top - my && y <= r.bottom + my;
}

function now() {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}
