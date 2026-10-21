/* Kalman de velocidad constante + media exponencial.
 *
 * La predicción cruda tiembla. No es ruido del estimador solamente: el ojo
 * humano **no se queda quieto** ni cuando uno cree que mira fijo -hay micro
 * sacadas de hasta medio grado-, así que un cursor fiel a la mirada vibra por
 * construcción. Suavizar no es esconder un defecto, es mostrar la intención en
 * vez de la fisiología.
 *
 * El Kalman asume velocidad constante, que es razonable entre sacadas, y la
 * media exponencial encima saca lo que queda. Dos etapas en vez de una porque
 * un Kalman lo bastante agresivo para quitar todo el temblor también se come
 * las sacadas de verdad, y entonces el cursor llega tarde a cada celda.
 *
 * Mismos parámetros que `config.py` del lado nativo: smoother="kalman_ema",
 * ema_alpha=0.25.
 */

const ALFA_EMA = 0.25;

export class Suavizador {
  /**
   * @param {number} q  ruido de proceso: cuánto se le cree al movimiento
   * @param {number} r  ruido de medición: cuánto se le cree a la predicción
   */
  constructor(q = 18, r = 260) {
    this.q = q; this.r = r;
    this.reiniciar();
  }

  reiniciar() {
    this.x = null;          // [px, py, vx, vy]
    this.P = null;          // covarianza 4x4, como matriz diagonal por bloques
    this.ema = null;
  }

  /**
   * @param {number} zx @param {number} zy  medición cruda
   * @param {number} dt  segundos desde la medición anterior
   * @returns {[number, number]} posición suavizada
   */
  paso(zx, zy, dt) {
    if (!Number.isFinite(zx) || !Number.isFinite(zy)) {
      return this.ema ? [this.ema[0], this.ema[1]] : [zx, zy];
    }
    if (this.x === null) {
      this.x = [zx, zy, 0, 0];
      this.P = [500, 500, 500, 500];
      this.ema = [zx, zy];
      return [zx, zy];
    }
    // dt se acota: si la pestaña estuvo en segundo plano, puede llegar un
    // salto de varios segundos y el modelo de velocidad constante proyectaría
    // el cursor fuera de la pantalla.
    dt = Math.min(Math.max(dt, 1 / 120), 0.2);

    // --- predicción ---
    this.x[0] += this.x[2] * dt;
    this.x[1] += this.x[3] * dt;
    for (let i = 0; i < 4; i++) this.P[i] += this.q * dt;

    // --- corrección, por eje (los ejes son independientes en este modelo) ---
    for (const [ip, iv, z] of [[0, 2, zx], [1, 3, zy]]) {
      const S = this.P[ip] + this.r;
      const kp = this.P[ip] / S;
      const kv = this.P[iv] / S;
      const y = z - this.x[ip];
      this.x[ip] += kp * y;
      this.x[iv] += kv * y;
      this.P[ip] *= (1 - kp);
      this.P[iv] *= (1 - kv);
    }

    this.ema[0] += ALFA_EMA * (this.x[0] - this.ema[0]);
    this.ema[1] += ALFA_EMA * (this.x[1] - this.ema[1]);
    return [this.ema[0], this.ema[1]];
  }
}

/* Ventana por TIEMPO y no por cantidad de cuadros.
 *
 * El lado nativo cometía el error de contar cuadros: a 14 fps una ventana de
 * 7 cuadros son medio segundo de retraso, y a 30 fps son 230 ms. El mismo
 * número producía dos comportamientos distintos según cuánto costara el
 * cuadro, que es la clase de cosa que hace imposible comparar dos sesiones. */
export class VentanaMediana {
  constructor(ms = 320) { this.ms = ms; this.buf = []; }

  empujar(x, y, t) {
    this.buf.push([x, y, t]);
    const corte = t - this.ms;
    while (this.buf.length && this.buf[0][2] < corte) this.buf.shift();
    return this.mediana();
  }

  mediana() {
    if (!this.buf.length) return null;
    const xs = this.buf.map((p) => p[0]).sort((a, b) => a - b);
    const ys = this.buf.map((p) => p[1]).sort((a, b) => a - b);
    const m = (a) => a.length % 2 ? a[(a.length - 1) / 2]
                                  : (a[a.length / 2 - 1] + a[a.length / 2]) / 2;
    return [m(xs), m(ys)];
  }

  limpiar() { this.buf.length = 0; }
}
