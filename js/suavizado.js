/* El filtro que usa la aplicación: mediana sobre una ventana que se ACORTA
 * sola cuando la mirada salta.
 *
 * ## Por qué reemplaza a la cadena Kalman + media exponencial + mediana
 *
 * Aquella cadena se midió con un escalón: la mirada salta 600 px y se mira qué
 * hace el cursor. A 30 cuadros por segundo:
 *
 * ```
 * 1.07 s después   todavía le faltan 69 px para llegar
 * 2.13 s           se PASÓ 143 px de largo
 * 3.47 s           sigue 61 px pasado
 * 4.07 s           recién queda dentro de 20 px
 * ```
 *
 * A 15 cuadros por segundo tarda 7 s. El Kalman tenía un estado de velocidad
 * sin amortiguar: acumulaba impulso durante el viaje y seguía de largo.
 *
 * Eso contaminaba todo lo que venía después:
 *
 * - en el tablero el cursor llegaba un segundo tarde, se pasaba a la celda
 *   vecina y volvía: marcaba otra celda y acercaba en otro lugar;
 * - la prueba de precisión espera 1 s y mide 1.2 s, o sea que medía justo
 *   mientras el filtro se pasaba de largo, y lo anotaba como sesgo;
 * - la fase de corrección medía la ganancia con el mismo artefacto, siempre
 *   en el sentido del salto anterior.
 *
 * ## Cómo funciona este
 *
 * Una mediana no tiene impulso: no se puede pasar de largo. El problema de la
 * mediana es el otro: una ventana larga -la que hace falta para que el cursor
 * no tiemble- tarda media ventana en enterarse de un salto.
 *
 * Así que hay dos ventanas sobre los mismos datos. La larga da la salida
 * mientras la mirada está quieta. La corta vigila: si su mediana se aparta de
 * la larga más de `salto` píxeles, la mirada se fue a otro lado, y lo viejo se
 * descarta. En una fijación se promedia mucho; en un salto se responde en un
 * décimo de segundo.
 */
export class FiltroMirada {
  constructor({ msLargo = 800, msCorto = 250, salto = 120, sigmas = 4 } = {}) {
    this.msLargo = msLargo;
    this.msCorto = msCorto;
    this.salto = salto;
    this.sigmas = sigmas;
    this.buf = [];
    this.salida = null;
    this.saltos = 0;
  }

  reiniciar() { this.buf.length = 0; this.salida = null; }

  /**
   * @param {number} x @param {number} y  predicción cruda, en px
   * @param {number} t  milisegundos
   * @returns {[number, number]|null}
   */
  paso(x, y, t) {
    if (!Number.isFinite(x) || !Number.isFinite(y)) return this.salida;
    const buf = this.buf;
    // El lazo corre a la frecuencia de la pantalla y el estimador a la de la
    // cámara, o más lento: la misma predicción llega repetida. Contarla dos
    // veces no agrega información y hace creer que hay más muestras de las
    // que hay, que es justo con lo que se decide si un salto es real.
    const ult = buf[buf.length - 1];
    if (!ult || ult[0] !== x || ult[1] !== y) buf.push([x, y, t]);
    while (buf.length && buf[0][2] < t - this.msLargo) buf.shift();
    if (!buf.length) { buf.push([x, y, t]); }

    let desde = buf.length;
    while (desde > 0 && buf[desde - 1][2] >= t - this.msCorto) desde -= 1;
    const corto = buf.slice(desde);

    const largo = medianaDe(buf);
    if (corto.length >= 3 && corto.length < buf.length) {
      const c = medianaDe(corto);
      // El umbral se adapta al ruido que hay: con una señal limpia alcanza
      // con `salto` píxeles, con una ruidosa la mediana de pocas muestras
      // baila sola y hay que pedirle más para creerle. Sin esto, con ruido
      // alto el filtro "veía" saltos que no existían y tiraba la ventana.
      const ruido = desvioRobusto(buf);
      const umbral = Math.max(this.salto,
        this.sigmas * 1.25 * ruido / Math.sqrt(corto.length));
      if (Math.hypot(c[0] - largo[0], c[1] - largo[1]) > umbral) {
        // La mirada saltó: lo anterior describe otro lugar.
        this.buf = corto;
        this.salida = c;
        this.saltos += 1;
        return c;
      }
    }
    this.salida = largo;
    return largo;
  }
}

/** Cuánto ruido trae la señal, estimado por las diferencias entre muestras
 *  SEGUIDAS.
 *
 * La primera versión medía la dispersión de la ventana alrededor de su
 * mediana. Falla justo cuando importa: si la mirada alterna entre dos celdas,
 * la ventana tiene dos grupos separados, la dispersión se infla, el umbral
 * sube con ella y el filtro deja de ver los saltos. La salida se quedaba
 * flotando entre las dos celdas, quieta, y eso se leía como una fijación.
 *
 * Las diferencias entre muestras seguidas no tienen ese problema: un salto
 * aporta UNA diferencia grande y la mediana la ignora.
 */
function desvioRobusto(muestras) {
  if (muestras.length < 4) return 0;
  const d = (k) => {
    const a = [];
    for (let i = 1; i < muestras.length; i++) {
      a.push(Math.abs(muestras[i][k] - muestras[i - 1][k]));
    }
    a.sort((p, q) => p - q);
    return a[a.length >> 1];
  };
  // mediana|diferencia| = 0.6745 * sigma * raiz(2); se promedian los dos ejes.
  return (d(0) + d(1)) / 2 / (0.6745 * Math.SQRT2);
}

function medianaDe(muestras) {
  const m = (a) => {
    a.sort((p, q) => p - q);
    const n = a.length;
    return n % 2 ? a[(n - 1) / 2] : (a[n / 2 - 1] + a[n / 2]) / 2;
  };
  return [m(muestras.map((p) => p[0])), m(muestras.map((p) => p[1]))];
}
