/* Corrección de la mirada después de calibrar, por eje y con ganancia.
 *
 * ## Qué problema resuelve, y por qué el anterior no alcanzaba
 *
 * La versión anterior medía un solo punto en el centro y restaba la
 * diferencia: una constante. Eso arregla un desvío parejo -todo corrido 200 px
 * a la derecha- y nada más.
 *
 * Medido con WebGazer sobre nueve puntos, el desvío NO era parejo:
 *
 * ```
 * objetivo x=384   ->  desvío medio   +18 px
 * objetivo x=768   ->  desvío medio  +165 px
 * objetivo x=1152  ->  desvío medio  +309 px
 * ```
 *
 * El desvío crece con la coordenada. Eso no es un corrimiento, es una
 * **ganancia equivocada**: la predicción se mueve 1.38 px por cada píxel que
 * se mueve el objetivo en horizontal, y 0.84 en vertical. Restar una constante
 * no puede arreglarlo, porque no hay ninguna constante que sea correcta en los
 * dos extremos a la vez: la que centra la izquierda descentra la derecha.
 *
 * Con una recta por eje -dos parámetros, ganancia y corrimiento- el sesgo
 * medido cae de **210 px a 45 px** sobre los mismos nueve puntos.
 *
 * ## Por qué una recta por eje y no algo más
 *
 * Porque es lo que los datos piden y nada más. El desvío horizontal depende
 * casi solo de x (+18, +165, +309 por columna) y el vertical casi solo de y
 * (+112, +97, +41 por fila); las dependencias cruzadas no muestran tendencia.
 * Una afín completa en 2D serían seis parámetros para explicar lo que explican
 * cuatro, y cada parámetro de más se ajusta a ruido.
 *
 * ## Por qué no se recalibra y listo
 *
 * Porque la ganancia equivocada no es un error de la calibración: es lo que el
 * modelo hace entre los puntos con los que se calibró. WebGazer ajusta 120
 * rasgos con 45 muestras y regularización casi nula, así que clava los nueve
 * puntos de entrenamiento -el recorrido da 100%- y entre medio hace lo que
 * quiere. Recalibrar reproduce el mismo modelo con el mismo comportamiento.
 *
 * Esto es la etapa de validación y corrección que hacen los medidores de
 * mirada de laboratorio después de calibrar, y existe por este motivo exacto.
 *
 * ## Los puntos de corrección NO son los de la calibración NI los de la prueba
 *
 * Calibración en 0.1/0.5/0.9, corrección en 0.2/0.5/0.8, prueba en
 * 0.25/0.5/0.75. Si la corrección se midiera sobre los puntos de calibración
 * no vería nada -ahí el modelo acierta- y si se midiera sobre los de la prueba,
 * la prueba dejaría de medir generalización.
 */

/** Ajuste por mínimos cuadrados de `pred = a * objetivo + b` en un eje.
 *
 * Con menos de dos objetivos distintos no hay recta posible: devuelve el
 * corrimiento puro, que es exactamente la corrección de un solo punto que
 * había antes. Es decir que esto generaliza aquello en vez de reemplazarlo.
 */
export function ajusteEje(objetivos, predicciones) {
  const n = objetivos.length;
  if (!n || n !== predicciones.length) return { a: 1, b: 0, puntos: 0 };
  const mt = objetivos.reduce((s, v) => s + v, 0) / n;
  const mp = predicciones.reduce((s, v) => s + v, 0) / n;
  let stt = 0, stp = 0;
  for (let i = 0; i < n; i++) {
    stt += (objetivos[i] - mt) ** 2;
    stp += (objetivos[i] - mt) * (predicciones[i] - mp);
  }
  // `corrimiento` es el ajuste SIN ganancia -la corrección de un solo punto-
  // y viaja junto con el otro porque es el repliegue si la ganancia no se
  // puede creer. Calculado a partir de `a` sería basura: `b` solo tiene
  // sentido acompañado de la pendiente con la que se calculó.
  const corrimiento = mp - mt;
  if (stt < 1) return { a: 1, b: corrimiento, corrimiento, puntos: n };
  const a = stp / stt;
  return { a, b: mp - a * mt, corrimiento, puntos: n };
}

/* Fuera de esta banda el ajuste no se cree. Una ganancia de 0.3 querría decir
 * que la predicción se mueve un tercio de lo que se mueve la mirada, y dividir
 * por ella multiplicaría el temblor por tres: la corrección haría más daño que
 * el error que arregla. */
const GANANCIA_MIN = 0.4;
const GANANCIA_MAX = 2.5;

export class Correccion {
  constructor(ex = { a: 1, b: 0 }, ey = { a: 1, b: 0 }) {
    this.ex = ex;
    this.ey = ey;
    this.listo = false;
  }

  /** @param {Array<[number,number]>} objetivos  dónde estaba el punto
   *  @param {Array<[number,number]>} crudas     dónde cayó la predicción */
  static ajustar(objetivos, crudas) {
    if (objetivos.length < 1 || objetivos.length !== crudas.length) {
      return new Correccion();
    }
    const ex = ajusteEje(objetivos.map((o) => o[0]), crudas.map((c) => c[0]));
    const ey = ajusteEje(objetivos.map((o) => o[1]), crudas.map((c) => c[1]));
    const c = new Correccion(sensata(ex), sensata(ey));
    c.listo = true;
    c.puntos = objetivos.length;
    return c;
  }

  /** Invierte la recta: de dónde cayó la predicción a dónde estaba mirando. */
  aplicar(p) {
    return [(p[0] - this.ex.b) / this.ex.a,
            (p[1] - this.ey.b) / this.ey.a];
  }

  get resumen() {
    const n = (v) => (v >= 0 ? "+" : "") + Math.round(v);
    return `x: ganancia ${this.ex.a.toFixed(2)} corrimiento ${n(this.ex.b)} px`
         + ` | y: ganancia ${this.ey.a.toFixed(2)} corrimiento ${n(this.ey.b)} px`;
  }
}

/** Una ganancia fuera de rango se degrada a corrimiento puro, que es seguro.
 *
 * Preferir no corregir la ganancia antes que corregirla mal: dividir por una
 * ganancia chica amplifica el temblor en la misma proporción, y el temblor es
 * lo único que hasta ahora salió bien en todas las mediciones.
 */
function sensata(e) {
  if (!Number.isFinite(e.a) || e.a < GANANCIA_MIN || e.a > GANANCIA_MAX) {
    const b = Number.isFinite(e.corrimiento) ? e.corrimiento : 0;
    return { a: 1, b, corrimiento: b, degradado: true };
  }
  return e;
}
