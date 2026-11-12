/* Elegir la constante de cresta de WebGazer por validación cruzada.
 *
 * ## Por qué hace falta
 *
 * WebGazer ajusta `(XᵀX + kI)⁻¹Xᵀy` en cada predicción, sin intercepto y sin
 * centrar, con `k = 0.00001` fijo. Los números concretos:
 *
 * - **120 rasgos** (dos parches de ojo de 10x6, ecualizados, en 0..255);
 * - **45 muestras** de calibración, que es lo máximo que entra en su ventana;
 * - diagonal de `XᵀX` del orden de 45 x 128² ≈ 7x10⁵.
 *
 * Con 45 muestras y 120 rasgos la matriz tiene rango 45 como mucho: es
 * singular. Y `k = 0.00001` está once órdenes de magnitud por debajo de su
 * diagonal, así que no la arregla; lo único que la vuelve invertible es el
 * redondeo de punto flotante. El modelo pasa exacto por los nueve puntos de
 * calibración -por eso el recorrido da 100%- y entre medio lo que hace lo
 * decide el espacio nulo, que no está determinado por los datos.
 *
 * Eso se midió. Dos sesiones seguidas, mismo banco, misma persona:
 *
 * ```
 * sesión 1:  desvío horizontal perfectamente recto   (curvatura  1 px)
 * sesión 2:  desvío horizontal doblado               (curvatura 27 px)
 * ```
 *
 * La forma del error cambia de una calibración a la siguiente. Por eso la
 * corrección lineal arregla mucho pero no todo: una recta puede seguir a otra
 * recta, no a una curva que además cambia.
 *
 * ## Por qué validación cruzada y no un valor elegido a mano
 *
 * Porque el estimador propio ya elige su alfa por validación cruzada dejando
 * un PUNTO afuera. Comparar un modelo con el hiperparámetro ajustado contra
 * otro con el hiperparámetro de fábrica no mide los dos estimadores, mide que
 * uno está ajustado y el otro no. Esto empareja el banco.
 *
 * No le cuesta nada al usuario: se corre sobre las 45 muestras que la
 * calibración ya grabó, sin pedir un punto más.
 *
 * ## Por qué dejar un PUNTO afuera y no una muestra
 *
 * Las cinco muestras de un mismo punto se tomaron con la mirada quieta en el
 * mismo lugar: son casi idénticas. Dejando una afuera, las otras cuatro la
 * delatan y la validación aprueba cualquier k, incluido el que memoriza. Hay
 * que dejar afuera el punto entero.
 */

/* Candidatos RELATIVOS a la diagonal media de XᵀX, no absolutos: así la escala
 * de los rasgos no cambia el significado del número. El primero, 1e-11, es
 * prácticamente el `0.00001` de fábrica, y está para que la comparación
 * incluya "no regularizar" en vez de darlo por perdido de entrada. */
export const RELATIVOS = [1e-11, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 2];

/** Resuelve `M z = B` por eliminación con pivoteo parcial.
 *
 * `M` es d x d y `B` es d x m, los dos planos y por filas. Se destruyen: el
 * resultado queda en `B`. Con m = 2 se resuelven los dos ejes de una sola
 * eliminación, que es donde está la mitad del ahorro.
 */
export function resolver(M, d, B, m) {
  for (let c = 0; c < d; c++) {
    let p = c, mx = Math.abs(M[c * d + c]);
    for (let r = c + 1; r < d; r++) {
      const v = Math.abs(M[r * d + c]);
      if (v > mx) { mx = v; p = r; }
    }
    if (!(mx > 0)) return null;
    if (p !== c) {
      for (let k = c; k < d; k++) {
        const t = M[c * d + k]; M[c * d + k] = M[p * d + k]; M[p * d + k] = t;
      }
      for (let k = 0; k < m; k++) {
        const t = B[c * m + k]; B[c * m + k] = B[p * m + k]; B[p * m + k] = t;
      }
    }
    const piv = M[c * d + c];
    for (let r = c + 1; r < d; r++) {
      const f = M[r * d + c] / piv;
      if (f === 0) continue;
      for (let k = c; k < d; k++) M[r * d + k] -= f * M[c * d + k];
      for (let k = 0; k < m; k++) B[r * m + k] -= f * B[c * m + k];
    }
  }
  for (let c = d - 1; c >= 0; c--) {
    const piv = M[c * d + c];
    for (let k = 0; k < m; k++) {
      let s = B[c * m + k];
      for (let j = c + 1; j < d; j++) s -= M[c * d + j] * B[j * m + k];
      B[c * m + k] = s / piv;
    }
  }
  return B;
}

/** Normales `XᵀX`, `Xᵀyx`, `Xᵀyy` sobre las filas que `tomar` acepte. */
function normales(X, yx, yy, d, tomar) {
  const A = new Float64Array(d * d);
  const bx = new Float64Array(d), by = new Float64Array(d);
  for (let k = 0; k < X.length; k++) {
    if (!tomar(k)) continue;
    const x = X[k], vx = yx[k], vy = yy[k];
    for (let i = 0; i < d; i++) {
      const xi = x[i];
      if (xi === 0) continue;
      bx[i] += xi * vx; by[i] += xi * vy;
      const fila = i * d;
      for (let j = i; j < d; j++) A[fila + j] += xi * x[j];
    }
  }
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < i; j++) A[i * d + j] = A[j * d + i];
  }
  return { A, bx, by };
}

/**
 * La constante de cresta que menos error de generalización deja.
 *
 * @param {number[][]} X       rasgos por muestra (45 x 120)
 * @param {number[]}   yx      objetivo horizontal de cada muestra, en px
 * @param {number[]}   yy      objetivo vertical
 * @param {number[]}   grupos  a qué punto de calibración pertenece cada muestra
 * @returns {{cresta:number, relativo:number, error:number, base:number,
 *            tabla:Array<{relativo:number, cresta:number, error:number}>,
 *            mejoraSobre:number}|null}
 *          `null` si no hay con qué validar: menos de tres puntos distintos.
 */
export function elegirCresta(X, yx, yy, grupos, relativos = RELATIVOS) {
  const n = X.length;
  if (!n || yx.length !== n || yy.length !== n || grupos.length !== n) return null;
  const puntos = [...new Set(grupos)];
  if (puntos.length < 3) return null;
  const d = X[0].length;

  // Escala de referencia: la diagonal media de XᵀX con TODO. Un relativo de
  // 0.01 quiere decir "la centésima parte de lo que ya hay en la diagonal",
  // que es una cantidad comparable sin importar en qué unidades vengan los
  // rasgos.
  const todo = normales(X, yx, yy, d, () => true);
  let traza = 0;
  for (let i = 0; i < d; i++) traza += todo.A[i * d + i];
  const base = traza / d;
  if (!(base > 0)) return null;

  const errores = relativos.map(() => ({ suma: 0, n: 0 }));
  const M = new Float64Array(d * d);
  const B = new Float64Array(d * 2);

  for (const g of puntos) {
    const dentro = normales(X, yx, yy, d, (k) => grupos[k] !== g);
    const fuera = [];
    for (let k = 0; k < n; k++) if (grupos[k] === g) fuera.push(k);
    if (!fuera.length) continue;

    for (let c = 0; c < relativos.length; c++) {
      const k = relativos[c] * base;
      M.set(dentro.A);
      for (let i = 0; i < d; i++) M[i * d + i] += k;
      for (let i = 0; i < d; i++) { B[i * 2] = dentro.bx[i]; B[i * 2 + 1] = dentro.by[i]; }
      if (!resolver(M, d, B, 2)) { errores[c].suma = Infinity; errores[c].n = 1; continue; }
      for (const f of fuera) {
        const x = X[f];
        let px = 0, py = 0;
        for (let i = 0; i < d; i++) { px += x[i] * B[i * 2]; py += x[i] * B[i * 2 + 1]; }
        errores[c].suma += Math.hypot(px - yx[f], py - yy[f]);
        errores[c].n += 1;
      }
    }
  }

  const tabla = relativos.map((r, c) => ({
    relativo: r,
    cresta: r * base,
    error: errores[c].n ? errores[c].suma / errores[c].n : NaN,
  })).filter((f) => Number.isFinite(f.error));
  if (!tabla.length) return null;

  const mejor = tabla.reduce((a, b) => (b.error < a.error ? b : a));
  // Qué se gana contra no regularizar, que es lo que hace WebGazer de fábrica.
  const sinNada = tabla[0];
  return {
    cresta: mejor.cresta,
    relativo: mejor.relativo,
    error: mejor.error,
    base,
    tabla,
    mejoraSobre: sinNada ? sinNada.error : NaN,
  };
}

/* ------------------------------------------------------------------------
 * Regresión con INTERCEPTO, para cuando el ajuste lo hace este proyecto y no
 * WebGazer.
 *
 * ## Por qué existe
 *
 * La validación de arriba se probó con datos reales y no encontró nada: con
 * 45 muestras el error validado fue de 109 px con la cresta de fábrica y de
 * 109 px con la mejor. No había nada que elegir, porque el problema no era el
 * valor de la cresta sino la forma del ajuste:
 *
 * - **sin intercepto**: la predicción es una suma de rasgos por coeficientes
 *   y nada más, así que el brillo medio del parche -que no dice nada de la
 *   mirada- tiene que explicarse con los mismos coeficientes;
 * - **nueve lugares**: el desvío medido cambia de +98 px a -155 px entre dos
 *   puntos vecinos de la misma fila. El modelo pasa por los nueve y entre
 *   medio no tiene con qué orientarse;
 * - **45 muestras como máximo**, por la ventana de WebGazer.
 *
 * Centrar los rasgos y los objetivos antes de ajustar es lo mismo que agregar
 * el intercepto, y deja a la cresta castigando solo lo que varía.
 */

/* Con los rasgos centrados la diagonal ya no carga con el brillo medio, así
 * que la cresta útil es, en proporción, bastante mayor que sin centrar. */
export const RELATIVOS_CENTRADO =
  [1e-6, 1e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1, 3];

function centrar(X, yx, yy, filas) {
  const d = X[0].length, n = filas.length;
  const media = new Float64Array(d);
  let mx = 0, my = 0;
  for (const k of filas) {
    const x = X[k];
    for (let i = 0; i < d; i++) media[i] += x[i];
    mx += yx[k]; my += yy[k];
  }
  for (let i = 0; i < d; i++) media[i] /= n;
  mx /= n; my /= n;
  const Xc = filas.map((k) => {
    const x = X[k], f = new Float64Array(d);
    for (let i = 0; i < d; i++) f[i] = x[i] - media[i];
    return f;
  });
  return { Xc, yxc: filas.map((k) => yx[k] - mx),
           yyc: filas.map((k) => yy[k] - my), media, mx, my };
}

function resolverCentrado(c, d, cresta, A0, M, B) {
  M.set(A0.A);
  for (let i = 0; i < d; i++) M[i * d + i] += cresta;
  for (let i = 0; i < d; i++) { B[i * 2] = A0.bx[i]; B[i * 2 + 1] = A0.by[i]; }
  if (!resolver(M, d, B, 2)) return null;
  const cx = new Float64Array(d), cy = new Float64Array(d);
  for (let i = 0; i < d; i++) { cx[i] = B[i * 2]; cy[i] = B[i * 2 + 1]; }
  return { cx, cy, media: c.media, mx: c.mx, my: c.my };
}

/** Ajusta el modelo con intercepto sobre las filas dadas (todas, si no se
 *  dicen). Devuelve lo que necesita `predecirCentrado`, o null. */
export function ajustarCentrado(X, yx, yy, cresta, filas = null) {
  if (!X.length) return null;
  const todas = filas ?? X.map((_, i) => i);
  const d = X[0].length;
  const c = centrar(X, yx, yy, todas);
  const A0 = normales(c.Xc, c.yxc, c.yyc, d, () => true);
  return resolverCentrado(c, d, cresta, A0,
    new Float64Array(d * d), new Float64Array(d * 2));
}

/** @returns {[number, number]} la posición en pantalla para unos rasgos */
export function predecirCentrado(m, f) {
  let x = m.mx, y = m.my;
  for (let i = 0; i < m.cx.length; i++) {
    const v = f[i] - m.media[i];
    x += v * m.cx[i]; y += v * m.cy[i];
  }
  return [x, y];
}

/** Lo que hace WebGazer de fábrica: sin intercepto y con la cresta que se le
 *  pase. Está para poder comparar contra eso, no para usarlo. */
export function ajustarCrudo(X, yx, yy, cresta) {
  if (!X.length) return null;
  const d = X[0].length;
  const A0 = normales(X, yx, yy, d, () => true);
  const c = { media: new Float64Array(d), mx: 0, my: 0 };
  return resolverCentrado(c, d, cresta, A0,
    new Float64Array(d * d), new Float64Array(d * 2));
}

/**
 * La cresta que menos error deja en un punto que el modelo no vio, con
 * intercepto. El centrado se recalcula en cada vuelta con las filas que
 * quedaron adentro: usar la media de todas dejaría que el punto que se saca
 * influya en el modelo que lo va a predecir.
 *
 * @returns {{cresta:number, relativo:number, error:number, base:number,
 *            enBorde:boolean,
 *            tabla:Array<{relativo:number, cresta:number, error:number}>}|null}
 */
export function elegirCrestaCentrada(X, yx, yy, grupos,
                                     relativos = RELATIVOS_CENTRADO) {
  const n = X.length;
  if (!n || yx.length !== n || yy.length !== n || grupos.length !== n) return null;
  const puntos = [...new Set(grupos)];
  if (puntos.length < 3) return null;
  const d = X[0].length;

  const todo = centrar(X, yx, yy, X.map((_, i) => i));
  const At = normales(todo.Xc, todo.yxc, todo.yyc, d, () => true);
  let traza = 0;
  for (let i = 0; i < d; i++) traza += At.A[i * d + i];
  const base = traza / d;
  if (!(base > 0)) return null;

  const errores = relativos.map(() => ({ suma: 0, n: 0 }));
  const M = new Float64Array(d * d), B = new Float64Array(d * 2);

  for (const g of puntos) {
    const adentro = [], afuera = [];
    for (let k = 0; k < n; k++) (grupos[k] === g ? afuera : adentro).push(k);
    if (!afuera.length || !adentro.length) continue;
    const c = centrar(X, yx, yy, adentro);
    const A0 = normales(c.Xc, c.yxc, c.yyc, d, () => true);
    for (let r = 0; r < relativos.length; r++) {
      const m = resolverCentrado(c, d, relativos[r] * base, A0, M, B);
      if (!m) { errores[r].suma = Infinity; errores[r].n = 1; continue; }
      for (const k of afuera) {
        const [px, py] = predecirCentrado(m, X[k]);
        errores[r].suma += Math.hypot(px - yx[k], py - yy[k]);
        errores[r].n += 1;
      }
    }
  }

  const tabla = relativos.map((r, i) => ({
    relativo: r, cresta: r * base,
    error: errores[i].n ? errores[i].suma / errores[i].n : NaN,
  })).filter((f) => Number.isFinite(f.error));
  if (!tabla.length) return null;
  const mejor = tabla.reduce((a, b) => (b.error < a.error ? b : a));
  return {
    cresta: mejor.cresta, relativo: mejor.relativo, error: mejor.error,
    base, tabla,
    enBorde: mejor === tabla[0] || mejor === tabla[tabla.length - 1],
  };
}
