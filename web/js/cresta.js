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
