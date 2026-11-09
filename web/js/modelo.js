/* Regresión ridge de 486 rasgos a dos coordenadas de pantalla.
 *
 * ## Por qué la forma dual
 *
 * La forma directa resuelve (XᵀX + λI)w = Xᵀy, o sea un sistema de 486x486.
 * Son ~38 millones de operaciones en la eliminación, en JavaScript, mientras
 * la persona espera después de calibrar.
 *
 * Con más rasgos que muestras -486 contra ~130- la forma dual da exactamente
 * la misma predicción resolviendo un sistema de n x n, que acá es 130x130:
 * unas 700 mil operaciones, dos órdenes de magnitud menos. La identidad es
 *
 *     w = Xᵀ(XXᵀ + λI)⁻¹ y
 *
 * y predecir es ŷ(x) = Σ αᵢ (xᵢ · x) con α = (XXᵀ + λI)⁻¹ y. No es una
 * aproximación: es la misma solución reordenada.
 *
 * ## La estandarización no es opcional
 *
 * Los 483 primeros rasgos son coordenadas normalizadas, con valores del orden
 * de 0.01-2. Los tres últimos son ángulos en radianes, del orden de 0.1-1.5.
 * Sin estandarizar, la penalización de ridge castiga a todos por igual en
 * valor absoluto, lo que en la práctica significa castigar más a los rasgos de
 * escala chica. Igual que el `StandardScaler` del pipeline de Python.
 */

export const ALFA_POR_DEFECTO = 1.0;

export class ModeloMirada {
  constructor() {
    this.listo = false;
    this.X = null;        // muestras estandarizadas
    this.alfaX = null;    // coeficientes duales para la coordenada x
    this.alfaY = null;
    this.media = null;
    this.desvio = null;
    this.n = 0;
  }

  /**
   * @param {Float64Array[]} muestras  vectores de rasgos
   * @param {Array<[number,number]>} objetivos  píxeles de pantalla
   */
  entrenar(muestras, objetivos, alfa = ALFA_POR_DEFECTO) {
    const n = muestras.length;
    if (n < 4) throw new Error("hacen falta al menos 4 muestras");
    const d = muestras[0].length;

    // --- estandarizar ---
    const media = new Float64Array(d), desvio = new Float64Array(d);
    for (const m of muestras) for (let j = 0; j < d; j++) media[j] += m[j];
    for (let j = 0; j < d; j++) media[j] /= n;
    for (const m of muestras) {
      for (let j = 0; j < d; j++) {
        const dif = m[j] - media[j];
        desvio[j] += dif * dif;
      }
    }
    // Un rasgo constante tiene desvío cero y dividirlo da NaN, que envenena
    // todo el vector. Se le pone 1: estandarizado queda en cero y ridge lo
    // ignora, que es lo correcto para un rasgo que no varía.
    for (let j = 0; j < d; j++) {
      desvio[j] = Math.sqrt(desvio[j] / n);
      if (desvio[j] < 1e-9) desvio[j] = 1;
    }

    const X = muestras.map((m) => {
      const z = new Float64Array(d);
      for (let j = 0; j < d; j++) z[j] = (m[j] - media[j]) / desvio[j];
      return z;
    });

    // --- núcleo lineal K = XXᵀ + λI ---
    const K = [];
    for (let i = 0; i < n; i++) {
      const fila = new Float64Array(n);
      for (let j = 0; j <= i; j++) {
        let s = 0;
        const a = X[i], b = X[j];
        for (let k = 0; k < d; k++) s += a[k] * b[k];
        fila[j] = s;
        if (j < i) K[j][i] = s;   // simétrica: se calcula media matriz
      }
      K.push(fila);
    }
    for (let i = 0; i < n; i++) K[i][i] += alfa;

    // Centrar los objetivos: ridge penaliza los coeficientes pero no debería
    // penalizar el nivel medio. Sin centrar, la predicción se arrastra hacia
    // el cero de la pantalla (la esquina superior izquierda).
    const mx = objetivos.reduce((s, o) => s + o[0], 0) / n;
    const my = objetivos.reduce((s, o) => s + o[1], 0) / n;

    this.alfaX = resolver(K, objetivos.map((o) => o[0] - mx));
    this.alfaY = resolver(K, objetivos.map((o) => o[1] - my));
    this.mediaObjX = mx; this.mediaObjY = my;
    this.X = X; this.media = media; this.desvio = desvio; this.n = n;
    this.listo = true;
    return this;
  }

  /** @param {Float64Array} rasgos @returns {[number,number]} */
  predecir(rasgos) {
    if (!this.listo) return [0, 0];
    const d = rasgos.length;
    const z = new Float64Array(d);
    for (let j = 0; j < d; j++) z[j] = (rasgos[j] - this.media[j]) / this.desvio[j];

    let sx = this.mediaObjX, sy = this.mediaObjY;
    for (let i = 0; i < this.n; i++) {
      const xi = this.X[i];
      let k = 0;
      for (let j = 0; j < d; j++) k += xi[j] * z[j];
      sx += this.alfaX[i] * k;
      sy += this.alfaY[i] * k;
    }
    return [sx, sy];
  }

  /** Error de entrenamiento, en píxeles. Diagnóstico honesto: NO es la
   *  precisión real, porque se mide sobre los mismos puntos con los que se
   *  entrenó. Sirve para detectar una calibración rota, no para reportarla. */
  errorEntrenamiento(muestras, objetivos) {
    let suma = 0;
    for (let i = 0; i < muestras.length; i++) {
      const [px, py] = this.predecir(muestras[i]);
      suma += Math.hypot(px - objetivos[i][0], py - objetivos[i][1]);
    }
    return suma / muestras.length;
  }
}

/* Eliminación gaussiana con pivoteo parcial. El pivoteo importa: K puede tener
 * filas casi dependientes cuando dos puntos de calibración salen parecidos
 * -alguien que no movió los ojos entre dos puntos- y sin pivoteo eso da
 * división por un número diminuto y coeficientes enormes. */
function resolver(A, b) {
  const n = b.length;
  const M = A.map((fila, i) => {
    const f = new Float64Array(n + 1);
    f.set(fila.subarray ? fila.subarray(0, n) : fila.slice(0, n));
    f[n] = b[i];
    return f;
  });

  for (let col = 0; col < n; col++) {
    let mejor = col;
    for (let f = col + 1; f < n; f++) {
      if (Math.abs(M[f][col]) > Math.abs(M[mejor][col])) mejor = f;
    }
    if (mejor !== col) { const t = M[col]; M[col] = M[mejor]; M[mejor] = t; }
    const piv = M[col][col];
    if (Math.abs(piv) < 1e-12) continue;   // fila degenerada: se deja en cero
    for (let f = col + 1; f < n; f++) {
      const factor = M[f][col] / piv;
      if (factor === 0) continue;
      for (let c = col; c <= n; c++) M[f][c] -= factor * M[col][c];
    }
  }

  const x = new Float64Array(n);
  for (let f = n - 1; f >= 0; f--) {
    let s = M[f][n];
    for (let c = f + 1; c < n; c++) s -= M[f][c] * x[c];
    x[f] = Math.abs(M[f][f]) < 1e-12 ? 0 : s / M[f][f];
  }
  return x;
}

/* Alphas a probar. Cubre seis órdenes de magnitud porque el valor correcto
 * depende de la relación señal-ruido de ESA calibración, no de la tarea. */
const ALFAS = [0.1, 1, 10, 50, 200, 1000, 5000, 20000];

/**
 * Elige alpha con validación cruzada dejando fuera un PUNTO completo, y
 * devuelve el modelo ya entrenado con todo.
 *
 * ## Por qué hacía falta
 *
 * El valor fijo anterior (5000) venía de una prueba sintética cuya estructura
 * de señal no se parecía a la real, y en uso colapsó el modelo: medido sobre
 * nueve puntos, predecía prácticamente el centro de la pantalla para cualquier
 * entrada. El sesgo era de 354 px con apenas 40 px de temblor, que es la firma
 * exacta de una predicción casi constante.
 *
 * ## Por qué se deja fuera un punto y no una muestra
 *
 * Las 30 muestras de un mismo punto son cuadros consecutivos de un ojo quieto:
 * casi idénticas. Dejar fuera una sola deja sus 29 gemelas dentro, así que
 * cualquier alpha parece perfecto y la validación no valida nada. Dejando
 * fuera el punto entero se mide lo único que importa: cuánto sirve en una
 * fijación que el modelo no vio.
 *
 * @param {Float64Array[]} muestras
 * @param {Array<[number,number]>} objetivos
 * @param {number[]} grupos  a qué punto de calibración pertenece cada muestra
 */
export function entrenarConCV(muestras, objetivos, grupos) {
  const puntos = [...new Set(grupos)];
  let mejor = { alfa: ALFAS[0], error: Infinity };

  if (puntos.length >= 3) {
    for (const alfa of ALFAS) {
      let suma = 0, n = 0;
      for (const fuera of puntos) {
        const idxDentro = [], idxFuera = [];
        grupos.forEach((g, i) => (g === fuera ? idxFuera : idxDentro).push(i));
        if (idxDentro.length < 4 || !idxFuera.length) continue;
        let m;
        try {
          m = new ModeloMirada().entrenar(
            idxDentro.map((i) => muestras[i]),
            idxDentro.map((i) => objetivos[i]), alfa);
        } catch (_) { continue; }
        for (const i of idxFuera) {
          const [px, py] = m.predecir(muestras[i]);
          if (!Number.isFinite(px) || !Number.isFinite(py)) { suma += 1e6; n++; continue; }
          suma += Math.hypot(px - objetivos[i][0], py - objetivos[i][1]);
          n += 1;
        }
      }
      if (n && suma / n < mejor.error) mejor = { alfa, error: suma / n };
    }
  }

  const modelo = new ModeloMirada().entrenar(muestras, objetivos, mejor.alfa);
  modelo.alfaElegido = mejor.alfa;
  // Este es el número honesto: error sobre fijaciones que el modelo NO vio.
  // El error de entrenamiento siempre se ve bien y no dice nada.
  modelo.errorCV = mejor.error;
  return modelo;
}
