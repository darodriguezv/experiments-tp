/* Prueba de precisión: cuánto error tiene la mirada, en píxeles.
 *
 * Es el equivalente de `Eyetrax/main.py accuracy` dentro del navegador, y hace
 * falta porque el puerto a JavaScript nunca se midió. Sin un número, "funciona
 * mal" no se distingue de tres problemas con tres arreglos distintos:
 *
 * - **error grande y sistemático** -> el mapeo está mal: calibración o rasgos;
 * - **error chico pero la grilla falla** -> las celdas son muy chicas para el
 *   error que hay, y hay que achicar la grilla, no el estimador;
 * - **temblor grande** -> el suavizado no alcanza.
 *
 * La descomposición en sesgo y temblor es la que resolvió esto del lado de
 * Python: sobre diez sesiones reales, las diez estaban dominadas por el sesgo
 * y el temblor era chico, así que un estimador más fino habría atacado el
 * término equivocado.
 *
 * ## Los puntos NO son los de la calibración
 *
 * Medir sobre los mismos nueve puntos con los que se ajustó el modelo informa
 * cuánto memorizó, no cuánto sirve. Esta grilla está desplazada a propósito.
 */

export const PUNTOS_PRUEBA = [
  [0.25, 0.20], [0.50, 0.20], [0.75, 0.20],
  [0.25, 0.50], [0.50, 0.50], [0.75, 0.50],
  [0.25, 0.80], [0.50, 0.80], [0.75, 0.80],
];

export const MS_ASENTAR = 1000;
/* Se descarta este tiempo al principio de cada punto. La mirada tarda en
 * llegar, y medido del lado nativo son ~1.7 s en el primero: contar la sacada
 * como error infla la media, el p95 y sobre todo el máximo. */
export const MS_MEDIR = 1200;

export class PruebaPrecision {
  constructor() {
    this.i = 0;
    this.fase = "asentar";
    this.desde = 0;
    this.buffer = [];
    this.porPunto = [];
    this.terminada = false;
  }

  get objetivo() {
    const p = PUNTOS_PRUEBA[this.i];
    return p ? [p[0] * innerWidth, p[1] * innerHeight] : null;
  }

  /** @returns {{avance:number, cambio:boolean}} */
  paso(t, muestra) {
    if (this.terminada) return { avance: 1, cambio: false };
    if (!this.desde) this.desde = t;
    const pasado = t - this.desde;

    if (this.fase === "asentar") {
      if (pasado > MS_ASENTAR) { this.fase = "medir"; this.desde = t; }
      return { avance: 0, cambio: false };
    }

    if (muestra && muestra.ok) this.buffer.push([muestra.x, muestra.y]);

    if (pasado > MS_MEDIR) {
      this._cerrarPunto();
      this.i += 1;
      this.buffer = [];
      this.fase = "asentar";
      this.desde = t;
      if (this.i >= PUNTOS_PRUEBA.length) this.terminada = true;
      return { avance: 1, cambio: true };
    }
    return { avance: Math.min(1, pasado / MS_MEDIR), cambio: false };
  }

  _cerrarPunto() {
    const [tx, ty] = [PUNTOS_PRUEBA[this.i][0] * innerWidth,
                      PUNTOS_PRUEBA[this.i][1] * innerHeight];
    if (!this.buffer.length) {
      this.porPunto.push({ tx, ty, n: 0, sesgo: NaN, temblor: NaN, err: [] });
      return;
    }
    const xs = this.buffer.map((p) => p[0]);
    const ys = this.buffer.map((p) => p[1]);
    // Mediana y no media: si la mirada todavía viaja hacia el punto cuando
    // empieza el registro, una media arrastra el centro hacia la fijación
    // anterior. Eso es una sacada, no error de estimación.
    const mx = mediana(xs), my = mediana(ys);
    this.porPunto.push({
      tx, ty, n: this.buffer.length,
      sesgo: Math.hypot(mx - tx, my - ty),
      temblor: mediana(this.buffer.map((p) => Math.hypot(p[0] - mx, p[1] - my))),
      err: this.buffer.map((p) => Math.hypot(p[0] - tx, p[1] - ty)),
      dx: mx - tx, dy: my - ty,
    });
  }

  resumen() {
    const validos = this.porPunto.filter((p) => p.n > 0);
    if (!validos.length) return null;
    const todos = validos.flatMap((p) => p.err);
    const sesgos = validos.map((p) => p.sesgo);
    const temblores = validos.map((p) => p.temblor);
    return {
      puntos: validos.length,
      muestras: todos.length,
      media: todos.reduce((a, b) => a + b, 0) / todos.length,
      mediana: mediana(todos),
      p95: percentil(todos, 95),
      sesgo: sesgos.reduce((a, b) => a + b, 0) / sesgos.length,
      temblor: temblores.reduce((a, b) => a + b, 0) / temblores.length,
      // Desvío del sesgo entre puntos: si es grande, el error no es un
      // corrimiento parejo sino que cambia según la zona de la pantalla.
      dispersion: desvio(sesgos),
      por_punto: validos.map((p) => ({
        x: Math.round(p.tx), y: Math.round(p.ty),
        sesgo: Math.round(p.sesgo), temblor: Math.round(p.temblor),
        dx: Math.round(p.dx), dy: Math.round(p.dy),
      })),
    };
  }
}

/** Qué se puede esperar del tablero con este error, por tamaño de grilla. */
export function aciertoPorGrilla(prueba) {
  const validos = prueba.porPunto.filter((p) => p.n > 0);
  const dxs = [], dys = [];
  for (const p of validos) {
    for (const [i] of p.err.entries()) void i;
  }
  // Se usan los desvíos crudos de cada muestra respecto del objetivo, no los
  // sesgos medianos: lo que decide si una celda se acierta es dónde cae CADA
  // mirada, no dónde cae el promedio.
  for (const p of validos) {
    dxs.push(Math.abs(p.dx)); dys.push(Math.abs(p.dy));
  }
  const filas = [];
  for (const [cols, fils] of [[5, 5], [4, 4], [3, 3], [3, 2], [2, 2]]) {
    const w = innerWidth / cols, h = (innerHeight * 0.855) / fils;
    let dentro = 0, total = 0;
    for (const p of validos) {
      for (const e of p.err) {
        total += 1;
        // Aproximación isotrópica: se usa el error radial contra el semieje
        // más corto, que es el que ata.
        if (e < Math.min(w, h) / 2) dentro += 1;
      }
    }
    filas.push({ grilla: `${cols}x${fils}`,
                 celda: `${Math.round(w)}x${Math.round(h)}`,
                 acierto: total ? dentro / total : 0 });
  }
  return filas;
}

function mediana(a) {
  const s = [...a].sort((x, y) => x - y);
  if (!s.length) return NaN;
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function percentil(a, p) {
  const s = [...a].sort((x, y) => x - y);
  if (!s.length) return NaN;
  return s[Math.min(s.length - 1, Math.floor((p / 100) * s.length))];
}

function desvio(a) {
  if (a.length < 2) return 0;
  const m = a.reduce((x, y) => x + y, 0) / a.length;
  return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length);
}

/** Lectura en texto, para copiar y pegar en un reporte. */
export function informe(prueba) {
  const r = prueba.resumen();
  if (!r) return "No se registró ninguna muestra válida.";
  const L = [];
  L.push(`${r.muestras} muestras en ${r.puntos} puntos`);
  L.push("");
  L.push(`error medio     ${r.media.toFixed(0)} px   (${(r.media / 40).toFixed(1)}°)`);
  L.push(`mediana         ${r.mediana.toFixed(0)} px`);
  L.push(`p95             ${r.p95.toFixed(0)} px`);
  L.push("");
  L.push(`sesgo           ${r.sesgo.toFixed(0)} px   (dónde cae la mediana)`);
  L.push(`temblor         ${r.temblor.toFixed(0)} px   (cuánto se dispersa)`);
  L.push(`dispersión      ${r.dispersion.toFixed(0)} px   (variación entre zonas)`);
  L.push("");
  L.push(r.sesgo > 3 * r.temblor
    ? "Domina el SESGO: la mirada cae en un lugar consistentemente equivocado."
    : r.temblor > r.sesgo
      ? "Domina el TEMBLOR: la mirada es ruidosa. El suavizado no alcanza."
      : "Sesgo y temblor son comparables.");
  return L.join("\n");
}
