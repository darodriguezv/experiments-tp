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

import { ajusteEje } from "./correccion.js";

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
      // Por muestra y por eje: una celda es más ancha que alta, así que lo que
      // decide si se acierta no es el error radial sino cada eje contra su
      // propio semieje.
      dxs: this.buffer.map((p) => Math.abs(p[0] - tx)),
      dys: this.buffer.map((p) => Math.abs(p[1] - ty)),
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

/** Qué se puede esperar del tablero con este error, por tamaño de grilla.
 *
 * Se evalúa cada eje contra su propio semieje -|dx| < ancho/2 Y |dy| < alto/2-
 * y no el error radial contra el semieje más corto, que es lo que hacía la
 * primera versión. Para una celda de 307x126 eso exigía caer dentro de 63 px
 * en cualquier dirección, cuando horizontalmente hay 153 px de margen: la
 * tabla salía mucho más pesimista que la realidad.
 */
export function aciertoPorGrilla(prueba, factorZoom = 1) {
  const validos = prueba.porPunto.filter((p) => p.n > 0);
  const dxs = validos.flatMap((p) => p.dxs || []);
  const dys = validos.flatMap((p) => p.dys || []);
  const filas = [];
  for (const [cols, fils] of [[5, 5], [4, 4], [3, 3], [3, 2], [2, 2]]) {
    const w = (innerWidth / cols) * factorZoom;
    const h = ((innerHeight * 0.855) / fils) * factorZoom;
    let dentro = 0;
    for (let i = 0; i < dxs.length; i++) {
      if (dxs[i] < w / 2 && dys[i] < h / 2) dentro += 1;
    }
    filas.push({ grilla: `${cols}x${fils}`,
                 celda: `${Math.round(w)}x${Math.round(h)}`,
                 acierto: dxs.length ? dentro / dxs.length : 0 });
  }
  return filas;
}

/**
 * Cuánto hay que acercar para que una grilla llegue a cierto acierto.
 *
 * Es la pregunta de diseño que importa: el error ya está medido y no se va a
 * mover, así que lo único ajustable es cuánto crece la celda.
 */
export function zoomNecesario(prueba, cols, fils, objetivo = 0.9) {
  const validos = prueba.porPunto.filter((p) => p.n > 0);
  const dxs = validos.flatMap((p) => p.dxs || []);
  const dys = validos.flatMap((p) => p.dys || []);
  if (!dxs.length) return NaN;
  for (let k = 1; k <= 8; k += 0.25) {
    const w = (innerWidth / cols) * k, h = ((innerHeight * 0.855) / fils) * k;
    let dentro = 0;
    for (let i = 0; i < dxs.length; i++) {
      if (dxs[i] < w / 2 && dys[i] < h / 2) dentro += 1;
    }
    if (dentro / dxs.length >= objetivo) return k;
  }
  return NaN;   // ni acercando 8x alcanza
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

/** La recta que mejor explica el desvío, por eje, y lo que quedaría al
 *  corregirla.
 *
 * `residuo` se mide sobre los MISMOS nueve puntos con los que se ajusta la
 * recta, así que es optimista. Con cuatro parámetros sobre nueve puntos lo es
 * poco, pero es un límite inferior y no una promesa: lo que de verdad queda
 * después de corregir lo dice la siguiente medición, con la corrección puesta.
 */
export function ajusteAfin(prueba) {
  const v = prueba.porPunto.filter((p) => p.n > 0);
  if (v.length < 3) return null;
  const ex = ajusteEje(v.map((p) => p.tx), v.map((p) => p.tx + p.dx));
  const ey = ajusteEje(v.map((p) => p.ty), v.map((p) => p.ty + p.dy));
  let s = 0;
  for (const p of v) {
    s += Math.hypot((p.tx + p.dx - ex.b) / ex.a - p.tx,
                    (p.ty + p.dy - ey.b) / ey.a - p.ty);
  }
  return { ex, ey, residuo: s / v.length };
}

/** Cuánto se aparta el desvío de una RECTA, por eje, en píxeles.
 *
 * Es el número que decide si vale la pena corregir más o si hay que arreglar
 * el modelo. Medido sobre WebGazer ya corregido, el desvío horizontal por
 * columna daba -61, +36, +54 px: el salto de izquierda a centro es +97 y el
 * de centro a derecha +18. Eso no es una recta con ruido, es una curva, y una
 * corrección lineal tiene un piso ahí por más puntos que se midan.
 *
 * Se mide sobre los PROMEDIOS por columna y por fila, no por punto: el ruido
 * de un punto suelto se promedia y lo que queda es la forma.
 */
export function curvatura(prueba) {
  const v = prueba.porPunto.filter((p) => p.n > 0);
  const eje = (clave, desvio) => {
    const grupos = new Map();
    for (const p of v) {
      const k = Math.round(p[clave]);
      if (!grupos.has(k)) grupos.set(k, []);
      grupos.get(k).push(p[clave] + p[desvio]);
    }
    if (grupos.size < 3) return 0;   // con dos grupos la recta pasa exacta
    const t = [...grupos.keys()];
    const p = t.map((k) => {
      const g = grupos.get(k);
      return g.reduce((a, b) => a + b, 0) / g.length;
    });
    const e = ajusteEje(t, p);
    return Math.max(...t.map((tk, i) => Math.abs(e.a * tk + e.b - p[i])));
  };
  return { x: eje("tx", "dx"), y: eje("ty", "dy") };
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

  /* El desvío de CADA punto, con signo, y la recta que lo explica. Es lo que
   * distingue TRES fallos que el promedio del sesgo confunde:
   *
   * - todos corridos en la MISMA dirección y en la misma medida: corrimiento
   *   uniforme, y una constante lo arregla;
   * - el desvío CRECE con la coordenada: la ganancia está mal, y ninguna
   *   constante lo arregla, porque la que centra un extremo descentra el otro;
   * - corridos HACIA el centro, más cuanto más lejos: el modelo colapsó.
   *
   * La primera versión de esto solo miraba si había una dirección común, y por
   * eso informó "desplazamiento uniforme" sobre una medición cuyo desvío
   * horizontal iba de +18 px a +309 px según la columna. Eran dos problemas
   * distintos, con dos arreglos distintos, leídos como el mismo. */
  L.push("");
  L.push("desvío por punto (x, y), en px:");
  for (const p of r.por_punto) {
    const sx = p.dx >= 0 ? "+" : "";
    const sy = p.dy >= 0 ? "+" : "";
    L.push(`  (${String(p.x).padStart(4)},${String(p.y).padStart(4)})`
         + `  ${sx}${p.dx}, ${sy}${p.dy}`);
  }
  const mdx = r.por_punto.reduce((a, p) => a + p.dx, 0) / r.por_punto.length;
  const mdy = r.por_punto.reduce((a, p) => a + p.dy, 0) / r.por_punto.length;
  L.push("");
  L.push(`desvío PROMEDIO: ${mdx >= 0 ? "+" : ""}${mdx.toFixed(0)}, `
       + `${mdy >= 0 ? "+" : ""}${mdy.toFixed(0)} px`);

  const af = ajusteAfin(prueba);
  if (af) {
    L.push("");
    L.push(`ganancia       x ${af.ex.a.toFixed(2)}   y ${af.ey.a.toFixed(2)}`
         + `   (1.00 sería correcta)`);
    L.push(`sesgo si se corrigiera la ganancia: ${af.residuo.toFixed(0)} px`
         + `   (ahora ${r.sesgo.toFixed(0)} px)`);
    const cv = curvatura(prueba);
    L.push(`curvatura      x ${cv.x.toFixed(0)} px   y ${cv.y.toFixed(0)} px`
         + `   (cuánto se aparta de una recta)`);
    const torcida = Math.abs(af.ex.a - 1) > 0.15 || Math.abs(af.ey.a - 1) > 0.15;
    /* Lo que queda después de restar el corrimiento promedio. Una medición
     * real dio un promedio de (-61, -53) px con desvíos por punto que iban de
     * +98 a -155 px en la misma fila, y el informe la llamó "uniforme" porque
     * solo miraba el tamaño del promedio. Uniforme es que restarlo alcance. */
    const resto = r.por_punto.reduce((a, p) =>
      a + Math.hypot(p.dx - mdx, p.dy - mdy), 0) / r.por_punto.length;
    L.push(torcida
      ? "La GANANCIA está mal: el desvío crece con la coordenada. Restar una "
        + "constante no alcanza, hace falta corregir la escala."
      : Math.hypot(mdx, mdy) > 0.6 * r.sesgo
        ? (resto < 0.4 * r.sesgo
          ? "Corrimiento UNIFORME: los nueve están corridos igual, y una "
            + "constante lo arregla."
          : `Hay un corrimiento común, pero quitarlo deja ${resto.toFixed(0)} px: `
            + "el resto cambia de un punto al de al lado y ninguna corrección "
            + "simple lo arregla.")
        : "Ni corrimiento parejo ni ganancia torcida: el error cambia según "
          + "la zona, sin una regla simple.");
    /* Si el desvío se dobla, agregarle parámetros a la corrección lineal
     * no lo arregla: lo que hay que cambiar es el modelo que lo produce. */
    const piso = Math.hypot(cv.x, cv.y);
    if (piso > 25) {
      L.push(`El desvío NO es una recta sino una curva: una corrección `
           + `lineal tiene un piso de unos ${piso.toFixed(0)} px por más `
           + `puntos que se midan.`);
    }
  }
  return L.join("\n");
}
