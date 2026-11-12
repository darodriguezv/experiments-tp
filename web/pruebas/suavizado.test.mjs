/* El filtro de la mirada, contra el fallo medido del anterior.
 *
 * La cadena vieja -Kalman con velocidad, media exponencial, mediana- se midió
 * con un escalón de 600 px a 30 cuadros por segundo:
 *
 *   1.07 s   todavía le faltan 69 px
 *   2.13 s   se pasó 143 px de largo
 *   4.07 s   recién queda dentro de 20 px
 *
 * La prueba de precisión espera 1 s y mide entre 1.0 y 2.2 s: medía justo
 * mientras el filtro se pasaba, y lo anotaba como sesgo de la mirada. Con
 * saltos de 384 px entre columnas eso predice desvíos de -94, +47, +47 px por
 * columna; lo medido en la segunda sesión de WebGazer fue -61, +36, +54.
 *
 * Lo que se fija acá: que el filtro llegue rápido, que NO deje error
 * sistemático en esa ventana, y que no invente saltos con señal ruidosa.
 */
import { FiltroMirada } from "../js/suavizado.js";

const fallos = [];
const ok = (n, c, x = "") =>
  console.log(c ? `  ok    ${n}` : (fallos.push(n), `  FALLA ${n} ${x}`));

let semilla = 777;
const azar = () => (semilla = (semilla * 1103515245 + 12345) & 0x7fffffff)
                 / 0x7fffffff;
const gauss = () => Math.sqrt(-2 * Math.log(azar() + 1e-12))
                  * Math.cos(2 * Math.PI * azar());

/** Fija la mirada, salta `salto` px y devuelve qué hizo el filtro. */
function escalon(fps, ruido, salto, repeticiones = 30) {
  let llega = 0, sesgo = 0, falsos = 0, temblor = 0;
  for (let r = 0; r < repeticiones; r++) {
    const f = new FiltroMirada();
    const paso = 1000 / fps;
    let t = 0, o;
    const quieto = [];
    for (let i = 0; i < fps * 6; i++) {
      t += paso;
      o = f.paso(200 + ruido * gauss(), 300 + ruido * gauss(), t);
      if (i > fps * 2) quieto.push(o);
    }
    falsos += f.saltos;
    const med = (a) => a.sort((p, q) => p - q)[a.length >> 1];
    const mx = med(quieto.map((p) => p[0])), my = med(quieto.map((p) => p[1]));
    temblor += med(quieto.map((p) => Math.hypot(p[0] - mx, p[1] - my)));

    const t0 = t;
    let primera = null;
    const enVentana = [];
    for (let i = 0; i < fps * 3; i++) {
      t += paso;
      o = f.paso(200 + salto + ruido * gauss(), 300 + ruido * gauss(), t);
      const e = o[0] - (200 + salto);
      if (primera === null && Math.abs(e) < 50) primera = t - t0;
      if (t - t0 >= 1000 && t - t0 <= 2200) enVentana.push(e);
    }
    llega += primera ?? 3000;
    sesgo += enVentana.reduce((p, q) => p + q, 0) / enVentana.length;
  }
  return { llega: llega / repeticiones, sesgo: sesgo / repeticiones,
           falsos: falsos / repeticiones, temblor: temblor / repeticiones };
}

console.log("  fps ruido salto   llega   sesgo en 1.0-2.2 s   temblor");
for (const fps of [30, 15]) {
  for (const ruido of [30, 60, 100]) {
    for (const salto of [600, 300, 126]) {
      const r = escalon(fps, ruido, salto);
      console.log(`  ${String(fps).padStart(3)} ${String(ruido).padStart(5)}`
        + ` ${String(salto).padStart(5)} ${r.llega.toFixed(0).padStart(6)} ms`
        + ` ${r.sesgo.toFixed(0).padStart(10)} px ${r.temblor.toFixed(0).padStart(12)} px`);
      const caso = `${fps} fps, ruido ${ruido}, salto ${salto}`;
      ok(`llega en menos de 0.7 s (${caso})`, r.llega < 700,
         `${r.llega.toFixed(0)} ms`);
      ok(`no deja error sistemático donde mide la prueba (${caso})`,
         Math.abs(r.sesgo) < 8, `${r.sesgo.toFixed(1)} px`);
      ok(`no inventa saltos mirando fijo (${caso})`, r.falsos < 0.2,
         r.falsos.toFixed(2));
    }
  }
}

/* Sin ruido se ve la forma pura de la respuesta: tiene que llegar y quedarse,
 * sin pasarse ni un píxel. Una mediana no tiene impulso. */
{
  const f = new FiltroMirada();
  let t = 0, maximo = -Infinity;
  for (let i = 0; i < 90; i++) { t += 33; f.paso(200, 300, t); }
  for (let i = 0; i < 150; i++) {
    t += 33;
    maximo = Math.max(maximo, f.paso(800, 300, t)[0]);
  }
  ok("sin ruido no se pasa del objetivo ni un píxel", maximo === 800,
     String(maximo));
}

/* Robustez de entrada. */
{
  const f = new FiltroMirada();
  ok("antes de la primera muestra no inventa una posición",
     f.paso(NaN, NaN, 0) === null);
  f.paso(100, 200, 10);
  const o = f.paso(NaN, 5, 20);
  ok("una medición rota devuelve la última salida buena",
     o && o[0] === 100 && o[1] === 200, String(o));

  // El lazo corre más rápido que el estimador: la misma predicción llega
  // repetida, y contarla de nuevo haría creer que hay más evidencia.
  const g = new FiltroMirada();
  for (let i = 0; i < 20; i++) g.paso(500, 500, i * 16);
  ok("una predicción repetida no se cuenta dos veces", g.buf.length === 1,
     String(g.buf.length));

  // Después de un parpadeo largo lo viejo ya caducó: se arranca de lo nuevo.
  const h = new FiltroMirada();
  for (let i = 0; i < 20; i++) h.paso(100 + i % 3, 100, i * 33);
  const tras = h.paso(900, 600, 5000);
  ok("tras una pausa larga no arrastra la posición vieja",
     tras[0] === 900 && tras[1] === 600, String(tras));
}

console.log(fallos.length ? `\n${fallos.length} FALLAS` : "\nTodo OK");
process.exit(fallos.length ? 1 : 0);
