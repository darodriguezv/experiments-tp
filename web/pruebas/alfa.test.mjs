/* Barre la regularización con muestras realistas.
 *
 * Lo que la primera versión de esta prueba no capturaba: las 14 muestras de un
 * mismo punto de calibración son cuadros consecutivos de un ojo quieto, o sea
 * casi idénticas. El tamaño de muestra EFECTIVO no es 126, es 9. Con 486
 * rasgos y 9 observaciones distintas, una regularización débil memoriza la
 * calibración y falla en uso real -que es exactamente el síntoma reportado:
 * calibra "bien" y después anda pésimo.
 */
import { ModeloMirada } from "../js/modelo.js";

function rng(s) {
  return () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
}
const D = 486;
const r = rng(11);

const pesosX = new Float64Array(D), pesosY = new Float64Array(D);
for (let j = 0; j < 80; j++) {
  pesosX[j] = (r() - 0.5) * 50;
  pesosY[j] = (r() - 0.5) * 50;
}

/** 9 fijaciones; cada una aporta `porPunto` cuadros casi iguales. */
function sesion(porPunto, jitterRasgo, ruidoPx) {
  const ms = [], os = [];
  for (let p = 0; p < 9; p++) {
    const base = new Float64Array(D);
    for (let j = 0; j < D; j++) base[j] = (r() - 0.5) * 2;
    let x = 960, y = 540;
    for (let j = 0; j < D; j++) { x += base[j] * pesosX[j]; y += base[j] * pesosY[j]; }
    for (let k = 0; k < porPunto; k++) {
      const m = new Float64Array(D);
      for (let j = 0; j < D; j++) m[j] = base[j] + (r() - 0.5) * jitterRasgo;
      ms.push(m);
      os.push([x + (r() - 0.5) * ruidoPx, y + (r() - 0.5) * ruidoPx]);
    }
  }
  return [ms, os];
}

const [mx, ox] = sesion(14, 0.04, 30);
const [tx, to] = sesion(3, 0.04, 30);   // fijaciones NUEVAS: uso real

console.log("9 fijaciones x 14 cuadros = 126 muestras, 486 rasgos");
console.log("(las 14 de cada punto son casi identicas: n efectivo = 9)\n");
console.log("    alpha   entrenamiento   fijaciones nuevas");
console.log("  " + "-".repeat(48));
let mejor = [Infinity, null];
for (const a of [1, 30, 100, 300, 1000, 3000, 10000, 40000, 150000]) {
  const m = new ModeloMirada().entrenar(mx, ox, a);
  const ent = m.errorEntrenamiento(mx, ox);
  let t = 0;
  for (let i = 0; i < tx.length; i++) {
    const [px, py] = m.predecir(tx[i]);
    t += Math.hypot(px - to[i][0], py - to[i][1]);
  }
  t /= tx.length;
  if (t < mejor[0]) mejor = [t, a];
  console.log(`  ${String(a).padStart(7)}   ${ent.toFixed(1).padStart(10)} px`
            + `   ${t.toFixed(0).padStart(13)} px`);
}
console.log(`\n  mejor: alpha=${mejor[1]} con ${mejor[0].toFixed(0)} px`);
console.log("  el valor que tiene el codigo hoy es 1.0");
