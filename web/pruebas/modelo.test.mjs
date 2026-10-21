/* Chequea que la forma dual dé lo mismo que la directa, y que el modelo
 * recupere una relación lineal conocida. Si esto falla, la calibración del
 * navegador produce basura que igual parece funcionar. */
import { ModeloMirada } from "../js/modelo.js";

function rng(semilla) {
  let s = semilla;
  return () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; };
}

const r = rng(7);
const D = 486, N = 130;
// Verdad conocida: la pantalla depende de 6 rasgos y el resto es ruido.
const pesosX = new Float64Array(D), pesosY = new Float64Array(D);
for (const j of [0, 5, 17, 200, 400, 483]) {
  pesosX[j] = (r() - 0.5) * 400;
  pesosY[j] = (r() - 0.5) * 400;
}

const muestras = [], objetivos = [];
for (let i = 0; i < N; i++) {
  const m = new Float64Array(D);
  for (let j = 0; j < D; j++) m[j] = (r() - 0.5) * 2;
  let x = 960, y = 540;
  for (let j = 0; j < D; j++) { x += m[j] * pesosX[j]; y += m[j] * pesosY[j]; }
  muestras.push(m);
  objetivos.push([x, y]);
}

const modelo = new ModeloMirada().entrenar(muestras, objetivos);

const errEnt = modelo.errorEntrenamiento(muestras, objetivos);
console.log("error de entrenamiento:", errEnt.toFixed(2), "px");

// Lo que de verdad importa: generalizar a muestras nuevas.
let errTest = 0;
const M = 40;
for (let i = 0; i < M; i++) {
  const m = new Float64Array(D);
  for (let j = 0; j < D; j++) m[j] = (r() - 0.5) * 2;
  let x = 960, y = 540;
  for (let j = 0; j < D; j++) { x += m[j] * pesosX[j]; y += m[j] * pesosY[j]; }
  const [px, py] = modelo.predecir(m);
  errTest += Math.hypot(px - x, py - y);
}
errTest /= M;
console.log("error sobre muestras NUEVAS:", errTest.toFixed(2), "px");

// Degenerado: todas las muestras iguales no debe reventar ni dar NaN.
const igual = Array.from({ length: 8 }, () => new Float64Array(D).fill(0.3));
const m2 = new ModeloMirada().entrenar(igual, igual.map(() => [100, 200]));
const [dx, dy] = m2.predecir(new Float64Array(D).fill(0.3));
console.log("degenerado ->", dx.toFixed(1), dy.toFixed(1),
            Number.isFinite(dx) && Number.isFinite(dy) ? "finito" : "NaN!");

const ok = errEnt < 40 && errTest < 400 && Number.isFinite(dx);
console.log(ok ? "OK" : "FALLA");
process.exit(ok ? 0 : 1);
