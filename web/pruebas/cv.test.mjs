/* La validación cruzada tiene que encontrar un alpha razonable sola, y sobre
 * todo NO elegir uno que colapse el modelo.
 *
 * El fallo que motiva esto: con alpha=5000 fijo, el modelo predecía
 * prácticamente el centro de la pantalla para cualquier entrada. Medido sobre
 * nueve puntos daba 354 px de sesgo con apenas 40 px de temblor, que es la
 * firma exacta de una predicción casi constante.
 */
import { ModeloMirada, entrenarConCV } from "../js/modelo.js";

const fallos = [];
const ok = (n, c, x = "") =>
  console.log(c ? `  ok    ${n}` : (fallos.push(n), `  FALLA ${n} ${x}`));

const D = 486;
function rng(s) {
  return () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
}

const W = 1536, H = 737;
/* La grilla REAL de calibración: nueve objetivos repartidos por toda la
 * pantalla, no agrupados.
 *
 * La primera versión de esta prueba generaba los objetivos a partir de rasgos
 * aleatorios, y salían amontonados cerca del centro. Con objetivos así,
 * predecir casi constante es casi óptimo, y la prueba aprobaba un alpha que en
 * uso real colapsaba el modelo. Es el mismo error que me hizo elegir 5000. */
const REJILLA = [0.1, 0.5, 0.9].flatMap(
  (fy) => [0.1, 0.5, 0.9].map((fx) => [fx * W, fy * H]));

/** Nueve fijaciones; las muestras de cada una son casi idénticas.
 *
 * Los rasgos se construyen AL REVÉS que antes: a partir del objetivo. Así es
 * la realidad -la geometría del ojo codifica hacia dónde se mira- y es lo que
 * hace que exista una señal que el modelo pueda recuperar. */
function sesion(semilla, porPunto, ruidoPx, objetivos = REJILLA) {
  const r = rng(semilla);
  const ms = [], os = [], gs = [];
  for (let p = 0; p < objetivos.length; p++) {
    const [tx, ty] = objetivos[p];
    const base = new Float64Array(D);
    // Unos pocos rasgos llevan la señal, linealmente en el objetivo; el resto
    // es geometría facial que no depende de la mirada.
    for (let j = 0; j < D; j++) base[j] = (r() - 0.5) * 2;
    base[0] = (tx / W - 0.5) * 1.6;
    base[7] = (ty / H - 0.5) * 1.6;
    base[21] = (tx / W - 0.5) * 0.9 + (ty / H - 0.5) * 0.4;
    base[250] = (ty / H - 0.5) * 1.1;
    base[483] = (tx / W - 0.5) * 0.5;
    for (let k = 0; k < porPunto; k++) {
      const m = new Float64Array(D);
      for (let j = 0; j < D; j++) m[j] = base[j] + (r() - 0.5) * 0.05;
      ms.push(m);
      os.push([tx + (r() - 0.5) * ruidoPx, ty + (r() - 0.5) * ruidoPx]);
      gs.push(p);
    }
  }
  return [ms, os, gs];
}

const [ms, os, gs] = sesion(11, 30, 30);
const modelo = entrenarConCV(ms, os, gs);
console.log(`  alpha elegido: ${modelo.alfaElegido}`);
console.log(`  error CV:      ${modelo.errorCV.toFixed(0)} px`);

ok("elige un alpha y lo reporta",
   Number.isFinite(modelo.alfaElegido) && Number.isFinite(modelo.errorCV));

/* Lo central: el modelo NO puede quedar casi constante. Si predice lo mismo
 * para entradas distintas, el error es sesgo puro y el tablero no sirve. */
const centro = [W / 2, H / 2];
let disp = 0;
const preds = [];
for (let p = 0; p < 9; p++) {
  const i = gs.indexOf(p);
  const q = modelo.predecir(ms[i]);
  preds.push(q);
  disp += Math.hypot(q[0] - centro[0], q[1] - centro[1]);
}
disp /= 9;
const rango = Math.hypot(
  Math.max(...preds.map((q) => q[0])) - Math.min(...preds.map((q) => q[0])),
  Math.max(...preds.map((q) => q[1])) - Math.min(...preds.map((q) => q[1])));
console.log(`  recorrido de las predicciones: ${rango.toFixed(0)} px`);
ok("las predicciones NO colapsan al centro", rango > 600, `${rango.toFixed(0)} px`);

/* Y la comparación directa contra el valor fijo que fallaba. */
const fijo = new ModeloMirada().entrenar(ms, os, 5000);
let errFijo = 0, errCV = 0;
const INTERMEDIOS = [0.25, 0.5, 0.75].flatMap(
  (fy) => [0.25, 0.5, 0.75].map((fx) => [fx * W, fy * H]));
const [ts, to] = sesion(99, 3, 30, INTERMEDIOS);   // puntos NUEVOS
for (let i = 0; i < ts.length; i++) {
  const a = fijo.predecir(ts[i]), b = modelo.predecir(ts[i]);
  errFijo += Math.hypot(a[0] - to[i][0], a[1] - to[i][1]);
  errCV += Math.hypot(b[0] - to[i][0], b[1] - to[i][1]);
}
errFijo /= ts.length; errCV /= ts.length;
console.log(`\n  alpha fijo 5000: ${errFijo.toFixed(0)} px`);
console.log(`  alpha por CV:    ${errCV.toFixed(0)} px`);

/* Con pocos puntos la CV no se puede hacer: tiene que degradar sin romper. */
const m2 = entrenarConCV(ms.slice(0, 60), os.slice(0, 60), gs.slice(0, 60));
ok("con 2 puntos no revienta, entrena igual", m2.listo);

console.log(fallos.length ? `\n${fallos.length} FALLAS` : "\nTodo OK");
process.exit(fallos.length ? 1 : 0);
