/* La puerta tiene que rechazar un modelo colapsado y dejar pasar uno sano.
 *
 * El fallo real: dos calibraciones del mismo código dieron 123 px y 316 px de
 * error. La mala tenía 318 px de sesgo con 15 de temblor, que es la firma de
 * una predicción casi constante — el modelo no sigue la mirada.
 *
 * No se usa el error de validación cruzada para esto. Sirve para ELEGIR alpha
 * -es una comparación relativa- pero como medida absoluta es pesimista por
 * construcción: al sacar un punto de una rejilla de 3x3 se saca la única
 * información sobre esa región y el modelo tiene que extrapolar. Medido con
 * datos sin un solo defecto, el LOO daba 282 px mientras el error real era 0.
 */
import { ModeloMirada, entrenarConCV } from "../js/modelo.js";

const D = 486, W = 1536, H = 737;
const fallos = [];
const ok = (n, c, x = "") =>
  console.log(c ? `  ok    ${n}` : (fallos.push(n), `  FALLA ${n} ${x}`));

let s = 5;
const r = () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
const REJILLA = [0.1, 0.5, 0.9].flatMap(
  (fy) => [0.1, 0.5, 0.9].map((fx) => [fx * W, fy * H]));

const ms = [], os = [], gs = [];
for (let p = 0; p < 9; p++) {
  const [tx, ty] = REJILLA[p];
  for (let k = 0; k < 30; k++) {
    const m = new Float64Array(D);
    for (let j = 0; j < D; j++) m[j] = (r() - 0.5) * 2;
    m[0] = (tx / W - 0.5) * 1.6;
    m[7] = (ty / H - 0.5) * 1.6;
    m[250] = (ty / H - 0.5) * 1.1;
    m[483] = (tx / W - 0.5) * 0.5;
    ms.push(m); os.push([tx, ty]); gs.push(p);
  }
}

const sano = entrenarConCV(ms, os, gs);
console.log(`  recorrido sano:       ${sano.recorrido.toFixed(2)}`);
ok("un modelo sano recorre casi todo", sano.recorrido > 0.8, sano.recorrido);

/* El alpha fijo que estaba antes, para comprobar que la puerta lo atrapa. */
const colapsado = new ModeloMirada().entrenar(ms, os, 5000);
const pr = [], ob = [];
for (let p = 0; p < 9; p++) {
  const i = gs.indexOf(p);
  pr.push(colapsado.predecir(ms[i]));
  ob.push(os[i]);
}
const amp = (a, j) =>
  Math.max(...a.map((q) => q[j])) - Math.min(...a.map((q) => q[j]));
const rec = Math.min(amp(pr, 0) / amp(ob, 0), amp(pr, 1) / amp(ob, 1));
console.log(`  recorrido con a=5000: ${rec.toFixed(2)}`);
ok("el alpha que fallaba queda por debajo del umbral", rec < 0.45, rec);
ok("y la puerta los separa", sano.recorrido > 0.45 && rec < 0.45);

console.log(fallos.length ? `\n${fallos.length} FALLAS` : "\nTodo OK");
process.exit(fallos.length ? 1 : 0);
