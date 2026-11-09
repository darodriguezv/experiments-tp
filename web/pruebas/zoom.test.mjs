/* La máquina de estados del acercamiento, y la cuenta que lo justifica. */
import { Zoom, aciertoEsperado } from "../js/zoom.js";

const fallos = [];
const ok = (n, c, x = "") =>
  console.log(c ? `  ok    ${n}` : (fallos.push(n), `  FALLA ${n} ${x}`));

// --- la máquina -------------------------------------------------------
let z = new Zoom();
ok("arranca alejado", !z.acercado && z.escala === 1);

z.acercar(800, 400, 1000);
ok("acercar sube un nivel", z.acercado && Math.abs(z.escala - 2.5) < 1e-9,
   z.escala);
const e = z.estilo();
ok("el origen ES el punto mirado",
   e.transformOrigin === "800px 400px", e.transformOrigin);
ok("la transformación escala", e.transform === "scale(2.5)", e.transform);

ok("no acerca más allá de los pasos configurados",
   z.acercar(900, 500, 1100) === false);

ok("no vence todavía", !z.venció(2000));
ok("vence pasados 4 s sin elegir", z.venció(1000 + 4100));

z.alejar();
ok("alejar vuelve al estado inicial",
   !z.acercado && z.escala === 1 && z.estilo().transform === "none");

// Dos pasos, para el caso configurable.
const z2 = new Zoom({ factor: 2, pasos: 2 });
z2.acercar(100, 100, 0); z2.acercar(100, 100, 10);
ok("dos pasos componen la escala", z2.escala === 4, z2.escala);
ok("y ahí se detiene", z2.acercar(100, 100, 20) === false);

// --- la cuenta que justifica todo esto --------------------------------
// Error sintético parecido al medido: mediana ~71 px horizontal, ~54 vertical.
function errores(n, escalaH, escalaV) {
  let s = 3;
  const r = () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  const dxs = [], dys = [];
  for (let i = 0; i < n; i++) {
    // Exponencial: cola larga, como el error real.
    dxs.push(-Math.log(1 - r()) * escalaH);
    dys.push(-Math.log(1 - r()) * escalaV);
  }
  return [dxs, dys];
}
const [dxs, dys] = errores(4000, 100, 78);

const celda = [1920 / 5, (1080 * 0.855) / 5];
const sin = aciertoEsperado(dxs, dys, celda[0], celda[1]);
const con = aciertoEsperado(dxs, dys, celda[0] * 2.5, celda[1] * 2.5);
console.log(`\n  sin acercar  ${(sin * 100).toFixed(0)}%`);
console.log(`  a 2.5x       ${(con * 100).toFixed(0)}%`);
ok("acercar mejora el acierto de forma sustancial", con - sin > 0.25,
   `${(sin * 100).toFixed(0)}% -> ${(con * 100).toFixed(0)}%`);

// Y el costo total, que es lo que decide: cada fallo cuesta borrar y rehacer.
const esp = (p) => (2 - p) / p;
const costoPlano = 1.2 * esp(sin);
const costoZoom = (0.45 + 1.2) * esp(con);
console.log(`  costo plano  ${costoPlano.toFixed(2)} s/palabra`);
console.log(`  costo zoom   ${costoZoom.toFixed(2)} s/palabra`);
ok("y aun con la permanencia extra sale más barato", costoZoom < costoPlano,
   `${costoZoom.toFixed(2)} vs ${costoPlano.toFixed(2)}`);

console.log(fallos.length ? `\n${fallos.length} FALLAS` : "\nTodo OK");
process.exit(fallos.length ? 1 : 0);
