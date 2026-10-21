/* Que todos los módulos parseen, exporten lo que los otros importan, y que el
 * tablero tenga el mismo vocabulario que board_layout.py. Un import mal
 * escrito no falla hasta que alguien abre la página. */
import * as rasgos from "../js/rasgos.js";
import * as modelo from "../js/modelo.js";
import * as suav from "../js/suavizado.js";
import * as perm from "../js/permanencia.js";
import * as tab from "../js/tablero.js";
import * as prompt from "../js/prompt.js";

const fallos = [];
const ok = (n, c, x = "") => console.log(c ? `  ok    ${n}`
  : (fallos.push(n), `  FALLA ${n} ${x}`));

ok("rasgos: 161 puntos -> 486 dimensiones",
   rasgos.INDICES.length === 161 && rasgos.RASGOS_DIM === 486,
   `${rasgos.INDICES.length} / ${rasgos.RASGOS_DIM}`);
ok("rasgos: ningún índice fuera del rango de MediaPipe",
   Math.max(...rasgos.INDICES) < 478, `max ${Math.max(...rasgos.INDICES)}`);
ok("rasgos: tolera landmarks faltantes sin reventar",
   rasgos.extraer(null) === null && rasgos.extraer([]) === null);

ok("modelo: exporta ModeloMirada", typeof modelo.ModeloMirada === "function");
ok("suavizado: exporta Suavizador y VentanaMediana",
   typeof suav.Suavizador === "function" && typeof suav.VentanaMediana === "function");
ok("permanencia: 1200 ms y 600 de refractario",
   perm.DWELL_MS === 1200 && perm.REFRACTARIO_MS === 600);

// El vocabulario tiene que ser EL MISMO que board_layout.py, o el costo en
// permanencias que mide la tesis no es el de esto. Se compara contra un
// volcado del lado de Python y no contra números escritos a mano acá: la
// primera version de este chequeo esperaba 36 celdas en el abecedario y el
// numero correcto es 35 -la ultima fila son cinco controles, no seis- asi
// que el chequeo fallaba por estar mal escrito, no por una divergencia.
import { readFileSync } from "node:fs";
const fixture = JSON.parse(readFileSync(
  new URL("./tablero.fixture.json", import.meta.url), "utf-8"));

for (const [pagina, filasPy] of Object.entries(fixture)) {
  const filasJs = tab.PAGINAS[pagina];
  ok(`tablero: existe la pagina ${pagina}`, !!filasJs);
  if (!filasJs) continue;
  const planoPy = filasPy.flat();
  const planoJs = filasJs.flat().map((c) =>
    Array.isArray(c) ? [c[0], c[1]] : [c, "w:" + c]);
  ok(`tablero: ${pagina} tiene ${planoPy.length} celdas`,
     planoJs.length === planoPy.length,
     `js=${planoJs.length} py=${planoPy.length}`);
  const dif = planoPy.filter((c, i) =>
    !planoJs[i] || planoJs[i][0] !== c[0] || planoJs[i][1] !== c[1]);
  ok(`tablero: ${pagina} coincide celda por celda con Python`,
     dif.length === 0, JSON.stringify(dif.slice(0, 3)));
}

ok("prompt: viene de prompts.py y pide JSON",
   prompt.PROMPT_SISTEMA.includes("candidatos")
   && prompt.PROMPT_SISTEMA.includes("ELA")
   && prompt.construirUsuario("AGUA") === "entrada: AGUA");

// El suavizador no debe devolver NaN ante una medición rota.
const s = new suav.Suavizador();
s.paso(100, 200, 0.03);
const [nx] = s.paso(NaN, NaN, 0.03);
ok("suavizado: una medición NaN no envenena el estado", Number.isFinite(nx), nx);

const v = new suav.VentanaMediana(300);
v.empujar(0, 0, 0); v.empujar(100, 100, 100); v.empujar(50, 50, 200);
ok("ventana: mediana, no media", v.mediana()[0] === 50, v.mediana());
v.empujar(10, 10, 900);
ok("ventana: descarta por tiempo", v.buf.length === 1, v.buf.length);

console.log(fallos.length ? `\n${fallos.length} FALLAS` : "\nTodo OK");
process.exit(fallos.length ? 1 : 0);
