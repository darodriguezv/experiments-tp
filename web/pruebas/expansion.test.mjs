/* El filtro de la página tiene que decidir EXACTAMENTE lo mismo que el de
 * Python, con el que se midieron todas las tablas.
 *
 * Los casos los genera `web/generar_fixture_filtro.py`: todas las respuestas
 * reales guardadas de quince modelos, más casos construidos para pegarle a
 * cada regla en su borde. Una sola diferencia es un error del puerto.
 */
import { readFileSync } from "node:fs";
import {
  normalizar, tokens, f1Tokens, filtrar, parsear, conLiteral, procesar,
  NEGACION, PARADA, SUSTANTIVOS_TABLERO, CORTESIA_SIN_NEGACION,
} from "../js/expansion.js";

const fallos = [];
const ok = (n, c, x = "") =>
  console.log(c ? `  ok    ${n}` : (fallos.push(n), `  FALLA ${n} ${x}`));
const igual = (a, b) => JSON.stringify(a) === JSON.stringify(b);

const fx = JSON.parse(readFileSync(
  new URL("./expansion.fixture.json", import.meta.url), "utf8"));

/* ----------------------------------------------------- las listas, iguales */

ok("la lista de negaciones es la de Python",
   igual(NEGACION, fx.constantes.NEGACION));
ok("la lista de órdenes de detenerse es la de Python",
   igual(PARADA, fx.constantes.PARADA));
ok("los sustantivos del tablero son los de Python",
   igual(SUSTANTIVOS_TABLERO, fx.constantes.SUSTANTIVOS_TABLERO));
ok("las fórmulas de cortesía son las de Python",
   igual(CORTESIA_SIN_NEGACION, fx.constantes.CORTESIA_SIN_NEGACION));

/* ------------------------------------------------------- las primitivas */

let malNormal = 0, malTokens = 0;
const ejemplosNormal = [];
for (const p of fx.primitivas) {
  if (normalizar(p.texto) !== p.normal) {
    malNormal += 1;
    if (ejemplosNormal.length < 3) ejemplosNormal.push(`${p.texto} -> ${normalizar(p.texto)} / ${p.normal}`);
  }
  if (!igual(tokens(p.texto), p.tokens)) malTokens += 1;
}
ok(`normalizar coincide en los ${fx.primitivas.length} textos`, malNormal === 0,
   ejemplosNormal.join(" | "));
ok(`la tokenización coincide, con clíticos`, malTokens === 0, String(malTokens));

let malF1 = 0;
for (const p of fx.f1) if (Math.abs(f1Tokens(p.a, p.b) - p.f1) > 1e-9) malF1 += 1;
ok(`el F1 de tokens coincide en ${fx.f1.length} pares`, malF1 === 0, String(malF1));

/* ------------------------------------------------- las decisiones del filtro */

const porOrigen = {};
const ejemplos = [];
for (const c of fx.casos) {
  const r = filtrar(c.entrada, c.candidatos);
  const bien = igual(r.candidatos, c.quedan)
    && igual(r.descartados, c.descartados)
    && r.sinEleccion === c.sinEleccion;
  const o = (porOrigen[c.origen] ??= { n: 0, mal: 0, descartes: 0 });
  o.n += 1;
  if (c.descartados.length) o.descartes += 1;
  if (!bien) {
    o.mal += 1;
    if (ejemplos.length < 4) {
      ejemplos.push(`${c.entrada} ${JSON.stringify(c.candidatos)}\n`
        + `      python ${JSON.stringify(c.quedan)} ${JSON.stringify(c.descartados)}\n`
        + `      js     ${JSON.stringify(r.candidatos)} ${JSON.stringify(r.descartados)}`);
    }
  }
}
for (const [origen, o] of Object.entries(porOrigen)) {
  console.log(`  ${origen}: ${o.n} casos, ${o.descartes} con algún descarte`);
  ok(`el filtro decide igual que Python en todos los casos ${origen}s`,
     o.mal === 0, `${o.mal} distintos\n    ${ejemplos.join("\n    ")}`);
}
ok("hay casos reales de verdad, no un fixture vacío",
   (porOrigen.real?.n ?? 0) > 1000 && (porOrigen.real?.descartes ?? 0) > 50,
   JSON.stringify(porOrigen.real));

/* ---------------------------------------------- los casos que importan */

const final = (e, txt) => procesar(e, txt).candidatos;

let r = final("NO BASTA DOLOR",
  '{"candidatos": ["me duele", "sigue, estoy bien", "basta, me duele"]}');
ok("NO BASTA DOLOR: nunca propone seguir",
   !r.some((c) => /sigue|estoy bien/.test(c)), JSON.stringify(r));
ok("y conserva la orden de parar", r.includes("basta, me duele"), JSON.stringify(r));

r = final("BASTA", '{"candidatos": ["STOP", "hasta ahora", "de nada"]}');
ok("BASTA: \"de nada\" no es una orden de detenerse (medido en la página)",
   igual(r, ["basta"]), JSON.stringify(r));

r = final("COMIDA NO", '{"candidatos": ["quiero comer", "no quiero comer", "comida"]}');
ok("COMIDA NO: descarta lo que perdió la negación",
   igual(r, ["no quiero comer", "comida no"]), JSON.stringify(r));

r = final("TAL VEZ", '{"candidatos": ["sí", "no", "puede ser"]}');
ok("TAL VEZ: no resuelve la duda por la persona",
   igual(r, ["puede ser", "tal vez"]), JSON.stringify(r));

r = final("BAÑO", '{"candidatos": ["necesito ir al baño", "tengo sed", "quiero comer"]}');
ok("BAÑO: descarta lo que habla de agua o comida",
   igual(r, ["necesito ir al baño", "baño"]), JSON.stringify(r));

r = final("AGUA", '{"candidatos": ["Agua.", "quiero agua", "Quiero agua."]}');
ok("no muestra dos veces la misma opción", igual(r, ["quiero agua", "agua"]),
   JSON.stringify(r));

r = final("AGUA", "basura sin formato que no es json\n\n");
ok("con una respuesta inútil queda el literal", r.at(-1) === "agua", JSON.stringify(r));

r = final("AGUA", "<think>\nrazono\n</think>\n{\"candidatos\": [\"quiero agua\"]}");
ok("ignora un bloque de razonamiento de Qwen3", igual(r, ["quiero agua", "agua"]),
   JSON.stringify(r));

ok("el literal va siempre al final",
   conLiteral(["a", "b", "c", "d"], "X").length === 4
   && conLiteral(["a", "b", "c", "d"], "X")[3] === "x");

ok("parsear acepta un arreglo suelto",
   igual(parsear('["uno", "dos"]'), ["uno", "dos"]));
ok("y líneas con viñetas", igual(parsear("- uno\n2. dos\n* tres\n- cuatro"),
   ["uno", "dos", "tres"]));

/* ------------------------------------- el modo grabado también filtra
 *
 * Cuando no hay WebGPU la página muestra salidas reales guardadas de
 * qwen2.5:7b. Sin filtro, 11 de las 132 traían algo que el filtro descarta:
 * SI NO -> "no" resolvía la duda, GRACIAS BASTA -> "gracias" perdía la orden
 * de parar, PIES FRIOS -> "me duele un pie" inventaba dolor. */
{
  const { SLM } = await import("../js/slm.js");
  const grabados = JSON.parse(readFileSync(
    new URL("../datos/candidatos.json", import.meta.url), "utf8"));
  const slm = new SLM(grabados);
  ok("sin WebGPU queda en modo grabado", slm.modo === "grabado" && !slm.soportado);
  const casos = {
    "SI NO": (c) => !c.some((x) => ["no", "sí", "si"].includes(x.trim().toLowerCase())),
    "GRACIAS BASTA": (c) => !c.some((x) => x.trim().toLowerCase() === "gracias"),
    "PIES FRIOS": (c) => !c.some((x) => /duele/.test(x)),
    "ME SIENTO MAL": (c) => !c.some((x) => /duele/.test(x)),
  };
  for (const [e, bien] of Object.entries(casos)) {
    const r = await slm.expandir(e);
    ok(`modo grabado, ${e}: filtrado`, r.modo === "grabado" && bien(r.candidatos),
       JSON.stringify(r.candidatos));
  }
  let inseguras = 0;
  for (const e of Object.keys(grabados)) {
    const r = await slm.expandir(e);
    if (filtrar(e, r.candidatos).descartados.length) inseguras += 1;
  }
  ok("ninguna de las 132 grabadas muestra algo que el filtro descartaría",
     inseguras === 0, String(inseguras));
}

console.log(fallos.length ? `\n${fallos.length} FALLAS` : "\nTodo OK");
process.exit(fallos.length ? 1 : 0);
