/* Corre el corpus contra la página real, en Chrome sin ventana, con WebGPU.
 *
 *   cd slm/banco_web && npm install && node correr.mjs <plan>
 *
 * Los planes están en `planes.mjs`. Cada respuesta se agrega a
 * `slm/results/web-<plan>.jsonl` apenas llega, y la corrida se puede cortar y
 * retomar: lo que ya está en el archivo no se repite.
 *
 * ## Por qué así y no con Ollama
 *
 * Todas las mediciones anteriores se hicieron con Ollama, temperatura 0.2,
 * semilla fija y JSON forzado por el decodificador. La página corre otro motor
 * (WebLLM), otra cuantización (q4f16_1), temperatura 0.9, sin JSON forzado, y
 * hasta este cambio sin el filtro. Esas tablas describían un sistema que no es
 * el que usa la persona. Esto mide el que usa.
 */
import http from "node:http";
import { readFileSync, existsSync, appendFileSync, mkdirSync, createReadStream, statSync } from "node:fs";
import { execFileSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";
import { PLANES } from "./planes.mjs";

const AQUI = path.dirname(fileURLToPath(import.meta.url));
const RAIZ = path.resolve(AQUI, "..", "..");          // experiments/
const SLM = path.join(RAIZ, "slm");
const CHROME = process.env.CHROME
  ?? "C:/Program Files/Google/Chrome/Application/chrome.exe";
// El perfil guarda los modelos bajados (GB). Fuera de OneDrive a propósito.
const PERFIL = process.env.PERFIL_BANCO
  ?? path.join(process.env.TEMP ?? "/tmp", "banco-slm-perfil");

const nombrePlan = process.argv[2];
const plan = PLANES[nombrePlan];
if (!plan) {
  console.error(`plan ${nombrePlan ?? "(ninguno)"} no existe. Hay: ${Object.keys(PLANES).join(", ")}`);
  process.exit(1);
}

/* ------------------------------------------------------------ el corpus */

const corpus = readFileSync(path.join(SLM, "data", "telegraphic_es.jsonl"), "utf8")
  .split(/\r?\n/).filter((l) => l.trim() && !l.startsWith("//")).map((l) => JSON.parse(l));

/* Los textos de las variantes salen de prompts.py, que es la fuente de verdad.
 * "desplegado" no es una variante de prompts.py: es web/js/prompt.js tal cual
 * está, que es lo que corre la página. */
const variantes = JSON.parse(execFileSync("python", ["-c",
  "import json,sys; from aac_semantic.prompts import VARIANTS; "
  + "sys.stdout.buffer.write(json.dumps({k: v.system for k, v in VARIANTS.items()}, ensure_ascii=False).encode('utf-8'))",
], { cwd: SLM, encoding: "utf8" }));

/* ------------------------------------------------- servidor de archivos */

const TIPOS = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                ".mjs": "text/javascript; charset=utf-8", ".json": "application/json" };
const servidor = http.createServer((q, r) => {
  const ruta = path.join(RAIZ, decodeURIComponent(new URL(q.url, "http://x").pathname));
  if (!ruta.startsWith(RAIZ) || !existsSync(ruta) || statSync(ruta).isDirectory()) {
    r.writeHead(404); r.end(); return;
  }
  r.writeHead(200, { "content-type": TIPOS[path.extname(ruta)] ?? "application/octet-stream",
                     "cache-control": "no-store" });
  createReadStream(ruta).pipe(r);
});
/* Puerto FIJO. Chrome guarda los modelos bajados por origen, y el origen
 * incluye el puerto: con un puerto al azar cada corrida era un origen nuevo y
 * volvía a bajar todo. Se midió así: un perfil con un solo modelo de 0.87 GB
 * pesaba 4.9 GB. */
const PUERTO_FIJO = Number(process.env.PUERTO_BANCO ?? 8740);
await new Promise((ok) => servidor.listen(PUERTO_FIJO, "127.0.0.1", ok));
const PUERTO = servidor.address().port;

/* ------------------------------------------------------------ la salida */

const SALIDA = path.join(SLM, "results", `web-${nombrePlan}.jsonl`);
mkdirSync(path.dirname(SALIDA), { recursive: true });
const clave = (o) => [o.modelo, o.variante, o.temperatura, o.perturbacion ?? "", o.item_id, o.k].join("|");
const hechos = new Set();
if (existsSync(SALIDA)) {
  for (const l of readFileSync(SALIDA, "utf8").split(/\r?\n/)) {
    if (!l.trim()) continue;
    const o = JSON.parse(l);
    if (!o.error) hechos.add(clave(o));
  }
}
console.log(`[banco] plan ${nombrePlan}: ${plan.length} celdas, ${hechos.size} respuestas ya guardadas en ${SALIDA}`);

/* ------------------------------------------------------------- Chrome */

/* Varios navegadores, no varias pestañas. Medido: dos pestañas del mismo
 * Chrome rinden 3.7 s por pedido contra 4.2 s con una, porque comparten el
 * proceso de GPU y los pasos de decodificación se turnan. Cada navegador tiene
 * su proceso de GPU propio. Cada uno necesita su propio perfil, y por lo tanto
 * su propia copia de los modelos bajados. */
const NAVEGADORES = Math.max(1, Number(process.env.NAVEGADORES ?? 1));
const navegadores = [];
for (let i = 0; i < NAVEGADORES; i++) {
  navegadores.push(await puppeteer.launch({
    executablePath: CHROME, headless: "new",
    userDataDir: i ? `${PERFIL}-${i}` : PERFIL,
    protocolTimeout: 0,
    args: ["--enable-unsafe-webgpu", "--no-first-run", "--no-default-browser-check",
           "--disable-background-timer-throttling", "--disable-renderer-backgrounding"],
  }));
}
const navegador = { newPage: (() => { let i = 0; return () => navegadores[i++ % NAVEGADORES].newPage(); })(),
                    close: () => Promise.all(navegadores.map((b) => b.close())) };
/* Varias pestañas, cada una con su propio motor. La decodificación de WebLLM
 * está limitada por la latencia de cada paso y no por el cálculo, así que
 * correr varios pedidos a la vez sube el rendimiento sin cambiar ninguna
 * respuesta: cada pedido sigue siendo uno solo, contra un motor solo. La
 * latencia que se guarda por pedido SÍ sube con la concurrencia, y por eso la
 * latencia se mide aparte, con una sola pestaña (plan "latencia"). */
const PARALELO = Math.max(1, Number(process.env.PARALELO ?? NAVEGADORES));
async function abrir() {
  const p = await navegador.newPage();
  p.setDefaultTimeout(0);
  p.on("pageerror", (e) => console.error("[página]", e.message));
  p.on("console", (m) => {
    if (m.type() === "error" && !m.text().includes("404")) console.error("[consola]", m.text());
  });
  await p.goto(`http://127.0.0.1:${PUERTO}/slm/banco_web/banco.html`);
  await p.waitForFunction("window.__listo === true");
  return p;
}
const paginas = [];
for (let i = 0; i < PARALELO; i++) paginas.push(await abrir());
const pagina = paginas[0];

const desplegado = await pagina.evaluate(() => window.banco.desplegado);
const v4 = variantes["v4-literal"];
console.log(`[banco] desplegado: ${desplegado.modelo}, t=${desplegado.temperatura}, `
  + `prompt ${desplegado.prompt === v4 ? "== v4-literal" : "DISTINTO de v4-literal"}`);

/* -------------------------------------------------------------- correr */

const perturbar = (await import("./perturbaciones.mjs")).perturbar;
let cargado = null;
const t0 = Date.now();
let hechasAhora = 0;

for (const celda of plan) {
  // `paso`: una entrada de cada tantas, parejo a lo largo del corpus (que está
  // ordenado por tipo de entrada, así que cortar el principio sesgaría).
  const items = (celda.items ? corpus.slice(0, celda.items) : corpus)
    .filter((_, i) => i % (celda.paso ?? 1) === 0);
  const pendientes = [];
  for (let k = 0; k < celda.k; k++) {
    for (const it of items) {
      for (const pert of celda.perturbaciones ?? [null]) {
        const o = { modelo: celda.modelo, variante: celda.variante, temperatura: celda.temperatura,
                    perturbacion: pert, item_id: it.id, k };
        if (!hechos.has(clave(o))) pendientes.push([o, it]);
      }
    }
  }
  const total = items.length * celda.k * (celda.perturbaciones?.length ?? 1);
  console.log(`[banco] ${celda.modelo} / ${celda.variante} / t=${celda.temperatura}`
    + `${celda.perturbaciones ? " / " + celda.perturbaciones.length + " perturbaciones" : ""}: `
    + `${total - pendientes.length}/${total} hechas`);
  if (!pendientes.length) continue;

  if (cargado !== celda.modelo) {
    let fallo = false;
    // De a una: la primera baja los pesos, las demás los toman de la caché.
    for (const p of paginas) {
      const c = await p.evaluate((m) => window.banco.cargar(m), celda.modelo);
      if (!c.listo) { console.error(`[banco] no cargó ${celda.modelo}: ${c.error}`); fallo = true; break; }
      console.log(`[banco] ${celda.modelo} cargado en ${(c.ms / 1000).toFixed(1)} s`);
      // Un pedido de calentamiento que no se guarda: el primero compila shaders.
      await p.evaluate((s) => window.banco.expandir("AGUA", { temperatura: 0.9, sistema: s }), null);
    }
    if (fallo) { cargado = null; continue; }
    cargado = celda.modelo;
  }

  const sistema = celda.variante === "desplegado" ? null : variantes[celda.variante];
  if (celda.variante !== "desplegado" && !sistema) throw new Error(`variante ${celda.variante} no existe`);

  let n = 0, siguiente = 0;
  const tCelda = Date.now();
  await Promise.all(paginas.map(async (p) => {
    while (siguiente < pendientes.length) {
      const [o, it] = pendientes[siguiente++];
      const entrada = o.perturbacion ? perturbar(it.input, o.perturbacion, o.k) : it.input;
      const r = await p.evaluate((e, opc) => window.banco.expandir(e, opc), entrada,
                                 { temperatura: o.temperatura, sistema });
      // Lo que identifica la respuesta va DESPUÉS de lo que devolvió la página:
      // si alguna vez coinciden nombres, gana la identificación.
      appendFileSync(SALIDA, JSON.stringify({ ...r, ...o, entrada, paralelo: PARALELO,
                                              t: Date.now() }) + "\n");
      n += 1; hechasAhora += 1;
      if (n % 50 === 0 || n === pendientes.length) {
        const seg = (Date.now() - tCelda) / 1000;
        console.log(`  ${n}/${pendientes.length}  ${(seg / n).toFixed(2)} s por pedido (rendimiento)`);
      }
    }
  }));
}

await navegador.close();
servidor.close();
console.log(`[banco] listo: ${hechasAhora} respuestas nuevas en ${((Date.now() - t0) / 60000).toFixed(1)} min`);
