/* El modo sin cámara, de punta a punta, en un Chrome de verdad.
 *
 *   cd slm/banco_web && node prueba_sin_camara.mjs
 *
 * Hace lo que haría alguien que prueba el tablero con el mouse: entra al modo,
 * arma una frase con palabras y letras, pide las opciones, elige una, usa el
 * teclado y vuelve al menú. Y comprueba lo que no tiene que pasar: que se pida
 * la cámara, o que la página tire un error.
 */
import http from "node:http";
import path from "node:path";
import { existsSync, createReadStream, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";

const RAIZ = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..", "web");
const TIPOS = { ".html": "text/html; charset=utf-8", ".js": "text/javascript",
                ".css": "text/css", ".json": "application/json" };
const srv = http.createServer((q, r) => {
  let f = path.join(RAIZ, decodeURIComponent(new URL(q.url, "http://x").pathname));
  if (existsSync(f) && statSync(f).isDirectory()) f = path.join(f, "index.html");
  if (!f.startsWith(RAIZ) || !existsSync(f)) { r.writeHead(404); return r.end(); }
  r.writeHead(200, { "content-type": TIPOS[path.extname(f)] ?? "application/octet-stream" });
  createReadStream(f).pipe(r);
}).listen(0, "127.0.0.1");
await new Promise((ok) => srv.on("listening", ok));

const fallos = [];
const ok = (n, c, x = "") =>
  console.log(c ? `  ok    ${n}` : (fallos.push(n), `  FALLA ${n} ${x}`));

const b = await puppeteer.launch({
  executablePath: process.env.CHROME ?? "C:/Program Files/Google/Chrome/Application/chrome.exe",
  headless: "new", args: ["--no-first-run"],
  userDataDir: path.join(process.env.TEMP ?? "/tmp", "prueba-sin-camara"),
});
const p = await b.newPage();
await p.setViewport({ width: 1280, height: 760 });
const errores = [];
p.on("pageerror", (e) => errores.push(e.message));
// Cualquier pedido de cámara queda registrado: este modo no puede hacer ninguno.
await p.evaluateOnNewDocument(() => {
  window.__camara = 0;
  const md = navigator.mediaDevices;
  if (md?.getUserMedia) {
    const orig = md.getUserMedia.bind(md);
    md.getUserMedia = (...a) => { window.__camara += 1; return orig(...a); };
  }
});
await p.goto(`http://127.0.0.1:${srv.address().port}/index.html`);
await p.waitForSelector("#opciones .opcion-modo");

const titulos = await p.$$eval("#opciones .titulo", (els) => els.map((e) => e.textContent));
ok("el menú ofrece el tablero sin cámara", titulos.includes("Tablero sin cámara"),
   JSON.stringify(titulos));

// Saber si el modelo ya está guardado obliga a bajar la biblioteca de WebLLM
// (unos 6 MB): el aviso inicial dura unos segundos.
await p.waitForFunction(() => !/Comprobando/.test(
  document.querySelector("#modelo-estado").textContent), { timeout: 60000 });
const estadoModelo = await p.$eval("#modelo-estado", (e) => e.textContent);
console.log(`  (estado del modelo: ${estadoModelo})`);
ok("el menú dice cómo están las frases (modelo o grabadas)",
   /modelo|grabadas|WebGPU/.test(estadoModelo), estadoModelo);

// Entrar.
await p.evaluate(() => [...document.querySelectorAll("#opciones .opcion-modo")]
  .find((b) => b.textContent.includes("Tablero sin cámara")).click());
await p.waitForSelector("#p-tablero:not([hidden]) .celda");
ok("entra directo al tablero, sin pantalla de inicio ni calibración",
   await p.$eval("#p-calibracion", (e) => e.hidden) && await p.$eval("#p-inicio", (e) => e.hidden));

const clic = async (etq) => {
  const el = await p.evaluateHandle((t) => [...document.querySelectorAll("#escena [data-accion]")]
    .find((c) => c.textContent.trim() === t), etq);
  if (!el.asElement()) throw new Error(`no hay celda ${etq}`);
  await el.asElement().click();
  await new Promise((r) => setTimeout(r, 60));
};
const mensaje = () => p.$eval("#mensaje", (e) => e.textContent);

// Palabras.
await clic("QUIERO"); await clic("AGUA");
ok("con clics se arma una frase de palabras", (await mensaje()) === "QUIERO AGUA", await mensaje());

// Letras, y vuelta a palabras.
await clic("ABC");
ok("ABC cambia a la página de letras",
   await p.$$eval("#escena [data-accion]", (e) => e.some((x) => x.textContent.trim() === "Ñ")));
await clic("F"); await clic("R"); await clic("I"); await clic("A");
ok("con clics se deletrea", (await mensaje()) === "QUIERO AGUA FRIA", await mensaje());
await clic("BORRAR");
ok("BORRAR saca la última letra", (await mensaje()) === "QUIERO AGUA FRI", await mensaje());
await clic("A");
await clic("PALABRAS");
ok("PALABRAS vuelve a la página de palabras",
   await p.$$eval("#escena [data-accion]", (e) => e.some((x) => x.textContent.trim() === "MEDICINA")));

// El teclado: Tab hasta una celda y Enter.
await p.evaluate(() => document.querySelector('#escena [data-accion="w:POR FAVOR"]').focus());
await p.keyboard.press("Enter");
await new Promise((r) => setTimeout(r, 60));
ok("Enter sobre una celda enfocada la elige", (await mensaje()).endsWith("POR FAVOR"), await mensaje());
ok("las celdas se alcanzan con Tab",
   await p.$$eval("#escena [data-accion]", (e) => e.every((x) => x.tabIndex === 0)));

// HABLAR: opciones, y elegir una.
await clic("LIMPIAR"); await clic("QUIERO"); await clic("AGUA");
await clic("HABLAR");
await p.waitForFunction(() => !document.querySelector(".candidatos h2")
  || document.querySelector(".candidatos h2").textContent !== "Pensando…", { timeout: 60000 });
const opciones = await p.$$eval(".candidatos .opcion", (e) => e.map((x) => x.textContent.trim()));
ok("HABLAR muestra las opciones, con el literal y VOLVER",
   opciones.includes("quiero agua") && opciones.includes("VOLVER AL TABLERO"), JSON.stringify(opciones));
const elegida = opciones.find((o) => o !== "VOLVER AL TABLERO");
await clic(elegida);
ok("elegir una opción la dice y limpia el mensaje",
   (await p.$("#escena .grilla")) !== null && !(await mensaje()).includes("QUIERO"), await mensaje());

// Lo que no tiene que haber pasado.
ok("nunca se pidió la cámara", (await p.evaluate(() => window.__camara)) === 0);
ok("el cursor de la mirada no aparece", await p.$eval("#cursor", (e) => getComputedStyle(e).display === "none"));
ok("la barra muestra la ayuda del modo, no los datos de la mirada",
   await p.$eval(".medidor .solo-puntero", (e) => getComputedStyle(e).display !== "none")
   && await p.$eval(".medidor .solo-mirada", (e) => getComputedStyle(e).display === "none"));

// Volver al menú.
await p.click("#btn-salir-tablero");
ok("Volver al menú sale del modo",
   !(await p.$eval("#p-menu", (e) => e.hidden))
   && !(await p.evaluate(() => document.body.classList.contains("sin-camara"))));

ok("sin errores de JavaScript en la página", errores.length === 0, errores.join(" | "));

await b.close();
srv.close();
console.log(fallos.length ? `\n${fallos.length} FALLAS` : "\nTodo OK");
process.exit(fallos.length ? 1 : 0);
