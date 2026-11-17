// ¿El modo sin ventana cambia la velocidad? Mismo modelo, mismo pedido.
import http from "node:http"; import path from "node:path"; import { existsSync, createReadStream } from "node:fs";
import puppeteer from "puppeteer-core";
const RAIZ = path.resolve("..", "..");
const srv = http.createServer((q, r) => { const f = path.join(RAIZ, decodeURIComponent(new URL(q.url,"http://x").pathname));
  if (!existsSync(f)) { r.writeHead(404); return r.end(); }
  r.writeHead(200, { "content-type": f.endsWith(".js") ? "text/javascript" : "text/html" }); createReadStream(f).pipe(r); }).listen(8732, "127.0.0.1");
for (const [nombre, extra] of [["sin ventana", { headless: "new" }], ["con ventana", { headless: false, args2: ["--window-position=-2400,-2400", "--window-size=900,700"] }],
                               ["sin ventana + d3d12 sin robustez", { headless: "new", args2: ["--enable-dawn-features=disable_robustness"] }]]) {
  const b = await puppeteer.launch({ executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe", headless: extra.headless,
    userDataDir: process.env.TEMP + "/banco-slm-perfil", protocolTimeout: 0,
    args: ["--enable-unsafe-webgpu", "--no-first-run", ...(extra.args2 ?? [])] });
  const p = await b.newPage(); p.setDefaultTimeout(0);
  await p.goto("http://127.0.0.1:8732/slm/banco_web/banco.html"); await p.waitForFunction("window.__listo === true");
  await p.evaluate(() => window.banco.cargar("Qwen2.5-1.5B-Instruct-q4f16_1-MLC"));
  await p.evaluate(() => window.banco.expandir("AGUA", { temperatura: 0.9, sistema: null }));
  const rs = [];
  for (const e of ["DOLOR", "QUIERO AGUA", "NO BASTA DOLOR", "VENTANA"]) rs.push(await p.evaluate((x) => window.banco.expandir(x, { temperatura: 0.9, sistema: null }), e));
  const dec = rs.map((r) => r.uso.extra.decode_tokens_per_s), ms = rs.map((r) => r.ms);
  const gpu = await p.evaluate(async () => { const a = await navigator.gpu.requestAdapter({powerPreference:"high-performance"}); return a.info?.vendor + "/" + a.info?.architecture; });
  console.log(nombre.padEnd(32), "decode tok/s", dec.map((d) => d.toFixed(1)).join(" "), "| ms", ms.join(" "), "|", gpu);
  await b.close();
}
srv.close();
