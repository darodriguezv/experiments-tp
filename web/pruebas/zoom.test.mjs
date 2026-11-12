/* La máquina de estados del acercamiento, y la cuenta que lo justifica. */
import { Zoom, aciertoEsperado, Fijacion, ParpadeoLargo, DWELL_ACERCAR_MS,
         MS_ORIENTARSE, MS_PAUSA_TRAS_ALEJAR, MS_PARPADEO_LARGO }
  from "../js/zoom.js";
import { FiltroMirada } from "../js/suavizado.js";
import { golpearEn, escalar } from "../js/tablero.js";

const fallos = [];
const ok = (n, c, x = "") =>
  console.log(c ? `  ok    ${n}` : (fallos.push(n), `  FALLA ${n} ${x}`));

// --- la máquina -------------------------------------------------------
let z = new Zoom();
ok("arranca alejado", !z.acercado && z.escala === 1);

z.acercar(800, 400, 1000);
ok("acercar sube un nivel", z.acercado && Math.abs(z.escala - 2.5) < 1e-9,
   z.escala);
/* `transform-origin` en píxeles se mide desde el borde del ELEMENTO, no desde
 * el viewport. La grilla tiene el panel del mensaje encima, así que arranca
 * bastante más abajo de 0: pasarle coordenadas de viewport acercaba centrado
 * arriba de donde la persona miraba. La versión anterior de esta prueba
 * afirmaba "800px 400px" y por eso el fallo pasó. */
const MARCO = { left: 0, top: 120, right: 1536, bottom: 737,
                width: 1536, height: 617 };
const e = z.estilo(MARCO);
ok("el origen está en coordenadas del elemento, no del viewport",
   e.transformOrigin === "800px 280px", e.transformOrigin);
ok("la transformación escala", e.transform === "scale(2.5)", e.transform);

ok("no acerca más allá de los pasos configurados",
   z.acercar(900, 500, 1100) === false);

ok("no vence todavía", !z.venció(2000));
ok("no vence a los 4 s: una selección lleva más de 1 s y hay que poder reintentar",
   !z.venció(1000 + 4100));
ok("vence pasados 5 s sin elegir", z.venció(1000 + 5100));

z.alejar();
ok("alejar vuelve al estado inicial",
   !z.acercado && z.escala === 1 && z.estilo(MARCO).transform === "none");

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


/* ------------------------------------------------ las cajas al acercar

 * El fallo que esto fija: `aplicarZoom()` fijaba el transform y medía las
 * cajas con `getBoundingClientRect()` en el mismo instante, pero `.grilla`
 * tiene `transition: transform .18s`. Lo medido era la geometría del ARRANQUE
 * de la animación, no la final, y no se volvía a medir nunca: las celdas
 * quedaban dibujadas 2.5x más grandes con el mapa de impacto de antes. El
 * puntero se veía sobre una celda y se marcaba otra.
 *
 * Calcular las cajas en vez de medirlas saca el reloj de la ecuación. */

const COLS = 5, FILS = 5;
const TAB = { left: 0, top: 120, width: 1536, height: 617 };
const base = new Map();
for (let fi = 0; fi < FILS; fi++) {
  for (let c = 0; c < COLS; c++) {
    const w = TAB.width / COLS, h = TAB.height / FILS;
    const left = TAB.left + c * w, top = TAB.top + fi * h;
    base.set(`${fi}:${c}`,
      { left, top, right: left + w, bottom: top + h, width: w, height: h });
  }
}

const ORIGEN = [700, 400], K = 2.5;
const acercadas = escalar(base, ORIGEN, K);

ok("el punto del acercamiento no se mueve",
   golpearEn(acercadas, ORIGEN[0], ORIGEN[1])
   === golpearEn(base, ORIGEN[0], ORIGEN[1]));

/* La propiedad exacta: la celda que se VE bajo un punto de la pantalla es la
 * que, deshaciendo la escala, contiene ese punto en las cajas de base. */
let correctos = 0, comoAntes = 0, total = 0;
for (let x = 20; x < 1536; x += 37) {
  for (let y = 130; y < 730; y += 41) {
    const verdadera = golpearEn(base,
      ORIGEN[0] + (x - ORIGEN[0]) / K,
      ORIGEN[1] + (y - ORIGEN[1]) / K);
    if (!verdadera) continue;
    total += 1;
    if (golpearEn(acercadas, x, y) === verdadera) correctos += 1;
    if (golpearEn(base, x, y) === verdadera) comoAntes += 1;
  }
}
console.log(`  ${total} puntos: calculadas ${correctos}, medidas-sin-acercar ${comoAntes}`);
ok("las cajas calculadas aciertan la celda que se ve, en todos los puntos",
   correctos === total, `${correctos}/${total}`);
ok("y con las cajas de antes de acercar se marcaría otra casi siempre",
   comoAntes < total * 0.35, `${comoAntes}/${total}`);

ok("acercar agranda las celdas por el factor",
   Math.abs(acercadas.get("0:0").width - base.get("0:0").width * K) < 1e-9);
ok("sin acercar, las cajas son las mismas",
   escalar(base, null, 1) === base && escalar(base, ORIGEN, 1) === base);

/* ------------------------------------------------ acercar por FIJACIÓN
 *
 * Dos versiones anteriores acercaban antes de tiempo. La primera contaba el
 * tiempo sobre la grilla entera; la segunda, dentro de una zona de un tercio
 * de pantalla. Ninguna medía lo que importa: si la mirada se quedó quieta.
 * Leyendo el tablero la mirada salta de celda en celda sin salir de una zona,
 * y con las dos versiones eso acercaba igual. */

const MARCO2 = { left: 0, top: 120, right: 1536, bottom: 737,
                 width: 1536, height: 617 };
const dentro = (x, y) => x >= MARCO2.left && x < MARCO2.right
                      && y >= MARCO2.top && y < MARCO2.bottom;

let sem = 4242;
const azar = () => (sem = (sem * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
const gauss = () => Math.sqrt(-2 * Math.log(azar() + 1e-12))
                  * Math.cos(2 * Math.PI * azar());

/** Corre una trayectoria de mirada y devuelve cuándo acercó, o null. */
function correr(fij, dondeMira, ms, { desde = 0, ruido = 20 } = {}) {
  for (let t = desde; t < desde + ms; t += 33) {
    const [x, y] = dondeMira(t - desde);
    const m = { x: x + ruido * gauss(), y: y + ruido * gauss(), ok: true };
    const r = fij.paso(m, dentro(m.x, m.y), t);
    if (r.disparo) return { t: t - desde, en: r.disparo };
  }
  return null;
}

// Mirar fijo una celda: acerca, y no antes de tiempo.
let fij = new Fijacion();
let res = correr(fij, () => [700, 400], 4000);
ok("mirar fijo un lugar acerca", res !== null);
ok("pero no antes de la permanencia completa",
   res && res.t >= DWELL_ACERCAR_MS, String(res?.t));
ok("y acerca centrado donde se estuvo mirando",
   res && Math.hypot(res.en[0] - 700, res.en[1] - 400) < 25,
   String(res?.en));

// Leer el tablero: saltar de celda en celda cada 600 ms, dentro de la misma
// zona. Es exactamente lo que antes acercaba solo.
fij = new Fijacion();
const celdas = [[160, 190], [460, 190], [160, 310], [460, 310], [160, 430]];
res = correr(fij, (t) => celdas[Math.floor(t / 600) % celdas.length], 15000);
ok("recorrer el tablero leyendo NO acerca", res === null, JSON.stringify(res));

// Pasar a la celda de al lado -307 px en el tablero más fino- ya cuenta
// como moverse.
fij = new Fijacion();
res = correr(fij, (t) => [t < 700 ? 500 : 807, 400], 1500);
ok("pasar a la celda vecina reinicia la cuenta", res === null,
   JSON.stringify(res));

// Fuera del tablero no se acumula nada.
fij = new Fijacion();
res = correr(fij, () => [700, 60], 4000, { ruido: 5 });
ok("mirar el mensaje, fuera del tablero, no acerca", res === null);

// Un parpadeo corto no rompe la fijación; perder la mirada medio segundo, sí.
fij = new Fijacion();
for (let t = 0; t < 600; t += 33) fij.paso({ x: 700, y: 400, ok: true }, true, t);
for (let t = 600; t < 800; t += 33) fij.paso({ x: 0, y: 0, ok: false }, false, t);
ok("un parpadeo corto no reinicia la fijación",
   fij.paso({ x: 702, y: 401, ok: true }, true, 810).avance > 0.5);
for (let t = 820; t < 1400; t += 33) fij.paso({ x: 0, y: 0, ok: false }, false, t);
ok("perder la mirada medio segundo sí la reinicia",
   fij.paso({ x: 702, y: 401, ok: true }, true, 1410).avance === 0);

// La pausa después de alejar.
fij = new Fijacion();
fij.bloquear(MS_PAUSA_TRAS_ALEJAR, 0);
res = correr(fij, () => [700, 400], MS_PAUSA_TRAS_ALEJAR + DWELL_ACERCAR_MS - 100);
ok("recién alejado no vuelve a acercar enseguida", res === null,
   JSON.stringify(res));

// Se alejó por tiempo: quedarse mirando el mismo lugar no puede volver a
// acercar, o la persona siente que no hay salida.
fij = new Fijacion();
fij.exigirMovimiento([700, 400]);
res = correr(fij, () => [720, 410], 6000);
ok("tras alejarse solo, mirar el mismo lugar no vuelve a acercar",
   res === null, JSON.stringify(res));
res = correr(fij, (t) => (t < 500 ? [1200, 600] : [720, 410]), 4000,
             { desde: 6000 });
ok("irse y volver sí permite acercar ahí de nuevo", res !== null);

/* La cadena completa: señal cruda ruidosa -> filtro -> fijación. Es lo que
 * corre de verdad; las pruebas de arriba le dan a la fijación una mirada ya
 * limpia. */
function cadena(dondeMira, ms, ruido) {
  const filtro = new FiltroMirada(), f = new Fijacion();
  for (let t = 0; t < ms; t += 33) {
    const [x, y] = dondeMira(t);
    const o = filtro.paso(x + ruido * gauss(), y + ruido * gauss(), t);
    const r = f.paso({ x: o[0], y: o[1], ok: true }, dentro(o[0], o[1]), t);
    if (r.disparo) return { t, en: r.disparo };
  }
  return null;
}
for (const ruido of [40, 80, 100]) {
  let acercó = 0, tarde = 0, lejos = 0;
  for (let k = 0; k < 20; k++) {
    const r = cadena(() => [900, 450], 6000, ruido);
    if (r) {
      acercó += 1; tarde += r.t;
      lejos = Math.max(lejos, Math.hypot(r.en[0] - 900, r.en[1] - 450));
    }
  }
  console.log(`  ruido ${ruido}: acercó ${acercó}/20, a los ${(tarde / acercó).toFixed(0)} ms,`
            + ` centro a ${lejos.toFixed(0)} px como mucho`);
  ok(`con señal cruda de ruido ${ruido}, mirar fijo acerca siempre`,
     acercó === 20, `${acercó}/20`);
  // El centro no puede ser mejor que el ruido que queda después de filtrar:
  // con señal cruda de 100 px eso son unos 30 px de desvío, y el peor de
  // veinte intentos se va a unos 65.
  ok(`y el centro queda cerca de lo mirado (ruido ${ruido})`,
     lejos < (ruido <= 40 ? 40 : 80),
     lejos.toFixed(0));

  let falsas = 0;
  for (let k = 0; k < 20; k++) {
    if (cadena((t) => celdas[Math.floor(t / 600) % celdas.length], 12000, ruido)) {
      falsas += 1;
    }
  }
  console.log(`  ruido ${ruido}: leyendo 12 s seguidos, acercó sin querer ${falsas}/20`);
  // Con señal limpia, nunca. Con señal muy ruidosa no se puede garantizar el
  // cero -el filtro tarda en enterarse de un salto corto- y lo que se fija
  // es que sea raro: menos de una vez cada minuto de lectura continua. Un
  // acercamiento de más se deshace cerrando los ojos; uno que no llega
  // nunca no tiene arreglo, y por eso el criterio no se aprieta más.
  ok(`y leyendo el tablero no acerca (ruido ${ruido})`,
     falsas <= (ruido <= 40 ? 0 : 4),
     `${falsas}/20`);
}

/* ------------------------------------------------- el parpadeo largo */

/* Abierto 0.45 y cerrado 0.17: el cerrado queda POR ENCIMA del umbral fijo de
 * 0.14 que usa el resto del sistema. Es el caso que un umbral fijo no ve
 * nunca, y por eso el detector se mide contra la apertura de cada persona. */
function parpadear(ms, { hayCara = true, abierto = 0.45, cerrado = 0.17 } = {}) {
  const p = new ParpadeoLargo();
  let avisos = 0;
  for (let t = 0; t < 900; t += 33) avisos += p.paso(abierto, true, t) ? 1 : 0;
  for (let t = 900; t < 900 + ms; t += 33) avisos += p.paso(cerrado, hayCara, t) ? 1 : 0;
  for (let t = 900 + ms; t < 1500 + ms; t += 33) avisos += p.paso(abierto, true, t) ? 1 : 0;
  return avisos;
}
ok("un parpadeo espontáneo de 150 ms no cancela nada", parpadear(150) === 0);
ok("uno de 300 ms tampoco", parpadear(300) === 0);
ok(`cerrar los ojos más de ${MS_PARPADEO_LARGO} ms cancela`,
   parpadear(MS_PARPADEO_LARGO + 120) === 1);
ok("y avisa UNA sola vez aunque se sigan cerrados", parpadear(2500) === 1);
ok("perder la cara no es parpadear", parpadear(2500, { hayCara: false }) === 0);
ok("funciona igual con ojos de apertura chica",
   parpadear(700, { abierto: 0.26, cerrado: 0.10 }) === 1);
ok("entrecerrar apenas los ojos no cancela",
   parpadear(2500, { abierto: 0.45, cerrado: 0.36 }) === 0);
{
  // Muchos parpadeos seguidos no pueden ir bajando la referencia.
  const p = new ParpadeoLargo();
  let t = 0;
  for (let k = 0; k < 40; k++) {
    for (let i = 0; i < 20; i++) { p.paso(0.45, true, t); t += 33; }
    for (let i = 0; i < 6; i++) { p.paso(0.12, true, t); t += 33; }
  }
  ok("parpadear seguido no baja la apertura aprendida", p.base > 0.42,
     p.base.toFixed(3));
}

ok("acercar pide una fijación de al menos 1 s", DWELL_ACERCAR_MS >= 1000,
   String(DWELL_ACERCAR_MS));
ok("y al entrar al tablero hay una pausa para orientarse", MS_ORIENTARSE > 0);

console.log(fallos.length ? `\n${fallos.length} FALLAS` : "\nTodo OK");
process.exit(fallos.length ? 1 : 0);
