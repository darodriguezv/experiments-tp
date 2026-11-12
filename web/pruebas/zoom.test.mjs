/* La máquina de estados del acercamiento, y la cuenta que lo justifica. */
import { Zoom, aciertoEsperado, zonaEn, cajaZona, ZONAS,
         DWELL_ACERCAR_MS, MS_ORIENTARSE } from "../js/zoom.js";
import { golpearEn, escalar } from "../js/tablero.js";
import { Permanencia } from "../js/permanencia.js";

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
ok("vence pasados 4 s sin elegir", z.venció(1000 + 4100));

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

/* --------------------------------------------------------- las zonas */

ok(`la rejilla gruesa tiene ${ZONAS}x${ZONAS} zonas`, ZONAS === 3);

const MARCO2 = { left: 0, top: 120, right: 1536, bottom: 737,
                 width: 1536, height: 617 };

const vistas = new Set();
for (let fi = 0; fi < ZONAS; fi++) {
  for (let c = 0; c < ZONAS; c++) {
    const r = cajaZona(`z:${fi}:${c}`, MARCO2);
    vistas.add(zonaEn((r.left + r.right) / 2, (r.top + r.bottom) / 2, MARCO2));
  }
}
ok("el centro de cada zona cae en esa misma zona",
   vistas.size === ZONAS * ZONAS && !vistas.has(null), [...vistas].join(" "));
ok("fuera del tablero no hay zona",
   zonaEn(-5, 400, MARCO2) === null && zonaEn(400, 50, MARCO2) === null
   && zonaEn(400, 900, MARCO2) === null);
ok("sin marco tampoco inventa una", zonaEn(400, 400, null) === null);
ok("una zona mide un tercio del tablero por lado",
   Math.abs(cajaZona("z:1:1", MARCO2).width - MARCO2.width / 3) < 1e-9);
ok("un id que no es de zona no devuelve caja",
   cajaZona("a:2:3", MARCO2) === null && cajaZona(null, MARCO2) === null);

/* Mover la mirada de un lado al otro TIENE que cambiar de zona: es lo que
 * reinicia la permanencia. Con el objetivo viejo -la grilla entera- esto daba
 * siempre el mismo valor y por eso acercaba sola. */
ok("mirar a izquierda y a derecha da zonas distintas",
   zonaEn(100, 400, MARCO2) !== zonaEn(1400, 400, MARCO2));
ok("y arriba y abajo también",
   zonaEn(700, 180, MARCO2) !== zonaEn(700, 700, MARCO2));

ok("acercar pide más que antes de los 450 ms de la grilla entera",
   DWELL_ACERCAR_MS >= 700, String(DWELL_ACERCAR_MS));
ok("y al entrar al tablero hay una pausa para orientarse",
   MS_ORIENTARSE > 0, String(MS_ORIENTARSE));


/* ------------------------- la permanencia de acercar, con zonas de verdad

 * El fallo reportado: "al mirar se hace zoom inmediatamente". El objetivo de
 * esta permanencia era la grilla ENTERA, o sea un valor constante. Como nunca
 * cambiaba, nunca se reiniciaba: la cuenta arrancaba al aparecer el tablero y
 * disparaba 450 ms después, mirara la persona donde mirara.
 *
 * Con zonas, mover la mirada cambia el objetivo y eso reinicia la cuenta. */

const dwellZoom = new Permanencia({ dwellMs: DWELL_ACERCAR_MS });
const mirarA = (x, y) => dwellZoom.paso(
  { x, y, ok: true },
  (px, py) => zonaEn(px, py, MARCO2),
  (id) => cajaZona(id, MARCO2));

const izq = cajaZona("z:1:0", MARCO2), der = cajaZona("z:1:2", MARCO2);
const cIzq = [(izq.left + izq.right) / 2, (izq.top + izq.bottom) / 2];
const cDer = [(der.left + der.right) / 2, (der.top + der.bottom) / 2];

const a = mirarA(cIzq[0], cIzq[1]);
const b = mirarA(cDer[0], cDer[1]);
ok("mirar a otra zona cambia el objetivo", a.objetivo !== b.objetivo,
   `${a.objetivo} -> ${b.objetivo}`);
ok("y la cuenta vuelve a empezar", b.avance < 0.05, String(b.avance));

/* El margen de pegado tiene que seguir aguantando el temblor: un desvío
 * chico dentro de la misma zona no puede reiniciar nada, o con 90 px de
 * error no se acercaría nunca. */
const c = mirarA(cDer[0] + 60, cDer[1] + 30);
ok("un temblor dentro de la zona NO reinicia", c.objetivo === b.objetivo,
   `${b.objetivo} -> ${c.objetivo}`);

/* Y mirar fuera del tablero -el panel del mensaje, de arriba- no acumula. */
const d = mirarA(700, MARCO2.top - 40);
ok("mirar fuera del tablero no acumula acercamiento",
   d.objetivo === null && d.avance === 0, String(d.objetivo));

ok("nada de esto disparó todavía",
   [a, b, c, d].every((r) => r.disparo === null));

console.log(fallos.length ? `\n${fallos.length} FALLAS` : "\nTodo OK");
process.exit(fallos.length ? 1 : 0);
