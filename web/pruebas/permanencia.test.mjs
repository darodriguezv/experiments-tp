/* Los cuatro comportamientos que hacen usable la permanencia. Cada uno existe
 * porque sin él el tablero falla de una forma concreta. */
import { Permanencia } from "../js/permanencia.js";

let reloj = 0;
globalThis.performance = { now: () => reloj };

const CAJAS = {
  a: { left: 0, top: 0, right: 100, bottom: 100, width: 100, height: 100 },
  b: { left: 110, top: 0, right: 210, bottom: 100, width: 100, height: 100 },
};
const caja = (id) => CAJAS[id] ?? null;
const golpear = (x, y) => {
  for (const [id, r] of Object.entries(CAJAS)) {
    if (x >= r.left && x < r.right && y >= r.top && y < r.bottom) return id;
  }
  return null;
};
const correr = (p, x, y, ms, ok = true) => {
  let ultimo = { avance: 0, disparo: null, objetivo: null };
  for (let i = 0; i < ms; i += 16) {
    reloj += 16;
    ultimo = p.paso({ x, y, ok }, golpear, caja);
    if (ultimo.disparo) return ultimo;
  }
  return ultimo;
};

const fallos = [];
const check = (nombre, cond, extra = "") =>
  console.log(cond ? `  ok    ${nombre}` : (fallos.push(nombre),
    `  FALLA ${nombre} ${extra}`));

// 1. Sostener dispara; soltar antes, no.
reloj = 0;
let p = new Permanencia();
check("sostener 1.2 s dispara", correr(p, 50, 50, 1300).disparo === "a");
reloj = 0; p = new Permanencia();
check("soltar antes no dispara", correr(p, 50, 50, 800).disparo === null);

// 2. Refractario: la misma fijación no dispara dos veces seguidas.
reloj = 0; p = new Permanencia();
correr(p, 50, 50, 1300);
check("refractario bloquea el segundo disparo inmediato",
      correr(p, 50, 50, 500).disparo === null);
check("pasado el refractario vuelve a disparar",
      correr(p, 50, 50, 1400).disparo === "a");

// 3. Pegado: temblar apenas afuera del borde NO reinicia.
reloj = 0; p = new Permanencia();
correr(p, 95, 50, 600);
const estado = p.paso({ x: 108, y: 50, ok: true }, golpear, caja);  // entre a y b
check("el temblor fuera del borde conserva el objetivo",
      estado.objetivo === "a", `-> ${estado.objetivo}`);
check("y completa igual", correr(p, 95, 50, 700).disparo === "a");

// 4. Parpadeo: perder la cara 200 ms no reinicia; 500 ms sí.
reloj = 0; p = new Permanencia();
correr(p, 50, 50, 600);
correr(p, 50, 50, 200, false);
check("un parpadeo corto no reinicia", p.objetivo === "a");
reloj = 0; p = new Permanencia();
correr(p, 50, 50, 600);
correr(p, 50, 50, 500, false);
check("perder la cara medio segundo sí reinicia", p.objetivo === null);

// 5. Cambiar de celda de verdad reinicia la cuenta.
reloj = 0; p = new Permanencia();
correr(p, 50, 50, 900);
const r5 = correr(p, 160, 50, 400);
check("cambiar de celda reinicia", r5.disparo === null && p.objetivo === "b");

console.log(fallos.length ? `\n${fallos.length} FALLAS` : "\nTodo OK");
process.exit(fallos.length ? 1 : 0);
