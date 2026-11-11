/* La elección de la constante de cresta, contra el problema real.
 *
 * Los datos de acá son sintéticos, y eso es una debilidad conocida en este
 * proyecto: cuatro intentos de reproducir fallos con datos inventados
 * fracasaron. Por eso lo que se simula NO es "ruido": es la geometría exacta
 * que hace fallar a WebGazer, y que sí está medida.
 *
 *   120 rasgos, 45 muestras  -> XᵀX tiene rango 45 como mucho: es SINGULAR
 *   rasgos muy correlacionados -> son parches de ojo, píxeles vecinos
 *   9 puntos x 5 muestras      -> las 5 de un punto son casi iguales
 *
 * Con esa forma, un ajuste sin regularizar pasa exacto por los nueve puntos y
 * entre medio lo decide el espacio nulo. La prueba es que la validación lo
 * note y elija otra cosa.
 *
 * Los parches se generan DESDE la mirada -que es la dirección causal real: la
 * mirada mueve el ojo, no al revés- y la regresión tiene que invertir eso.
 */
import { elegirCresta, resolver, RELATIVOS } from "../js/cresta.js";

const fallos = [];
const ok = (n, c, x = "") =>
  console.log(c ? `  ok    ${n}` : (fallos.push(n), `  FALLA ${n} ${x}`));

/* ------------------------------------------------- el solucionador primero */

// Si `resolver` estuviera mal, todo lo de abajo mediría otra cosa.
{
  const M = Float64Array.from([2, 1, 0, 1, 3, 1, 0, 1, 2]);
  const B = Float64Array.from([5, 1, 10, 2, 7, 3]);   // 3x2, por filas
  const z = resolver(M, 3, B, 2);
  // Sistema: comprobado contra la solución exacta de 2x+y=5, x+3y+z=10, y+2z=7
  const esperado = [1.5, 2, 2.5];
  let peor = 0;
  for (let i = 0; i < 3; i++) peor = Math.max(peor, Math.abs(z[i * 2] - esperado[i]));
  ok("resolver da la solución exacta de un sistema conocido", peor < 1e-9,
     `${z[0]}, ${z[2]}, ${z[4]}`);
}

/* ----------------------------------------------- datos con la forma mala */

let semilla = 20251111;
const azar = () => (semilla = (semilla * 1103515245 + 12345) & 0x7fffffff)
                 / 0x7fffffff;

const W = 1536, H = 737, D = 120, POR_PUNTO = 5;
const PUNTOS = [0.1, 0.5, 0.9].flatMap(
  (fy) => [0.1, 0.5, 0.9].map((fx) => [fx * W, fy * H]));

/** Un parche de ojo falso, con la jerarquía real de lo que lo mueve.
 *
 * Un parche de ojo depende MUCHO más de la pose de la cabeza y de la luz que
 * de hacia dónde se mira: la mirada es una modulación chica encima de una
 * apariencia que domina otra cosa. Y dentro de un mismo punto de calibración
 * -cinco muestras en poco más de un segundo- la pose y la luz son las MISMAS.
 *
 * Esa es la estructura que importa, y sin ella la simulación no sirve:
 *
 * - un ajuste sin regularizar aprovecha las direcciones de la pose, que no
 *   tienen nada que ver con la mirada, para pasar exacto por los nueve puntos;
 * - en un punto nuevo esa pose no está y el modelo se va a cualquier lado;
 * - y dejando afuera una MUESTRA en vez del punto entero, sus cuatro gemelas
 *   llevan la misma pose y la delatan, así que la validación aprueba todo.
 */
/* El nivel de estorbo está calibrado contra el único número real que hay:
 * con 10, el ajuste sin regularizar erra 186 px de media, y WebGazer sin
 * corregir midió 206 px sobre esta misma pantalla. No se eligió para que la
 * prueba pase, sino para que el problema simulado sea tan difícil como el
 * medido. Con estorbos mucho mayores la mirada deja de ser recuperable y la
 * simulación describiría un problema que no es el de WebGazer. */
function generar({ mirada = 30, pose = 10, ruido = 6 } = {}) {
  // Direcciones LISAS: píxeles vecinos se parecen, como en una imagen.
  const liso = (f) => {
    const v = new Float64Array(D);
    for (let i = 0; i < D; i++) v[i] = Math.sin(i * f) + 0.5 * Math.cos(i * f * 0.37);
    return v;
  };
  const base = liso(0.11), dirX = liso(0.05), dirY = liso(0.08);
  const estorbos = [liso(0.023), liso(0.041), liso(0.067), liso(0.095)];

  const X = [], yx = [], yy = [], grupos = [];
  for (let p = 0; p < PUNTOS.length; p++) {
    const [gx, gy] = PUNTOS[p];
    // Pose y luz de ESTE punto: iguales para sus cinco muestras.
    const coef = estorbos.map(() => pose * (azar() - 0.5));
    for (let m = 0; m < POR_PUNTO; m++) {
      const fila = new Array(D);
      for (let i = 0; i < D; i++) {
        let v = 128 + 40 * base[i]
              + mirada * dirX[i] * (gx / W - 0.5)
              + mirada * dirY[i] * (gy / H - 0.5)
              + ruido * (azar() - 0.5);
        for (let e = 0; e < estorbos.length; e++) v += coef[e] * estorbos[e][i];
        fila[i] = Math.max(0, Math.min(255, v));
      }
      X.push(fila); yx.push(gx); yy.push(gy); grupos.push(p);
    }
  }
  return { X, yx, yy, grupos };
}

const { X, yx, yy, grupos } = generar();
console.log(`  ${X.length} muestras x ${D} rasgos en ${PUNTOS.length} puntos`);
ok("los datos quedan subdeterminados, como los de verdad", X.length < D,
   `${X.length} vs ${D}`);

const t0 = performance.now();
const sel = elegirCresta(X, yx, yy, grupos);
const ms = performance.now() - t0;

console.log(`  tabla (relativo -> error px):`);
for (const f of sel.tabla) console.log(`    ${String(f.relativo).padStart(7)}  ${f.error.toFixed(0)}`);
console.log(`  elegido ${sel.relativo}  (${sel.cresta.toPrecision(3)})  en ${ms.toFixed(0)} ms`);

ok("elige algo, no se cae", sel !== null);
ok("NO elige el casi-cero, que es lo que trae WebGazer de fábrica",
   sel.relativo > 1e-11, String(sel.relativo));
ok("y el error validado baja respecto de no regularizar",
   sel.error < sel.mejoraSobre,
   `${sel.error.toFixed(0)} vs ${sel.mejoraSobre.toFixed(0)}`);
ok("la mejora es real, no decorativa", sel.error < sel.mejoraSobre * 0.85,
   `${(100 * (1 - sel.error / sel.mejoraSobre)).toFixed(0)}%`);

/* El óptimo tiene que caer DENTRO de la escalera de candidatos. Si cayera en
 * un extremo querría decir que el verdadero está más allá y que la escalera
 * está mal puesta: elegiría el borde y parecería que funcionó igual. */
ok("el óptimo cae dentro de la escalera, no en un borde",
   sel.relativo !== RELATIVOS[0]
   && sel.relativo !== RELATIVOS[RELATIVOS.length - 1],
   String(sel.relativo));

/* Tiene que correr una vez al cerrar la calibración, con la persona esperando.
 * Si esto se va a varios segundos hay que enterarse acá y no en la cara del
 * usuario. */
ok("tarda menos de 4 s con el tamaño real", ms < 4000, `${ms.toFixed(0)} ms`);

/* --------------------------------- demasiada cresta tampoco puede ganar */

// El extremo opuesto: con un relativo enorme los coeficientes se van a cero y
// el modelo predice el origen. La validación tiene que rechazarlo sola.
const grande = elegirCresta(X, yx, yy, grupos, [1e-11, 0.01, 1e6]);
ok("una cresta descomunal no gana", grande.relativo !== 1e6,
   String(grande.relativo));

/* ------------------------------------------------------- los repliegues */

ok("con menos de tres puntos no valida nada",
   elegirCresta(X.slice(0, 10), yx.slice(0, 10), yy.slice(0, 10),
                grupos.slice(0, 10)) === null);
ok("con las etiquetas corridas no inventa un resultado",
   elegirCresta(X, yx, yy, grupos.slice(0, -1)) === null);
ok("sin datos tampoco", elegirCresta([], [], [], []) === null);

/* ----------------------- dejar un PUNTO afuera, no una muestra suelta */

// Si se dejara afuera una muestra en vez del punto entero, sus cuatro gemelas
// se quedan en el entrenamiento y la delatan: cualquier cresta aprueba, y la
// validación deja de distinguir. Se comprueba que con grupos de a uno el
// casi-cero ya no se descarta.
const sueltos = grupos.map((_, i) => i);
const porMuestra = elegirCresta(X, yx, yy, sueltos);
console.log(`  dejando una MUESTRA afuera elegiría ${porMuestra.relativo}`
          + ` (error ${porMuestra.error.toFixed(0)} px)`);
// No hace falta que la trampa sea enorme para ser trampa: alcanza con que
// el error validado salga sistemáticamente MÁS CHICO que el real, porque
// entonces la elección se hace sobre un número que no es el que importa.
ok("dejar una muestra afuera subestima el error: es la trampa",
   porMuestra.error < sel.error * 0.8,
   `${porMuestra.error.toFixed(0)} vs ${sel.error.toFixed(0)}`);

console.log(fallos.length ? `\n${fallos.length} FALLAS` : "\nTodo OK");
process.exit(fallos.length ? 1 : 0);
