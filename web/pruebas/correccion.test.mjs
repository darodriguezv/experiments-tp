/* La corrección de ganancia, contra la medición real que la motivó.
 *
 * Los nueve desvíos de abajo son los de una sesión de WebGazer medida de
 * verdad. Están acá y no generados sintéticamente a propósito: cuatro intentos
 * de reproducir fallos de este proyecto con datos inventados fracasaron, y uno
 * de ellos aprobó un alpha que en uso real colapsaba el modelo. Los datos
 * reales no se pueden equivocar sobre lo que pasó.
 *
 * Lo que tienen que demostrar:
 *
 * 1. que el veredicto viejo -"desplazamiento uniforme"- era FALSO sobre esta
 *    medición, y que el nuevo lo llama por su nombre;
 * 2. que una recta por eje baja el sesgo de 210 px a unos 45;
 * 3. que la corrección se degrada a corrimiento puro cuando no puede confiar
 *    en la ganancia, en vez de amplificar el temblor.
 */
import { ajusteEje, Correccion } from "../js/correccion.js";
import { PruebaPrecision, ajusteAfin, informe, curvatura }
  from "../js/precision.js";

const fallos = [];
const ok = (n, c, x = "") =>
  console.log(c ? `  ok    ${n}` : (fallos.push(n), `  FALLA ${n} ${x}`));

/* Medición real: WebGazer 3.5.3, 825 muestras en 9 puntos, 1536x737.
 * (objetivo x, objetivo y, desvío x, desvío y) */
const MEDIDO = [
  [384, 147, 25, 141], [768, 147, 173, 88], [1152, 147, 263, 106],
  [384, 368, -13, 161], [768, 368, 164, 54], [1152, 368, 328, 76],
  [384, 588, 43, 93], [768, 588, 157, -19], [1152, 588, 335, 50],
];

/** Arma una prueba ya cerrada a partir de desvíos por punto. */
function pruebaCon(filas, temblor = 26) {
  const p = new PruebaPrecision();
  p.porPunto = filas.map(([tx, ty, dx, dy]) => ({
    tx, ty, n: 90, dx, dy,
    sesgo: Math.hypot(dx, dy),
    temblor,
    err: [Math.hypot(dx, dy)],
    dxs: [Math.abs(dx)], dys: [Math.abs(dy)],
  }));
  p.terminada = true;
  return p;
}

/* ------------------------------------------- el diagnóstico sobre lo real */

const real = pruebaCon(MEDIDO);
const af = ajusteAfin(real);
console.log(`  ganancia medida:  x ${af.ex.a.toFixed(3)}  y ${af.ey.a.toFixed(3)}`);
console.log(`  sesgo ahora ${real.resumen().sesgo.toFixed(0)} px`
          + ` -> corregido ${af.residuo.toFixed(0)} px`);

ok("la ganancia horizontal sale muy por encima de 1",
   af.ex.a > 1.3 && af.ex.a < 1.45, af.ex.a.toFixed(3));
ok("la vertical, por debajo", af.ey.a > 0.78 && af.ey.a < 0.90,
   af.ey.a.toFixed(3));
ok("corregir la ganancia baja el sesgo a menos de un cuarto",
   af.residuo < real.resumen().sesgo / 4,
   `${af.residuo.toFixed(0)} vs ${real.resumen().sesgo.toFixed(0)}`);

const texto = informe(real);
ok("el informe dice que el problema es la GANANCIA",
   /La GANANCIA está mal/.test(texto));
ok("y YA NO dice que sea un desplazamiento uniforme",
   !/desplazamiento uniforme/i.test(texto)
   && !/Corrimiento UNIFORME/.test(texto),
   texto.split("\n").slice(-2).join(" | "));

/* Un corrimiento de verdad parejo tiene que seguir llamándose así: el
 * veredicto nuevo no puede ser "siempre es la ganancia". */
const parejo = pruebaCon(MEDIDO.map(([tx, ty]) => [tx, ty, 180, 120]));
ok("un corrimiento realmente parejo se sigue llamando uniforme",
   /Corrimiento UNIFORME/.test(informe(parejo)));
ok("y ahí la ganancia sale 1.00",
   Math.abs(ajusteAfin(parejo).ex.a - 1) < 1e-9);

/* ------------------------------------------------- la corrección en sí */

// Cinco puntos, con el mismo defecto que se midió: pred = 1.378*t - 126 en x.
const CINCO = [[0.2, 0.2], [0.8, 0.2], [0.5, 0.5], [0.2, 0.8], [0.8, 0.8]]
  .map(([fx, fy]) => [fx * 1536, fy * 737]);
const crudas = CINCO.map(([x, y]) => [1.378 * x - 126, 0.841 * y + 142]);

const c = Correccion.ajustar(CINCO, crudas);
console.log(`  corrección ajustada: ${c.resumen}`);
ok("recupera la ganancia horizontal", Math.abs(c.ex.a - 1.378) < 0.01,
   c.ex.a.toFixed(3));
ok("recupera la vertical", Math.abs(c.ey.a - 0.841) < 0.01, c.ey.a.toFixed(3));

// Y sobre puntos DISTINTOS de los cinco con los que se ajustó.
let peor = 0;
for (const [fx, fy] of [[0.25, 0.25], [0.75, 0.5], [0.5, 0.75], [0.9, 0.9]]) {
  const t = [fx * 1536, fy * 737];
  const cruda = [1.378 * t[0] - 126, 0.841 * t[1] + 142];
  const [x, y] = c.aplicar(cruda);
  peor = Math.max(peor, Math.hypot(x - t[0], y - t[1]));
}
ok("y corrige también donde no se ajustó", peor < 1, `${peor.toFixed(2)} px`);

/* ------------------------------------------------- los repliegues seguros */

// Un solo punto: no hay recta posible, tiene que quedar el corrimiento puro.
// Es exactamente la corrección de deriva que había antes, así que esto es
// generalizarla y no reemplazarla.
const uno = Correccion.ajustar([[768, 368]], [[968, 468]]);
ok("con un solo punto la ganancia queda en 1", uno.ex.a === 1 && uno.ey.a === 1);
const [ux, uy] = uno.aplicar([968, 468]);
ok("y el corrimiento se resta igual",
   Math.abs(ux - 768) < 1e-6 && Math.abs(uy - 368) < 1e-6, `${ux}, ${uy}`);

// Una ganancia absurda -el modelo casi no se mueve- no se debe invertir:
// dividir por 0.1 multiplicaría el temblor por diez.
const chata = Correccion.ajustar(
  CINCO, CINCO.map(([x, y]) => [0.08 * x + 700, 0.08 * y + 330]));
ok("una ganancia fuera de rango se degrada a corrimiento",
   chata.ex.a === 1 && chata.ex.degradado === true, String(chata.ex.a));
ok("y el corrimiento que queda es el promedio, no el de la recta",
   Math.abs(chata.ex.b - chata.ex.corrimiento) < 1e-9);

/* ------------------------------------------------------------- el ajuste */

ok("sin variación en el objetivo no inventa pendiente",
   ajusteEje([500, 500, 500], [700, 710, 690]).a === 1);
ok("y ahí devuelve el corrimiento medio",
   Math.abs(ajusteEje([500, 500, 500], [700, 710, 690]).b - 200) < 1e-9);


/* -------------------------- la segunda medición: con la corrección puesta
 *
 * Misma sesión de WebGazer, pero ya con la corrección de ganancia aplicada.
 * El sesgo bajó de 210 px a 89, y sin embargo la ganancia que queda sigue sin
 * ser 1.00. El motivo no es que falten parámetros: es que el desvío se DOBLA.
 *
 * Por columna el desvío horizontal da -61, +36, +54 px. El salto de izquierda
 * a centro es +97 y el de centro a derecha +18: ninguna recta pasa por los
 * tres. Por eso la recta de la corrección -ajustada sobre 0.2/0.5/0.8- y la
 * que mide la prueba -sobre 0.25/0.5/0.75- coinciden en el centro y se separan
 * en los bordes: es como se separan dos cuerdas de una misma curva.
 *
 * Una afín 2D completa, con seis parámetros en vez de cuatro, baja el residuo
 * medio de 44 px a 33 y deja el máximo igual. Más parámetros no es la palanca;
 * esta prueba está para que no se vuelva a intentar por ahí. */

const MEDIDO2 = [
  [384, 147, 6, 24], [768, 147, 68, 14], [1152, 147, 63, 5],
  [384, 368, -90, -55], [768, 368, 21, -63], [1152, 368, 79, -92],
  [384, 588, -98, -81], [768, 588, 20, -73], [1152, 588, 19, -148],
];

const corregida = pruebaCon(MEDIDO2, 24);
const cv = curvatura(corregida);
console.log("  curvatura: x " + cv.x.toFixed(0) + " px   y " + cv.y.toFixed(0) + " px");

ok("la corrección bajó el sesgo a menos de la mitad",
   corregida.resumen().sesgo < real.resumen().sesgo / 2,
   corregida.resumen().sesgo.toFixed(0) + " vs " + real.resumen().sesgo.toFixed(0));

ok("el desvío horizontal se dobla, no es una recta", cv.x > 20, cv.x.toFixed(0));
ok("y el informe avisa que lo lineal tiene un piso",
   /una curva/.test(informe(corregida)));

/* Un desvío que SÍ es una recta no puede dar curvatura: si no, el aviso
 * aparecería siempre y dejaría de significar algo. */
const recto = pruebaCon(MEDIDO2.map(([tx, ty]) =>
  [tx, ty, 0.3 * tx - 230, -0.2 * ty + 74]));
const cvRecto = curvatura(recto);
ok("un desvío perfectamente lineal da curvatura cero",
   cvRecto.x < 1e-6 && cvRecto.y < 1e-6, cvRecto.x + ", " + cvRecto.y);
ok("y ahí el informe NO habla de curva", !/una curva/.test(informe(recto)));

/* ------------------- la tercera medición: la primera sin el filtro viejo
 *
 * Con el filtro que no se pasa de largo, la ganancia quedó en 0.92 y 1.02:
 * el problema de ganancia de las dos sesiones anteriores era, en buena parte,
 * el filtro. Lo que queda es otra cosa: un error que cambia de un punto al
 * vecino (de +98 a -155 px en la fila de arriba). El informe lo llamaba
 * "uniforme" porque el promedio era grande. */
const MEDIDO3 = [
  [384, 147, -39, 98], [768, 147, -41, -70], [1152, 147, -24, -155],
  [384, 368, -8, 7], [768, 368, -123, -105], [1152, 368, -100, -154],
  [384, 588, -10, -16], [768, 588, -96, 0], [1152, 588, -110, -85],
];
const tercera = pruebaCon(MEDIDO3, 11);
const af3 = ajusteAfin(tercera);
ok("sin el filtro viejo la ganancia medida queda cerca de 1",
   Math.abs(af3.ex.a - 1) < 0.15 && Math.abs(af3.ey.a - 1) < 0.15,
   `${af3.ex.a.toFixed(2)} / ${af3.ey.a.toFixed(2)}`);
ok("y el informe ya NO la llama uniforme",
   !/Corrimiento UNIFORME/.test(informe(tercera)),
   informe(tercera).split("\n").slice(-2).join(" | "));
ok("dice que el error cambia de un punto al de al lado",
   /cambia de un punto al de al lado/.test(informe(tercera)));

console.log(fallos.length ? `\n${fallos.length} FALLAS` : "\nTodo OK");
process.exit(fallos.length ? 1 : 0);
