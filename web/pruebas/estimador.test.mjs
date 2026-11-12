/* Lo que tiene que cumplir el estimador de WebGazer para que la comparación
 * contra el propio signifique algo.
 *
 * El riesgo acá no es que WebGazer ande mal: es que ande mal POR UN MOTIVO QUE
 * NO ES WEBGAZER, y que la conclusión termine siendo "los dos fallan igual"
 * cuando en realidad lo que falló fue la integración. Dos cosas lo provocan:
 *
 * 1. **El límite de 50 muestras.** WebGazer guarda el entrenamiento en un
 *    buffer circular de cincuenta posiciones (`DataWindow(50)` en
 *    `util_regression.InitRegression`). Las 270 muestras que graba el
 *    estimador propio dejarían vivas solo las últimas 50, o sea el último
 *    punto y medio de los nueve. El modelo habría quedado entrenado contra una
 *    esquina, y el síntoma -sesgo enorme, temblor chico- es exactamente el que
 *    estamos persiguiendo del otro lado.
 *
 * 2. **El cupo tiene que entrar en la ventana de grabado.** Las muestras se
 *    separan en el tiempo para que no salgan todas de la misma décima de
 *    segundo. Si la separación crece sin que crezca la ventana, los puntos se
 *    quedan cortos y nadie se entera.
 *
 * Y la puerta de calidad tiene que funcionar con los dos estimadores, midiendo
 * lo mismo: si `recorrido` se calculara distinto en cada uno, un umbral de
 * 0.45 querría decir dos cosas diferentes según cuál esté puesto.
 */
import { readFileSync } from "node:fs";
import { recorridoDe } from "../js/modelo.js";
import { FalloEstimador } from "../js/estimador.js";
import { EstimadorWebGazer, PUNTOS_16 } from "../js/webgazer_est.js";
import { elegirCresta, ajustarCrudo, predecirCentrado } from "../js/cresta.js";

const fallos = [];
const ok = (n, c, x = "") =>
  console.log(c ? `  ok    ${n}` : (fallos.push(n), `  FALLA ${n} ${x}`));

const W = 1536, H = 737;
const REJILLA = [0.1, 0.5, 0.9].flatMap(
  (fy) => [0.1, 0.5, 0.9].map((fx) => [fx * W, fy * H]));

/* ------------------------------------------- el presupuesto de muestras */

const wgz = new EstimadorWebGazer();
const VENTANA_WG = 50;   // DataWindow(50), ver util_regression.InitRegression

const total = wgz.muestrasPorPunto * 9;
console.log(`  ${wgz.muestrasPorPunto} muestras x 9 puntos = ${total} `
          + `(entran ${VENTANA_WG})`);
ok("las muestras de los nueve puntos entran en la ventana de WebGazer",
   total <= VENTANA_WG, `${total} > ${VENTANA_WG}`);
ok("y sobra margen para algún clic suelto",
   VENTANA_WG - total >= 3, `solo sobran ${VENTANA_WG - total}`);

/* La ventana de grabado la fija `app.js`. Se lee del archivo en vez de
 * copiarla acá: una copia se desincroniza en silencio, que es justo el fallo
 * que esta prueba existe para evitar. */
const fuente = readFileSync(new URL("../js/app.js", import.meta.url), "utf8");
const MS_GRABAR = Number(/const MS_GRABAR = (\d+)/.exec(fuente)?.[1]);
ok("se pudo leer MS_GRABAR de app.js", Number.isFinite(MS_GRABAR), MS_GRABAR);

// La primera muestra de cada punto sale enseguida (el reloj arranca en cero),
// así que las que entran son 1 + floor(ventana / separacion).
const caben = 1 + Math.floor(MS_GRABAR / wgz.msEntreMuestras);
console.log(`  en ${MS_GRABAR} ms, separadas ${wgz.msEntreMuestras} ms, `
          + `caben ${caben}`);
ok("el cupo se alcanza dentro de la ventana de grabado",
   caben >= wgz.muestrasPorPunto,
   `caben ${caben}, se piden ${wgz.muestrasPorPunto}`);

/* ------------------------------------------------- un regresor de mentira */

/** Imita lo que el estimador le pide de verdad al regresor de WebGazer:
 *  el contador de muestras, los datos guardados y `predict`.
 *
 * @param {(eyes:number)=>[number,number]} mapa  qué predice para cada muestra
 */
/** Rasgos de ojo de mentira: 120 valores en 0..255 que dependen de hacia
 *  dónde se está mirando, que es lo que hace falta para que la validación
 *  cruzada tenga algo real que validar. */
function rasgosDe([ox, oy], i) {
  const v = new Array(120);
  for (let k = 0; k < 120; k++) {
    v[k] = Math.max(0, Math.min(255, 128
      + 50 * Math.sin(k * 0.05) * (ox / W - 0.5)
      + 50 * Math.cos(k * 0.08) * (oy / H - 0.5)
      + 6 * Math.sin(k * 1.7 + i)));
  }
  return v;
}

function regresorFalso(mapa, ventana = VENTANA_WG, { conRasgos = true } = {}) {
  return {
    ridgeParameter: 1e-5,
    eyeFeaturesClicks: { data: [], length: 0 },
    screenXClicksArray: { data: [] },
    screenYClicksArray: { data: [] },
    dataClicks: { data: [] },
    predict(eyes) {
      const [x, y] = mapa(eyes);
      return { x, y };
    },
    _tomar(objetivo) {
      if (this.dataClicks.data.length >= ventana) return false;   // se satura
      const i = this.dataClicks.data.length;
      this.dataClicks.data.push({ eyes: i, screenPos: objetivo, type: "click" });
      if (conRasgos) {
        this.eyeFeaturesClicks.data.push(rasgosDe(objetivo, i));
        this.screenXClicksArray.data.push([objetivo[0]]);
        this.screenYClicksArray.data.push([objetivo[1]]);
      }
      this.eyeFeaturesClicks.length = this.dataClicks.data.length;
      return true;
    },
  };
}
function conRegresor(reg, { sinRasgos = false } = {}) {
  const e = new EstimadorWebGazer();
  e.wg = {
    getRegression: () => [reg],
    recordScreenPosition: (x, y) => { if (!sinRasgos) reg._tomar([x, y]); },
    clearData: async () => {},
  };
  return e;
}

/** Calibra los nueve puntos con el cupo completo. */
function calibrar(e) {
  let grabadas = 0;
  for (let p = 0; p < 9; p++) {
    for (let k = 0; k < e.muestrasPorPunto; k++) {
      if (e.grabar(null, REJILLA[p], p)) grabadas += 1;
    }
  }
  return grabadas;
}

/* --------------------------------------------------- grabar confirma solo */

// Sin rasgos de ojo, `recordScreenPosition` no hace NADA y lo hace en
// silencio. El estimador tiene que devolver false y no contar la muestra, o la
// barra de la calibración avanzaría sobre datos que no existen.
const mudo = conRegresor(regresorFalso(() => [0, 0]), { sinRasgos: true });
ok("sin rasgos de ojo, grabar devuelve false",
   mudo.grabar(null, [100, 100], 0) === false);
ok("y no deja grupos colgados", mudo._grupos.length === 0,
   String(mudo._grupos.length));

// Saturada la ventana, tampoco puede confirmar: así `_grupos` nunca se
// desalinea de `dataClicks`, que es de donde sale el diagnóstico.
const lleno = conRegresor(regresorFalso(() => [0, 0], 4));
for (let i = 0; i < 10; i++) lleno.grabar(null, [i, i], 0);
ok("una vez llena la ventana deja de confirmar",
   lleno._grupos.length === 4, String(lleno._grupos.length));

/* ------------------------------------------------ la puerta de calidad */

// Sano: predice cerca del objetivo con el que se grabó.
const sano = conRegresor(regresorFalso((i) => {
  const p = REJILLA[Math.floor(i / 5)] || REJILLA[8];
  return [p[0] + 40, p[1] - 30];
}));
// Una sola vez: `ok` evalúa el tercer argumento siempre, así que meter
// `calibrar(sano)` ahí adentro grabaría otra tanda y ensuciaría el ajuste.
const grabadasSano = calibrar(sano);
ok("calibra las 45", grabadasSano === 45, String(grabadasSano));

const dSano = await sano.ajustar();
console.log(`  recorrido sano:      ${dSano.recorrido.toFixed(2)}`);
ok("un estimador sano recorre casi todo", dSano.recorrido > 0.8,
   dSano.recorrido?.toFixed(2));
/* Lo que esta prueba tiene que fijar NO es si la validación cruzada elige
 * bien -eso lo mide `cresta.test.mjs`- sino si el estimador APLICA lo que
 * eligió. Por eso se compara contra el valor exacto que devuelve `elegirCresta`
 * con las mismas entradas, y no contra "distinto del de fábrica": con rasgos
 * fáciles la elección puede dar casi el de fábrica por motivos legítimos, y
 * esa afirmación pasaría o fallaría por casualidad.
 *
 * El fallo que esto atrapa es el de verdad caro: que la cresta se calcule, se
 * informe en el diagnóstico, y nunca llegue al regresor que predice. */
const reg2 = sano.wg.getRegression()[0];
const esperada = elegirCresta(
  reg2.eyeFeaturesClicks.data,
  reg2.screenXClicksArray.data.map((v) => v[0]),
  reg2.screenYClicksArray.data.map((v) => v[0]),
  sano._grupos);
console.log(`  cresta elegida:      ${reg2.ridgeParameter.toPrecision(3)}`
          + `  (de fábrica 0.00001)`);
ok("el regresor queda con la cresta que eligió la validación",
   reg2.ridgeParameter === esperada.cresta,
   `${reg2.ridgeParameter} vs ${esperada.cresta}`);
ok("y esa cresta es la que el diagnóstico informa",
   dSano.alfa === esperada.cresta && dSano.errorCV === esperada.error,
   `${dSano.alfa} / ${dSano.errorCV}`);
ok("y el diagnóstico informa el error validado, en píxeles",
   Number.isFinite(dSano.errorCV) && /dejando un punto afuera/.test(dSano.nota),
   dSano.nota);

/* Si los rasgos no estuvieran guardados, validar contra etiquetas que no les
 * corresponden sería peor que no validar. Tiene que replegarse y decirlo. */
const ciego = conRegresor(regresorFalso(() => [700, 300], VENTANA_WG,
                                        { conRasgos: false }));
calibrar(ciego);
const dCiego = await ciego.ajustar();
ok("sin rasgos guardados no valida y deja la de fábrica",
   ciego.wg.getRegression()[0].ridgeParameter === 1e-5
   && /de fábrica/.test(dCiego.nota), dCiego.nota);

// Colapsado: predice casi el centro mire donde mire. Es el fallo que se está
// persiguiendo, y la puerta tiene que atraparlo también acá.
const roto = conRegresor(regresorFalso((i) =>
  [W / 2 + (i % 3), H / 2 - (i % 2)]));
calibrar(roto);
const dRoto = await roto.ajustar();
console.log(`  recorrido colapsado: ${dRoto.recorrido.toFixed(2)}`);
ok("un estimador colapsado queda bajo el umbral", dRoto.recorrido < 0.45,
   dRoto.recorrido?.toFixed(2));
ok("y la puerta los separa", dSano.recorrido > 0.45 && dRoto.recorrido < 0.45);

/* Pocas muestras: tiene que decirlo, no entregar un diagnóstico inventado. */
const flaco = conRegresor(regresorFalso(() => [0, 0]));
for (let p = 0; p < 2; p++) flaco.grabar(null, REJILLA[p], p);
let aviso = null;
try { await flaco.ajustar(); } catch (e) { aviso = e; }
ok("con pocas muestras avisa en vez de inventar",
   aviso instanceof FalloEstimador, String(aviso));

/* ------------------------------- el recorrido es el MISMO en los dos lados */

const pred = REJILLA.map(([x, y]) => [x * 0.5 + W * 0.25, y * 0.5 + H * 0.25]);
ok("recorridoDe mide la mitad cuando las predicciones se encogen a la mitad",
   Math.abs(recorridoDe(pred, REJILLA) - 0.5) < 1e-9,
   String(recorridoDe(pred, REJILLA)));
ok("una predicción constante da recorrido cero",
   recorridoDe(REJILLA.map(() => [700, 300]), REJILLA) === 0);
ok("con menos de dos puntos no inventa un número",
   Number.isNaN(recorridoDe([[1, 1]], [[1, 1]])));

/* ------------------------------ la regresión propia sobre WebGazer
 *
 * Medido sin el artefacto del filtro, WebGazer erró 105 px, y elegirle la
 * cresta no cambió nada (109 px validados con la de fábrica, 109 con la
 * mejor). Lo que se prueba acá es el reemplazo: los rasgos siguen siendo los
 * de WebGazer, pero el ajuste es propio -con intercepto, dieciséis lugares y
 * sin tope de muestras-.
 *
 * Los parches se simulan como en `cresta.test.mjs`: dependen de la mirada,
 * pero sobre todo de la pose y la luz, que son las mismas dentro de un punto.
 * Y tienen un brillo medio de 128 que no dice nada de la mirada: es lo que un
 * ajuste sin intercepto tiene que explicar con los mismos coeficientes.
 */
{
  let sem = 99991;
  const azar = () => (sem = (sem * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  const liso = (f) => Array.from({ length: 120 },
    (_, i) => Math.sin(i * f) + 0.5 * Math.cos(i * f * 0.37));
  const base = liso(0.11), dX = liso(0.05), dY = liso(0.08);
  const estorbos = [liso(0.023), liso(0.041), liso(0.067), liso(0.095)];

  /** Un "ojo" mirando a (x, y), con la pose de ese momento. */
  const ojo = ([x, y], pose) => {
    const f = new Array(120);
    for (let i = 0; i < 120; i++) {
      let v = 128 + 40 * base[i] + 30 * dX[i] * (x / W - 0.5)
            + 30 * dY[i] * (y / H - 0.5) + 6 * (azar() - 0.5);
      for (let e = 0; e < 4; e++) v += pose[e] * estorbos[e][i];
      f[i] = Math.max(0, Math.min(255, v));
    }
    return { f };
  };
  const poseNueva = () => estorbos.map(() => 10 * (azar() - 0.5));

  /** Un WebGazer de mentira con lo que el estimador usa de verdad. */
  function webgazerFalso() {
    const reg = {
      eyeFeaturesClicks: { data: [], length: 0 },
      addData() { this.llamadasViejas = (this.llamadasViejas || 0) + 1; },
      predict() { return { x: -1, y: -1 }; },
    };
    const wg = {
      ojoActual: null,
      util: { getEyeFeats: (eyes) => eyes.f },
      getRegression: () => [reg],
      recordScreenPosition(x, y, tipo) { reg.addData(wg.ojoActual, [x, y], tipo); },
      clearData: async () => {},
    };
    return { wg, reg };
  }

  const { wg, reg } = webgazerFalso();
  const e = new EstimadorWebGazer();
  e.wg = wg;
  ok("toma la regresión cuando WebGazer expone sus rasgos",
     e._tomarRegresion() === true && e._propia === true);
  ok("y pasa a calibrar en dieciséis lugares", e.puntos === PUNTOS_16
     && PUNTOS_16.length === 16);
  ok("los dieciséis son distintos entre sí",
     new Set(PUNTOS_16.map((p) => p.join())).size === 16);
  const prueba9 = [0.25, 0.5, 0.75].flatMap(
    (fx) => [0.2, 0.5, 0.8].map((fy) => [fx, fy].join()));
  ok("y ninguno coincide con los de la prueba de precisión",
     PUNTOS_16.every((p) => !prueba9.includes(p.join())));
  const salto = Math.max(...PUNTOS_16.slice(1).map((p, i) =>
    Math.hypot(p[0] - PUNTOS_16[i][0], p[1] - PUNTOS_16[i][1])));
  ok("la serpentina no tiene saltos largos", salto < 0.3, salto.toFixed(2));
  ok("el cupo por punto entra en la ventana de grabado",
     1 + Math.floor(MS_GRABAR / e.msEntreMuestras) >= e.muestrasPorPunto,
     `${e.muestrasPorPunto} cada ${e.msEntreMuestras} ms`);

  ok("antes de calibrar no predice nada", reg.predict({ f: ojo([1, 1], [0, 0, 0, 0]).f }) === null);

  // Sin ojos todavía: no puede confirmar la muestra.
  wg.ojoActual = null;
  ok("sin rasgos de ojo, grabar devuelve false",
     e.grabar(null, [100, 100], 0) === false && e._grupos.length === 0);

  // La calibración completa.
  let grabadas = 0;
  PUNTOS_16.forEach(([fx, fy], g) => {
    const objetivo = [fx * W, fy * H], pose = poseNueva();
    for (let k = 0; k < e.muestrasPorPunto; k++) {
      wg.ojoActual = ojo(objetivo, pose);
      if (e.grabar(null, objetivo, g)) grabadas += 1;
    }
  });
  ok("graba todas, sin el tope de 50 de WebGazer", grabadas === 192
     && e._X.length === 192, String(grabadas));

  const d = await e.ajustar();
  console.log(`  ${d.nota}`);
  ok("el diagnóstico trae el error validado y el recorrido",
     Number.isFinite(d.errorCV) && d.recorrido > 0.6, JSON.stringify(d));
  ok("y dice que la regresión es la propia", /regresión propia/.test(d.nota));

  // Lo que importa: lugares que NO se calibraron, con poses nuevas.
  const noVistos = [0.25, 0.5, 0.75].flatMap(
    (fx) => [0.2, 0.5, 0.8].map((fy) => [fx * W, fy * H]));
  const errorDe = (predecir) => {
    let s = 0, n = 0;
    for (let rep = 0; rep < 6; rep++) {
      for (const t of noVistos) {
        const p = predecir(ojo(t, poseNueva()));
        s += Math.hypot(p[0] - t[0], p[1] - t[1]); n += 1;
      }
    }
    return s / n;
  };
  const errPropia = errorDe((o) => { const q = reg.predict(o); return [q.x, q.y]; });

  // Contra lo que hacía WebGazer: nueve lugares, cinco muestras, sin
  // intercepto y con la cresta de fábrica.
  const X9 = [], yx9 = [], yy9 = [];
  for (const t of REJILLA) {
    const pose = poseNueva();
    for (let k = 0; k < 5; k++) { X9.push(ojo(t, pose).f); yx9.push(t[0]); yy9.push(t[1]); }
  }
  const fabrica = ajustarCrudo(X9, yx9, yy9, 1e-5);
  const errFabrica = errorDe((o) => predecirCentrado(fabrica, o.f));
  console.log(`  en lugares no vistos:  de fábrica ${errFabrica.toFixed(0)} px`
            + `   propia ${errPropia.toFixed(0)} px`);
  // En esta simulación la ventaja es moderada -del orden del 10 al 15%-
  // porque lo que domina es la pose de cada punto, que ningún ajuste puede
  // adivinar. No se le exige más que eso: cuánto gana de verdad lo dice la
  // medición en la máquina, no este número.
  ok("la regresión propia erra menos donde no se calibró",
     errPropia < errFabrica * 0.95,
     `${errPropia.toFixed(0)} vs ${errFabrica.toFixed(0)}`);

  // Recalibrar empieza de cero y conserva los reemplazos.
  await e.limpiarDatos();
  ok("limpiar borra el modelo y las muestras",
     e._X.length === 0 && e._modelo === null && reg.predict(ojo([5, 5], [0, 0, 0, 0])) === null);
  wg.ojoActual = ojo([300, 300], poseNueva());
  ok("y después de limpiar se sigue grabando por la vía propia",
     e.grabar(null, [300, 300], 0) === true && !reg.llamadasViejas);

  // Calibración pobre: avisa.
  let aviso = null;
  try { await e.ajustar(); } catch (err) { aviso = err; }
  ok("con una sola muestra avisa en vez de ajustar", aviso instanceof FalloEstimador);

  // Si WebGazer no expone la función de rasgos, no se toca nada.
  const viejo = new EstimadorWebGazer();
  viejo.wg = { getRegression: () => [{}], util: {} };
  ok("sin la función de rasgos queda todo como de fábrica",
     viejo._tomarRegresion() === false && viejo._propia === false
     && viejo.puntos === null && viejo.muestrasPorPunto === 5);
}

console.log(fallos.length ? `\n${fallos.length} FALLAS` : "\nTodo OK");
process.exit(fallos.length ? 1 : 0);
