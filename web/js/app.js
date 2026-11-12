/* El dispositivo completo, en el navegador.
 *
 * Orquesta las cuatro piezas: cámara -> puntos faciales -> rasgos -> modelo
 * de mirada -> permanencia -> tablero -> expansión semántica.
 *
 * ## Qué corre dónde
 *
 * TODO corre en la máquina de quien abre la página. El video nunca se sube: se
 * procesa cuadro a cuadro con MediaPipe en WebAssembly y se descarta. Lo que
 * la persona quiere decir tampoco sale, porque el modelo de lenguaje también
 * corre acá con WebGPU. La página se hostea; el cómputo no.
 *
 * Eso no es una decisión de arquitectura por comodidad: es el mismo argumento
 * del aparato. Un dispositivo que dice "me duele" no puede depender de una
 * conexión, y el contenido es médico.
 *
 * ## La calibración es obligatoria, siempre
 *
 * No hay forma de saltearla ni de reusar una anterior. Entre dos sesiones la
 * persona se movió, cambió la luz, se corrió la silla: una calibración vieja
 * arrastra un error sistemático que ningún suavizado corrige, y ese error
 * aparecería como si el estimador fuera malo.
 */

import { UMBRAL_PARPADEO } from "./rasgos.js";
import { crearEstimador, ESTIMADORES, FalloEstimador } from "./estimador.js";
import { FiltroMirada } from "./suavizado.js";
import { Permanencia } from "./permanencia.js";
import { PAGINAS, dibujarPagina, medir, golpearEn, escalar }
  from "./tablero.js";
import { SLM } from "./slm.js";
import { Zoom, Fijacion, ParpadeoLargo, MS_ORIENTARSE,
         MS_PAUSA_TRAS_ALEJAR, MS_MIRAR_AFUERA } from "./zoom.js";
import { Correccion } from "./correccion.js";
import { PruebaPrecision, PUNTOS_PRUEBA, aciertoPorGrilla,
         zoomNecesario, informe as informePrecision }
  from "./precision.js";

/* Nueve puntos, como el lado nativo. Cinco dejan las esquinas sin cubrir y el
 * error crece hacia los bordes, que es justo donde viven las celdas. */
const PUNTOS = [
  [0.1, 0.1], [0.5, 0.1], [0.9, 0.1],
  [0.9, 0.5], [0.5, 0.5], [0.1, 0.5],
  [0.1, 0.9], [0.5, 0.9], [0.9, 0.9],
];
/* Serpentina: las filas alternan de dirección, como hace el lado de Python.
 *
 * Recorriendo por filas, el salto del final de una fila al principio de la
 * siguiente cruza la pantalla entera: 6732 px de recorrido total contra 4274
 * en serpentina. Cada píxel de recorrido es tiempo que la mirada pasa EN
 * TRÁNSITO, y las muestras que caen ahí le enseñan al modelo que los rasgos de
 * estar mirando el punto anterior corresponden a las coordenadas del actual. */
/* Cuántas muestras se graban por punto lo decide el ESTIMADOR, no esta
 * pantalla: el propio aguanta 30 y WebGazer solo guarda las últimas 50 en
 * total, así que ahí son 5. Ver `webgazer_est.js`. */
const MS_ASENTAR = 1400;
const MS_ASENTAR_PRIMERO = 2200;
/* El primer punto necesita mucho más. Viniendo de la pantalla de colocación la
 * mirada no sabe dónde está el punto, y medido del lado nativo tarda ~1.7 s en
 * llegar - contra los 0.7 que sobran una vez que ya viene siguiendo puntos. Sin
 * esto el primer punto graba la sacada, y con nueve puntos uno contaminado
 * alcanza para que la validación cruzada elija un alpha que colapsa el modelo. */
const MS_GRABAR = 1500;

/* Los modos que ofrece el menú.
 *
 * `cfg` sale de `protocolo.CONFIG_CONDICION` del lado de Python, exportado a
 * `datos/estudio.json`. Que venga de ahí y no esté escrito acá es lo que
 * impide que la web y el estudio midan cosas distintas creyendo que miden lo
 * mismo.
 */
const MODOS = {
  libre: {
    nombre: "Tablero libre",
    detalle: "Usar el tablero sin tareas ni guion. Para probar el equipo.",
    cfg: { expandir: true, contexto: false, pagina_inicial: "core" },
    conjunto: null,
  },
  letra: { conjunto: 0 },
  sin_contexto: { conjunto: 1 },
  con_contexto: { conjunto: 2 },
  precision: {
    nombre: "Medir la precisión",
    detalle: "Calibra y después mide el error en píxeles sobre nueve puntos "
           + "distintos. Dice si el problema es el estimador o el tamaño de "
           + "las celdas.",
    cfg: { expandir: false, contexto: false, pagina_inicial: "core" },
    conjunto: null,
  },
  completo: {
    nombre: "Flujo completo",
    detalle: "Las tres condiciones seguidas, con calibración entre cada una. "
           + "Unos 25 minutos.",
    secuencia: ["letra", "sin_contexto", "con_contexto"],
  },
};

/* Cuál estimador está puesto. Se recuerda entre recargas porque medir uno
 * contra el otro obliga a recalibrar, y perder la elección en cada F5 hace que
 * uno termine midiendo el mismo dos veces sin darse cuenta. */
const CLAVE_ESTIMADOR = "caa.estimador";

function estimadorElegido() {
  try {
    const v = localStorage.getItem(CLAVE_ESTIMADOR);
    if (ESTIMADORES.some((e) => e.clave === v)) return v;
  } catch (_) { /* modo privado */ }
  return "webgazer";
}

const est = {
  fase: "menu",
  modo: "libre", cola: [], protocolo: null,
  guion: null, tarea: 0,
  mirada: null,                 // el estimador en uso
  estimador: estimadorElegido(),
  diag: null,                   // lo que devolvió ajustar()
  filtro: new FiltroMirada(),
  dwell: new Permanencia(),
  // Dos permanencias distintas: acercar es barato de equivocar,
  // elegir no. Ver zoom.js.
  fijacion: new Fijacion(),
  parpadeo: new ParpadeoLargo(),
  afueraDesde: 0,
  zoom: new Zoom(),
  correccion: new Correccion(),   // identidad hasta que se ajuste
  deriva: null,
  slm: null,
  pagina: "core", palabras: [], letras: "",
  // `cajasBase` son las cajas SIN acercar, que es de lo único que se
  // puede medir con confianza: ver `escalar` en tablero.js. `cajas` son
  // las vigentes, calculadas a partir de aquellas.
  cajasBase: new Map(),
  cajas: new Map(),
  marcoBase: null,
  candidatos: null,
  ultimoT: 0, fps: 0,
  stats: { dwells: 0, t0: null },
  calib: { i: 0, n: 0, ultima: 0, fase: "asentar", desde: 0 },
};

const $ = (s) => document.querySelector(s);
const pantallas = {};

/* ------------------------------------------------------------- arranque */

async function iniciar() {
  ["menu", "caso", "inicio", "permiso", "colocacion", "calibracion",
   "tablero", "precision", "deriva", "resultado", "error"]
    .forEach((n) => { pantallas[n] = $("#p-" + n); });

  est.protocolo = await cargarProtocolo();
  est.slm = new SLM(await cargarGrabados());
  pintarMenu();

  $("#btn-menu")?.addEventListener("click", () => mostrar("menu"));
  $("#btn-caso-volver")?.addEventListener("click", () => mostrar("menu"));
  $("#btn-caso-seguir")?.addEventListener("click", () => mostrar("inicio"));
  $("#btn-res-menu")?.addEventListener("click", () => mostrar("menu"));
  $("#btn-res-copiar")?.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(est.ultimoResultado || "");
      $("#btn-res-copiar").textContent = "Copiado";
    } catch (_) {
      // Algunos navegadores lo rechazan: se selecciona el texto en su lugar.
      const r = document.createRange();
      r.selectNodeContents($("#res-texto"));
      getSelection().removeAllRanges();
      getSelection().addRange(r);
    }
  });
  $("#webgpu").textContent = est.slm.soportado
    ? "Este navegador puede correr el modelo localmente."
    : "Este navegador no tiene WebGPU: las frases van a venir de una corrida grabada.";

  pintarEstimadores();

  $("#btn-empezar").addEventListener("click", arrancarCamara);
  $("#btn-reintentar").addEventListener("click", () => {
    if (est.reintentarCalibracion && est.mirada) {
      // Ya hay cámara y detector: no hace falta recargar la página entera,
      // solo rehacer la calibración.
      est.reintentarCalibracion = false;
      $("#btn-reintentar").textContent = "Volver a empezar";
      est.colocadoDesde = 0;
      return mostrar("colocacion");
    }
    location.reload();
  });
  mostrar("menu");
}

async function cargarProtocolo() {
  try {
    const r = await fetch("datos/estudio.json", { cache: "no-store" });
    return await r.json();
  } catch (_) {
    return null;   // sin protocolo queda el tablero libre, que no lo necesita
  }
}

function cond(clave) {
  return est.protocolo?.condiciones?.find((c) => c.clave === clave) ?? null;
}

function pintarMenu() {
  const cont = $("#opciones");
  if (!cont) return;
  const items = [["libre", MODOS.libre.nombre, MODOS.libre.detalle]];
  for (const clave of ["letra", "sin_contexto", "con_contexto"]) {
    const c = cond(clave);
    if (c) items.push([clave, c.nombre, c.instrucciones]);
  }
  items.push(["completo", MODOS.completo.nombre, MODOS.completo.detalle]);
  items.push(["precision", MODOS.precision.nombre, MODOS.precision.detalle]);

  cont.replaceChildren(...items.map(([clave, nombre, detalle], i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "opcion-modo";
    b.innerHTML = `<span class="n">${clave === "libre" ? "·"
      : clave === "completo" ? "∗" : i}</span>`;
    const c = document.createElement("span");
    c.className = "cuerpo";
    c.innerHTML = `<span class="titulo"></span><span class="detalle"></span>`;
    c.querySelector(".titulo").textContent = nombre;
    c.querySelector(".detalle").textContent = detalle || "";
    b.appendChild(c);
    b.addEventListener("click", () => elegirModo(clave));
    return b;
  }));
}

/** El selector de estimador.
 *
 * Está en el menú y no escondido en una opción de desarrollo porque ahora
 * mismo es la variable que se está midiendo. Cambiarlo recarga la página: el
 * estimador se queda con la cámara y con su propio lazo de cuadros, y
 * desmontarlo en caliente sería otra fuente de fallos sin ningún beneficio.
 */
function pintarEstimadores() {
  const cont = $("#estimadores");
  if (!cont) return;
  cont.replaceChildren(...ESTIMADORES.map((e) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "opcion-est" + (e.clave === est.estimador ? " puesto" : "");
    const t = document.createElement("span");
    t.className = "titulo";
    t.textContent = e.nombre;
    const d = document.createElement("span");
    d.className = "detalle";
    d.textContent = e.detalle;
    b.append(t, d);
    b.addEventListener("click", () => {
      if (e.clave === est.estimador) return;
      try { localStorage.setItem(CLAVE_ESTIMADOR, e.clave); } catch (_) {}
      location.reload();
    });
    return b;
  }));
}

function elegirModo(clave) {
  est.modo = clave;
  est.cola = clave === "completo" ? MODOS.completo.secuencia.slice() : [clave];
  prepararSiguiente();
}

/** Deja listo el próximo bloque de la cola y muestra la antesala. */
function prepararSiguiente() {
  const clave = est.cola[0] ?? "libre";
  const c = cond(clave);
  const cfg = c ? c.cfg : MODOS.libre.cfg;
  est.cfgActual = cfg;
  est.pagina = cfg.pagina_inicial || "core";

  const conjunto = MODOS[clave]?.conjunto;
  est.guion = (conjunto !== null && conjunto !== undefined
               && est.protocolo?.tareas?.[conjunto])
    ? est.protocolo.tareas[conjunto] : null;
  est.tarea = 0;

  $("#inicio-titulo").textContent = c ? c.nombre : MODOS.libre.nombre;
  $("#inicio-instrucciones").textContent =
    c ? c.instrucciones : MODOS.libre.detalle;
  $("#inicio-tareas").textContent = est.guion
    ? `Se le van a pedir ${est.guion.length} frases, una por vez.`
    : "Sin tareas: use el tablero libremente.";

  // El guion del caso solo se lee en la condición contextualizada: es la que
  // necesita que la persona interprete a alguien.
  if (cfg.contexto && est.protocolo?.caso) {
    $("#caso-titulo").textContent =
      `Su papel: ${est.protocolo.caso.nombre}`;
    $("#caso-guion").textContent = est.protocolo.caso.guion;
    return mostrar("caso");
  }
  mostrar("inicio");
}

async function cargarGrabados() {
  try {
    const r = await fetch("datos/candidatos.json", { cache: "force-cache" });
    return await r.json();
  } catch (_) {
    return {};   // sin grabados queda solo el literal, que igual alcanza
  }
}

function mostrar(nombre) {
  est.fase = nombre;
  Object.entries(pantallas).forEach(([n, el]) => { if (el) el.hidden = n !== nombre; });
}

function fallar(msg, detalle = "") {
  $("#error-msg").textContent = msg;
  $("#error-detalle").textContent = detalle;
  mostrar("error");
}

/* --------------------------------------------------------------- cámara */

async function arrancarCamara() {
  /* En el flujo completo se vuelve a esta pantalla entre condiciones, y ahí la
   * cámara y el detector ya están andando: hay que recalibrar, no volver a
   * arrancar todo. Con WebGazer la diferencia deja de ser una optimización y
   * pasa a ser necesaria, porque un segundo `begin()` monta un segundo video y
   * un segundo lazo de cuadros sobre la misma cámara. */
  if (est.mirada) {
    est.colocadoDesde = 0;
    return mostrar("colocacion");
  }
  mostrar("permiso");
  // WebGazer baja unos 8 MB de WebAssembly la primera vez, y eso pasa dentro
  // de su arranque. Sin avisarlo, la pantalla dice "esperando la camara"
  // durante medio minuto y parece colgada.
  const detalle = $("#permiso-detalle");
  if (detalle && est.estimador === "webgazer") {
    detalle.textContent = "El navegador solicitará autorización. La primera "
      + "vez, WebGazer descarga unos 8 MB de modelo: puede demorar medio "
      + "minuto antes de que aparezca el encuadre.";
  }
  try {
    est.mirada = await crearEstimador(est.estimador);
    await est.mirada.iniciar();
  } catch (e) {
    if (e instanceof FalloEstimador) return fallar(e.message, e.detalle);
    return fallar("No se pudo arrancar el estimador de mirada.",
                  String(e?.message || e));
  }
  est.resolucion = est.mirada.resolucion;
  console.log("[estimador]", est.mirada.nombre, est.resolucion);

  // El espejo del óvalo comparte el MISMO stream que usa el detector: dos
  // <video> sobre un MediaStream es seguro y evita pedir la cámara dos veces.
  const espejo = $("#espejo");
  if (espejo && est.mirada.stream) {
    espejo.srcObject = est.mirada.stream;
    espejo.play().catch(() => {});
  }

  mostrar("colocacion");
  requestAnimationFrame(lazo);
}

/* ------------------------------------------------------- lazo principal */

function lazo(t) {
  requestAnimationFrame(lazo);
  const dt = est.ultimoT ? (t - est.ultimoT) / 1000 : 1 / 30;
  est.ultimoT = t;
  est.fps = est.fps * 0.9 + (1 / Math.max(dt, 1e-3)) * 0.1;

  let datos = null;
  try { datos = est.mirada?.leer(t) ?? null; } catch (_) { /* un cuadro perdido */ }

  // Diagnostico barato y permanente. Sin esto, "funciona mal" no se puede
  // distinguir de "el detector no ve nada" ni de "la prediccion esta
  // congelada", que son tres problemas con tres arreglos distintos.
  if (datos) {
    est.nLandmarks = est.mirada.nPuntosCara;
    const firma = datos.centro[0] + datos.centro[1] + datos.distOjos;
    if (est.firmaPrevia !== undefined && firma !== est.firmaPrevia) {
      est.rasgosVivos = (est.rasgosVivos || 0) + 1;
    }
    est.firmaPrevia = firma;
  }

  if (est.fase === "colocacion") return pasoColocacion(datos);
  if (est.fase === "calibracion") return pasoCalibracion(datos, t);
  if (est.fase === "tablero") return pasoTablero(datos, dt, t);
  if (est.fase === "precision") return pasoPrecision(datos, dt, t);
  if (est.fase === "deriva") return pasoDeriva(datos, dt, t);
}

/* ---------------------------------------------------------- colocación */

function pasoColocacion(datos) {
  const guia = $("#guia");
  const ok = datos
    && datos.centro[0] > 0.33 && datos.centro[0] < 0.67
    && datos.centro[1] > 0.25 && datos.centro[1] < 0.62
    && datos.distOjos > 0.055 && datos.distOjos < 0.16;

  guia.classList.toggle("ok", !!ok);
  $("#colocacion-estado").textContent = !datos
    ? "No veo su cara. Acérquese y busque más luz."
    : ok ? "Perfecto. Quédese así."
    : datos.distOjos <= 0.055 ? "Acérquese un poco a la pantalla."
    : datos.distOjos >= 0.16 ? "Aléjese un poco."
    : "Centre su cara en el óvalo.";

  if (ok) {
    est.colocadoDesde = est.colocadoDesde || performance.now();
    const listo = performance.now() - est.colocadoDesde;
    $("#colocacion-barra").style.width =
      Math.min(100, (listo / 1500) * 100) + "%";
    if (listo > 1500) empezarCalibracion();
  } else {
    est.colocadoDesde = 0;
    $("#colocacion-barra").style.width = "0%";
  }
}

/* -------------------------------------------------------- calibración */

async function empezarCalibracion() {
  // `limpiarDatos` espera -WebGazer borra su almacenamiento- y mientras tanto
  // el lazo sigue corriendo en colocación, así que sin este cerrojo se
  // arrancarían varias calibraciones encimadas.
  if (est.arrancando) return;
  est.arrancando = true;
  est.calib = { i: 0, n: 0, ultima: 0, fase: "asentar",
                desde: performance.now() };
  await est.mirada.limpiarDatos();
  est.calib.desde = performance.now();
  est.arrancando = false;
  mostrar("calibracion");
  pintarPuntoCalib();
}

function pintarPuntoCalib() {
  const [fx, fy] = PUNTOS[est.calib.i];
  const p = $("#punto");
  p.style.left = (fx * 100) + "%";
  p.style.top = (fy * 100) + "%";
  $("#calib-n").textContent = `${est.calib.i + 1} / ${PUNTOS.length}`;
}

function pasoCalibracion(datos, t) {
  const c = est.calib;
  const p = $("#punto");
  const transcurrido = t - c.desde;

  if (c.fase === "asentar") {
    p.classList.remove("grabando");
    p.style.setProperty("--avance", "0");
    const espera = c.i === 0 ? MS_ASENTAR_PRIMERO : MS_ASENTAR;
    if (transcurrido > espera) { c.fase = "grabar"; c.desde = t; }
    return;
  }

  p.classList.add("grabando");
  p.style.setProperty("--avance", Math.min(1, transcurrido / MS_GRABAR));

  // El parpadeo NO se graba: durante un parpadeo la geometría del ojo es otra
  // y esas muestras corren el modelo hacia una cara que no está mirando.
  //
  // Cuántas muestras entran y cada cuánto lo dice el estimador. WebGazer solo
  // conserva las últimas 50 en total, así que ahí son cinco por punto y hay
  // que repartirlas a lo largo del grabado en vez de tomar cinco cuadros
  // seguidos, que serían la misma muestra cinco veces.
  const cupo = est.mirada.muestrasPorPunto;
  const espaciado = est.mirada.msEntreMuestras;
  if (datos && datos.ear > UMBRAL_PARPADEO && c.n < cupo
      && t - c.ultima >= espaciado) {
    const [fx, fy] = PUNTOS[c.i];
    const objetivo = [fx * window.innerWidth, fy * window.innerHeight];
    if (est.mirada.grabar(datos, objetivo, c.i)) { c.n += 1; c.ultima = t; }
  }

  if (transcurrido > MS_GRABAR) {
    c.grabadas = (c.grabadas || 0) + c.n;
    c.n = 0; c.ultima = 0;
    c.i += 1;
    if (c.i >= PUNTOS.length) return terminarCalibracion();
    c.fase = "asentar"; c.desde = t;
    pintarPuntoCalib();
  }
}

async function terminarCalibracion() {
  let diag;
  try {
    diag = await est.mirada.ajustar();
  } catch (e) {
    if (e instanceof FalloEstimador) return fallar(e.message, e.detalle);
    return fallar("No se pudo ajustar el modelo.", String(e?.message || e));
  }
  est.diag = diag;

  // Puerta de calidad. Medido: dos calibraciones del mismo código dieron 123 px
  // y 316 px de error, y la segunda era un modelo colapsado que el sistema
  // dejaba usar igual -la persona lo descubría peleando con el tablero-.
  //
  // Se mira el RECORRIDO y no el error de validación cruzada. El LOO sirve para
  // elegir alpha, pero como medida absoluta es sistemáticamente pesimista: al
  // sacar un punto de una rejilla de 3x3 se saca la única información sobre esa
  // región, así que el modelo extrapola. Con datos perfectos el LOO daba 282 px
  // y el error real era 0.
  //
  // El recorrido compara cuánto se mueven las predicciones contra cuánto se
  // mueven los objetivos. Por debajo de 0.45 el modelo responde a la mirada
  // menos de la mitad de lo que debería, y eso no se arregla con zoom. Es el
  // mismo número en los dos estimadores, a propósito: ver `recorridoDe`.
  const rec = diag.recorrido;
  console.log("[calibración]", est.mirada.nombre, diag.nota,
              "recorrido", Number(rec).toFixed(2));
  if (Number.isFinite(rec) && rec < 0.45) {
    return calibracionMala(rec);
  }

  est.filtro.reiniciar();
  est.stats = { dwells: 0, t0: null };
  // El contexto va al prompt SOLO en la condición contextualizada. Es la
  // variable independiente de esa comparación: todo lo demás -modelo, prompt,
  // tablero- queda idéntico entre las dos condiciones con modelo.
  est.slm.contexto = est.cfgActual?.contexto
    ? (est.protocolo?.caso?.contexto || "") : "";
  est.correccion = new Correccion();
  iniciarDeriva();
}

/* ------------------------------------------------------------- tablero */

/** Pasa a la tarea siguiente; al agotarse, al bloque siguiente de la cola. */
function avanzarTarea() {
  if (!est.guion) return;
  est.tarea += 1;
  if (est.tarea < est.guion.length) return;

  est.cola.shift();
  if (est.cola.length) {
    // Entre condiciones se recalibra. No es una molestia del protocolo: entre
    // bloques la persona se levanta, cambia de postura, y una calibración
    // vieja arrastra un error sistemático que se le atribuiría a la condición.
    est.mirada.limpiarDatos();
    est.diag = null;
    prepararSiguiente();
  } else {
    mostrar("menu");
  }
}

function bandaTarea() {
  if (!est.guion || est.tarea >= est.guion.length) return "";
  const t = est.guion[est.tarea];
  return `${est.tarea + 1}/${est.guion.length}   ${t.pista}`;
}

/* ------------------------------------------------------- precisión */

function pintarPuntoPrueba() {
  const [fx, fy] = PUNTOS_PRUEBA[est.prueba.i] || [0.5, 0.5];
  const el = $("#prec-punto");
  el.style.left = (fx * 100) + "%";
  el.style.top = (fy * 100) + "%";
  $("#prec-n").textContent = `${est.prueba.i + 1} / ${PUNTOS_PRUEBA.length}`;
}

function pasoPrecision(datos, dt, t) {
  let muestra = { x: 0, y: 0, ok: false };
  if (datos && datos.cruda && datos.ear > UMBRAL_PARPADEO) {
    const [px, py] = datos.cruda;
    const med = est.filtro.paso(px, py, t);
    if (med) {
      const [cx, cy] = est.correccion.aplicar(med);
      muestra = { x: cx, y: cy, ok: true };
      const c = $("#cursor");
      c.style.transform = `translate(${cx}px, ${cy}px)`;
      c.hidden = false;
    }
  }

  const r = est.prueba.paso(t, muestra);
  const el = $("#prec-punto");
  el.classList.toggle("grabando", est.prueba.fase === "medir");
  el.style.setProperty("--avance", r.avance);

  if (est.prueba.terminada) return mostrarResultado();
  if (r.cambio) pintarPuntoPrueba();
}

function mostrarResultado() {
  const prueba = est.prueba;
  $("#cursor").hidden = true;
  $("#res-texto").textContent = informePrecision(prueba)
    + `

estimador: ${est.mirada.etiqueta}`
    + `
seguimiento de la calibración: `
    + `${((est.diag?.recorrido ?? NaN) * 100).toFixed(0)}%`
    + `
${est.diag?.nota ?? ""}`
    + `
corrección: ${est.correccion.listo ? est.correccion.resumen : "ninguna"}`;

  const tabla = $("#res-tabla");
  const filas = aciertoPorGrilla(prueba);
  const conZoom = aciertoPorGrilla(prueba, est.zoom.factor);
  tabla.innerHTML = "<tr><th>grilla</th><th>celda</th>"
                  + "<th class='num'>sola</th>"
                  + `<th class='num'>con zoom ${est.zoom.factor}x</th></tr>`;
  filas.forEach((f, i) => {
    const z = conZoom[i];
    const tr = document.createElement("tr");
    if (z.acierto >= 0.9) tr.className = "bien";
    else if (z.acierto < 0.7) tr.className = "mal";
    tr.innerHTML = `<td>${f.grilla}</td><td class="num">${f.celda}</td>`
                 + `<td class="num">${Math.round(f.acierto * 100)}%</td>`
                 + `<td class="num">${Math.round(z.acierto * 100)}%</td>`;
    tabla.appendChild(tr);
  });

  // Cuánto habría que acercar para llegar al 90% sobre la grilla de 5x5. Es
  // la única perilla que queda: el error ya está medido y no se va a mover.
  const k = zoomNecesario(prueba, 5, 5, 0.9);
  const nota = $("#res-nota");
  if (nota) {
    nota.textContent = Number.isFinite(k)
      ? `Para acertar el 90% sobre la grilla de 5x5 hace falta acercar `
        + `${k.toFixed(2)}x. Ahora está configurado en ${est.zoom.factor}x.`
      : "Ni acercando 8x se llega al 90%: el error es demasiado grande para "
        + "esta grilla.";
  }
  est.ultimoResultado = [
    informePrecision(prueba),
    "",
    ...filas.map((f) =>
      `${f.grilla}  ${f.celda}  ${Math.round(f.acierto * 100)}%`),
    "",
    `estimador ${est.mirada.nombre} - ${est.mirada.etiqueta}`,
    `camara ${est.resolucion}, ${est.nLandmarks} puntos faciales`,
    `calibracion: ${est.diag?.nota ?? "?"}, `
      + `recorrido ${((est.diag?.recorrido ?? NaN) * 100).toFixed(0)}%`,
    `correccion: ${est.correccion.listo ? est.correccion.resumen : "ninguna"}`,
  ].join("\n");
  mostrar("resultado");
}

/* ------------------------------------------------------------- zoom */

function aplicarZoom() {
  const g = $("#escena").firstElementChild;
  if (!g) return;
  Object.assign(g.style, est.zoom.estilo(est.marcoBase));
  // Las cajas se CALCULAN, no se vuelven a medir. Medirlas acá devolvía la
  // geometría del arranque de la transición de 180 ms y no la final, así que
  // el mapa de impacto se quedaba con el tamaño de antes de acercar: el
  // puntero se veía sobre una celda y se marcaba otra.
  est.cajas = escalar(est.cajasBase, est.zoom.origen, est.zoom.escala);
  const ind = $("#zoom-ind");
  if (ind) {
    ind.hidden = !est.zoom.acercado;
    ind.textContent = est.zoom.acercado
      ? `${est.zoom.escala.toFixed(1)}x · cierre los ojos para volver` : "";
  }
}

/** El anillo del cursor se completa mientras se acumula el acercamiento. */
function pintarAvanceZoom(avance) {
  const c = $("#cursor");
  if (c) c.style.setProperty("--zoom-avance", avance || 0);
}

/** La calibración salió mal: se dice, con el detalle, y se repite.
 *
 * Deja ver el error de cada punto porque casi siempre hay uno o dos que
 * arruinan el ajuste -un parpadeo largo, mirar fuera de la pantalla- y saber
 * cuál permite repetir prestando atención ahí en vez de repetir a ciegas.
 *
 * Ese desglose solo lo sabe dar el estimador propio: reajusta el modelo
 * dejando cada punto afuera. WebGazer entrena en línea y no expone nada
 * equivalente, así que ahí el aviso va sin la lista y lo dice.
 */
function calibracionMala(rec) {
  const peores = (est.mirada.erroresPorPunto?.() ?? [])
    .map((e, i) => [i + 1, e])
    .filter(([, e]) => Number.isFinite(e))
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3);

  $("#error-msg").textContent = "La calibración no salió bien";
  $("#error-detalle").innerHTML =
    `El modelo solo sigue la mirada un <strong>`
    + `${(rec * 100).toFixed(0)}%</strong> de lo que debería: predice casi el `
    + `mismo punto mire donde mire. Con eso el tablero no puede funcionar, `
    + `ni siquiera con el acercamiento.<br><br>`
    + (peores.length
        ? `Los puntos que peor salieron fueron el `
          + peores.map(([n, e]) => `<strong>${n}</strong> (${e.toFixed(0)} px)`)
                 .join(", ")
          + `. Suele ser un parpadeo largo o haber mirado fuera de la `
          + `pantalla.<br><br>`
        : "")
    + `Repita la calibración siguiendo cada punto hasta que el anillo se `
    + `complete, con buena luz de frente y sin mover la cabeza.`;
  $("#btn-reintentar").textContent = "Repetir la calibración";
  est.reintentarCalibracion = true;
  mostrar("error");
}

/* ------------------------------------------- corrección de la ganancia */

/* Después de calibrar, cinco puntos más, y con ellos se ajusta una recta por
 * eje: `predicción = ganancia * objetivo + corrimiento`.
 *
 * La versión anterior medía un solo punto en el centro y restaba una
 * constante. Medido con WebGazer sobre nueve puntos, el desvío horizontal iba
 * de +18 px en la columna izquierda a +309 px en la derecha: no hay ninguna
 * constante correcta en los dos extremos a la vez. Con la recta, ese sesgo
 * medido cae de 210 px a 45 px.
 *
 * Con un solo punto la recta se degrada sola al corrimiento puro, así que
 * esto generaliza la corrección anterior en vez de reemplazarla. Ver
 * `correccion.js`.
 *
 * Son cinco y no nueve porque la recta tiene dos parámetros por eje: cinco
 * puntos ya la determinan de sobra, y cada punto cuesta dos segundos de la
 * persona. Están en 0.2/0.5/0.8, distintos de los de calibración (0.1/0.5/0.9)
 * y de los de la prueba (0.25/0.5/0.75): sobre los de calibración el modelo
 * acierta y no se vería nada, y sobre los de la prueba la prueba dejaría de
 * medir generalización. */
const PUNTOS_DERIVA = [
  [0.5, 0.5], [0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8],
];
const MS_DERIVA_ASENTAR = 1100;
/* El primero viene de la última esquina de la calibración y está en el centro,
 * así que es el salto más largo de toda la secuencia. */
const MS_DERIVA_PRIMERO = 1700;
const MS_DERIVA_MEDIR = 1000;

function iniciarDeriva() {
  est.deriva = { i: 0, fase: "asentar", desde: 0, buffer: [],
                 objetivos: [], crudas: [] };
  pintarPuntoDeriva();
  mostrar("deriva");
}

function pintarPuntoDeriva() {
  const [fx, fy] = PUNTOS_DERIVA[est.deriva.i];
  const el = $("#deriva-punto");
  el.style.left = (fx * 100) + "%";
  el.style.top = (fy * 100) + "%";
  const n = $("#deriva-n");
  if (n) n.textContent = `${est.deriva.i + 1} / ${PUNTOS_DERIVA.length}`;
}

function pasoDeriva(datos, dt, t) {
  const d = est.deriva;
  if (!d.desde) d.desde = t;
  const pasado = t - d.desde;
  const el = $("#deriva-punto");

  let muestra = null;
  if (datos && datos.cruda && datos.ear > UMBRAL_PARPADEO) {
    // Se predice SIN la corrección vigente: lo que se quiere medir es el
    // desvío crudo del modelo, no el que queda después de corregir.
    const [px, py] = datos.cruda;
    const med = est.filtro.paso(px, py, t);
    if (med) muestra = med;
  }

  if (d.fase === "asentar") {
    el.classList.remove("grabando");
    el.style.setProperty("--avance", "0");
    const espera = d.i === 0 ? MS_DERIVA_PRIMERO : MS_DERIVA_ASENTAR;
    if (pasado > espera) { d.fase = "medir"; d.desde = t; }
    return;
  }

  el.classList.add("grabando");
  el.style.setProperty("--avance", Math.min(1, pasado / MS_DERIVA_MEDIR));
  if (muestra) d.buffer.push(muestra);

  if (pasado > MS_DERIVA_MEDIR) {
    // Mediana y no media: si la mirada todavía viaja hacia el punto cuando
    // empieza el registro, la media arrastra el centro hacia el punto
    // anterior. Eso es una sacada, no un desvío del estimador.
    if (d.buffer.length >= 4) {
      const m = (a) => [...a].sort((x, y) => x - y)[a.length >> 1];
      const [fx, fy] = PUNTOS_DERIVA[d.i];
      d.objetivos.push([fx * innerWidth, fy * innerHeight]);
      d.crudas.push([m(d.buffer.map((q) => q[0])),
                     m(d.buffer.map((q) => q[1]))]);
    }
    d.buffer = [];
    d.i += 1;
    if (d.i >= PUNTOS_DERIVA.length) return cerrarDeriva();
    d.fase = "asentar"; d.desde = t;
    pintarPuntoDeriva();
  }
}

function cerrarDeriva() {
  const d = est.deriva;
  // Sin al menos tres puntos no se ajusta una recta con ninguna confianza, y
  // se prefiere no corregir antes que corregir mal: una ganancia equivocada
  // amplifica el temblor, que es lo único que hasta ahora salió bien en todas
  // las mediciones.
  if (d.objetivos.length >= 3) {
    est.correccion = Correccion.ajustar(d.objetivos, d.crudas);
    console.log("[corrección]", est.correccion.resumen);
  } else {
    console.log("[corrección] solo", d.objetivos.length,
                "puntos válidos: se sigue sin corregir");
  }
  seguirDespuesDeCalibrar();
}


function seguirDespuesDeCalibrar() {
  if (est.modo === "precision") {
    est.prueba = new PruebaPrecision();
    pintarPuntoPrueba();
    return mostrar("precision");
  }
  mostrar("tablero");
  redibujar();
  // Al entrar, la mirada viene del último punto de la prueba y puede estar
  // parada sobre cualquier zona. Sin esta pausa acercaría antes de que la
  // persona llegue a mirar el tablero.
  est.fijacion.reiniciar();
  est.fijacion.bloquear(MS_ORIENTARSE, performance.now());
}

function redibujar() {
  const cont = $("#escena");
  if (est.candidatos) dibujarCandidatos(cont);
  else { dibujarPagina(cont, est.pagina); }
  // Medir ANTES de transformar: estas son las cajas sin acercar, y de ellas
  // salen las acercadas. El elemento acaba de nacer, así que todavía no tiene
  // ninguna transformación encima.
  est.cajasBase = medir(cont);
  const g = cont.firstElementChild;
  est.marcoBase = g ? g.getBoundingClientRect() : null;
  aplicarZoom();
  pintarMensaje();
}

function texto() {
  const p = est.palabras.join(" ");
  return (p + (est.letras ? (p ? " " : "") + est.letras : "")).trim();
}

function pintarMensaje() {
  const t = texto();
  const el = $("#mensaje");
  el.textContent = t || "Mire una celda y sostenga la mirada";
  el.classList.toggle("vacio", !t);
  // La consigna tiene que estar a la vista todo el tiempo: si hay que
  // acordarse de memoria, el estudio mide memoria.
  const banda = $("#consigna");
  if (banda) {
    const b = bandaTarea();
    banda.textContent = b;
    banda.hidden = !b;
  }
}

function pasoTablero(datos, dt, t) {
  const cursor = $("#cursor");
  let muestra = { x: 0, y: 0, ok: false };

  // Cerrar los ojos un momento cancela todo: es la salida que no depende de
  // acertarle a nada con la mirada.
  if (est.parpadeo.paso(datos ? datos.ear : NaN, !!datos, t)) {
    cancelar(t, false);
    return pintarMedidor();
  }

  // Con los ojos cerrados el estimador sigue entregando una posición, y es
  // basura: no se usa, o arrastraría el cursor mientras se parpadea.
  if (datos && datos.cruda && datos.ear > UMBRAL_PARPADEO
      && !est.parpadeo.cerrado) {
    const [px, py] = datos.cruda;
    const med = est.filtro.paso(px, py, t);
    if (med) {
      const [cx, cy] = est.correccion.aplicar(med);
      muestra = { x: cx, y: cy, ok: true };
      cursor.style.transform = `translate(${cx}px, ${cy}px)`;
      cursor.hidden = false;
    }
  }

  const m = est.marcoBase;
  const enTablero = muestra.ok && !!m
    && muestra.x >= m.left && muestra.x < m.right
    && muestra.y >= m.top && muestra.y < m.bottom;

  // --- primer tiempo: acercar donde la mirada se quedó quieta -----------
  if (!est.zoom.acercado && !est.candidatos) {
    const z = est.fijacion.paso(muestra, enTablero, t);
    pintarAvanceZoom(z.avance);
    if (z.disparo) {
      est.zoom.acercar(z.disparo[0], z.disparo[1], t);
      aplicarZoom();
      est.afueraDesde = 0;
      est.dwell.reiniciar();
      est.dwell.bloquear(250);
      pintarAvanceZoom(0);
    }
    return pintarMedidor();
  }

  // --- acercado: mirar fuera del tablero también aleja ------------------
  if (est.zoom.acercado) {
    if (muestra.ok && !enTablero) {
      if (!est.afueraDesde) est.afueraDesde = t;
      else if (t - est.afueraDesde > MS_MIRAR_AFUERA) {
        cancelar(t, false);
        return pintarMedidor();
      }
    } else if (muestra.ok) {
      est.afueraDesde = 0;
    }
  }

  // --- segundo tiempo: elegir, ya con las celdas grandes ----------------
  // Acercada, la grilla sigue más allá del borde visible: lo que quedó
  // tapado no se puede elegir.
  const r = est.dwell.paso(
    muestra,
    (x, y) => ((est.zoom.acercado && !enTablero)
      ? null : golpearEn(est.cajas, x, y)),
    (id) => est.cajas.get(id) ?? null);

  // Pasó el tiempo sin elegir nada. No se corta una selección que ya va por
  // la mitad: faltarían décimas para completarla.
  if (est.zoom.venció(t) && r.avance < 0.5 && !r.disparo) {
    cancelar(t, true);
    return pintarMedidor();
  }

  est.cajas.forEach((_, id) => {
    const el = $(`[data-id="${CSS.escape(id)}"]`);
    if (!el) return;
    const activo = id === r.objetivo;
    el.classList.toggle("viva", activo);
    el.style.setProperty("--avance", activo ? r.avance : 0);
  });

  if (r.disparo) {
    const el = $(`[data-id="${CSS.escape(r.disparo)}"]`);
    est.zoom.alejar();
    aplicarZoom();
    est.afueraDesde = 0;
    est.fijacion.reiniciar();
    est.fijacion.bloquear(MS_PAUSA_TRAS_ALEJAR, t);
    if (el) aplicar(el.dataset.accion);
  }
  pintarMedidor();
}

/** Aleja y reinicia lo que se estuviera acumulando.
 *
 * @param {boolean} porTiempo  si se alejó solo, por pasar el tiempo. En ese
 *   caso la persona estuvo mirando ahí sin elegir, y no se vuelve a acercar
 *   en el mismo lugar hasta que la mirada se haya ido a otro lado. Si lo
 *   pidió ella -cerrando los ojos o mirando afuera- no hace falta.
 */
function cancelar(t, porTiempo) {
  const donde = est.zoom.origen;
  est.zoom.alejar();
  aplicarZoom();
  est.afueraDesde = 0;
  est.dwell.reiniciar();
  est.dwell.bloquear(MS_PAUSA_TRAS_ALEJAR);
  est.fijacion.reiniciar();
  est.fijacion.bloquear(MS_PAUSA_TRAS_ALEJAR, t);
  if (porTiempo && donde) est.fijacion.exigirMovimiento(donde);
  pintarAvanceZoom(0);
  document.querySelectorAll("#escena .viva").forEach((el) => {
    el.classList.remove("viva");
    el.style.setProperty("--avance", 0);
  });
}

function pintarMedidor() {
  $("#fps").textContent = est.fps.toFixed(0) + " fps";
  $("#dwells").textContent = est.stats.dwells;
  const diag = $("#diag");
  if (diag) {
    diag.textContent = `${est.mirada?.nombre ?? "?"} · `
      + `${est.nLandmarks || 0} pts · ${est.resolucion || "?"}`
      + (est.rasgosVivos ? "" : " · SEÑAL CONGELADA");
  }
}

async function aplicar(accion) {
  est.stats.dwells += 1;
  if (!est.stats.t0) est.stats.t0 = Date.now();

  if (accion.startsWith("w:")) {
    const v = accion.slice(2);
    if (est.pagina === "abc") est.letras += v;
    else {
      if (est.letras) { est.palabras.push(est.letras); est.letras = ""; }
      est.palabras.push(v);
    }
  } else if (accion.startsWith("page:")) {
    if (est.letras) { est.palabras.push(est.letras); est.letras = ""; }
    est.pagina = accion.slice(5);
    // La mirada que cambió de página no puede disparar la celda que quedó
    // debajo: es la selección accidental más frecuente del tablero.
    est.dwell.bloquear(800);
  } else if (accion === "space") {
    if (est.letras) { est.palabras.push(est.letras); est.letras = ""; }
  } else if (accion === "delete") {
    if (est.letras) est.letras = est.letras.slice(0, -1);
    else est.palabras.pop();
  } else if (accion === "clear") {
    est.palabras = []; est.letras = "";
  } else if (accion === "speak") {
    if (est.letras) { est.palabras.push(est.letras); est.letras = ""; }
    const t = texto();
    if (!t) return redibujar();
    // Condición de línea de base: lo que se escribió es lo que se dice, sin
    // pantalla de candidatos. Elegir entre opciones es justamente la función
    // que esta condición existe para NO tener.
    if (est.cfgActual && est.cfgActual.expandir === false) {
      hablar(t);
      est.palabras = []; est.letras = "";
      avanzarTarea();
      est.dwell.bloquear(800);
      return redibujar();
    }
    est.candidatos = { pensando: true, lista: [], modo: "", ms: 0 };
    redibujar();
    const r = await est.slm.expandir(t);
    est.candidatos = { pensando: false, lista: r.candidatos,
                       modo: r.modo, ms: r.ms };
    est.dwell.bloquear(600);
    return redibujar();
  } else if (accion.startsWith("say:")) {
    hablar(accion.slice(4));
    est.palabras = []; est.letras = ""; est.candidatos = null;
    avanzarTarea();
    est.dwell.bloquear(800);
    return redibujar();
  } else if (accion === "back") {
    est.candidatos = null;
    est.dwell.bloquear(600);
    return redibujar();
  }
  redibujar();
}

function dibujarCandidatos(cont) {
  const c = est.candidatos;
  const caja = document.createElement("div");
  caja.className = "candidatos";

  const cab = document.createElement("div");
  cab.className = "cab";
  cab.innerHTML = `<h2>${c.pensando ? "Pensando…" : "¿Cuál quiso decir?"}</h2>`;
  const sello = document.createElement("span");
  sello.className = "sello";
  sello.textContent = c.pensando ? ""
    : c.modo === "vivo" ? `modelo local · ${c.ms} ms`
    : c.modo === "grabado" ? `qwen2.5:7b · corrida grabada`
    : "sin respuesta guardada para esta frase";
  cab.appendChild(sello);
  caja.appendChild(cab);

  if (!c.pensando) {
    const lit = texto().toLowerCase();
    c.lista.forEach((frase) => {
      const el = document.createElement("div");
      el.className = "opcion celda";
      if (frase.toLowerCase() === lit) el.classList.add("literal");
      el.dataset.id = "cand:" + frase;
      el.dataset.accion = "say:" + frase;
      el.innerHTML = '<span class="relleno"></span><span class="trazo"></span>';
      const t = document.createElement("span");
      t.className = "etq";
      t.textContent = frase;
      el.appendChild(t);
      caja.appendChild(el);
    });

    const volver = document.createElement("div");
    volver.className = "opcion celda ctrl";
    volver.dataset.id = "cand:volver";
    volver.dataset.accion = "back";
    volver.innerHTML = '<span class="relleno"></span><span class="trazo"></span>'
      + '<span class="etq">VOLVER AL TABLERO</span>';
    caja.appendChild(volver);
  }

  cont.replaceChildren(caja);
}

function hablar(frase) {
  try {
    const u = new SpeechSynthesisUtterance(frase);
    u.lang = "es-ES";
    speechSynthesis.cancel();
    speechSynthesis.speak(u);
  } catch (_) { /* sin voz, el texto ya está en pantalla */ }
}

addEventListener("resize", () => {
  if (est.fase !== "tablero") return;
  // Se aleja primero: con la transformación puesta, lo que se mide son las
  // cajas ya acercadas, y de ahí no se puede volver a las de base.
  est.zoom.alejar();
  est.fijacion.reiniciar();
  redibujar();
});

iniciar();
