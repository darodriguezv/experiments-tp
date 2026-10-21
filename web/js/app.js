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

import { extraer, RASGOS_DIM, UMBRAL_PARPADEO } from "./rasgos.js";
import { ModeloMirada } from "./modelo.js";
import { Suavizador, VentanaMediana } from "./suavizado.js";
import { Permanencia } from "./permanencia.js";
import { PAGINAS, dibujarPagina, medir, golpearEn } from "./tablero.js";
import { SLM } from "./slm.js";

const MP = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";
const MODELO_CARA =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/" +
  "face_landmarker/float16/1/face_landmarker.task";

/* Nueve puntos, como el lado nativo. Cinco dejan las esquinas sin cubrir y el
 * error crece hacia los bordes, que es justo donde viven las celdas. */
const PUNTOS = [
  [0.1, 0.1], [0.5, 0.1], [0.9, 0.1],
  [0.1, 0.5], [0.5, 0.5], [0.9, 0.5],
  [0.1, 0.9], [0.5, 0.9], [0.9, 0.9],
];
const MUESTRAS_POR_PUNTO = 14;
const MS_ASENTAR = 900;    // tiempo para que la mirada llegue antes de grabar
const MS_GRABAR = 1100;

const est = {
  fase: "inicio",
  video: null, landmarker: null,
  modelo: new ModeloMirada(),
  suave: new Suavizador(),
  ventana: new VentanaMediana(320),
  dwell: new Permanencia(),
  slm: null,
  pagina: "core", palabras: [], letras: "",
  cajas: new Map(),
  candidatos: null,
  ultimoT: 0, fps: 0,
  stats: { dwells: 0, t0: null },
  calib: { i: 0, muestras: [], objetivos: [], buffer: [], fase: "asentar",
           desde: 0 },
};

const $ = (s) => document.querySelector(s);
const pantallas = {};

/* ------------------------------------------------------------- arranque */

async function iniciar() {
  ["inicio", "permiso", "colocacion", "calibracion", "tablero", "error"]
    .forEach((n) => { pantallas[n] = $("#p-" + n); });

  est.slm = new SLM(await cargarGrabados());
  $("#webgpu").textContent = est.slm.soportado
    ? "Este navegador puede correr el modelo localmente."
    : "Este navegador no tiene WebGPU: las frases van a venir de una corrida grabada.";

  $("#btn-empezar").addEventListener("click", arrancarCamara);
  $("#btn-reintentar").addEventListener("click", () => location.reload());
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
  mostrar("permiso");
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 },
               facingMode: "user" },
      audio: false,
    });
    est.video = $("#video");
    est.video.srcObject = stream;
    await est.video.play();
  } catch (e) {
    return fallar(
      "No se pudo usar la cámara.",
      "Hace falta permiso de cámara y una conexión segura (https). " +
      String(e?.message || e));
  }

  try {
    const vision = await import(MP + "/+esm");
    const archivos = await vision.FilesetResolver.forVisionTasks(MP + "/wasm");
    est.landmarker = await vision.FaceLandmarker.createFromOptions(archivos, {
      baseOptions: { modelAssetPath: MODELO_CARA, delegate: "GPU" },
      // VIDEO y no IMAGE: el modo video mantiene estado entre cuadros y da
      // puntos mucho más estables, que es de donde sale la mitad del temblor.
      runningMode: "VIDEO",
      numFaces: 1,
      // Los puntos del iris son los que más informan sobre la mirada.
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: false,
    });
  } catch (e) {
    return fallar("No se pudo cargar el detector facial.",
                  String(e?.message || e));
  }

  mostrar("colocacion");
  requestAnimationFrame(lazo);
}

/* ------------------------------------------------------- lazo principal */

let ultimoTs = -1;

function leerCara() {
  const v = est.video;
  if (!v || v.readyState < 2) return null;
  // MediaPipe exige marcas de tiempo estrictamente crecientes; dos cuadros en
  // el mismo milisegundo lo hacen tirar excepción y cortar el lazo.
  let ts = Math.round(performance.now());
  if (ts <= ultimoTs) ts = ultimoTs + 1;
  ultimoTs = ts;
  const r = est.landmarker.detectForVideo(v, ts);
  return r?.faceLandmarks?.[0] ?? null;
}

function lazo(t) {
  requestAnimationFrame(lazo);
  const dt = est.ultimoT ? (t - est.ultimoT) / 1000 : 1 / 30;
  est.ultimoT = t;
  est.fps = est.fps * 0.9 + (1 / Math.max(dt, 1e-3)) * 0.1;

  let lm = null;
  try { lm = leerCara(); } catch (_) { /* un cuadro perdido no corta nada */ }
  const datos = lm ? extraer(lm) : null;

  if (est.fase === "colocacion") return pasoColocacion(datos);
  if (est.fase === "calibracion") return pasoCalibracion(datos, t);
  if (est.fase === "tablero") return pasoTablero(datos, dt, t);
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

function empezarCalibracion() {
  est.calib = { i: 0, muestras: [], objetivos: [], buffer: [],
                fase: "asentar", desde: performance.now() };
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
    if (transcurrido > MS_ASENTAR) { c.fase = "grabar"; c.desde = t; }
    return;
  }

  p.classList.add("grabando");
  p.style.setProperty("--avance", Math.min(1, transcurrido / MS_GRABAR));

  // El parpadeo NO se graba: durante un parpadeo la geometría del ojo es otra
  // y esas muestras corren el modelo hacia una cara que no está mirando.
  if (datos && datos.ear > UMBRAL_PARPADEO && c.buffer.length < MUESTRAS_POR_PUNTO) {
    c.buffer.push(datos.rasgos);
  }

  if (transcurrido > MS_GRABAR) {
    const [fx, fy] = PUNTOS[c.i];
    const objetivo = [fx * window.innerWidth, fy * window.innerHeight];
    for (const r of c.buffer) { c.muestras.push(r); c.objetivos.push(objetivo); }
    c.buffer = [];
    c.i += 1;
    if (c.i >= PUNTOS.length) return terminarCalibracion();
    c.fase = "asentar"; c.desde = t;
    pintarPuntoCalib();
  }
}

function terminarCalibracion() {
  const c = est.calib;
  if (c.muestras.length < PUNTOS.length * 4) {
    return fallar(
      "La calibración no juntó suficientes muestras.",
      `Se grabaron ${c.muestras.length}. Suele ser falta de luz o la cara ` +
      `fuera de cuadro. Pruebe de nuevo con la cara bien iluminada.`);
  }
  try {
    est.modelo.entrenar(c.muestras, c.objetivos);
  } catch (e) {
    return fallar("No se pudo ajustar el modelo.", String(e?.message || e));
  }
  const err = est.modelo.errorEntrenamiento(c.muestras, c.objetivos);
  // Error de ENTRENAMIENTO: se mide sobre los mismos puntos con los que se
  // ajustó, así que subestima el error real. Sirve para detectar una
  // calibración rota, no para reportar precisión.
  $("#ajuste").textContent = `ajuste ${err.toFixed(0)} px`;

  est.suave.reiniciar();
  est.ventana.limpiar();
  est.stats = { dwells: 0, t0: null };
  mostrar("tablero");
  redibujar();
}

/* ------------------------------------------------------------- tablero */

function redibujar() {
  const cont = $("#escena");
  if (est.candidatos) dibujarCandidatos(cont);
  else { dibujarPagina(cont, est.pagina); }
  est.cajas = medir(cont);
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
}

function pasoTablero(datos, dt, t) {
  const cursor = $("#cursor");
  let muestra = { x: 0, y: 0, ok: false };

  if (datos && datos.ear > UMBRAL_PARPADEO) {
    const [px, py] = est.modelo.predecir(datos.rasgos);
    const [sx, sy] = est.suave.paso(px, py, dt);
    const med = est.ventana.empujar(sx, sy, t);
    if (med) {
      muestra = { x: med[0], y: med[1], ok: true };
      cursor.style.transform = `translate(${med[0]}px, ${med[1]}px)`;
      cursor.hidden = false;
    }
  }

  const r = est.dwell.paso(
    muestra,
    (x, y) => golpearEn(est.cajas, x, y),
    (id) => est.cajas.get(id) ?? null);

  est.cajas.forEach((_, id) => {
    const el = $(`[data-id="${CSS.escape(id)}"]`);
    if (!el) return;
    const activo = id === r.objetivo;
    el.classList.toggle("viva", activo);
    el.style.setProperty("--avance", activo ? r.avance : 0);
  });

  if (r.disparo) {
    const el = $(`[data-id="${CSS.escape(r.disparo)}"]`);
    if (el) aplicar(el.dataset.accion);
  }

  $("#fps").textContent = est.fps.toFixed(0) + " fps";
  $("#dwells").textContent = est.stats.dwells;
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
  if (est.fase === "tablero") est.cajas = medir($("#escena"));
});

iniciar();
