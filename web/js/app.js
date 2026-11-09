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
import { ModeloMirada, entrenarConCV } from "./modelo.js";
import { Suavizador, VentanaMediana } from "./suavizado.js";
import { Permanencia } from "./permanencia.js";
import { PAGINAS, dibujarPagina, medir, golpearEn } from "./tablero.js";
import { SLM } from "./slm.js";
import { Zoom, DWELL_ACERCAR_MS } from "./zoom.js";
import { PruebaPrecision, PUNTOS_PRUEBA, aciertoPorGrilla,
         zoomNecesario, informe as informePrecision }
  from "./precision.js";

const MP = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";
const MODELO_CARA =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/" +
  "face_landmarker/float16/1/face_landmarker.task";

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
const MUESTRAS_POR_PUNTO = 30;
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

const est = {
  fase: "menu",
  modo: "libre", cola: [], protocolo: null,
  guion: null, tarea: 0,
  video: null, landmarker: null,
  modelo: new ModeloMirada(),
  suave: new Suavizador(),
  ventana: new VentanaMediana(320),
  dwell: new Permanencia(),
  // Dos permanencias distintas: acercar es barato de equivocar,
  // elegir no. Ver zoom.js.
  dwellZoom: new Permanencia({ dwellMs: DWELL_ACERCAR_MS,
                               refractarioMs: 250 }),
  zoom: new Zoom(),
  correccion: [0, 0],
  deriva: null,
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

  $("#btn-empezar").addEventListener("click", arrancarCamara);
  $("#btn-reintentar").addEventListener("click", () => {
    if (est.reintentarCalibracion && est.landmarker) {
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
    // El espejo del ovalo comparte el MISMO stream: dos <video> sobre un
    // MediaStream es seguro y evita pedir la camara dos veces.
    const espejo = $("#espejo");
    if (espejo) { espejo.srcObject = stream; espejo.play().catch(() => {}); }

    // Si el elemento no tiene dimensiones reales, el detector recibe una
    // imagen inservible y nada de lo que sigue significa algo. Vale la pena
    // comprobarlo y decirlo, en vez de dejar que falle silenciosamente.
    await new Promise((r) => {
      if (est.video.videoWidth) return r();
      est.video.addEventListener("loadedmetadata", r, { once: true });
      setTimeout(r, 3000);
    });
    est.resolucion = `${est.video.videoWidth}x${est.video.videoHeight}`;
    console.log("[camara]", est.resolucion);
    if (est.video.videoWidth < 320) {
      return fallar("La cámara entrega una imagen demasiado chica.",
                    `${est.resolucion}. Se necesita al menos 320 px de ancho `
                    + `para ubicar los puntos del iris.`);
    }
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

  // Diagnostico barato y permanente. Sin esto, "funciona mal" no se puede
  // distinguir de "el detector no ve nada" ni de "los rasgos estan
  // congelados", que son tres problemas con tres arreglos distintos.
  if (lm) {
    est.nLandmarks = lm.length;
    if (datos) {
      const firma = datos.rasgos[0] + datos.rasgos[250] + datos.rasgos[483];
      if (est.firmaPrevia !== undefined && firma !== est.firmaPrevia) {
        est.rasgosVivos = (est.rasgosVivos || 0) + 1;
      }
      est.firmaPrevia = firma;
    }
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

function empezarCalibracion() {
  est.calib = { i: 0, muestras: [], objetivos: [], grupos: [], buffer: [],
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
    const espera = c.i === 0 ? MS_ASENTAR_PRIMERO : MS_ASENTAR;
    if (transcurrido > espera) { c.fase = "grabar"; c.desde = t; }
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
    for (const r of c.buffer) {
      c.muestras.push(r); c.objetivos.push(objetivo); c.grupos.push(c.i);
    }
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
    // Alpha lo elige la propia calibración, por validación cruzada dejando
    // fuera un punto entero. El valor fijo que había antes (5000) colapsaba
    // el modelo: predecía casi el centro de la pantalla para cualquier
    // entrada, con 354 px de sesgo y 40 px de temblor.
    est.modelo = entrenarConCV(c.muestras, c.objetivos, c.grupos);
  } catch (e) {
    return fallar("No se pudo ajustar el modelo.", String(e?.message || e));
  }
  // Se muestra el error de VALIDACIÓN CRUZADA, no el de entrenamiento: el de
  // entrenamiento siempre se ve bien -se mide sobre los mismos puntos con los
  // que se ajustó- y por eso una calibración rota pasaba desapercibida.
  const cv = est.modelo.errorCV;

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
  // menos de la mitad de lo que debería, y eso no se arregla con zoom.
  const rec = est.modelo.recorrido;
  console.log("[calibración] alpha", est.modelo.alfaElegido,
              "LOO", Math.round(cv), "recorrido", rec?.toFixed(2));
  if (Number.isFinite(rec) && rec < 0.45) {
    return calibracionMala(rec, cv, c);
  }

  est.suave.reiniciar();
  est.ventana.limpiar();
  est.stats = { dwells: 0, t0: null };
  // El contexto va al prompt SOLO en la condición contextualizada. Es la
  // variable independiente de esa comparación: todo lo demás -modelo, prompt,
  // tablero- queda idéntico entre las dos condiciones con modelo.
  est.slm.contexto = est.cfgActual?.contexto
    ? (est.protocolo?.caso?.contexto || "") : "";
  est.correccion = [0, 0];
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
    est.modelo = new ModeloMirada();
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
  if (datos && datos.ear > UMBRAL_PARPADEO) {
    const [px, py] = est.modelo.predecir(datos.rasgos);
    const [sx, sy] = est.suave.paso(px, py, dt);
    const med = est.ventana.empujar(sx, sy, t);
    if (med) {
      const cx = med[0] + est.correccion[0];
      const cy = med[1] + est.correccion[1];
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

seguimiento de la calibración: `
    + `${((est.modelo.recorrido ?? NaN) * 100).toFixed(0)}%`
    + ` (alpha ${est.modelo.alfaElegido})`;

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
    `camara ${est.resolucion}, ${est.nLandmarks} landmarks`,
  ].join("\n");
  mostrar("resultado");
}

/* ------------------------------------------------------------- zoom */

function aplicarZoom() {
  const g = $("#escena").firstElementChild;
  if (!g) return;
  Object.assign(g.style, est.zoom.estilo());
  // Las cajas hay que volver a medirlas: getBoundingClientRect() ya tiene en
  // cuenta la transformación, así que el impacto sobre las celdas funciona
  // solo, pero con los valores viejos apuntaría a donde estaban antes.
  est.cajas = medir($("#escena"));
  const ind = $("#zoom-ind");
  if (ind) {
    ind.hidden = !est.zoom.acercado;
    ind.textContent = est.zoom.acercado
      ? `${est.zoom.escala.toFixed(1)}x` : "";
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
 */
function calibracionMala(rec, cv, c) {
  const porPunto = errorPorPuntoCalib(c);
  const peores = porPunto
    .map((e, i) => [i + 1, e])
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .filter(([, e]) => Number.isFinite(e));

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

/** Error de cada punto de calibración, dejándolo fuera del ajuste. */
function errorPorPuntoCalib(c) {
  const puntos = [...new Set(c.grupos)];
  return puntos.map((fuera) => {
    const dentro = [], afuera = [];
    c.grupos.forEach((g, i) => (g === fuera ? afuera : dentro).push(i));
    if (dentro.length < 4 || !afuera.length) return NaN;
    try {
      const m = new ModeloMirada().entrenar(
        dentro.map((i) => c.muestras[i]), dentro.map((i) => c.objetivos[i]),
        est.modelo.alfaElegido);
      let s = 0;
      for (const i of afuera) {
        const [px, py] = m.predecir(c.muestras[i]);
        s += Math.hypot(px - c.objetivos[i][0], py - c.objetivos[i][1]);
      }
      return s / afuera.length;
    } catch (_) {
      return NaN;
    }
  });
}

/* ------------------------------------------------- corrección de desvío */

/* Medido en cuatro sesiones: los nueve puntos salían corridos en la MISMA
 * dirección, +259 y +282 px de promedio. Un desvío uniforme no se arregla
 * recalibrando -vuelve a aparecer- pero se mide y se resta en dos segundos.
 *
 * Es la corrección de deriva que hacen los medidores de mirada de laboratorio
 * entre ensayos, y por el mismo motivo: la relación entre la geometría del ojo
 * y la pantalla deriva con la postura, y volver a ajustar el modelo entero es
 * caro mientras que recentrarlo es casi gratis. */
const MS_DERIVA_ASENTAR = 1400;
const MS_DERIVA_MEDIR = 1200;

function iniciarDeriva() {
  est.deriva = { fase: "asentar", desde: 0, buffer: [] };
  const el = $("#deriva-punto");
  el.style.left = "50%";
  el.style.top = "50%";
  mostrar("deriva");
}

function pasoDeriva(datos, dt, t) {
  const d = est.deriva;
  if (!d.desde) d.desde = t;
  const pasado = t - d.desde;
  const el = $("#deriva-punto");

  let muestra = null;
  if (datos && datos.ear > UMBRAL_PARPADEO) {
    // Se predice SIN la corrección vigente: lo que se quiere medir es el
    // desvío crudo del modelo, no el que queda después de corregir.
    const [px, py] = est.modelo.predecir(datos.rasgos);
    const [sx, sy] = est.suave.paso(px, py, dt);
    const med = est.ventana.empujar(sx, sy, t);
    if (med) muestra = med;
  }

  if (d.fase === "asentar") {
    el.classList.remove("grabando");
    if (pasado > MS_DERIVA_ASENTAR) { d.fase = "medir"; d.desde = t; }
    return;
  }

  el.classList.add("grabando");
  el.style.setProperty("--avance", Math.min(1, pasado / MS_DERIVA_MEDIR));
  if (muestra) d.buffer.push(muestra);

  if (pasado > MS_DERIVA_MEDIR) {
    if (d.buffer.length >= 5) {
      const xs = d.buffer.map((q) => q[0]).sort((a, b) => a - b);
      const ys = d.buffer.map((q) => q[1]).sort((a, b) => a - b);
      const m = (a) => a[a.length >> 1];
      est.correccion = [innerWidth / 2 - m(xs), innerHeight / 2 - m(ys)];
      console.log("[deriva] corrección", est.correccion.map(Math.round));
    }
    seguirDespuesDeCalibrar();
  }
}

function seguirDespuesDeCalibrar() {
  if (est.modo === "precision") {
    est.prueba = new PruebaPrecision();
    pintarPuntoPrueba();
    return mostrar("precision");
  }
  mostrar("tablero");
  redibujar();
}

function redibujar() {
  const cont = $("#escena");
  if (est.candidatos) dibujarCandidatos(cont);
  else { dibujarPagina(cont, est.pagina); }
  aplicarZoom();
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

  if (datos && datos.ear > UMBRAL_PARPADEO) {
    const [px, py] = est.modelo.predecir(datos.rasgos);
    const [sx, sy] = est.suave.paso(px, py, dt);
    const med = est.ventana.empujar(sx, sy, t);
    if (med) {
      const cx = med[0] + est.correccion[0];
      const cy = med[1] + est.correccion[1];
      muestra = { x: cx, y: cy, ok: true };
      cursor.style.transform = `translate(${cx}px, ${cy}px)`;
      cursor.hidden = false;
    }
  }

  // --- primer tiempo: acercar donde cae la mirada ----------------------
  if (!est.zoom.acercado && !est.candidatos) {
    const z = est.dwellZoom.paso(
      muestra,
      // Cualquier punto sobre la grilla sirve: no se elige una región de una
      // lista, se acerca donde está mirando. Por eso no puede fallar.
      (x, y) => (golpearEn(est.cajas, x, y) ? "grilla" : null),
      () => ({ left: 0, top: 0, right: innerWidth, bottom: innerHeight,
               width: innerWidth, height: innerHeight }));
    pintarAvanceZoom(z.avance);
    if (z.disparo) {
      est.zoom.acercar(muestra.x, muestra.y, t);
      aplicarZoom();
      est.dwell.reiniciar();
      est.dwell.bloquear(200);
    }
    return;
  }

  // Si se acercó a la zona equivocada, se vuelve solo. Sin esto habría que
  // adivinar cuál celda es la de salir.
  if (est.zoom.venció(t)) {
    est.zoom.alejar();
    aplicarZoom();
    est.dwellZoom.reiniciar();
    return;
  }

  // --- segundo tiempo: elegir, ya con las celdas grandes ----------------
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
    est.zoom.alejar();
    aplicarZoom();
    est.dwellZoom.reiniciar();
    if (el) aplicar(el.dataset.accion);
  }

  $("#fps").textContent = est.fps.toFixed(0) + " fps";
  $("#dwells").textContent = est.stats.dwells;
  const diag = $("#diag");
  if (diag) {
    diag.textContent = `${est.nLandmarks || 0} pts · ${est.resolucion || "?"}`
      + (est.rasgosVivos ? "" : " · RASGOS CONGELADOS");
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
  if (est.fase === "tablero") est.cajas = medir($("#escena"));
});

iniciar();
