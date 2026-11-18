/* Lo que pasa entre la respuesta del modelo y lo que la persona ve.
 *
 * Tres etapas, en este orden:
 *
 * 1. `parsear`  - sacar los candidatos del texto que devolvió el modelo;
 * 2. `filtrar`  - descartar los que dicen algo que la entrada no dice;
 * 3. `conLiteral` - garantizar que lo escrito tal cual sea siempre una opción.
 *
 * ## Por qué existe este archivo
 *
 * El filtro de la etapa 2 existía en Python (`slm/aac_semantic/guard.py`) y
 * todas las tablas de la tesis se midieron con él puesto. **La página no lo
 * tenía.** Lo que veía la persona en el navegador era la salida cruda del
 * modelo, con la tasa cruda de vuelcos de polaridad: medido sobre el corpus,
 * entre el 5 % y el 16 % de los conjuntos de candidatos perdían una negación o
 * una orden de detenerse. Para la entrada `NO BASTA DOLOR` eso es proponer, con
 * la voz de la persona, que sigan haciendo lo que le duele.
 *
 * Es un puerto línea por línea, no una reinterpretación. Se comprueba contra
 * el de Python en `web/pruebas/expansion.test.mjs`, sobre las decisiones que
 * el filtro de Python tomó con miles de respuestas reales de los modelos:
 * cualquier diferencia es un error de este archivo.
 *
 * Y está separado de `slm.js` para que el banco de pruebas del navegador
 * (`slm/banco_web/`) importe ESTE MISMO archivo. Lo medido es lo desplegado.
 */

/* ---------------------------------------------------------- normalización */

const PUNTUACION = /[^\p{L}\p{N}_ ]+/gu;
const ESPACIOS = /\s+/gu;
const PALABRA = /[\p{L}\p{N}_]+/gu;

/** Minúsculas, sin puntuación, espacios colapsados. Los acentos y la ñ se
 *  quedan: en español son letras, no adorno. Igual a `prompts.normalise`. */
export function normalizar(texto) {
  return String(texto ?? "").toLowerCase()
    .replace(PUNTUACION, " ").replace(ESPACIOS, " ").trim();
}

const CLITICOS = ["melo", "mela", "selo", "sela", "noslo", "nosla", "telo",
  "tela", "me", "te", "se", "nos", "le", "les", "lo", "la", "los", "las"];
const FIN_VERBAL = ["ar", "er", "ir", "ndo", "á", "é", "í"];

/** `acostarme` -> `acostar`, `me`. Solo si lo que queda parece un verbo:
 *  `clase` no se parte. Igual a `metrics._sin_clitico`. */
function sinClitico(palabra) {
  // Python cuenta caracteres, no unidades de UTF-16: con letras del español da
  // lo mismo, pero se cuenta igual para que la regla no dependa de eso.
  if ([...palabra].length < 5) return [palabra];
  for (const c of CLITICOS) {
    if (!palabra.endsWith(c)) continue;
    const raiz = palabra.slice(0, palabra.length - c.length);
    if ([...raiz].length >= 3 && FIN_VERBAL.some((f) => raiz.endsWith(f))) {
      return [raiz, c];
    }
  }
  return [palabra];
}

export function tokens(texto) {
  const salida = [];
  for (const p of normalizar(texto).match(PALABRA) ?? []) {
    salida.push(...sinClitico(p));
  }
  return salida;
}

/** F1 de tokens como multiconjunto. Igual a `metrics.token_f1`. */
export function f1Tokens(a, b) {
  const ta = tokens(a), tb = tokens(b);
  if (!ta.length || !tb.length) return !ta.length && !tb.length ? 1 : 0;
  const cuenta = new Map();
  for (const t of tb) cuenta.set(t, (cuenta.get(t) ?? 0) + 1);
  let comunes = 0;
  for (const t of ta) {
    const n = cuenta.get(t) ?? 0;
    if (n > 0) { comunes += 1; cuenta.set(t, n - 1); }
  }
  if (!comunes) return 0;
  const p = comunes / ta.length, r = comunes / tb.length;
  return (2 * p * r) / (p + r);
}

/** ¿Aparece alguna de estas palabras o frases, como palabra entera? Así "no"
 *  no se encuentra dentro de "noche". Igual a `metrics._has_any`. */
export function tieneAlguna(texto, agujas) {
  const t = ` ${normalizar(texto)} `;
  return agujas.some((a) => t.includes(` ${normalizar(a)} `));
}

/* ------------------------------------------- las listas, copiadas de Python
 *
 * Copiadas tal cual de `metrics.py` y `guard.py`. La prueba compara estas
 * listas contra las de Python, elemento por elemento: si alguien agrega una
 * palabra de un lado y no del otro, falla. */

export const NEGACION = ["no", "nada", "ninguna", "ninguno", "ningun",
  "ningún", "sin", "tampoco", "nunca", "jamas", "jamás", "quita", "saca"];

export const PARADA = [
  "detente", "detené", "detene", "detenete", "deténgase", "detengase",
  "frená", "frena", "cortala", "cortá la", "pará ya", "basta ya",
  "basta", "para", "para ya", "pará", "paren", "pare", "pará",
  "deja", "dejá", "dejen", "dejalo", "suficiente", "no",
  "ya esta", "ya está", "alcanza", "terminá", "termina",
  "no sigas", "no mas", "no más"];

export const SUSTANTIVOS_TABLERO = {
  agua: ["agua", "sed", "beber", "tomar agua"],
  comida: ["comida", "comer", "hambre", "almuerzo", "cena", "desayuno"],
  baño: ["baño", "inodoro", "orinar", "hacer pis"],
  dolor: ["dolor", "duele", "duelen"],
  medicina: ["medicina", "medicamento", "remedio", "remedios", "pastilla"],
};

/* ------------------------------------------------------------- el filtro */

/** Sustantivos del tablero que el candidato dice y la entrada no. Si la
 *  entrada dice BAÑO y un candidato habla de agua, está hablando de otra
 *  cosa. Igual a `guard.foreign_nouns`. */
export function sustantivosAjenos(entrada, candidato) {
  const e = normalizar(entrada);
  const intrusos = [];
  for (const [sust, formas] of Object.entries(SUSTANTIVOS_TABLERO)) {
    if (e.includes(sust) || formas.some((f) => tieneAlguna(e, [f]))) continue;
    if (formas.some((f) => tieneAlguna(candidato, [f]))) intrusos.push(sust);
  }
  return intrusos;
}

/** Fórmulas con una palabra negativa que no niegan nada. "de nada" tiene
 *  "nada" y pasaba como orden de detenerse: medido en la página, `BASTA` ->
 *  "de nada". Igual a `metrics.CORTESIA_SIN_NEGACION`. */
export const CORTESIA_SIN_NEGACION = ["de nada"];

export function sinCortesia(texto) {
  let t = ` ${normalizar(texto)} `;
  for (const f of CORTESIA_SIN_NEGACION) t = t.split(` ${normalizar(f)} `).join(" ");
  return t.trim();
}

const niega = (t) => tieneAlguna(sinCortesia(t), NEGACION);
const detiene = (t) => {
  const s = sinCortesia(t);
  return tieneAlguna(s, PARADA) || tieneAlguna(s, NEGACION);
};

export const pideNegacion = (entrada) => tieneAlguna(entrada, ["no"]);
export const pideParada = (entrada) => tieneAlguna(entrada, ["basta"]);

/** La entrada expresa duda: TAL VEZ, o SI y NO seguidos. La salida no puede
 *  resolverla por la persona. Igual a `guard.is_uncertain`. */
export function esDuda(entrada) {
  const t = normalizar(entrada);
  if (t.includes("tal vez")) return true;
  const toks = t.split(" ");
  return toks.includes("si") && toks.includes("no");
}

/** "mañana mañana mañana": una palabra repetida y nada más. */
function repeticionPura(texto) {
  const p = normalizar(texto).split(" ").filter(Boolean);
  return p.length > 1 && new Set(p).size === 1;
}

/** Quita lo que es la misma idea escrita distinto. Igual a `guard.dedupe`. */
export function deduplicar(candidatos, umbral = 0.9) {
  const salida = [];
  for (const c of candidatos) {
    if (repeticionPura(c)) continue;
    const aqui = new Set(normalizar(c).split(" ").filter(Boolean));
    const repetido = salida.some((k) => {
      if (f1Tokens(c, k) >= umbral) return true;
      const alla = new Set(normalizar(k).split(" ").filter(Boolean));
      return aqui.size > 0 && aqui.size === alla.size
        && [...aqui].every((t) => alla.has(t));
    });
    if (!repetido) salida.push(c);
  }
  return salida;
}

/**
 * Descarta los candidatos que violan lo que la entrada dice. Nunca reescribe:
 * solo descarta, y agrega el literal al final si no estaba.
 *
 * Igual a `guard.guarded` (deduplicar y después `apply`).
 *
 * @returns {{candidatos: string[], descartados: Array<[string, string]>,
 *            sinEleccion: boolean}}
 */
export function filtrar(entrada, candidatos, minimo = 2) {
  const literal = String(entrada ?? "").trim().toLowerCase();
  const quedan = [];
  const descartados = [];

  const exigeParada = pideParada(entrada);
  // "NO BASTA DOLOR" es una sola orden: con la parada alcanza, no se exige
  // además una negación léxica. Ver el comentario en guard.py.
  const exigeNegacion = pideNegacion(entrada) && !exigeParada;
  const duda = esDuda(entrada);

  for (const c of deduplicar(candidatos)) {
    const texto = String(c ?? "").trim();
    if (!texto) continue;
    if (exigeParada && !detiene(texto)) {
      descartados.push([texto, "perdió la orden de detenerse"]);
      continue;
    }
    if (exigeNegacion && !niega(texto)) {
      descartados.push([texto, "perdió la negación"]);
      continue;
    }
    if (duda && ["si", "sí", "no"].includes(normalizar(texto))) {
      descartados.push([texto, "resolvió la duda a sí/no"]);
      continue;
    }
    const intrusos = sustantivosAjenos(entrada, texto);
    if (intrusos.length) {
      descartados.push([texto,
        "habla de " + intrusos.join(", ") + " y la entrada no lo dice"]);
      continue;
    }
    quedan.push(texto);
  }

  if (literal && !quedan.some((c) => normalizar(c) === normalizar(literal))) {
    quedan.push(literal);
  }
  return { candidatos: quedan, descartados, sinEleccion: quedan.length < minimo };
}

/* ------------------------------------------------------ parsear y el literal */

/** Saca los candidatos del texto del modelo. Tolera JSON con "candidatos", un
 *  arreglo suelto o líneas con viñetas: un modelo que acierta pero contesta
 *  en prosa es un fallo de formato, no de la tarea. */
export function parsear(texto) {
  let t = (texto || "").trim();
  // Qwen3 puede abrir con un bloque de razonamiento aunque se le pida que no.
  t = t.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
  if (!t) return [];
  try {
    const bloque = t.match(/\{[\s\S]*\}|\[[\s\S]*\]/);
    if (bloque) {
      const j = JSON.parse(bloque[0]);
      const arr = Array.isArray(j) ? j
        : (j.candidatos || j.candidates || j.opciones || []);
      if (Array.isArray(arr) && arr.length) {
        return arr.map((s) => String(s).trim()).filter(Boolean).slice(0, 3);
      }
    }
  } catch (_) { /* no era JSON: se sigue por líneas */ }

  return t.split(/\r?\n/)
    .map((l) => l.replace(/^[\s\-*•\d.)"']+/, "").replace(/["',]+$/, "").trim())
    .filter((l) => l && l.length < 90)
    .slice(0, 3);
}

/** El literal SIEMPRE disponible, y al final. Si las propuestas están mal, la
 *  persona tiene que poder decir exactamente lo que escribió.
 *
 *  Se compara normalizado y no en minúsculas a secas: con la comparación
 *  anterior, "Agua." y el literal "agua" contaban como distintos y la persona
 *  veía la misma opción dos veces, gastando una celda. */
export function conLiteral(lista, telegrama) {
  const lit = String(telegrama ?? "").trim().toLowerCase();
  const n = normalizar(lit);
  const fuera = lista.filter((c) => c && normalizar(c) !== n).slice(0, 3);
  return [...fuera, lit];
}

/** La cadena completa, como la usa la página. */
export function procesar(telegrama, textoModelo) {
  const f = filtrar(telegrama, parsear(textoModelo));
  return {
    candidatos: conLiteral(f.candidatos, telegrama),
    descartados: f.descartados,
  };
}

/** Lo que hacía la página antes de este archivo: sin filtro. Queda solo para
 *  que el banco de pruebas pueda medir la diferencia; la página no lo usa. */
export function procesarSinFiltro(telegrama, textoModelo) {
  return { candidatos: conLiteral(parsear(textoModelo), telegrama),
           descartados: [] };
}
