/* Cómo llega de verdad una entrada escrita con la mirada.
 *
 * El corpus está escrito limpio: mayúsculas, acentos, sin errores. Lo que
 * escribe alguien con un tablero de mirada no siempre es así, y el modelo
 * tiene que tolerar las diferencias que no cambian el sentido. Cada
 * perturbación es determinista dada la entrada y la repetición `k`, para que
 * una corrida se pueda reproducir.
 */

const VECINAS = {
  // Letras vecinas en el abecedario del tablero (filas de 7): con 90 px de
  // error, es la confusión más probable al deletrear.
  A: "BH", B: "ACI", C: "BDJ", D: "CEK", E: "DFL", F: "EGM", G: "FN",
  H: "AIO", I: "HJBP", J: "IKCQ", K: "JLDR", L: "KMES", M: "LFT", N: "GU",
  O: "HPV", P: "OQIW", Q: "PRJX", R: "QSKY", S: "RTLZ", T: "SM", U: "N",
  V: "OW", W: "VXP", X: "WYQ", Y: "XZR", Z: "YS", Ñ: "N",
};

/** Un número de 0 a 1 estable para (texto, k). */
function azar(texto, k, sal = 0) {
  let h = 2166136261 ^ (k * 2654435761) ^ sal;
  for (const c of texto) { h ^= c.codePointAt(0); h = Math.imul(h, 16777619); }
  return ((h >>> 0) % 100000) / 100000;
}

const SIN_ACENTO = { Á: "A", É: "E", Í: "I", Ó: "O", Ú: "U", Ñ: "N", Ü: "U" };

export const PERTURBACIONES = {
  /** Todo en minúsculas: no debería cambiar nada. */
  minusculas: (t) => t.toLowerCase(),
  /** Sin tildes ni eñe: el tablero puede no tenerlas, o la persona omitirlas. */
  sin_acentos: (t) => t.replace(/[ÁÉÍÓÚÑÜ]/g, (c) => SIN_ACENTO[c]),
  /** Espacios de más, como cuando se elige ESPACIO dos veces. */
  espacios: (t) => " " + t.split(" ").join("  ") + " ",
  /** Una palabra repetida, como cuando la permanencia dispara dos veces. */
  palabra_doble: (t, k) => {
    const p = t.split(" ");
    const i = Math.floor(azar(t, k, 1) * p.length);
    p.splice(i, 0, p[i]);
    return p.join(" ");
  },
  /** Una letra cambiada por una vecina en el tablero, en una palabra
   *  deletreada de al menos cuatro letras. Si no hay, queda igual. */
  letra_vecina: (t, k) => {
    const p = t.split(" ");
    const largas = p.map((w, i) => [w, i]).filter(([w]) => w.length >= 4);
    if (!largas.length) return t;
    const [w, i] = largas[Math.floor(azar(t, k, 2) * largas.length)];
    const j = 1 + Math.floor(azar(t, k, 3) * (w.length - 1));
    const vec = VECINAS[w[j]];
    if (!vec) return t;
    p[i] = w.slice(0, j) + vec[Math.floor(azar(t, k, 4) * vec.length)] + w.slice(j + 1);
    return p.join(" ");
  },
};

export function perturbar(texto, nombre, k) {
  const f = PERTURBACIONES[nombre];
  if (!f) throw new Error(`perturbación ${nombre} no existe`);
  return f(texto, k);
}
