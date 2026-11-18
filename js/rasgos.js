/* Extracción de rasgos: de 478 puntos faciales a un vector que predice la mirada.
 *
 * Port del `extract_features` de eyetrax (gaze.py), con los mismos índices y
 * la misma normalización. Que sea el mismo diseño importa: lo que se mide en
 * Python con `main.py accuracy` tiene que ser lo que corre acá, o son dos
 * sistemas distintos que se parecen.
 *
 * ## La normalización es lo que hace que esto funcione
 *
 * Los puntos crudos de MediaPipe se mueven cuando la cabeza se mueve, y eso
 * domina cualquier señal del ojo. La normalización saca tres cosas:
 *
 * 1. **Posición**: se resta el punto medio entre las comisuras externas de los
 *    ojos, así el origen viaja con la cara.
 * 2. **Rotación**: se arma un marco local (x hacia el ojo derecho, y hacia la
 *    coronilla, z el producto cruz) y se rota todo a ese marco.
 * 3. **Escala**: se divide por la distancia entre ojos, así acercarse o
 *    alejarse de la cámara no cambia los números.
 *
 * Lo que queda es la geometría del ojo *relativa a la cara*, que es donde vive
 * la dirección de la mirada. Los tres ángulos de pose se agregan aparte,
 * porque la pose sí informa: mirar la misma celda con la cabeza girada da una
 * geometría ocular distinta.
 */

export const RASGOS_DIM = 486;   // 161 puntos x 3 + yaw/pitch/roll

/* Índices de MediaPipe FaceLandmarker, copiados de eyetrax/constants.py.
 * Incluyen los puntos del iris (468-477), que son los que más informan: por eso
 * hace falta un modelo con refineLandmarks / iris activado. */
export const IZQ = [
  107, 66, 105, 63, 70, 55, 65, 52, 53, 46, 468, 469, 470, 471, 472, 133,
  33, 173, 157, 158, 159, 160, 161, 246, 155, 154, 153, 145, 144, 163, 7,
  243, 190, 56, 28, 27, 29, 30, 247, 130, 25, 110, 24, 23, 22, 26, 112,
  244, 189, 221, 222, 223, 224, 225, 113, 226, 31, 228, 229, 230, 231,
  232, 233, 193, 245, 128, 121, 120, 119, 118, 117, 111, 35, 124, 143, 156
];
export const DER = [
  336, 296, 334, 293, 300, 285, 295, 282, 283, 276, 473, 476, 475, 474,
  477, 362, 263, 398, 384, 385, 386, 387, 388, 466, 382, 381, 380, 374,
  373, 390, 249, 463, 414, 286, 258, 257, 259, 260, 467, 359, 255, 339,
  254, 253, 252, 256, 341, 464, 413, 441, 442, 443, 444, 445, 342, 446,
  261, 448, 449, 450, 451, 452, 453, 417, 465, 357, 350, 349, 348, 347,
  346, 340, 265, 353, 372, 383
];
export const MUTUO = [
  4, 10, 151, 9, 152, 234, 454, 58, 288
];

export const INDICES = [...IZQ, ...DER, ...MUTUO];

/* Puntos para detectar el parpadeo. Los mismos que usa eyetrax. */
const OJO_IZQ = { interno: 133, externo: 33, arriba: 159, abajo: 145 };
const OJO_DER = { interno: 362, externo: 263, arriba: 386, abajo: 374 };

const ANCLA_IZQ = 33, ANCLA_DER = 263, CORONILLA = 10;

function resta(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function norma(v) { return Math.hypot(v[0], v[1], v[2]); }
function escala(v, k) { return [v[0] * k, v[1] * k, v[2] * k]; }
function punto(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function cruz(a, b) {
  return [a[1] * b[2] - a[2] * b[1],
          a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]];
}
function normalizar(v) { return escala(v, 1 / (norma(v) + 1e-9)); }

/**
 * @param {Array<{x:number,y:number,z:number}>} lm  los 478 puntos de MediaPipe
 * @returns {{rasgos: Float64Array, pose: {yaw:number,pitch:number,roll:number},
 *            ear: number, centro: [number,number], distOjos: number} | null}
 */
export function extraer(lm) {
  if (!lm || lm.length <= Math.max(...INDICES)) return null;

  const p = (i) => [lm[i].x, lm[i].y, lm[i].z];
  const izq = p(ANCLA_IZQ), der = p(ANCLA_DER), arriba = p(CORONILLA);

  const centroOjos = [(izq[0] + der[0]) / 2,
                      (izq[1] + der[1]) / 2,
                      (izq[2] + der[2]) / 2];

  // Marco local. `y` se ortogonaliza contra `x` (Gram-Schmidt) porque la
  // coronilla no está exactamente perpendicular a la línea de los ojos, y sin
  // ortogonalizar el marco se deforma al inclinar la cabeza.
  const ejeX = normalizar(resta(der, izq));
  const yAprox = normalizar(resta(arriba, centroOjos));
  const ejeY = normalizar(resta(yAprox, escala(ejeX, punto(yAprox, ejeX))));
  const ejeZ = normalizar(cruz(ejeX, ejeY));

  // R es la matriz [x|y|z] por columnas; rotar al marco local es multiplicar
  // por su transpuesta, que para una rotación es su inversa.
  const rotar = (v) => [punto(v, ejeX), punto(v, ejeY), punto(v, ejeZ)];

  const izqRot = rotar(resta(izq, centroOjos));
  const derRot = rotar(resta(der, centroOjos));
  const distOjos = norma(resta(derRot, izqRot));
  const k = distOjos > 1e-7 ? 1 / distOjos : 1;

  const rasgos = new Float64Array(RASGOS_DIM);
  for (let i = 0; i < INDICES.length; i++) {
    const r = rotar(resta(p(INDICES[i]), centroOjos));
    rasgos[i * 3]     = r[0] * k;
    rasgos[i * 3 + 1] = r[1] * k;
    rasgos[i * 3 + 2] = r[2] * k;
  }

  // Ángulos de pose desde R. Mismas fórmulas que eyetrax: R[fila][columna]
  // con las columnas siendo ejeX/ejeY/ejeZ.
  const yaw   = Math.atan2(ejeX[1], ejeX[0]);
  const pitch = Math.atan2(-ejeX[2], Math.hypot(ejeY[2], ejeZ[2]));
  const roll  = Math.atan2(ejeY[2], ejeZ[2]);
  const base = INDICES.length * 3;
  rasgos[base] = yaw; rasgos[base + 1] = pitch; rasgos[base + 2] = roll;

  return {
    rasgos,
    pose: { yaw, pitch, roll },
    ear: (razonOjo(lm, OJO_IZQ) + razonOjo(lm, OJO_DER)) / 2,
    centro: [centroOjos[0], centroOjos[1]],
    distOjos,
  };
}

/* Eye Aspect Ratio: alto del párpado sobre ancho del ojo. Cae a casi cero al
 * parpadear. Se usa para NO congelar el cursor durante un parpadeo, que de
 * otro modo se lee como que la mirada saltó. */
function razonOjo(lm, o) {
  const alto = Math.hypot(lm[o.arriba].x - lm[o.abajo].x,
                          lm[o.arriba].y - lm[o.abajo].y);
  const ancho = Math.hypot(lm[o.externo].x - lm[o.interno].x,
                           lm[o.externo].y - lm[o.interno].y);
  return ancho > 1e-9 ? alto / ancho : 0;
}

export const UMBRAL_PARPADEO = 0.14;
