/* Las páginas del tablero y cómo se dibujan.
 *
 * Mismo vocabulario y mismo reparto que `board_layout.py`: 25 celdas en la
 * página de palabras, 36 en la de letras, y la última fila siempre de
 * controles. Que sea idéntico no es prolijidad - si las celdas estuvieran en
 * otro lado, el costo en permanencias que mide la tesis no sería el de esto.
 */

export const PAGINAS = {
  core: [
    ["SÍ", "NO", "POR FAVOR", "GRACIAS", "HOLA"],
    ["QUIERO", "NECESITO", "ME SIENTO", "AYUDA", "BASTA"],
    ["AGUA", "COMIDA", "BAÑO", "DOLOR", "MEDICINA"],
    ["MÁS", "MENOS", "BIEN", "MAL", "TAL VEZ"],
    [["ABC", "page:abc"], ["ESPACIO", "space"], ["BORRAR", "delete"],
     ["LIMPIAR", "clear"], ["HABLAR", "speak"]],
  ],
  abc: [
    ["A", "B", "C", "D", "E", "F"],
    ["G", "H", "I", "J", "K", "L"],
    ["M", "N", "Ñ", "O", "P", "Q"],
    ["R", "S", "T", "U", "V", "W"],
    ["X", "Y", "Z", ".", ",", "?"],
    [["PALABRAS", "page:core"], ["ESPACIO", "space"], ["BORRAR", "delete"],
     ["LIMPIAR", "clear"], ["HABLAR", "speak"]],
  ],
};

export function dibujarPagina(cont, pagina) {
  const filas = PAGINAS[pagina];
  const g = document.createElement("div");
  g.className = "grilla";
  g.style.setProperty("--cols", Math.max(...filas.map((f) => f.length)));

  filas.forEach((fila, r) => fila.forEach((c, i) => {
    const [etq, accion] = Array.isArray(c) ? c : [c, "w:" + c];
    const el = document.createElement("div");
    el.className = "celda";
    if (accion === "speak") el.classList.add("hablar");
    else if (!accion.startsWith("w:")) el.classList.add("ctrl");
    // Las celdas de control van un escalón MÁS OSCURAS que el vocabulario:
    // son herramientas, no es lo que la persona quiere decir. HABLAR es la
    // excepción, porque es la celda que emite.
    el.dataset.id = `${pagina}:${r}:${i}`;
    el.dataset.accion = accion;
    el.innerHTML = '<span class="relleno"></span><span class="trazo"></span>';
    const t = document.createElement("span");
    t.className = "etq";
    t.textContent = etq;
    el.appendChild(t);
    g.appendChild(el);
  }));

  cont.replaceChildren(g);
  return g;
}

/** Mapa id -> rectángulo, en coordenadas de viewport. Se recalcula cuando
 *  cambia la página o el tamaño: usar valores viejos manda la mirada a donde
 *  las celdas ya no están. */
export function medir(cont) {
  const m = new Map();
  cont.querySelectorAll("[data-id]").forEach((el) => {
    m.set(el.dataset.id, el.getBoundingClientRect());
  });
  return m;
}

export function golpearEn(cajas, x, y) {
  for (const [id, r] of cajas) {
    if (x >= r.left && x < r.right && y >= r.top && y < r.bottom) return id;
  }
  return null;
}

/** Las mismas cajas, como quedan después de un `scale(escala)` alrededor de
 *  `origen`.
 *
 * Se CALCULAN en vez de volver a medirlas con `getBoundingClientRect()`, y el
 * motivo es un fallo medido: `.grilla` tiene `transition: transform .18s`, así
 * que medir justo después de fijar el estilo devuelve la geometría del
 * ARRANQUE de la animación, no la final. Las cajas quedaban con el tamaño de
 * antes de acercar y no se volvían a medir nunca, así que el puntero se veía
 * sobre una celda y se marcaba otra.
 *
 * Esto no depende del reloj: da la geometría final en el mismo cuadro en que
 * se dispara el acercamiento.
 *
 * @param {Map<string, DOMRect>} cajas  las cajas SIN transformar
 * @param {[number,number]|null} origen centro del acercamiento, en viewport
 * @param {number} escala
 */
export function escalar(cajas, origen, escala) {
  if (!origen || !(escala > 0) || escala === 1) return cajas;
  const [ox, oy] = origen;
  const m = new Map();
  for (const [id, r] of cajas) {
    const left = ox + (r.left - ox) * escala;
    const top = oy + (r.top - oy) * escala;
    m.set(id, {
      left, top,
      right: ox + (r.right - ox) * escala,
      bottom: oy + (r.bottom - oy) * escala,
      width: r.width * escala,
      height: r.height * escala,
    });
  }
  return m;
}
