/* El conductor, del lado del navegador.
 *
 * Sin framework: son cuatro pantallas y un sondeo. Agregar React obligaría a
 * un paso de compilación en un proyecto que tiene que poder correr en un
 * equipo dedicado donde lo único garantizado es Python.
 *
 * El flujo de una sesión es deliberadamente estrecho -una sola cosa hacer en
 * cada pantalla- porque quien lo maneja está además atendiendo a una persona
 * en la sala. Una pantalla con seis opciones es una oportunidad de equivocarse
 * mientras se mira para otro lado.
 */

const $ = (sel, raiz = document) => raiz.querySelector(sel);
const vista = $('#vista');

const estado = {
  pestana: 'sesion',
  estudio: null,
  participante: null,
  ficha: null,
  corrida: null,
  resultados: null,
  mostrandoGuion: false,
};

const api = {
  async get(ruta) {
    const r = await fetch(ruta, { cache: 'no-store' });
    return r.json();
  },
  async post(ruta, cuerpo) {
    const r = await fetch(ruta, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(cuerpo || {}),
    });
    return r.json();
  },
};

const esc = (s) => String(s ?? '').replace(/[&<>"']/g,
  (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

const pct = (v) => (v === null || v === undefined || Number.isNaN(v))
  ? '—' : `${v > 0 ? '+' : ''}${Math.round(v * 100)}%`;

// ---------------------------------------------------------------- arranque

async function iniciar() {
  estado.estudio = await api.get('/api/estudio');
  $('#subtitulo').textContent =
    `Caso ${estado.estudio.caso.nombre} · ${estado.estudio.n} participante(s) ` +
    `de ${estado.estudio.objetivo} para equilibrio exacto`;

  $('#tabs').addEventListener('click', (ev) => {
    const b = ev.target.closest('button[data-vista]');
    if (!b) return;
    estado.pestana = b.dataset.vista;
    [...$('#tabs').children].forEach((x) =>
      x.classList.toggle('activa', x === b));
    dibujar();
  });

  sondear();
  setInterval(sondear, 1500);
}

async function sondear() {
  const previa = estado.corrida && estado.corrida.corriendo;
  estado.corrida = await api.get('/api/corrida');
  const ahora = estado.corrida.corriendo;
  // Al cerrarse el tablero hay que recargar la ficha: cambió qué bloque toca.
  if (previa && !ahora && estado.participante) {
    estado.ficha = await api.get(`/api/participante/${estado.participante}`);
    estado.resultados = null;
  }
  dibujar();
}

function dibujar() {
  if (estado.corrida && estado.corrida.corriendo) return vistaEsperando();
  if (estado.pestana === 'resultados') return vistaResultados();
  if (estado.pestana === 'diseno') return vistaDiseno();
  return vistaSesion();
}

// ------------------------------------------------------------------ sesión

function vistaEsperando() {
  const c = estado.corrida.corriendo;
  const nombre = estado.estudio.condiciones
    .find((x) => x.clave === c.condicion)?.nombre || c.condicion;
  vista.innerHTML = `
    <div class="tarjeta esperando">
      <div class="punto"></div>
      <h2>El tablero está abierto</h2>
      <p class="tenue">Participante ${c.participante} · ${esc(nombre)}</p>
      <p class="tenue chico" style="max-width:46ch;margin:22px auto 0">
        Primero la calibración, después las seis frases. Esta pantalla se
        actualiza sola cuando el bloque termina.
      </p>
    </div>`;
}

function vistaSesion() {
  if (!estado.participante) return vistaElegirParticipante();
  const f = estado.ficha;
  if (!f) return;

  if (f.completo) {
    vista.innerHTML = `
      <div class="tarjeta">
        <div class="entre">
          <h2 style="margin:0">Participante ${f.participante}</h2>
          <span class="etiqueta ok">Las tres condiciones hechas</span>
        </div>
        <p class="tenue">Ya no queda nada por correr con esta persona.</p>
        <div class="fila" style="margin-top:16px">
          <button class="suave" onclick="volver()">Elegir otro participante</button>
          <button onclick="irA('resultados')">Ver resultados</button>
        </div>
      </div>
      ${listaBloques(f)}`;
    return;
  }

  const sig = f.siguiente;
  const ultimo = estado.corrida && estado.corrida.ultimo;

  if (estado.mostrandoGuion) return vistaGuion(f, sig);

  vista.innerHTML = `
    ${ultimo && ultimo.error ? `
      <div class="tarjeta" style="border-color:var(--aviso)">
        <h2 style="color:var(--aviso)">El último bloque falló</h2>
        <pre class="plano">${esc(ultimo.error)}</pre>
      </div>` : ''}
    ${ultimo && !ultimo.error && ultimo.enunciados !== undefined ? `
      <div class="tarjeta">
        <div class="entre">
          <h2 style="margin:0">Bloque anterior guardado</h2>
          <span class="etiqueta ${ultimo.incompleto ? 'aviso' : 'ok'}">
            ${ultimo.enunciados} enunciado(s)${ultimo.incompleto ? ' · incompleto' : ''}
          </span>
        </div>
        ${ultimo.incompleto ? `<p class="chico tenue">Se cerró antes de
          terminar las seis tareas. Queda marcado como incompleto; no se
          descarta.</p>` : ''}
      </div>` : ''}

    <div class="tarjeta">
      <div class="entre">
        <h2 style="margin:0">Participante ${f.participante}</h2>
        <span class="etiqueta">Le toca ${esc(sig.nombre)}</span>
      </div>
      <p class="tenue chico">
        Bloque ${sig.posicion} de 3 · conjunto de frases F${sig.conjunto}.
        El orden lo fija el contrabalanceo, no se elige.
      </p>

      <div class="aviso-caja" style="margin:18px 0">
        ${esc(sig.instrucciones)}
      </div>

      <h3>Antes de empezar, en voz alta</h3>
      <ul class="chico tenue" style="margin:0 0 18px;padding-left:20px">
        <li>Primero viene la calibración. Mire cada punto hasta que desaparezca.</li>
        <li>Van a aparecer seis consignas, una por vez, arriba del tablero.</li>
        <li>Diga cada una con sus palabras. No tiene que ser exacta.</li>
        <li>Si algo sale mal, siga: el error también es un dato.</li>
      </ul>

      <div class="campo" style="max-width:260px">
        <label for="monitor">Pantalla del tablero</label>
        <select id="monitor">
          <option value="ask">Preguntar al abrir</option>
          <option value="0">Pantalla 0</option>
          <option value="1">Pantalla 1</option>
        </select>
      </div>

      <div class="fila">
        ${sig.con_guion
          ? `<button class="grande" onclick="verGuion()">Leer el papel de
               ${esc(estado.estudio.caso.nombre)} →</button>`
          : `<button class="grande" onclick="lanzar()">Abrir el tablero</button>`}
        <button class="suave" onclick="volver()">Cambiar de participante</button>
      </div>
    </div>
    ${listaBloques(f)}`;
}

function vistaGuion(f, sig) {
  const caso = estado.estudio.caso;
  vista.innerHTML = `
    <div class="tarjeta">
      <h2>Su papel en esta parte</h2>
      <p class="tenue chico">Léalo con calma. Cuando termine, avise.</p>
      <div class="guion-caja" style="margin:18px 0">
        <div class="guion">${esc(caso.guion)}</div>
      </div>
      <div class="fila">
        <button class="grande" onclick="lanzar()">Ya lo leí, empecemos</button>
        <button class="suave" onclick="cerrarGuion()">Volver</button>
      </div>
    </div>`;
}

function listaBloques(f) {
  return `
    <div class="tarjeta">
      <h2>Las tres condiciones de esta persona</h2>
      <ol class="pasos">
        ${f.bloques.map((b) => `
          <li class="paso ${b.hecho ? 'listo' : ''}">
            <div class="num">${b.hecho ? '✓' : b.posicion}</div>
            <div class="cuerpo">
              <div class="titulo">${esc(b.nombre)}</div>
              <div class="detalle">
                Conjunto F${b.conjunto}${b.hecho
                  ? ` · ${b.enunciados} enunciado(s) medidos`
                  : ' · pendiente'}
              </div>
            </div>
          </li>`).join('')}
      </ol>
    </div>`;
}

function vistaElegirParticipante() {
  const lista = estado.estudio.participantes;
  vista.innerHTML = `
    <div class="tarjeta">
      <h2>¿Con quién vamos a trabajar?</h2>
      ${lista.length ? `
        <div class="fila" style="margin:14px 0">
          ${lista.map((n) => `
            <button class="suave" onclick="elegir(${n})">Participante ${n}</button>
          `).join('')}
        </div>` : `
        <p class="tenue">Todavía no hay nadie dado de alta.</p>`}

      <h3>Dar de alta a alguien nuevo</h3>
      <div class="campo" style="max-width:420px">
        <label for="notas">Notas (opcional, sin datos que identifiquen)</label>
        <input id="notas" placeholder="usa anteojos; sesión de la mañana">
      </div>
      <button onclick="alta()">Dar de alta</button>
      <p class="chico tenue" style="margin-top:14px;max-width:58ch">
        Acá la persona es solo un número. El consentimiento se firma en papel y
        se guarda aparte: nada de lo que se escriba en esta pantalla debería
        servir para saber quién es.
      </p>
    </div>`;
}

// ------------------------------------------------------------- resultados

async function vistaResultados() {
  if (!estado.resultados) {
    vista.innerHTML = '<div class="tarjeta"><p class="tenue">Calculando…</p></div>';
    estado.resultados = await api.get('/api/resultados');
  }
  const r = estado.resultados;
  const ag = r.agregado;

  const fila = (cond) => {
    const a = ag[cond] || {};
    return `<tr>
      <td>${esc(r.nombres[cond])}</td>
      <td class="num">${pct(a.ahorro_permanencias_p50)}</td>
      <td class="num">${pct(a.ahorro_segundos_p50)}</td>
      <td class="num">${pct(a.acierto_p50).replace('+', '')}</td>
      <td class="num">${a.n ?? 0}</td>
    </tr>`;
  };

  vista.innerHTML = `
    <div class="tarjeta">
      <div class="entre">
        <h2 style="margin:0">Ahorro contra el deletreo</h2>
        <span class="etiqueta">${ag.n_completos} de ${ag.n_participantes} completos</span>
      </div>
      <p class="tenue chico">
        Mediana del ahorro de cada persona contra su propia línea de base.
        Intra-sujeto: así la enorme variabilidad entre personas se cancela.
      </p>
      <table style="margin-top:14px">
        <thead><tr>
          <th>Condición</th><th class="num">Permanencias</th>
          <th class="num">Segundos</th><th class="num">Acierto</th>
          <th class="num">n</th>
        </tr></thead>
        <tbody>${fila('sin_contexto')}${fila('con_contexto')}</tbody>
      </table>
      <p class="chico tenue" style="margin-top:16px;max-width:62ch">
        Sin pruebas de significancia a propósito: con diez participantes
        cualquier p se leería como más de lo que es. El dato está por
        participante y por tarea para que el contraste emparejado lo haga quien
        corresponda.
      </p>
    </div>

    ${r.pendientes.length ? `
    <div class="tarjeta">
      <div class="entre">
        <h2 style="margin:0">Esperan su juicio</h2>
        <span class="etiqueta aviso">${r.pendientes.length}</span>
      </div>
      <p class="tenue chico" style="max-width:62ch">
        A la persona se le pidió una intención, no una frase, así que hay
        respuestas correctas que no se parecen en las palabras. Estas quedaron
        en la zona gris y las resuelve usted. Queda registrado.
      </p>
      <table style="margin-top:12px">
        <thead><tr>
          <th>P</th><th>Se le pidió</th><th>Dijo</th>
          <th class="num">F1</th><th></th>
        </tr></thead>
        <tbody>
          ${r.pendientes.map((p) => `
            <tr>
              <td>${p.participante}</td>
              <td class="tenue">${esc(p.intencion)}</td>
              <td><strong>${esc(p.dicho)}</strong></td>
              <td class="num">${p.f1}</td>
              <td class="num" style="white-space:nowrap">
                <button class="suave" onclick="juzgar(${p.participante},
                  '${p.condicion}','${p.tarea}','acierto')">Comunica</button>
                <button class="suave" onclick="juzgar(${p.participante},
                  '${p.condicion}','${p.tarea}','fallo')">No</button>
              </td>
            </tr>`).join('')}
        </tbody>
      </table>
    </div>` : ''}

    <div class="tarjeta">
      <h2>Por participante</h2>
      ${r.participantes.length ? `
      <table>
        <thead><tr>
          <th>P</th><th>Condición</th>
          <th class="num">Permanencias</th><th class="num">Segundos</th>
          <th class="num">Acierto</th>
        </tr></thead>
        <tbody>
          ${r.participantes.flatMap((f) =>
            ['sin_contexto', 'con_contexto'].map((c) => {
              const a = f[`ahorro_${c}`] || {};
              const s = f[c] || {};
              if (!s.n) return '';
              return `<tr>
                <td>${f.participante}</td>
                <td>${esc(r.nombres[c])}</td>
                <td class="num">${pct(a.ahorro_permanencias)}</td>
                <td class="num">${pct(a.ahorro_segundos)}</td>
                <td class="num">${s.tasa_acierto === undefined ? '—'
                  : Math.round(s.tasa_acierto * 100) + '%'}</td>
              </tr>`;
            })).join('')}
        </tbody>
      </table>` : '<p class="tenue">Todavía no hay ningún bloque medido.</p>'}
    </div>`;
}

// ----------------------------------------------------------------- diseño

function vistaDiseno() {
  vista.innerHTML = `
    <div class="tarjeta">
      <h2>Por qué cada persona hace las condiciones en otro orden</h2>
      <p style="max-width:62ch">
        Para cuando llega a la tercera condición, la persona lleva veinte
        minutos manejando un cursor con los ojos: es bastante más diestra que
        al empezar. Si todos hicieran el mismo orden y la tercera condición
        ganara, no habría forma de distinguir cuánto fue el contexto y cuánto
        la práctica.
      </p>
      <pre class="plano">${esc(estado.estudio.balance)}</pre>
    </div>
    <div class="tarjeta">
      <h2>El caso</h2>
      <p class="tenue chico">
        Es el mismo para todos. Con diez participantes el caso no puede ser una
        variable: fijarlo lo convierte en constante, que es lo único defendible
        con esta muestra. El precio es que los resultados valen
        <strong>para este caso</strong>.
      </p>
      <div class="guion-caja" style="margin-top:16px">
        <div class="guion">${esc(estado.estudio.caso.guion)}</div>
      </div>
    </div>`;
}

// ------------------------------------------------------------- acciones

async function alta() {
  const notas = $('#notas') ? $('#notas').value : '';
  const r = await api.post('/api/participante', { notas });
  estado.estudio = await api.get('/api/estudio');
  elegir(r.participante);
}

async function elegir(n) {
  estado.participante = n;
  estado.ficha = await api.get(`/api/participante/${n}`);
  estado.mostrandoGuion = false;
  dibujar();
}

function volver() {
  estado.participante = null;
  estado.ficha = null;
  estado.mostrandoGuion = false;
  dibujar();
}

function verGuion() { estado.mostrandoGuion = true; dibujar(); }
function cerrarGuion() { estado.mostrandoGuion = false; dibujar(); }
function irA(p) {
  estado.pestana = p;
  [...$('#tabs').children].forEach((x) =>
    x.classList.toggle('activa', x.dataset.vista === p));
  dibujar();
}

async function lanzar() {
  const sig = estado.ficha.siguiente;
  const monitor = $('#monitor') ? $('#monitor').value : 'ask';
  estado.mostrandoGuion = false;
  const r = await api.post('/api/lanzar', {
    participante: estado.ficha.participante,
    condicion: sig.condicion,
    monitor,
  });
  if (!r.ok) { alert(r.error); return; }
  sondear();
}

async function juzgar(participante, condicion, tarea, veredicto) {
  await api.post('/api/adjudicar',
    { participante, condicion, tarea, veredicto });
  estado.resultados = null;
  vistaResultados();
}

iniciar();
