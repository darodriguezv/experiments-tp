/* Vuelve a pasar las respuestas crudas del banco por el `expansion.js` actual.
 *
 *   node reprocesar.mjs ../results/web-cribado.jsonl > salida.jsonl
 *
 * El banco guarda el texto crudo de cada respuesta. Si después se corrige el
 * filtro -como pasó con "de nada" a mitad de esta medición- lo que vería la
 * persona se recalcula acá en segundos, con el código que está desplegado, en
 * vez de volver a correr horas de modelos.
 *
 * Agrega `igualAlBanco`: si lo recalculado coincide con lo que calculó la
 * página en el momento. Si el filtro no cambió, tiene que dar siempre true; es
 * un control de que el reproceso no se aparta de lo que corre en el navegador.
 */
import { readFileSync } from "node:fs";
import { parsear, procesar, procesarSinFiltro } from "../../web/js/expansion.js";

for (const ruta of process.argv.slice(2)) {
  for (const linea of readFileSync(ruta, "utf8").split(/\r?\n/)) {
    if (!linea.trim()) continue;
    const o = JSON.parse(linea);
    if (o.error || typeof o.crudo !== "string") {
      process.stdout.write(JSON.stringify(o) + "\n");
      continue;
    }
    const con = procesar(o.entrada, o.crudo);
    const salida = {
      ...o,
      propuestos: parsear(o.crudo),
      sinFiltro: procesarSinFiltro(o.entrada, o.crudo).candidatos,
      conFiltro: con.candidatos,
      descartados: con.descartados,
      conFiltroBanco: o.conFiltro,
      igualAlBanco: JSON.stringify(con.candidatos) === JSON.stringify(o.conFiltro),
    };
    process.stdout.write(JSON.stringify(salida) + "\n");
  }
}
