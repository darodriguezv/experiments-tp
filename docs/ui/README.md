# La interfaz del tablero

Figuras: `tablero.png` (vocabulario), `letras.png` (deletreo), `candidatos.png`
(lo que propone el SLM). Se regeneran con `python docs/ui/render.py .`, que
dibuja llamando al mismo código que corre en el dispositivo.

## Por qué estos colores

La paleta anterior era ámbar sobre marrón. Se cambió por una pizarra fría con
un acento verde azulado, pero el motivo no es el gusto:

- **El acento es lo único saturado de toda la pantalla.** Eso lo reserva para
  una sola cosa: dónde está la mirada y cuánto falta para que la celda dispare.
  Si el tablero entero fuera de colores, el avance de la permanencia sería un
  color más.
- **Las celdas de control van más oscuras que el vocabulario.** BORRAR, ESPACIO
  y cambiar de página son herramientas; lo que la persona quiere decir va
  primero. Antes iban más claras, o sea exactamente al revés.
- **HABLAR es la excepción** (`TILE_COMMIT`): es la celda que emite, y hundirla
  junto a BORRAR la volvía lo más difícil de encontrar siendo lo más buscado.

## Las dos señales de la permanencia

La celda bajo la mirada muestra el avance de **dos** formas a la vez: el borde
se completa en sentido horario desde arriba a la izquierda (cuánto falta) y el
relleno se aclara hacia el acento (que el sistema te está siguiendo a vos).
Mirando el centro de la celda -que es lo que uno hace- el borde se percibe
apenas por el rabillo del ojo; con solo el relleno no se sabe cuánto queda.

## Un solo tamaño de letra por página

Ajustar cada etiqueta a su celda dejaba «SÍ» gigante al lado de «MEDICINA»
diminuta. No es solo feo: con tamaños desparejos el ojo salta a la palabra más
grande en vez de recorrer la grilla, y acá la grilla se recorre con la mirada.
Manda la etiqueta más larga (`ui.uniform_text_size`).

## Lo que las figuras NO muestran

Son renders estáticos a 1920x1080. No dicen nada sobre el temblor del cursor,
la latencia ni la tasa de misclick, que es donde se juega si el tablero sirve.
Para eso están `main.py accuracy` y los CSV de sesión.
