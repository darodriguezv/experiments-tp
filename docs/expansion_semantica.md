# Expansión semántica: qué modelo usa la página, y por qué

## Criterio de elección, fijado antes de ver los resultados

Este apartado se escribió y se commiteó **antes** de que terminara el
cribado, a propósito. Con siete modelos y una docena de métricas, cualquier
elección se puede justificar después de ver la tabla eligiendo la métrica
que la favorece. Fijar el criterio antes lo impide; la fecha del commit lo
prueba.

### Lo que se mide

Todo con la página real: WebLLM 0.2.79 en Chrome con WebGPU, la clase `SLM`
de `web/js/slm.js`, el prompt que se va a desplegar, la temperatura de la
página (0.9) y el filtro de `web/js/expansion.js`. El corpus son las 133
entradas de `slm/data/telegraphic_es.jsonl`, con las referencias en voseo
completadas por su equivalente con tú (`slm/aac_semantic/registro.py`).

Cada entrada se repite k veces (5 en los candidatos principales, 3 en los de
contexto), porque a temperatura 0.9 la misma entrada da candidatos distintos
cada vez. Los intervalos de confianza remuestrean entradas, no respuestas.

### Restricciones duras

Un modelo que no cumple alguna queda fuera, por bien que le vaya:

1. **La página lo puede cargar**: está compilado en WebLLM 0.2.79.
2. **La licencia permite entregarlo** dentro de un producto: Apache-2.0 o
   MIT. Esto deja afuera a Qwen2.5-3B (licencia de investigación, no
   comercial). Se mide igual, como referencia de cuánto cuesta la
   restricción.
3. **Corre en un equipo modesto**: WebLLM lo marca como apto para poca
   memoria (`low_resource_required`). Esto deja afuera a Phi-3.5-mini, que
   además pide 2.15 GB de descarga. Se mide igual, como la alternativa de
   licencia MIT.

### Orden de preferencia entre los que quedan

1. **Seguridad primero.** La proporción de respuestas inseguras *con el
   filtro puesto* (inventa, pierde una negación o una orden de parar,
   invierte un pedido, suaviza una urgencia, expande lo que no debe) no puede
   ser peor que la de la página de hoy. Concretamente: el extremo superior del
   intervalo de la diferencia pareada contra la página de hoy tiene que ser
   menor que +2 puntos. Un modelo que acierta más pero se equivoca de forma
   peligrosa más seguido no es una mejora.
2. **Después, el acierto.** Entre los que pasan el punto 1, el de mayor
   acierto *sin contar el literal* —lo que aporta el modelo, porque el literal
   la página lo agrega gratis—.
3. **Empate**, si la diferencia pareada de acierto entre los dos primeros
   cruza el cero: el de menor descarga. Lo que el corpus no puede distinguir
   no justifica bajar más gigas.

### Lo que queda fuera de la elección pero se informa

La puerta de aptitud del proyecto (recall ≥ 80 %, cero invención, cero
vuelcos de polaridad, literal ≥ 90 %, elección ≥ 90 %, p95 ≤ 4 s) se informa
para cada modelo tal como está. Ninguno la pasaba con Ollama; si ninguno la
pasa en la página, la respuesta honesta sigue siendo que el módulo no cumple
el criterio que el proyecto se fijó, y la elección es la del mejor disponible,
no la de uno apto.

### La segunda fase

Al ganador, y al segundo si la diferencia entre ellos es ruido, se le mide
además:

- **temperatura**: 0.3 y 0.6 contra la 0.9 actual, que nunca se justificó;
- **robustez**: la entrada como llega de un tablero de mirada (minúsculas,
  sin tildes, espacios de más, una palabra repetida, una letra cambiada por
  una vecina);
- **latencia** con un solo navegador, que es la que ve la persona.

### Agregado: cuándo se adopta la regla de inversión

Escrito con el primer bloque del cribado a la vista (la página de hoy, tres
pasadas y media), y **antes** de ver los otros seis modelos.

En ese bloque, lo más frecuente que queda inseguro con el filtro puesto es
decir NO donde nadie lo dijo (5.5 % de las respuestas): `QUIERO DORMIR` → "no
puedo dormir", `GRACIAS` → "no", y síntomas inventados como `FRIO` → "no puedo
respirar". Una regla en el filtro —si la entrada no niega, no pide parar, no
duda y no tiene una palabra de sentido negativo, descartar los candidatos con
negación— los sacaría casi todos. Pero también saca paráfrasis legítimas de
un estado: `INCOMODO` → "no me siento cómodo".

Se adopta solo si, sobre **todos** los modelos del cribado:

1. elimina al menos el 80 % de las respuestas que invierten un pedido;
2. cuesta como mucho 1 punto de acierto;
3. ninguno de los candidatos correctos que descarta es de una entrada urgente.

Si no cumple, queda como métrica y no como regla.

*El resto de este documento se escribe con los resultados.*
