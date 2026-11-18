# Expansión semántica: qué modelo usa la página, y por qué

## La respuesta corta

**Qwen3-4B, con el prompt en español neutro (`v12-neutro`), a temperatura
0.6, con el filtro determinista de `web/js/expansion.js`.**

Medido con la página real (WebLLM en Chrome con WebGPU) sobre las 133
entradas del corpus, repetidas entre 3 y 5 veces, contra lo que la página
corría hasta ahora (Qwen2.5-1.5B, prompt en voseo, temperatura 0.9, **sin
filtro**):

| | la página hasta ahora | la elegida |
|---|---|---|
| la frase buscada está entre las opciones | 60 % | **76 %** |
| … sin contar el literal, que la página agrega siempre | 27 % | **55 %** |
| respuestas inseguras que llegan a la persona | **20 %** | **1 %** |
| entradas que fueron inseguras alguna vez en 5 intentos | 26 % | **0.8 %** |
| misma entrada, mismas opciones (constancia, 0 a 1) | 0.37 | **0.79** |
| el modelo ofrece por sí mismo la lectura literal | 36 % | **83 %** |
| latencia mediana en una notebook con RTX 4060 | ~3 s | **~4.7 s** |
| descarga, una sola vez | 0.87 GB | **2.26 GB** |

Lo que cuesta, dicho sin rodeos: casi dos segundos más por expansión y 1.4 GB
más de descarga. Y **sigue sin pasar la puerta de aptitud que el proyecto se
fijó**: el acierto queda en 76 % (pide 80 %) y el p95 de latencia en unos
6.5 s (pide 4 s). Es el mejor disponible, no uno apto.

El resto del documento explica de dónde sale cada número, qué se encontró en
el camino, y por qué ninguna de las alternativas gana.

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

## Lo que se encontró antes de poder medir

Cinco problemas, todos de la página o del instrumento y no de un modelo, que
habrían invalidado cualquier comparación hecha encima:

1. **La página no tenía el filtro de seguridad.** Todas las tablas anteriores
   se midieron con el filtro determinista de `slm/aac_semantic/guard.py`
   puesto; `web/js/slm.js` mostraba la salida cruda del modelo. Sobre las 5 509
   respuestas reales guardadas de quince modelos, el 21 % traía al menos un
   candidato que el filtro descarta. El modo grabado de la página mostraba 11
   de 132 entradas así: `SI NO` → "no" resolvía la duda por la persona,
   `GRACIAS BASTA` → "gracias" perdía la orden de parar, `PIES FRIOS` → "me
   duele un pie" inventaba dolor. Se portó el filtro línea por línea a
   `web/js/expansion.js`, y una prueba le exige decidir **exactamente** lo mismo
   que el de Python en los 5 509 casos reales y en casos construidos para cada
   borde. Coinciden todos.
2. **La página nunca cargaba el modelo.** `SLM.cargar` estaba definido y nadie
   lo llamaba, desde el primer commit. Todo lo que propuso la página salió de
   132 respuestas grabadas; cualquier otra frase mostraba solo el literal.
3. **Lo medido no era lo que corre la página.** La grilla anterior usaba
   Ollama, temperatura 0.2 con semilla fija y JSON forzado por el
   decodificador. La página usa WebLLM, otra cuantización (q4f16_1),
   temperatura 0.9 y texto libre. Por eso esta medición se hizo con la página
   misma: `slm/banco_web/` la abre en Chrome sin ventana, con WebGPU, y usa la
   clase `SLM` de la página, no una copia.
4. **El prompt estaba en voseo**, ejemplos incluidos ("prendé la luz"), y 56
   de las 455 frases aceptadas del corpus también. Una respuesta neutra
   correcta podía contar como fallo. Se agregó una variante idéntica salvo el
   registro (`v12-neutro`; el smoke verifica que deshacer sus sustituciones
   devuelva `v4-literal` exacto) y se completaron las referencias con su
   equivalente con tú, sin quitar ninguna.
5. **La métrica de acierto tenía tres huecos**, encontrados mirando
   respuestas una por una: contaba como acierto "no quiero comer más" para
   `COMIDA MAS` (F1 0.857 contra "quiero comer más"); contaba "necesito ayuda"
   para `NECESITO AYUDA BASTA` aunque perdía la orden de parar; y el filtro
   dejaba pasar "de nada" como orden de detenerse porque "nada" está en la
   lista de negaciones. Los tres se cerraron en la métrica, en el filtro de
   Python y en el de JavaScript a la vez.

Y dos métricas que no existían, porque nadie miraba esos fallos:

- **acierto del modelo**: el acierto sin contar el literal. La página lo
  agrega siempre, y si el corpus acepta la palabra sola, eso ya cuenta como
  acierto aunque el modelo no haya aportado nada. En la página de hoy el
  acierto es 60 % y el del modelo, 27 %: el literal hacía más de la mitad del
  trabajo.
- **invierte un pedido**: decir NO donde nadie lo dijo. `QUIERO COMIDA` → "no
  tengo hambre". Ni la métrica de polaridad ni el filtro miraban este lado.

## Cómo se midió

- **Motor**: WebLLM 0.2.79 en Chrome 157 sin ventana, WebGPU sobre una RTX 4060
  de notebook. La misma versión de WebLLM que importa la página.
- **Código**: la clase `SLM` de `web/js/slm.js` y el filtro de
  `web/js/expansion.js`, sin copias. El banco solo cambia, con las opciones
  que la clase expone, el modelo, la temperatura y el texto del prompt.
- **Corpus**: las 133 entradas de `slm/data/telegraphic_es.jsonl` más 56
  referencias neutras. 27 urgentes, 17 negaciones, 10 órdenes de parar, 3
  dudas, 6 que no deben expandirse, 76 con palabras deletreadas.
- **Repeticiones**: 5 pasadas del corpus para la página de hoy, el mismo
  modelo con el prompt neutro y los dos Qwen3 entregables; 3 para cada
  temperatura y para los modelos de contexto. Unas 6 000 respuestas en total.
- **Estadística**: diferencias pareadas entrada por entrada, con intervalos
  del 95 % por remuestreo de entradas (4 000 réplicas). Las k respuestas de
  una entrada no son independientes, y remuestrear respuestas sueltas daría
  intervalos falsamente angostos.
- **Reproceso**: el banco guarda el texto crudo de cada respuesta, y el
  análisis lo vuelve a pasar por el `expansion.js` actual, así una corrección
  del filtro se re-mide sin volver a correr modelos. Lo reprocesado coincide
  con lo que calculó la página en el momento en todas las respuestas
  posteriores al cierre de "de nada".

Las tablas completas, generadas, están en
[expansion_semantica_tablas.md](expansion_semantica_tablas.md).

## Prompt y modelo, por separado

Cambiar a la vez el prompt y el modelo no dice cuál de los dos compró la
mejora. El cribado incluye el modelo de hoy con el prompt neutro, que separa
los dos efectos:

| comparación (diferencia pareada) | acierto | acierto del modelo | inseguro |
|---|---|---|---|
| **solo el prompt**: Qwen2.5-1.5B, neutro contra el de hoy | +3 (+0 a +6) | +1 (−3 a +6) | −3 (−5 a −1) |
| **solo el modelo**: Qwen3-1.7B contra Qwen2.5-1.5B, mismo prompt | +5 (+1 a +10) | +7 (+0 a +13) | −4 (−7 a −0) |
| **solo el modelo**: Qwen3-4B contra Qwen2.5-1.5B, mismo prompt | +13 (+8 a +19) | +28 (+21 a +36) | −4 (−6 a −1) |
| **Qwen3-4B contra Qwen3-1.7B**, mismo prompt | +7 (+2 a +12) | +19 (+12 a +26) | −0 (−3 a +2) |

El prompt neutro mejora un poco **por sí solo**, y no por el registro: con
cualquiera de los dos prompts el voseo en las salidas es mínimo (0.15 a
0.30 %), así que la hipótesis de que los modelos copiaban el registro del
prompt no se sostuvo. El salto grande es del modelo.

## La elección, aplicando el criterio

1. **Restricciones duras.** Quedan Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen3-0.6B,
   Qwen3-1.7B y Qwen3-4B. Phi-3.5-mini (no apto para poca memoria) y
   Qwen2.5-3B (licencia no comercial) se miden solo como referencia.
2. **Seguridad no peor que la página de hoy.** Los dos Qwen3 la mejoran:
   −7 puntos de respuestas inseguras con el filtro puesto, con intervalos que
   no tocan el cero.
3. **Mayor acierto del modelo.** Qwen3-4B: 54 % contra 35 % de Qwen3-1.7B, una
   diferencia de +19 (+12 a +26). No es empate, así que la menor descarga no
   entra en juego.

**Qwen3-4B.** Los modelos de contexto, más chicos o fuera por restricción,
no pueden cambiar esta conclusión; sus resultados están más abajo.

### Por tipo de entrada

Acierto / respuestas inseguras, con filtro:

| | urgentes | negaciones | parar | ambiguas | deletreo | postura |
|---|---|---|---|---|---|---|
| página de hoy | 42 % / 6 % | 89 % / 1 % | 88 % / 2 % | 32 % / 3 % | 49 % / 8 % | 8 % / 2 % |
| Qwen3-4B a 0.6 | 60 % / 0 % | 88 % / 0 % | 79 % / 0 % | 56 % / 0 % | 69 % / 0 % | 28 % / 0 % |

Las urgentes pasan de 42 a 60 % de acierto y de 6 a 0 % de respuestas
inseguras, que es lo que más importa. La postura (`MOVER BRAZO`, `SENTAR`)
sigue siendo lo peor de todos los modelos: 28 %. Las órdenes de parar bajan
algo (88 → 79 %), con intervalos que se superponen; el filtro garantiza que
ninguna opción pierda la orden, pero no que haya una opción buena.

## Temperatura

La página usaba 0.9 y eso nunca se justificó. Con Qwen3-4B:

| temp. | acierto | inseguro | cobertura de lecturas | constancia |
|---|---|---|---|---|
| 0.3 | 74 % | 1 % | 42 % | 0.87 |
| **0.6** | **76 %** | **1 %** | **47 %** | **0.79** |
| 0.9 | 75 % | 2 % | 41 % | 0.68 |

Acierto y seguridad no distinguen las tres: todas las diferencias cruzan el
cero. Lo que sí cambia es **constancia** —para una misma entrada, cuánto se
parecen las opciones de una vez a la siguiente; con 1 son siempre las
mismas— y **cobertura de lecturas** —en las entradas ambiguas, qué parte de
las lecturas aceptadas propone—. Bajar la temperatura sube la constancia,
que es lo que permite aprender dónde aparece lo que uno busca; pero a 0.3 la
cobertura baja, y para `VENTANA` lo útil es ofrecer abrir **y** cerrar.

0.6 supera o empata a 0.9 en todo: cobertura +6 (+1 a +13), constancia 0.79
contra 0.68, y acierto y seguridad iguales. Contra 0.3 gana cobertura (+5,
+0 a +12, en el borde) y pierde constancia. **Se eligió 0.6.** Para la
temperatura no se fijó un criterio antes de medir; esta elección es posterior
a ver los datos y se declara así.

## El filtro: lo que gana y lo que cuesta

| | respuestas salvadas | aciertos perdidos | candidatos correctos descartados |
|---|---|---|---|
| página de hoy | 11.1 % | 0 | 0 |
| Qwen3-1.7B | 10.2 % | 1.1 % | 10 |
| Qwen3-4B a 0.9 | 3.3 % | 2.0 % | 14 |

Con el modelo de hoy el filtro es gratis: salva una de cada nueve respuestas
y no le quita ningún acierto. Con Qwen3-4B salva menos —el modelo ya se
equivoca menos solo— y cuesta dos puntos. Los aciertos que quita son casi
todos de una misma regla, la que descarta "dolor" cuando la entrada no lo
dice: `PIE MAL` → "me duele el pie", `QUEMA PIERNA` → "me duele la pierna".
El corpus acepta esas frases; el filtro no.

Es un desacuerdo de criterio, y se resolvió a favor del filtro. "Pie mal" no
dice dolor, aunque sea la lectura más probable; hacer que una parte del
cuerpo más MAL autorice dolor dejaría pasar `ME SIENTO MAL` → "me duele", que
sí es una invención, y la persona conserva siempre el literal "pie mal". Pero
es un costo real y queda medido.

## La regla de inversión

*Pendiente: se decide con los siete modelos completos, según el criterio
fijado arriba.*

## Robustez: la entrada como llega de verdad

*Pendiente: corriendo.*

## Latencia

*Pendiente: corriendo, con un solo navegador.*

## Los modelos de contexto

*Pendiente: corriendo.*

## Lo que sigue sin cumplirse

La puerta de aptitud del proyecto, para Qwen3-4B a 0.6 en la página:

| requisito | pide | da |
|---|---|---|
| acierto | ≥ 80 % | **76 %** (69–83) |
| invención | 0 % | 0.0 % en esta celda; 0.5–1.2 % a otras temperaturas |
| vuelcos de polaridad | 0 % | 0 %, por construcción del filtro |
| literal disponible | ≥ 90 % | 100 % (la página lo agrega siempre) |
| hay elección | ≥ 90 % | 95 % |
| p95 de latencia | ≤ 4 s | **~6.5 s** en esta máquina |

No pasa. Lo más lejos está la latencia, que no se arregla con prompt: es el
precio del modelo de 4B en WebGPU. Y la postura (28 %) es el punto más débil
del acierto, en todos los modelos medidos.

## Cómo reproducirlo

```
cd slm/banco_web && npm install
PARALELO=1 node correr.mjs cribado            # unas 6 h en esta máquina
FINALISTAS=Q3_4 PARALELO=1 node correr.mjs temperatura
FINALISTAS=Q3_4 PARALELO=1 node correr.mjs robustez
NAVEGADORES=1 PARALELO=1 node correr.mjs latencia
cd .. && python informe_web.py               # regenera las tablas
```

Dos advertencias medidas: con dos pestañas a la vez y un modelo de 4B, la
GPU de 8 GB se llena y cada pedido pasa de 4.5 a 16 s; y Chrome guarda los
modelos por origen, así que el puerto del banco es fijo (8740) o vuelve a
bajar todo en cada corrida.
