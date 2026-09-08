# Expansión y enriquecimiento semántico (CAA / ELA)

Tercer módulo del proyecto. Los otros dos —[Eyetrax](../Eyetrax) y
[RealsenseV1](../RealsenseV1)— resuelven *dónde mira* el usuario. Este resuelve
qué hacer con lo que escribió: convertir `AGUA NO` en algo que una persona
pueda decir en voz alta.

Modelos **pequeños, abiertos y locales** (SLM, ≤ 4B parámetros), por
prompting. Sin ajuste fino todavía.

## La decisión de diseño, primero

El contrato obvio sería «telegráfico entra, una frase fluida sale». **Acá está
mal**, y vale explicar por qué antes que cualquier métrica.

A ~800 ms por permanencia, el usuario pagó caro cada palabra. Si el modelo
convierte `AGUA NO` en «no quiero agua, gracias» pero quiso decir «no hay
agua», el dispositivo **habló una falsedad en su nombre**, y corregirla cuesta
otras diez permanencias. Para alguien cuyo único canal de salida es este
tablero, eso no es una imprecisión de UX.

Así que el contrato es otro:

> El módulo propone **N candidatos**; el usuario elige con la misma
> permanencia; y **el texto literal es siempre una de las opciones**.

De ahí sale todo lo demás. La métrica no es «¿acertó la frase?» sino
**recall@N** —¿aparece la frase buscada entre las opciones?— más una puerta
dura de no-invención. N = 3, porque cada candidato extra es otra celda que hay
que leer, y leer no es gratis cuando la mirada es también el puntero.

## Por qué no BLEU

La métrica obvia para «telegráfico → frase» es solapamiento contra una
referencia. Es el instrumento equivocado, por tres razones:

- **No hay una referencia única.** `AGUA` es legítimamente «quiero agua», «¿me
  das agua?» o «tengo sed». Puntuar contra una castiga a las otras dos por ser
  distintas, no por estar mal.
- **La salida es un conjunto, no una cadena.** Importa si la frase buscada está
  *entre* los candidatos, no cuán cerca quedó el primero.
- **Los fallos peligrosos son invisibles al solapamiento.** «no quiero agua» y
  «quiero agua» difieren en un token corto y puntúan casi igual bajo cualquier
  métrica de n-gramas. Son mensajes opuestos. Mientras tanto «quiero un vaso de
  agua fría» puntúa bien e inventa una temperatura y un recipiente.

Por eso las métricas de [metrics.py](aac_semantic/metrics.py) son **asimétricas
a propósito**: la fluidez se mide flojo, la invención y la polaridad se miden
duro. Un modelo rígido pero fiel es usable. Uno fluido e infiel es peor que el
tablero que reemplaza — el tablero al menos solo decía lo que se apretaba.

| métrica | qué mide | umbral |
| --- | --- | --- |
| `recall@3` | la frase buscada aparece entre los 3 candidatos | ≥ 80 % |
| `invención` | dijo contenido que la entrada no dice | **0 %** |
| `polaridad` | perdió la negación o la orden de detenerse | **0 %** |
| `suavizado` | diluyó un mensaje urgente o un dolor | **0 %** |
| `literal` | ofreció la lectura literal como salida de escape | ≥ 90 % |
| `cobertura de lecturas` | en entradas ambiguas, cuántas lecturas distintas cubre | — |
| `p95` de latencia | | ≤ 4 s |

Los ceros son literales, no retóricos: un dispositivo que fabrica un síntoma
una vez cada veinte lo va a hacer en la única conversación que importa.

## El corpus

40 ítems en [data/telegraphic_es.jsonl](data/telegraphic_es.jsonl), construidos
sobre el **vocabulario real** del tablero (las 20 palabras de
[board_layout.py](../RealsenseV1/aac_realsense/board_layout.py)) más deletreo
libre. Vocabulario cerrado y chico, así que las entradas se pueden enumerar en
vez de inventar.

Cada ítem trae:

- `readings` — las lecturas válidas, cada una con varias frases aceptables.
- `forbidden_content` — lo que **no** debe decir. Escrito a mano, por ítem.
- `must_keep_polarity`, `urgency` — las puertas duras.
- `expect_refusal` — entradas que **no deben expandirse** (una sola letra, un
  verbo sin objeto, vacío).

Composición: 7 ambiguos, 12 urgentes, 3 que no deben expandirse, 30 con trampa
de invención. `main.py corpus --tag urgente` los muestra.

Los casos que más trabajo dieron son los que deciden si esto sirve:

- **`NO BASTA DOLOR`** — si el modelo lo suaviza, alguien sigue haciendo algo
  que causa dolor.
- **`MEDICINA DOLOR`** — dos lecturas que piden acciones **opuestas**: dar el
  remedio, o suspenderlo porque lo causa.
- **`VENTANA`** — tres lecturas, dos de ellas acciones contrarias (abrir /
  cerrar). El caso donde ofrecer candidatos gana claramente a adivinar.
- **`QUIERO HIJA`** — `forbidden_content` incluye «despedirme», una inferencia
  devastadora y completamente inventada.
- **`A`** y **`QUIERO`** — completar la palabra que falta es adivinar; debe
  quedarse literal.

## Las variantes de prompt

Seis, en [prompts.py](aac_semantic/prompts.py), **anidadas**: cada una agrega
exactamente una cosa a la anterior. Eso es lo que hace interpretable la tabla
de ablación — si v4 gana, se sabe *cuál* de los cuatro agregados lo compró.

| variante | agrega |
| --- | --- |
| `v0-minimo` | nada: instrucción desnuda. El piso. |
| `v1-reglas` | las 7 reglas (no inventar, polaridad, urgencia, primera persona) |
| `v2-ejemplos` | 5 ejemplos, elegidos sobre los casos difíciles |
| `v3-json` | contrato JSON + decodificación restringida |
| `v4-literal` | obliga a incluir la lectura literal |
| `v5-lecturas` | primero enumera lecturas distintas, después expande una por lectura |

`v3` existe para separar dos fallos que un formato por líneas confunde: un
modelo que entendió la tarea pero contestó en prosa, y uno que la entendió mal.

## Por qué SLM y no LLM

Todos los modelos acá tienen **4B parámetros o menos**. No es una restricción de
presupuesto: son tres requisitos del dispositivo empujando en la misma
dirección.

- **Sin internet.** Un aparato que dice «me duele» no puede depender de una
  llamada de red. Si se cae el wifi, el usuario se queda sin voz.
- **Latencia.** Cada permanencia cuesta ~800 ms del esfuerzo del usuario. Un
  modelo que tarda ocho segundos en proponer candidatos gasta más de su tiempo
  que el que costó escribir.
- **Privacidad.** El contenido es médico y lo dice alguien que quizá no pueda
  volver a dar consentimiento después. No debería salir de la habitación.

Un SLM que corre en 1 GB de RAM y responde en 300 ms es una respuesta mejor a
este problema que un modelo de 70B en la nube, incluso si el de 70B redacta
mejor. La pregunta del experimento no es cuál escribe más lindo, es **cuál es el
más chico que todavía es fiel**.

## Los modelos

| modelo | params | tamaño | licencia | por qué está |
| --- | --- | --- | --- | --- |
| `qwen2.5:0.5b` | 0.5B | 0.4 GB | Apache-2.0 | el piso: ¿puede algo tan chico? Si sí, corre en cualquier cosa |
| `qwen2.5:1.5b` | 1.5B | 1.0 GB | Apache-2.0 | el candidato realista: licencia limpia, entra en RAM sin GPU |
| `qwen2.5:3b` | 3B | 1.9 GB | Qwen Research | techo de la familia |
| `llama3.2:1b` | 1B | 1.3 GB | Llama 3.2 | segunda familia; Llama 3.2 declara español oficialmente soportado |
| `llama3.2:3b` | 3B | 2.0 GB | Llama 3.2 | el par de `qwen2.5:3b`: mismo tamaño, otra receta |
| `gemma2:2b` | 2B | 1.6 GB | Gemma Terms | tercera familia, en el medio del rango |
| `phi3.5` | 3.8B | 2.2 GB | MIT | el más grande que sigue siendo SLM, y la licencia más limpia para distribuir |
| `smollm2:1.7b` | 1.7B | 1.8 GB | Apache-2.0 | control: entrenado sobre todo en inglés |

Dos elecciones deliberadas en esa lista:

**La familia Qwen aparece en tres tamaños.** Misma receta de entrenamiento, solo
cambia el número de parámetros — es la única forma limpia de preguntar «¿importa
el tamaño *para esta tarea*?» en vez de «¿es el modelo A mejor que el B?».

**`smollm2` está como control negativo.** Está entrenado casi todo en inglés. Si
va notablemente peor que un Qwen del mismo tamaño, el resultado no es «smollm2
es malo» sino **«el español en el entrenamiento pesa más que los parámetros»**,
que es una conclusión mucho más útil para elegir el modelo del dispositivo.

## Estado

**El arnés está completo y validado. Todavía no corrió ningún modelo real.**

```powershell
.\run.cmd smoke      # 14/14
```

Las 14 pruebas no necesitan modelo y validan **el instrumento**, que es lo que
hay que validar primero: una métrica que no detecta una invención no reporta un
bug, reporta un buen modelo — el peor fallo posible en un arnés de evaluación, y
uno que podría sobrevivir una tesis entera sin que nadie lo note.

Cada prueba mete un conjunto de candidatos **conocidamente malo** por el
puntuador real y exige que lo atrape: una temperatura inventada, una negación
volteada, un dolor suavizado, una letra completada a una palabra. Más el corpus:
las 87 trampas se disparan al inyectarlas, y las 60 referencias se reconocen a
sí mismas.

### Dos bugs del instrumento que encontraron esas pruebas

Vale registrarlos porque los dos habrían inflado los resultados de todos los
modelos por igual, que es la clase de error más difícil de notar.

1. **El chequeo de invención usaba `in` a secas.** `"quiero agua"` está prohibido
   en el ítem `AGUA NO`, y es subcadena de `"no quiero agua"` — que es **la
   respuesta correcta**. La marcaba como invención. Arreglado moviendo los
   volteos de polaridad a `must_keep_polarity`, que es el chequeo que entiende
   la negación, y comparando por límites de palabra y por candidato en vez de
   sobre el texto concatenado.

2. **La cobertura de lecturas contaba de más.** `"agua no"` puntúa 0.80 contra
   `"no hay agua"`, porque `hay` —justo la palabra que distingue «no quiero
   agua» de «no hay agua»— es uno de tres tokens cortos. Con umbral
   independiente por lectura, tres parafraseos de una lectura decían cubrir dos.
   Arreglado asignando cada candidato a la lectura que mejor le calza y contando
   lecturas distintas.

Y una limitación que quedó **documentada y fijada con un test** en vez de
arreglada: la diversidad léxica no mide diversidad semántica. Estos dos
conjuntos puntúan idéntico (0.24):

```
["no quiero agua", "agua no", "no, agua no"]   -> tres formas de una lectura
["no quiero agua", "no hay agua", "agua no"]   -> dos lecturas opuestas
```

`reading_coverage` es la métrica que responde la pregunta semántica. La
diversidad se reporta al lado, solo para detectar repetición literal, y hay un
test que falla si alguien empieza a usarla como proxy de cobertura.

### Falta

Instalar Ollama y correr la grilla. Nada más:

```powershell
# 1. Ollama (~700 MB)  ->  https://ollama.com/download
#    o la versión portable, sin instalador ni permisos de admin:
#    tools/ollama/ollama.exe serve
.\run.cmd models        # qué hay y qué falta
.\run.cmd pull          # los comandos de descarga (~12 GB todos, 3.3 GB los tres mínimos)

# 2. Mirar una entrada suelta antes de gastar la grilla
.\run.cmd try --model qwen2.5:7b --input "AGUA NO"

# 3. La grilla: 6 modelos x 6 variantes x 40 ítems
.\run.cmd run --limit 10          # prueba corta primero
.\run.cmd run                     # completa
```

## Reejecutar sin volver a inferir

`run` guarda **cada respuesta cruda** en `results/run-*.jsonl`. Así:

```powershell
.\run.cmd rescore results/run-20260825-190000.jsonl
```

re-mide con las métricas actuales sin llamar a ningún modelo. Cuando una métrica
resulte estar mal —y una va a estar mal, ya pasó dos veces— el arreglo se
verifica contra la corrida de ayer en un segundo, en vez de costar otros veinte
minutos de grilla.

Esa decisión es deuda pagada: el módulo de mirada descartaba sus muestras de
calibración después de ajustar, y la primera calibración real en hardware
terminó con todos los parámetros del montaje clavados en sus límites — el paso
siguiente obvio era reajustar con límites más anchos, y era imposible porque los
datos ya no estaban.

## Distribución de archivos

| archivo | rol |
| --- | --- |
| [main.py](main.py) | CLI |
| [data/telegraphic_es.jsonl](data/telegraphic_es.jsonl) | el corpus, editable a mano |
| [backends.py](aac_semantic/backends.py) | Ollama + doble de prueba |
| [prompts.py](aac_semantic/prompts.py) | las 6 variantes y el parser tolerante |
| [metrics.py](aac_semantic/metrics.py) | recall@N, invención, polaridad, suavizado, cobertura |
| [runner.py](aac_semantic/runner.py) | la grilla, y el guardado que hace barato re-medir |
| [report.py](aac_semantic/report.py) | tabla, ablación, fallos verbatim, deriva léxica |
| [smoke.py](aac_semantic/smoke.py) | los 14 chequeos del instrumento |

## Notas para el proyecto

- **El corpus lo escribí yo, y eso es una limitación real.** Las referencias
  aceptables y las trampas son mi criterio sobre qué es fiel, no el de un
  usuario de CAA ni el de un fonoaudiólogo. Antes de que estos números
  sustenten una conclusión, alguien con experiencia clínica tiene que revisar
  `readings` y `forbidden_content`. La `deriva léxica` del reporte está pensada
  para eso: lista las palabras que los modelos introdujeron y que nadie prohibió
  todavía.
- **Sin usuario en el lazo, `recall@3` es un techo, no una medida de utilidad.**
  Que la frase correcta esté entre las tres no dice que el usuario la encuentre
  rápido ni que la prefiera. Eso necesita una sesión con alguien usando el
  tablero, y hasta entonces `recall@3` mide el modelo, no el sistema.
- **Falta el contexto, y es la mejora más grande disponible.** Ahora mismo cada
  entrada se expande aislada. La hora del día, la última frase dicha y quién
  está en la habitación resolverían buena parte de la ambigüedad que hoy se
  paga con candidatos: `MEDICINA DOLOR` a las 3 de la mañana es casi
  seguramente un pedido, no un reporte de efecto adverso.
- **La temperatura está en 0.2 y la semilla fija.** No es escritura creativa: la
  misma entrada tiene que dar los mismos candidatos hoy y la semana que viene, o
  el usuario no puede construir memoria muscular y el experimento no se puede
  replicar.
- **Lo que sigue después del prompting**, en orden de rendimiento esperado:
  contexto (arriba), después ajuste fino con LoRA sobre el corpus una vez
  revisado clínicamente, y solo después modelos más grandes. El orden importa:
  un modelo más grande con el mismo prompt sin contexto sigue sin saber si son
  las 3 de la mañana.
