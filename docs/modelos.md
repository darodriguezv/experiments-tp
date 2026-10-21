# Selección del modelo de lenguaje

*Generado el 2026-10-21 con `slm/informe_modelos.py` desde `run-20261021-084846.jsonl`. No editar a mano: regenerar.*

15 modelos, 133 entradas del corpus, 2 variantes de prompt por modelo (`v4-literal`, `v6-glosario`), 3990 respuestas en total.

Cada modelo compite con la variante de prompt que mejor le va. Fijar una sola redacción para todos mediría qué familia le sienta mejor a esa redacción, no qué familia es mejor.

## La tabla completa

| modelo | prompt | recall@3 | literal | invención | polaridad | diversidad | p50 modelo | disco | licencia | entregable |
|---|---|---|---|---|---|---|---|---|---|---|
| `llama3.1:8b` | v4-literal | **80%** | 58% | 3% | 6% | 0.47 | 0.96 s | 4.6 GB | Llama 3.1 | **no** |
| `qwen2.5:3b` | v4-literal | **77%** | 52% | 2% | 16% | 0.57 | 0.39 s | 1.8 GB | Qwen Research | **no** |
| `qwen2.5:7b` | v4-literal | **77%** | 81% | 1% | 9% | 0.50 | 0.87 s | 4.4 GB | Apache-2.0 | sí |
| `gemma2:2b` | v6-glosario | **74%** | 63% | 1% | 9% | 0.53 | 0.65 s | 1.5 GB | Gemma Terms | **no** |
| `qwen2.5:3b-instruct-q8_0` | v4-literal | **74%** | 48% | 2% | 17% | 0.61 | 0.65 s | 3.1 GB | Qwen Research | **no** |
| `mistral:7b` | v4-literal | **72%** | 38% | 2% | 7% | 0.50 | 1.04 s | 4.1 GB | Apache-2.0 | sí |
| `llama3.2:3b` | v4-literal | **67%** | 55% | 2% | 8% | 0.45 | 0.47 s | 1.9 GB | Llama 3.2 | **no** |
| `qwen2.5:1.5b` | v6-glosario | **63%** | 43% | 3% | 5% | 0.49 | 0.23 s | 0.9 GB | Apache-2.0 | sí |
| `qwen2.5:1.5b-instruct-fp16` | v6-glosario | **62%** | 39% | 5% | 6% | 0.48 | 0.54 s | 2.9 GB | Apache-2.0 | sí |
| `qwen2.5:1.5b-instruct-q8_0` | v4-literal | **61%** | 49% | 5% | 8% | 0.48 | 0.26 s | 1.5 GB | Apache-2.0 | sí |
| `phi3.5` | v6-glosario | **61%** | 29% | 5% | 11% | 0.65 | 0.59 s | 2.0 GB | MIT | sí |
| `llama3.2:1b` | v6-glosario | **49%** | 48% | 2% | 8% | 0.56 | 0.31 s | 1.2 GB | Llama 3.2 | **no** |
| `qwen2.5:0.5b` | v6-glosario | **45%** | 44% | 1% | 8% | 0.25 | 0.11 s | 0.4 GB | Apache-2.0 | sí |
| `qwen2.5:0.5b-instruct-fp16` | v6-glosario | **45%** | 42% | 2% | 11% | 0.36 | 0.17 s | 0.9 GB | Apache-2.0 | sí |
| `smollm2:1.7b` | v4-literal | **42%** | 37% | 4% | 6% | 0.51 | 0.49 s | 1.7 GB | Apache-2.0 | sí |

**Qué mide cada columna**

- **recall@3**: la frase buscada aparece entre los 3 candidatos. Se cuenta por F1 de tokens ≥ 0.75 contra las referencias escritas a mano, no por coincidencia exacta: *quiero agua por favor* tiene que contar como acierto de *quiero agua, por favor*. A 0.5, *no quiero agua* empezaría a matchear *quiero agua*, que es la única confusión que este módulo no puede cometer.
- **literal**: ofreció la lectura literal como salida de escape. Si las tres propuestas están mal, la persona tiene que poder decir exactamente lo que escribió — el tablero pelado ya podía.
- **invención**: dijo algo que la entrada no dice. Debe ser 0%.
- **polaridad**: perdió una negación o una orden de detenerse. El filtro determinista lo lleva a 0% por construcción; esta columna es **antes** del filtro, o sea la tendencia cruda del modelo.
- **diversidad**: 0 = tres parafraseos de lo mismo, 1 = tres cosas distintas.

## La puerta de aptitud

Un modelo es admisible solo si cumple **todas**: cero invención, cero vuelcos de polaridad, cero suavizado de urgencia, literal presente ≥ 90%, recall@3 ≥ 80%, al menos dos candidatos el 90% de las veces, y p95 del modelo ≤ 4 s.

**Ninguno pasa.** No es un empate ni un detalle de calibración: mientras no la pase nadie, la respuesta honesta a *¿qué modelo usamos?* es *ninguno cumple todavía el criterio que fijamos nosotros*, y lo que hay que mover es el prompt o el corpus, no el umbral.

Los más cerca, y **en qué** fallan. Reportar solo la distancia al recall sería engañoso: un modelo puede llegar al 80% y quedar afuera por inventar, que es una falla de otra naturaleza y no se compensa con recall.

- `llama3.1:8b` (80% de recall): inventa 3% (tiene que ser 0%); vuelca polaridad 6% antes del filtro; ofrece el literal solo 58% (hace falta 90%)
- `qwen2.5:3b` (77% de recall): inventa 2% (tiene que ser 0%); vuelca polaridad 16% antes del filtro; ofrece el literal solo 52% (hace falta 90%); recall 77% (faltan 3%)
- `qwen2.5:7b` (77% de recall): inventa 1% (tiene que ser 0%); vuelca polaridad 9% antes del filtro; ofrece el literal solo 81% (hace falta 90%); recall 77% (faltan 3%)

## ¿Por qué Qwen y no otra familia?

Solo tiene sentido a tamaño igualado.

| comparación | recall@3 | Δ | veredicto | qué controla |
|---|---|---|---|---|
| `qwen2.5:3b` vs `llama3.2:3b` | 77% vs 67% | -10% | ruido (z=-1.78) | 3B contra 3B, misma talla y otra receta |
| `qwen2.5:1.5b` vs `llama3.2:1b` | 63% vs 49% | -14% | real* (z=-2.37) | el extremo chico de cada familia |
| `qwen2.5:1.5b` vs `gemma2:2b` | 63% vs 74% | +11% | real* (z=+2.00) | Qwen contra Gemma en el rango de ~2B |
| `qwen2.5:3b` vs `phi3.5` | 77% vs 61% | -16% | real (z=-2.82) | contra la licencia más limpia (MIT) |
| `qwen2.5:7b` vs `llama3.1:8b` | 77% vs 80% | +4% | ruido (z=+0.75) | los dos grandes |
| `qwen2.5:7b` vs `mistral:7b` | 77% vs 72% | -5% | ruido (z=-0.84) | Qwen contra Mistral a 7B |
| `qwen2.5:1.5b` vs `smollm2:1.7b` | 63% vs 42% | -21% | real (z=-3.52) | control: smollm2 se entrenó casi solo en inglés |

*ruido* significa que el corpus no alcanza para distinguirlos, **no** que sean iguales. Con 133 entradas hace falta una diferencia de ~12% para que se lea; por debajo de eso la elección se decide por costo y licencia, no por calidad.

## ¿Cuánto cuesta cuantizar?

Mismo modelo y misma receta; lo único que cambia son los bits por peso.

| modelo | precisión | recall@3 | Δ | veredicto |
|---|---|---|---|---|
| 0.5B | q4_K_M → fp16 | 45% vs 45% | +0% | ruido (z=+0.00) |
| 1.5B | q4_K_M → q8_0 | 63% vs 61% | -2% | ruido (z=-0.38) |
| 1.5B | q4_K_M → fp16 | 63% vs 62% | -1% | ruido (z=-0.13) |
| 3B | q4_K_M → q8_0 | 77% vs 74% | -3% | ruido (z=-0.57) |

## Al mismo disco: ¿parámetros o precisión?

La pregunta práctica para un equipo dedicado: con los mismos GB, ¿conviene un modelo más grande y más cuantizado, o uno más chico con más precisión?

| presupuesto | menos params, más bits | más params, menos bits | Δ | veredicto |
|---|---|---|---|---|
| ~0.9 GB | `qwen2.5:0.5b-instruct-fp16` 45% | `qwen2.5:1.5b` 63% | +18% | real (z=+3.00) |
| ~1.5-1.8 GB | `qwen2.5:1.5b-instruct-q8_0` 61% | `qwen2.5:3b` 77% | +16% | real (z=+2.82) |
| ~3 GB | `qwen2.5:1.5b-instruct-fp16` 62% | `qwen2.5:3b-instruct-q8_0` 74% | +11% | real* (z=+1.99) |

## ¿Cuánto compra el tamaño?

Dentro de Qwen, a receta constante: lo único que cambia es la cantidad de parámetros.

| modelo | recall@3 | literal | invención | p50 |
|---|---|---|---|---|
| `qwen2.5:0.5b` | 45% | 44% | 1% | 0.11 s |
| `qwen2.5:1.5b` | 63% | 43% | 3% | 0.23 s |
| `qwen2.5:3b` | 77% | 52% | 2% | 0.39 s |
| `qwen2.5:7b` | 77% | 81% | 1% | 0.87 s |

De `qwen2.5:0.5b` a `qwen2.5:7b`: +32% de recall, **real** (z=+5.58).

Y en la salida de escape literal: 44% → 81%, +37%, **real** (z=+6.72).

## Costo medido en esta máquina

| modelo | params | cuantización | disco | VRAM | caliente p50 | recarga tibia | tok/s |
|---|---|---|---|---|---|---|---|
| `qwen2.5:0.5b` | 494.03M | Q4_K_M | 0.4 GB | 0.4 GB | 0.04 s | 2.1 s | 366 |
| `qwen2.5:1.5b` | 1.5B | Q4_K_M | 0.9 GB | 1.0 GB | 0.11 s | 7.5 s | 181 |
| `qwen2.5:0.5b-instruct-fp16` | 494.03M | F16 | 0.9 GB | 1.0 GB | 0.09 s | 7.9 s | 183 |
| `llama3.2:1b` | 1.2B | Q8_0 | 1.2 GB | 1.3 GB | 0.06 s | 2.8 s | 170 |
| `gemma2:2b` | 2.6B | Q4_0 | 1.5 GB | 1.7 GB | 0.22 s | 4.1 s | 102 |
| `qwen2.5:1.5b-instruct-q8_0` | 1.5B | Q8_0 | 1.5 GB | 1.6 GB | 0.11 s | 9.0 s | 120 |
| `smollm2:1.7b` | 1.7B | Q8_0 | 1.7 GB | 2.1 GB | 0.10 s | 2.7 s | 115 |
| `qwen2.5:3b` | 3.1B | Q4_K_M | 1.8 GB | 1.9 GB | 0.15 s | 8.8 s | 103 |
| `llama3.2:3b` | 3.2B | Q4_K_M | 1.9 GB | 2.2 GB | 0.14 s | 9.7 s | 103 |
| `phi3.5` | 3.8B | Q4_0 | 2.0 GB | 2.8 GB | 0.15 s | 9.8 s | 97 |
| `qwen2.5:1.5b-instruct-fp16` | 1.5B | F16 | 2.9 GB | 3.0 GB | 0.19 s | 10.7 s | 74 |
| `qwen2.5:3b-instruct-q8_0` | 3.1B | Q8_0 | 3.1 GB | 3.2 GB | 0.19 s | 6.2 s | 69 |
| `mistral:7b` | 7.2B | Q4_K_M | 4.1 GB | 4.4 GB | 0.29 s | 7.4 s | 54 |
| `qwen2.5:7b` | 7.6B | Q4_K_M | 4.4 GB | 4.3 GB | 0.26 s | 11.5 s | 53 |
| `llama3.1:8b` | 8.0B | Q4_K_M | 4.6 GB | 4.7 GB | 0.26 s | 14.9 s | 51 |

**Sobre la recarga.** Es *tibia*: el modelo salió de RAM pero su archivo sigue en la caché de páginas del sistema. Una primera lectura real desde disco es varias veces peor — se midió 4.3 s contra 39 s en el mismo modelo — y no se puede automatizar sin vaciar la caché del sistema operativo. Para el dispositivo importa la tibia, porque el equipo queda encendido; la fría solo aparece tras reiniciar.

## Qué se concluye

1. **Cuantizar sale gratis.** Las 4 comparaciones de precisión a modelo fijo difieren como mucho 3% y ninguna es legible con este corpus. Para esta tarea, Q4_K_M no cuesta calidad: la cuantización pasa a ser una decisión de costo pura y no una concesión.

2. **Al mismo disco conviene gastar en parámetros, no en bits.** En 3 de 3 presupuestos el modelo más grande y más cuantizado le gana al más chico y más preciso. La intuición contraria -que un modelo chico sin cuantizar rinda más que uno grande cuantizado- no tiene respaldo acá.

3. **Contra `llama3.2:3b`, `llama3.1:8b`, `mistral:7b` no hay diferencia demostrable.** A tamaño igualado el corpus no los distingue de Qwen, así que la elección entre ellos se decide por licencia y costo, no por calidad. Afirmar superioridad técnica sería ir más allá del dato.

4. **Qwen sí les gana a `llama3.2:1b` (-14%), `phi3.5` (-16%), `smollm2:1.7b` (-21%)**, con diferencias por encima del umbral de legibilidad del corpus.

5. **Y pierde contra `gemma2:2b` (+11%).** No se puede omitir: es un resultado que desordena la narrativa pero está en el dato.

6. **El español en el entrenamiento pesa más que el tamaño.** `smollm2:1.7b`, entrenado casi solo en inglés y **más grande** en disco que `qwen2.5:1.5b`, saca 42% contra 63%. Es la diferencia más limpia de la tabla y explica *por qué* funciona lo que funciona, que es más de lo que da cualquier ranking.

7. **Entre el 3b y el 7b lo que cambia no es el recall sino la salida de escape.** El recall es indistinguible, pero el literal pasa de 52% a 81% (real, z=+5.33). Con el 3b, una de cada dos veces la persona no recibe la opción de decir exactamente lo que escribió. Es la diferencia que se nota usándolo y la que un ranking por recall esconde.

## Límites de esta medición

- **El corpus tiene 133 entradas y las escribimos nosotros.** Las categorías salen de la literatura de CAA y cuidados en ELA, pero las frases concretas y las referencias aceptadas son juicio del tesista, no de un clínico ni de un usuario de CAA. Sirven para comparar configuraciones entre sí, que es para lo que se usan acá. No sirven para afirmar que el módulo dice lo correcto.
- **Potencia.** Con 133 entradas solo se leen diferencias de ~12 puntos. Para resolver 5 puntos harían falta unas 380 entradas; para 3 puntos, más de mil. Ampliar el corpus compra más capacidad de decisión que probar otro modelo.
- **Las etiquetas por defecto de Ollama no usan la misma cuantización.** Qwen viene en Q4_K_M, pero otras familias vienen en Q8_0 o Q4_0. Donde una familia pierde corriendo con *más* precisión, el resultado se refuerza; donde pierde con menos, parte del déficit puede ser el redondeo y no la familia. La tabla de costo dice la cuantización real de cada uno.
- **Todo corrió en GPU.** La RAM medida es ~0 en los quince: la carga está en VRAM. Las latencias de la tabla son de GPU, y un equipo dedicado de CAA probablemente no tenga una placa dedicada. Ahí los tokens por segundo serían sustancialmente peores.
- **La polaridad de la tabla es anterior al filtro determinista.** En el dispositivo, `guard.py` descarta por construcción lo que perdió la negación, así que la polaridad efectiva es 0%. La columna muestra la tendencia cruda del modelo, que es lo que hay que conocer para saber cuánto trabajo está haciendo el filtro.

