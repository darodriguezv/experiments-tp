# ¿Conviene cambiar el estimador de mirada?

*Investigación de 2026-10-26. La conclusión corta: **no**, y los datos dicen
por qué.*

Escrito para quien tenga que decidir si vale la pena reemplazar eyetrax por un
modelo de aprendizaje profundo más moderno.

## Lo que mide cada número, antes de comparar nada

Esta es la confusión que invalida casi toda comparación publicada: **no todos
los "grados de error" miden lo mismo.**

| qué se mide | qué significa | quién lo reporta |
|---|---|---|
| error angular de **dirección de mirada** | cuánto se desvía el vector de mirada estimado, sin calibrar, en personas que el modelo nunca vio | casi todos los papers de CNN |
| error de **punto en pantalla** tras calibración personal | cuántos píxeles separan la mirada estimada del punto que la persona mira, después de calibrar con esa persona | este proyecto, y los pocos trabajos de calibración personal |

Son problemas distintos. Un modelo con 4° de error direccional, usado para
apuntar a una pantalla, rinde **peor** que uno con 2° de error de punto tras
calibrar. Poner L2CS-Net en lugar de eyetrax no sería una mejora: sería una
degradación.

## Dónde está eyetrax hoy

Medido sobre diez sesiones reales guardadas en `Eyetrax/logs/accuracy-*.csv`,
con la conversión aproximada de 40 px por grado que usa `accuracy.py`:

| | error medio |
|---|---|
| mejor sesión | **82 px ≈ 2.05°** |
| sesiones buenas | 109–178 px ≈ 2.7–4.5° |
| peor sesión | 768 px ≈ 19° |

La conversión px→grados es aproximada y el propio código lo dice, así que los
grados valen para comparar con la literatura a grandes rasgos, no al decimal.

## Comparación con lo publicado

| método | error | condición |
|---|---|---|
| **eyetrax (este proyecto)** | **2.05° / 2.7–4.5°** | 9 puntos, por sesión |
| EMC-Gaze (2026), landmarks | 5.79° general · **2.92° cabeza quieta** | 9 puntos, por sesión |
| Mapeos de regresión 2D (revisión) | 2–4° | calibrados |
| CNN persona-específica, cara completa | **1.14–1.5°** | controlado |
| Métodos clásicos bien controlados | ~1° | calibrados |
| CNN persona-independiente | 4–6° | sin calibrar |
| L2CS-Net | 4° (MPIIGaze) · 9–10° (Gaze360) | sin calibrar |
| 3DGazeNet (en robótica real) | ~10° | sin calibrar |
| OpenVINO gaze-vector | 6.95° | 4 puntos |
| FAZE | 2.44° | **~1 Hz** |

**eyetrax está dentro de la banda esperada para su categoría.** La revisión que
cubre mapeos de regresión 2D los ubica entre 2° y 4°; eyetrax mide eso.

El techo disponible es **~1.1–1.5°**, de CNN persona-específicas con entrada de
cara completa en condiciones controladas. Es una mejora de aproximadamente 2×,
no de un orden de magnitud. Y FAZE, el único que combina buena precisión con
método moderno, corre a ~1 Hz: inservible para un tablero por permanencia que
necesita 20 fps.

## Pero la precisión media no es el problema de este proyecto

Descomponiendo el error de las diez sesiones en **sesgo** (dónde cae la mediana
de la mirada respecto del objetivo) y **temblor** (cuánto se dispersa alrededor
de esa mediana):

| sesión | error | sesgo | temblor | temblor crudo |
|---|---|---|---|---|
| 20260922-100340 | 82 px | 79 px | 17 px | 15 px |
| 20260922-100153 | 109 px | 109 px | 26 px | 24 px |
| 20260821-143626 | 160 px | 127 px | 31 px | 33 px |
| 20260922-092116 | 178 px | 173 px | 61 px | 52 px |
| 20260818-194801 | 330 px | 348 px | 55 px | 84 px |
| 20260922-093750 | 378 px | 361 px | 59 px | 88 px |
| 20260825-140351 | 768 px | 828 px | 162 px | 317 px |

**En las diez sesiones el error está dominado por el sesgo.** La mirada cae en
un lugar consistentemente equivocado, no en un lugar ruidoso. El temblor se
mantiene entre 17 y 162 px mientras el sesgo recorre de 79 a 828.

Dos consecuencias:

1. **El filtro de Kalman ya hace su trabajo.** El temblor crudo y el filtrado
   son casi iguales en las sesiones buenas; no queda mucho por ganar ahí.
2. **Un modelo más preciso atacaría el término equivocado.** Mejoraría los 17
   px de temblor que ya no molestan y no tocaría los 828 px de sesgo que
   arruinan una sesión.

### No es extrapolación hacia los bordes

| zona de la pantalla | sesgo mediano |
|---|---|
| centro | 189 px ≈ 4.7° |
| medio | 114 px ≈ 2.9° |
| borde | 157 px ≈ 3.9° |

El error **no** crece hacia los bordes, así que la hipótesis de que el modelo
extrapola fuera de los nueve puntos de calibración queda descartada. El sesgo
es de **sesión entera**: algo del armado de cada sesión -luz, postura, calidad
de la calibración- fija el nivel de error de toda ella.

## Lo que la evidencia dice que sí mejoraría

1. **Calibración con fijación y seguimiento suave combinados.** El único
   estudio que comparó las dos formas en webcam encontró que juntas, promediadas
   sobre tres momentos de la sesión, dan la mejor precisión de fijación: 2.58°
   con su mejor pipeline. El seguimiento suave cuesta ~26 s contra ~14 s de la
   fijación por punto.
2. **Más puntos de calibración.** Ese estudio usa 16 de fijación; este proyecto
   usa 9.
3. **Estabilidad de cabeza.** EMC-Gaze reporta 2.92° con la cabeza quieta
   contra 5.79° general: moverse aproximadamente duplica el error. Otro estudio
   de 2025 mide 0.38° de diferencia entre cabeza fija y en movimiento. Coincide
   con lo observado en este proyecto desde el principio.

## Recomendación

**No reemplazar el estimador.** Atacar el sesgo por sesión, que es donde está
el error.

Lo primero es gratis: `Eyetrax/aac_eyetrax/headaware.py` ya existe y **nunca se
midió**. Agrega 6 términos de posición de cabeza a los 486 rasgos de eyetrax.
Lo que está probado es el mecanismo —los rasgos de eyetrax son ciegos al
movimiento: trasladar la cabeza un 5% del ancho los cambia 2e-07, mientras los
términos nuevos cambian 0.31— pero **no que el error baje**. Eso es justo el
mecanismo que la literatura señala como principal.

    python Eyetrax/main.py pose

Calibra, mide, pide moverse, mide de nuevo, para los dos estimadores, y reporta
la degradación de cada uno. Unos diez minutos.

Si después de eso se quiere probar un modelo, el camino con sentido **no es
reemplazar eyetrax** sino usar una CNN de cara completa como extractor de
rasgos alimentando el mismo ridge por sesión: eso es lo que produce los 1.14°.
`hysts/pytorch_mpiigaze_demo` es MIT y trae pesos de ETH-XGaze, que generaliza
mejor que Gaze360 según el benchmark zero-shot de Gaze4HRI. Los datasets detrás
de esos pesos tienen sus propias condiciones de uso y hay que leerlas antes de
meterlos en algo que se distribuya.

## Adenda: WebGazer puesto a medir en el mismo banco

Esta investigación concluyó que un estimador más preciso atacaría el término
equivocado, porque las diez sesiones estaban dominadas por el sesgo y no por el
temblor. El puerto a navegador no reprodujo ese resultado: el sesgo quedó entre
123 y 392 px según la sesión, con el mismo código, y cuatro rondas de arreglos
-alpha por validación cruzada, puerta de colapso, orden en serpentina,
corrección de deriva- no lo cerraron.

Eso deja dos hipótesis que la medición de Python no puede separar:

1. el estimador propio tiene un defecto que solo aparece en el navegador;
2. el banco de medición del navegador mide mal, y el estimador está bien.

**WebGazer sirve para separarlas.** Se integró como estimador alternativo
seleccionable desde el menú, con todo lo demás idéntico: la misma calibración
de nueve puntos, el mismo filtro de `suavizado.js` -se le apaga el Kalman
propio-, la misma corrección de deriva y la misma prueba de precisión sobre los
mismos nueve puntos de control. Si WebGazer anda y el propio no, el problema es
el estimador. Si los dos fallan igual, el problema es el banco.

No es un reemplazo decidido. Mide otra cosa -los parches de los ojos reducidos
a 10x6 píxeles, 120 rasgos, contra la geometría de 161 landmarks del propio- y
su licencia es **GPL-3.0-or-later**, contagiosa para un aparato que se
distribuya. El encabezado del paquete agrega que empresas valuadas en menos de
un millón de dólares pueden usarlo bajo LGPLv3. Para medir y para una tesis no
cambia nada; para distribuir, sí. Esa decisión queda para después de la
medición, no antes.

### El detalle de integración que casi arruina la comparación

WebGazer guarda el entrenamiento en un buffer circular de **cincuenta**
posiciones. Las 270 muestras que graba el estimador propio -30 por punto-
habrían dejado vivas solo las últimas 50: el último punto y medio de los nueve.
El modelo habría quedado ajustado contra una esquina de la pantalla, y el
síntoma -sesgo enorme con temblor chico- es exactamente el que se está
persiguiendo del otro lado. La conclusión habría sido "los dos fallan igual"
por un motivo que no tiene nada que ver con WebGazer.

Por eso con WebGazer son cinco muestras por punto, 45 de las 50, que es el
tamaño para el que está hecho: su propio ejemplo de calibración usa nueve
puntos y cinco clics en cada uno. `web/pruebas/estimador.test.mjs` fija ese
presupuesto para que no vuelva a subir por simetría con el otro estimador.

## Límites de esta investigación

- **Una sola persona.** Todas las sesiones son del mismo usuario en el mismo
  equipo. El sesgo por sesión podría comportarse distinto en otra cara.
- **La conversión px→grados es aproximada**, y el código que la usa lo declara.
  Las comparaciones con la literatura valen a grandes rasgos.
- **Las condiciones de cada sesión no quedaron registradas.** No hay anotación
  de luz, hora ni distancia, así que "algo del armado fija el nivel de error" es
  una inferencia por descarte y no una medición de la causa.
- **No se encontró una comparación controlada** entre rasgos de CNN preentrenada
  y rasgos de landmarks bajo la misma calibración personal. Esa es la
  comparación que decidiría el punto y no parece estar publicada.

## Fuentes

- [L2CS-Net: Fine-Grained Gaze Estimation in Unconstrained Environments](https://arxiv.org/pdf/2203.03339)
- [Deployment-Oriented Session-wise Meta-Calibration for Landmark-Based Webcam Gaze Tracking (EMC-Gaze, 2026)](https://arxiv.org/pdf/2603.12388)
- [Person-Specific Gaze Estimation from Low-Quality Webcam Images](https://pmc.ncbi.nlm.nih.gov/articles/PMC10147084/)
- [Eye Gaze Estimation and Its Applications (Zhang, 2021)](https://cix.cs.uni-saarland.de/wp-content/uploads/2022/05/Zhang2021_Chapter_EyeGazeEstimationAndItsApplica.pdf)
- [Towards efficient calibration for webcam eye-tracking in online experiments](https://pure.mpg.de/rest/items/item_3386007_3/component/file_3477673/content)
- [Robust Camera-Based Eye-Tracking Method Allowing Head Movements](https://pmc.ncbi.nlm.nih.gov/articles/PMC12734114/)
- [Gaze Estimation for Human-Robot Interaction: NICO Platform](https://arxiv.org/html/2509.24001)
- [Gaze4HRI: Zero-shot Benchmarking Gaze Estimation Neural-Networks](https://arxiv.org/html/2605.04770)
- [pytorch_mpiigaze_demo (MIT)](https://github.com/hysts/pytorch_mpiigaze_demo)
- [yakhyo/gaze-estimation (ONNX)](https://github.com/yakhyo/gaze-estimation)
