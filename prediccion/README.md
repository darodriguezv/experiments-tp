# Predicción personalizada (CAA / ELA)

Cuarto módulo. Los de mirada —[Eyetrax](../Eyetrax), [RealsenseV1](../RealsenseV1),
[WebcamV1](../WebcamV1)— resuelven *dónde mira*. [slm](../slm) resuelve *qué
quiso decir*. Este resuelve **qué dice habitualmente**, para que lo repetido
cueste menos.

La referencia obvia es SwiftKey: un teclado que aprende de vos y sugiere. La
idea es correcta; **el traslado literal al tablero no funciona**, y entender por
qué es la mitad de este módulo.

## El hallazgo, primero

En un teclado, «necesito» cuesta ocho pulsaciones y sugerirla entera ahorra
siete. En este tablero **NECESITO es una celda**: cuesta una permanencia.
Sugerirla en una casilla de predicción también cuesta una.

| situación | sin predicción | con | ahorro |
| --- | --- | --- | --- |
| palabra de la página actual | 1 | 1 | **0** |
| palabra de la otra página | 2 (cambio + celda) | 1 | 1 |
| frase de 4 palabras ya dicha | 4 | 1 | **3** |
| palabra que el tablero no tiene («hermana») | 9 (cambio + 7 letras + espacio) | 1 | **8** |

**Predecir la próxima palabra no ahorra casi nada. Lo que ahorra son las frases
enteras y lo que hay que deletrear.** Un port ingenuo de SwiftKey agregaría
cinco celdas de lectura a cada selección para devolver cero permanencias.

Por eso el ranking no ordena por probabilidad sino por **probabilidad ×
permanencias ahorradas**, y una predicción correcta con ahorro cero nunca ocupa
una casilla. Está fijado por un test —`ahorro cero nunca ocupa casilla`— porque
es la clase de decisión que una refactorización bienintencionada deshace.

## El tablero no se reordena. Nunca.

SwiftKey reordena sus sugerencias, pero **el teclado de abajo no se mueve**.
Acá la razón es más fuerte: quien maneja el tablero con la mirada aprende dónde
está cada celda y va directo. Mover AGUA porque esta semana se pidió mucha
destruye esa memoria espacial y obliga a barrer el tablero otra vez — se pierde
más en búsqueda visual que lo que cualquier predicción ahorra, y se pierde justo
cuando la persona tiene apuro.

Las predicciones viven en una franja de posición fija. `PAGES` no se toca.

## Lo que se mide, y lo que se cobra

| métrica | qué es |
| --- | --- |
| **ahorro** | permanencias evitadas / permanencias sin predicción |
| **aciertos** | fracción de selecciones que usaron una casilla |
| **celdas por selección** | lo que la vista recorre — **el costo** |

La tercera es la que se olvida siempre. Una franja de 5 casillas agrega 5 celdas
a **toda** selección, también a las que no la usan. Es el mismo error que en el
módulo de expansión, donde `recall@3` contaba un acierto aunque el usuario
recibiera un solo candidato: medir el beneficio sin cobrar su costo.

## Resultados (usuario sintético, 60 días, 1024 enunciados)

```
permanencias sin predecir 6101
permanencias prediciendo  4450
AHORRO                    27.1 %
por enunciado             6.0 -> 4.3
aciertos                  4.6 % de las selecciones
celdas por selección      31.3 -> 34.3  (+5 casillas)
```

**4.6 % de aciertos y 27 % de ahorro** es la confirmación numérica del hallazgo:
pocos aciertos, cada uno enorme. El ahorro medio por acierto es de unas ocho
permanencias — son las palabras deletreadas.

### Barrido

| configuración | ahorro | celdas/sel |
| --- | --- | --- |
| 1 casilla | 15.6 % | 31.6 |
| 3 casillas | 22.5 % | 33.4 |
| 5 casillas | 27.1 % | 34.3 |
| 8 casillas | 30.6 % | 36.4 |
| 5 casillas, vida media 3 d | 14.0 % | 34.7 |
| 5 casillas, **sin olvido** | **26.8 %** | 34.3 |

Dos lecturas incómodas:

**Más casillas siempre ahorra más permanencias**, pero el barrido no puede
decidir cuántas poner: no sabe cuánto cuesta leer una celda más. Ese número sale
de medir tiempo de búsqueda visual con una persona frente al tablero, y no está
medido. 5 es una elección, no un óptimo.

**La recencia no se ganó su lugar.** 30 días da 27.1 % y no olvidar nunca da
26.8 %: la misma cifra. Olvidar rápido empeora mucho. El decaimiento queda
porque es casi gratis y porque una deriva real es probablemente más brusca que
la del simulador — pero **afirmar que ayuda sería inventar un resultado**.

## Qué NO dicen estos números

Son de un **usuario sintético**. Sirven para validar el mecanismo, barrer
parámetros y atrapar errores del arnés. **No predicen el ahorro sobre una
persona real.**

El simulador está construido para no poder halagar: no es un modelo de n-gramas
(es una persona con rutinas por hora del día), incluye 15 % de enunciados
novedosos —impredecibles por construcción— y una deriva a los 30 días. El test
`el simulador no es el modelo` falla si el ahorro sube de 95 %.

Aun así, un simulador valida el instrumento, no el resultado.

## El arnés no mira el futuro

Cada enunciado se predice con lo aprendido de los **anteriores** y recién
después se observa. Romper esto es fácil y silencioso: entrenar con todo el
historial y medir sobre el mismo historial da números excelentes y falsos.

El test `el arnés no mira el futuro` mide las dos formas y **exige que la
tramposa dé mejor**: honesto 20.2 % contra 33.6 % con trampa. Si algún día esa
diferencia desaparece, el arnés dejó de medir predicción.

## Cómo se consiguen datos reales

El tablero **ya los está juntando**. `SessionLog` de RealsenseV1 escribe un CSV
por sesión con una fila por selección. `sessions.py` reconstruye de ahí los
enunciados:

```
python main.py logs ../RealsenseV1/logs
```

Dos decisiones sobre esa reconstrucción:

- **Solo cuenta lo que llegó a HABLAR.** Un mensaje escrito y borrado es lo que
  la persona decidió *no* decir. Aprenderlo empujaría el sistema hacia sus
  errores — y hacia los del estimador de mirada, porque una celda tocada por
  error y corregida es ruido del sensor, no vocabulario.
- **BORRAR y LIMPIAR se aplican.** Se aprende la frase dicha, no el camino
  accidentado hasta ella.

Dado que este proyecto no prueba con pacientes, «datos reales» significa las
sesiones del propio tesista usando el tablero.

## Privacidad

El modelo del usuario es un JSON con las frases de una persona concreta sobre su
cuerpo y su tratamiento. Nunca sale del equipo y nunca va al repositorio: el
`.gitignore` excluye `*.user.json` y `models/`.

## Uso

```
python main.py smoke    # 11 chequeos del instrumento
python main.py sim      # ahorro sobre el usuario sintético
python main.py sweep    # barre casillas y vida media
python main.py logs ../RealsenseV1/logs --save mio.user.json
python main.py show --model mio.user.json
```

## Lo que falta

- **El costo de una celda** en tiempo de búsqueda visual. Sin ese número, elegir
  cuántas casillas poner es una corazonada. Es medible con el arnés de
  `accuracy` que ya existe.
- **Enchufarlo al tablero.** Hoy es un módulo de medición: no dibuja la franja.
- **Reordenar los candidatos del SLM** por lo que la persona eligió antes. Es la
  unión natural con [slm](../slm) y no está hecha.
- **Deletreo asistido.** Predecir la palabra a mitad de deletrear es donde queda
  el ahorro más grande sin tocar: hoy se predice antes de empezar o nada.
