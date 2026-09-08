# Módulo de mirada sobre Intel RealSense D435i (AAC / ELA)

Segunda iteración del módulo de detección de mirada para el dispositivo CAA. La
primera está en [../Eyetrax](../Eyetrax) y usa una webcam con
[`eyetrax`](https://pypi.org/project/eyetrax/); dio resultados razonables pero no
suficientes. Este directorio cambia la cámara **y el modelo**, porque el problema
no era solo la cámara.

## Por qué no alcanzaba la webcam

De los propios registros de [../Eyetrax/logs](../Eyetrax/logs), la mejor corrida
(21-ago):

| | error |
| --- | --- |
| media | 160 px |
| p95 | 263 px |
| centro de la pantalla (objetivos 3, 4) | 56-89 px |
| **esquina superior izquierda** (objetivo 0) | **405 px** |

El tablero actual usa celdas de 384 px y la regla `2 x p95` pide 526 px. Pero lo
importante no es el promedio: es que el centro anda bien y las esquinas se
derrumban. Eso no es ruido, es **sesgo sistemático**, y ningún suavizado lo
corrige.

La causa es estructural. EyeTrax aprende una regresión de 486 coordenadas de
puntos faciales *sin escala* directamente a píxeles de pantalla. Esa función solo
es válida alrededor de la posición de cabeza con la que se entrenó: en cuanto la
persona se mueve, el mapeo entero se corre. El README de la primera iteración ya
lo decía —"un modelo de mirada solo vale para, más o menos, la posición de cabeza
con la que se calibró"— y respondía con una pantalla de colocación que **exige**
volver a esa postura.

Para este proyecto esa respuesta no sirve. La postura de una persona con ELA
deriva a lo largo de una sesión, y deriva **más** a medida que la enfermedad
avanza. Un sistema que pide una postura reproducible está pidiendo algo que el
usuario no puede dar.

## Qué cambia con la D435i

La profundidad da **escala métrica real**, y eso permite reemplazar la regresión
por un **modelo geométrico**:

1. Se mide la posición 3D métrica de la cabeza, cuadro a cuadro.
2. El centro del globo ocular es un punto **fijo en coordenadas de la cabeza**,
   no de la cámara. Se mueve con ella gratis.
3. La pupila está sobre una esfera alrededor de ese centro. El rayo de la cámara
   que pasa por el píxel del iris corta esa esfera en la pupila.
4. Centro-a-pupila es el eje **óptico**; el eje **visual** —donde la persona
   mira— está desviado un ángulo por ojo (kappa), porque la fóvea no está en el
   fondo del ojo.
5. Ese rayo se cruza con el **plano físico de la pantalla**.

Nada de eso depende de dónde esté la cabeza. La invariancia a la postura no es
una mejora que se ajusta: es una consecuencia de la construcción.

El modelo tiene **11 parámetros** (3 por centro ocular, 2 por kappa, 1 radio
compartido) contra los cientos de una regresión sobre 486 rasgos. Por eso una
calibración de nueve puntos alcanza, y por eso extrapola a posturas que nunca
vio.

## Estado: qué está verificado y qué no

Todo lo de abajo se escribió **antes** de tener la cámara, comprobando contra la
**solución exacta** en vez de contra una medición: el problema directo tiene
forma cerrada, así que [synth.py](aac_realsense/synth.py) construye una escena a
partir de parámetros conocidos, la pasa por el estimador real y exige que
devuelva el píxel del que se partió. Esos chequeos siguen siendo la red de
seguridad y siguen corriendo sin hardware.

**La cámara ya está conectada** (una D435, sin IMU) y el camino de hardware
funciona de punta a punta: ver "Medido en la cámara real" más abajo. Todavía no
hay números de `accuracy`, que son los que deciden si el enfoque sirve.

```powershell
.\run.cmd smoke      # 27/27
```

| Chequeo | Resultado |
| --- | --- |
| Mirada, ida y vuelta (7 objetivos, las 4 esquinas incluidas) | **0.0000 px** |
| **Invariancia a la postura** (3 objetivos x 5 poses, 0.45-0.70 m, ±25° de giro) | **0.0004 px** |
| Modelo vectorizado del ajuste == camino de inferencia | 2.3e-13 px |
| Pose de cabeza por profundidad (Kabsch) | exacta |
| Pose de cabeza por PnP, **sin profundidad** | 0.00 mm |
| Plantilla craneal métrica, 30 cuadros con 1 mm de ruido | distancias dentro de **0.68 mm** |
| Ajuste de calibración desde el arranque real (6.8 mm de error) con montaje mal medido (15 mm, 2°) | RMS **12.8 px**, radio corneal recuperado 13.17 mm (verdad 12.8) |
| **El mismo ajuste, en una postura NUNCA calibrada** | **22 px** |
| Extremo a extremo: mirada → celda → permanencia, en postura no calibrada | celda correcta, **1.0 px** del centro |
| Inclinación desde el acelerómetro (0°, 5°, 10°, 20°, roll 8°) | exacta |
| Kabsch robusto con 3/20 puntos arruinados | 9.23° → 0.000° |
| Nuestra deproyección contra `rs2_deproject_pixel_to_point` del SDK | **exacta** (0.05 mm en el peor modelo, límite de float32 del SDK) |
| Los 38 nombres de la API de pyrealsense2 que usa este código | presentes en 2.58.3 |
| MediaPipe: 478 puntos con iris, e índices 468/473 en el ojo correcto | verificado |

> **Estos números son validación geométrica, no una predicción de precisión
> real.** Dicen que la matemática es correcta y que la implementación coincide
> con la matemática. No dicen nada sobre el ruido de MediaPipe sobre un iris
> real, la profundidad sobre piel real, la refracción de la córnea o la
> asfericidad corneal, que el modelo de esfera no representa.
>
> La expectativa honesta, según la literatura de mirada basada en modelo con
> cámara de profundidad, es **1-2° de error angular**. En esta pantalla eso son
> **38-76 px** (la escala medida es 37.8 px por grado a 58 cm). Contra los 263 px
> de p95 y 405 px en esquina de la webcam, sería entre 3× y 7× mejor, con el
> derrumbe de esquina eliminado por construcción. Habrá que medirlo con
> `accuracy` cuando llegue la cámara.

Dos cosas que **sí** se pudieron verificar contra las librerías reales, sin
cámara:

- **La deproyección contra la del SDK.** `frames.Intrinsics` reimplementa
  `rs2_deproject_pixel_to_point` en numpy para correr sin el SDK y para convertir
  un arreglo entero de puntos de una vez. Un `rs.intrinsics` se puede llenar a
  mano, así que las dos implementaciones se comparan sin hardware. Encontró un
  error real: el modelo `inverse_brown_conrady` de librealsense **ya no** se
  deproyecta con una evaluación directa (como decía la semántica antigua del
  nombre) sino con la misma undistorsión iterativa que `brown_conrady`. La
  versión directa discrepaba **32 mm a 0.6 m, en dirección opuesta**. Corregido y
  ahora chequeado en cada corrida.
- **La convención izquierda/derecha de MediaPipe.** Los índices 468 y 473 son los
  centros de iris del ojo *del sujeto*, así que el ojo derecho aparece a la
  izquierda de la imagen. Confundirlos espeja la mirada horizontal y la
  calibración lo tapa a medias. Verificado sobre una cara sintética dibujada a
  propósito: el 468 salió en x=584 con la pupila derecha dibujada en x=585.

### Medido en la cámara real (D435, 913522071989)

| | |
| --- | --- |
| Apertura del pipeline, con warm-up | 3.3 s |
| Color 1280x720, fx medido | 913.5 px, HFOV 70.0° |
| Coeficientes de distorsión | **todos cero** (así que la rama Brown-Conrady no interviene en la práctica) |
| Escala de profundidad | 0.001 m/unidad |
| **fps con MediaPipe en el lazo, 720p** | **30.0** (tope del stream, no del CPU) |
| fps a 1920x1080 | **20.6** — MediaPipe no sostiene 30 a 1080p en esta máquina |
| Cara detectada | 87/90 cuadros |
| Radio del iris a 46 cm, 720p | 11.0 px (diámetro 22 px) |
| Cobertura de profundidad sobre la cara, preset High Accuracy | **62 %** |
| Cobertura con presets de menor confianza | **~96 %** |

Esa última fila cambió un valor por defecto: ver `visual_preset` en
[config.py](aac_realsense/config.py). High Accuracy descarta los píxeles de baja
confianza, y en una cara los de baja confianza son los periféricos —sienes,
bordes de pómulo— que son justo los puntos más separados del conjunto rígido y
por lo tanto los que fijan la rotación de la cabeza. Perderlos no agrega ruido,
agrega **sesgo**, y el ajuste robusto absorbe ruido pero no puede inventar un
punto que no está. El mismo razonamiento ya había forzado la banda de rechazo de
profundidad de 60 a 100 mm.

### Dos bugs que solo aparecieron con la cámara enchufada

Vale registrarlos porque los dos habían pasado los 27 chequeos sintéticos.

1. **`frames()` terminaba la sesión al primer cuadro perdido**, en lugar de
   saltearlo como decía su propio docstring. Un fallo transitorio de USB cerraba
   la vista previa o la calibración sin imprimir nada. Ahora una cámara viva
   tolera 3 s de fallos seguidos; una grabación sigue terminando en el primero,
   porque ahí un fallo *sí* es el final.

2. **El eje "hacia atrás de la cabeza" estaba inclinado 49°.** Se derivaba de
   `centro_de_ojos − punta_de_nariz`, y en una cara real la nariz está unos 35 mm
   por debajo y 30 mm por delante de los ojos, así que ese vector no es la normal
   facial: está a `atan2(35, 30)` de ella. Resultado medido en la cámara: **44.9°
   de inclinación reportada para alguien sentado de frente**, lo que bloqueaba la
   guía de colocación por completo.

   La mirada nunca estuvo afectada — una rotación constante entre el frame de la
   plantilla y la anatomía se cancela entre `TemplateBuilder` y `fit_pose`. Lo
   que rompía era todo ángulo *reportado*, y la compuerta que depende de uno.

   Se corrigió derivando el eje "abajo" de los pómulos, que *encierran* a las
   comisuras oculares en protrusión y dejan unos 2° de sesgo en vez de 49°.
   Después del arreglo, la misma persona en la misma silla: **−4.4°**.

   **Por qué no lo atrapó ninguna prueba:** el fixture sintético ponía la punta
   de la nariz a la misma altura que los ojos, precisamente para que esa
   construcción diera la identidad y las aserciones quedaran más limpias. Un
   fixture con la forma que le conviene al código que prueba no puede encontrar
   esta clase de bug. Ahora la nariz está donde va, y
   `cabeza de frente lee ~0°` es la regresión.

Falta todavía, y necesita a alguien sentado adelante: `calibrate`, `accuracy`,
`live`, `board`. La parte con más riesgo residual es la cadena de filtros de
profundidad —ver la nota en [source.py](aac_realsense/source.py).

## Instalación

```powershell
conda create -y -p .\.conda python=3.12 pip
.\.conda\python.exe -m pip install -r requirements.txt
```

`run.cmd` reenvía al intérprete del entorno, así que no hace falta activar nada.

Notas:

- **`pyrealsense2` tiene ruedas oficiales para Windows en CPython 3.9-3.14.**
  Fuera de ese rango pip intenta compilar desde fuente y falla sin MSVC.
- A diferencia de la primera iteración, **acá nada fija `numpy<2`**. Ese pin era
  de `eyetrax`; al no depender de él, desaparece.
- La primera corrida baja el modelo de MediaPipe (`face_landmarker.task`, ~3 MB)
  a `~/.cache/aac_realsense/`.
- El SDK de RealSense **no es necesario** para `smoke`: toda la geometría corre
  sin él. Eso es deliberado, y es lo que permitió escribir y validar esto sin la
  cámara. Si está instalado, `smoke` agrega dos chequeos más: que existan todos
  los nombres de la API que usa el código, y que nuestra deproyección coincida
  con la del SDK.
- Verificado en este entorno: python 3.12.13, numpy 2.5.2, pyrealsense2 2.58.3,
  mediapipe 1.0.1, opencv 5.0.0, scipy 1.18.1.

## Lo único que hay que medir a mano

Un modelo geométrico necesita saber **dónde está la pantalla** en coordenadas de
la cámara. Son seis grados de libertad, y se reparten así:

| DOF | Cómo se obtiene |
| --- | --- |
| inclinación (pitch), rotación (roll) | **el acelerómetro los mide** — `.\run.cmd mount` |
| desplazamientos x, y, z | cinta métrica, del centro de la pantalla al centro del lente |
| giro (yaw) | cinta métrica; normalmente 0, la cámara mira de frente |

Que el IMU se encargue de los dos ángulos difíciles es la mejor razón para
preferir la **D435i** sobre la D435: no se puede apoyar un goniómetro sobre una
cámara sujeta a un bisel, pero la cámara sí puede medir su propia actitud contra
la gravedad con una fracción de grado de error.

Los desplazamientos se miden desde el **centro del área activa** de la pantalla
hasta el centro del lente de color:

```
        +---------------------------------------+
        |                                       |
        |                                       |   offset_y_mm > 0  ->  hacia ABAJO
        |                 (centro)              |   offset_x_mm > 0  ->  hacia la DERECHA
        |                    +                  |   offset_z_mm > 0  ->  hacia el USUARIO
        |                    |                  |                        (cuánto sobresale
        |                    | offset_y         |                         el lente)
        |                    v                  |
        +--------------------o------------------+
                          [cámara]
```

Ejemplo, cámara centrada en el bisel inferior de un monitor de 24" (área activa
527 x 296 mm):

```powershell
.\run.cmd mount        # dice: inclinación +7.4°, rotación -0.3°
.\run.cmd calibrate --pitch 7.4 --offset-x 0 --offset-y 163 --offset-z 18
```

`offset_y = 296/2 + 15` = 163 mm: media pantalla más los 15 mm de bisel hasta el
lente.

**No hace falta que sea exacto.** La calibración refina el montaje hasta 30 mm y
5°, así que basta con quedar dentro de ese margen. Lo que **sí** importa es que
el punto de partida no esté fuera: si el montaje está mal por 5 cm, el ajuste
tapa el error metiéndolo en el kappa, y ahí se pierde la invariancia a la postura
que es todo el motivo del cambio.

El tamaño físico de la pantalla se lee del EDID vía `screeninfo`. Si su monitor
no lo informa, pase `--screen-mm 527x296`: el diagonal nominal en pulgadas no
sirve porque todos los fabricantes lo redondean, y un 1% de error de tamaño es un
1% de error de mirada en toda la pantalla.

## Inicio rápido

```powershell
.\run.cmd devices      # ¿está la cámara? ¿por USB 3? ¿tiene IMU?
.\run.cmd preview      # ¿hay profundidad sobre la cara? ¿cuántos píxeles de iris?
.\run.cmd mount        # inclinación real de la cámara
.\run.cmd pipeline     # calibrar -> medir -> tablero
```

Ese orden no es decorativo: los tres primeros comandos detectan, cada uno, un
fallo que después se disfraza de "el algoritmo anda mal".

## Los comandos

**`devices`** — detecta cámaras RealSense y reporta modelo, firmware, tipo de
USB, presencia de IMU, escala de profundidad y modos disponibles. Avisa
explícitamente de las cuatro cosas que degradan la precisión en silencio: USB 2
en vez de 3, una D435 sin IMU, firmware viejo, o un modelo distinto al esperado.

**`preview`** — color y profundidad superpuestos, sin calibrar nada. Muestra
distancia en cm, radio del iris en píxeles, cobertura de profundidad sobre la
cara y ángulos de cabeza. Es lo primero que hay que correr con una cámara nueva,
porque cada uno de esos números es un techo físico que ningún modelo levanta.

**`mount`** — lee el acelerómetro y dice la inclinación real de la cámara.

**`calibrate`** — nueve puntos, **dos pasadas** por defecto. Entre pasadas pide
cambiar de postura; ver abajo por qué eso no es opcional. Guarda
`models\gaze_model.json`.

**`live`** — cursor de mirada. La tecla **`H`** activa la **prueba de postura**:
mire un punto fijo, muévase, y la pantalla informa cuánto se movió la cabeza y
cuánto se corrió la mirada. Es la forma de *ver* la invariancia a la postura en
diez segundos, en vez de creerla.

**`accuracy`** — mide el error contra objetivos conocidos. Tres cosas que la
versión de webcam no podía hacer:

- **Grados reales.** La primera iteración tenía que fijar 40 px/grado a mano.
  Acá la distancia se mide cuadro a cuadro, así que los grados se calculan.
- **Prueba de invariancia a la postura.** Ajusta una recta del error contra el
  desplazamiento de la cabeza y reporta la pendiente en px/cm. Es la medición
  que decide si valió la pena cambiar de cámara: una regresión de apariencia
  tiene pendiente positiva por construcción, un modelo geométrico correcto debe
  dar plano. Falsable, no retórico.
- **Auditoría de respaldos.** Cuántos cuadros resolvieron la pose por PnP en vez
  de por profundidad, y cuánta profundidad hubo sobre la cara. Una sesión que
  corrió sin profundidad todo el tiempo está midiendo otra cosa.

Se mantiene la separación **temblor vs desvío** de la primera iteración, porque
esa distinción estaba bien y siguen teniendo arreglos opuestos.

Las primeras ocho columnas del CSV son idénticas a las de
[../Eyetrax](../Eyetrax), así que las dos corridas se comparan sin script de
conversión.

**`board`** — el tablero CAA. Igual que el de la primera iteración (mismo
vocabulario, misma permanencia, mismo CSV de sesión) más tres cambios que salen
de tener confianza:

- La permanencia **se sostiene** en cuadros de baja confianza, en vez de avanzar.
  Los dos errores no son simétricos: avanzar produce selecciones equivocadas, que
  cuestan un borrado y un reintento; reiniciar tira el esfuerzo del usuario.
  Sostener no hace ninguna de las dos.
- Un borde ámbar y el texto "SEÑAL DÉBIL" cuando eso pasa, así una permanencia
  detenida es legible. No poder distinguir "me está ignorando" de "no me puede
  ver" es un problema real cuando el tablero es el único canal de salida.
- El CSV registra el estado del sensor junto a cada selección, así una selección
  equivocada queda diagnosticable después.

**`record`** — graba a `.bag`. Vale la pena en toda sesión real: una grabación se
reproduce por el mismo código, así que evaluar un estimador nuevo pasa a ser una
pregunta offline en vez de otra cita con un participante. Para un estudio con
personas cuyo cuadro avanza, poder reanalizar la sesión de ayer no es una
comodidad.

```powershell
.\run.cmd record --seconds 30
.\run.cmd accuracy --playback recordings\session-20260822-141530.bag
```

## Por qué la calibración pide moverse (y por qué eso invierte la intuición)

En la primera iteración, mover la cabeza entre calibración y uso era lo que
rompía todo. Acá es al revés: **mover la cabeza *durante* la calibración es lo
que hace que funcione.**

El motivo es una degeneración. Correr el centro del ojo 5 mm hacia la pantalla y
correr la pantalla 5 mm hacia el ojo producen casi las mismas predicciones —
desde **una sola** posición de cabeza. El ajuste no puede distinguirlas, así que
reparte el error de cualquier manera, reporta un residuo chico, y después deriva
en cuanto la persona se mueve.

Pero el centro del ojo está fijo en coordenadas de la **cabeza** y la pantalla en
coordenadas de la **cámara**. En el momento en que la cabeza ocupa dos
posiciones realmente distintas, los dos parámetros dejan de ser intercambiables.

Diez centímetros alcanzan. Con una sola pasada, `calibrate` avisa:

```
avisos:
  - la cabeza se movió muy poco durante la calibración (3 mm). La posición del
    ojo y la de la pantalla quedan confundidas entre sí, así que el ajuste vale
    cerca de esa postura y no lejos de ella.
```

Ese aviso está probado ([`la degeneración se detecta`](aac_realsense/smoke.py)):
una calibración de una sola postura ajusta **perfectamente** sus propios
objetivos —RMS 0.0 px— y queda avisada igual. Ese es exactamente el caso
peligroso, porque el número se ve bien.

## Qué hacer si la precisión no alcanza

`accuracy` imprime una lista ordenada. En orden de rendimiento típico:

1. **Píxeles sobre el iris.** Es el techo físico: el centro de la pupila queda
   cuantizado por el tamaño del iris. `--color 1920x1080` da un 50% más de ancho
   que el default de 720p. `preview` dice cuántos píxeles hay.
2. **Profundidad sobre la cara.** Sin profundidad no hay escala métrica y todo
   cae al respaldo por PnP, que funciona pero es más sensible al ruido de los
   puntos faciales. El sol directo satura el emisor infrarrojo de 850 nm: es la
   causa número uno.
3. **La geometría del montaje.** En el camino geométrico, un desvío constante es
   casi siempre un error de traslación del montaje, no del modelo. `accuracy`
   convierte el sesgo en milímetros de `offset_y_mm` para que se pueda corregir
   con la cinta.
4. **Dos pasadas de calibración**, si se usó una.
5. **Suavizado**, al final y solo si `accuracy` dice que el problema es temblor y
   no desvío. Esconde el temblor a cambio de retardo, y el retardo lo siente
   directamente quien escribe una frase de a una permanencia.

Note que la lista es distinta de la de la primera iteración. Ahí el consejo
número uno era "probá un modelo no lineal" porque `ridge` se derrumbaba en los
bordes. Acá ese modo de falla no existe, así que las causas que quedan son
físicas.

## Distribución de archivos

| Archivo | Rol |
| --- | --- |
| [main.py](main.py) | CLI |
| [geometry.py](aac_realsense/geometry.py) | matemática pura: rotaciones, intersecciones, Kabsch, `ScreenPlane` |
| [frames.py](aac_realsense/frames.py) | intrínsecos y distorsión, `Frame`; sin pyrealsense2 |
| [config.py](aac_realsense/config.py) | `Settings` y `CameraMount` (la geometría del montaje) |
| [rsdevice.py](aac_realsense/rsdevice.py) | detección de la cámara y reporte de capacidades |
| [source.py](aac_realsense/source.py) | cámara en vivo, reproducción de `.bag`, fuente sintética, IMU |
| [landmarks.py](aac_realsense/landmarks.py) | MediaPipe (478 puntos con iris), detector inyectable, parpadeo |
| [face3d.py](aac_realsense/face3d.py) | plantilla craneal métrica y pose de cabeza (Kabsch + PnP) |
| [eyemodel.py](aac_realsense/eyemodel.py) | esfera corneal, kappa, rayo de mirada |
| [estimator.py](aac_realsense/estimator.py) | estimador geométrico, brazo híbrido, corrección de residuo, `GazeModel` |
| [fitting.py](aac_realsense/fitting.py) | el ajuste de calibración por mínimos cuadrados acotados |
| [smoothing.py](aac_realsense/smoothing.py) | suavizadores, todos sensibles a la confianza |
| [tracker.py](aac_realsense/tracker.py) | **la única superficie que ve la capa CAA** |
| [synth.py](aac_realsense/synth.py) | escenas sintéticas con verdad de terreno exacta |
| [smoke.py](aac_realsense/smoke.py) | los 25 chequeos |
| [positioning.py](aac_realsense/positioning.py) | guía de colocación, ahora un chequeo de sensor |
| [accuracy.py](aac_realsense/accuracy.py) | medición de error y diagnóstico |
| [calibrate.py](aac_realsense/calibrate.py) | presentación de objetivos, recolección y ajuste |
| [preview.py](aac_realsense/preview.py) | `preview`, `mount`, `record` |
| [aac_board.py](aac_realsense/aac_board.py), [board_layout.py](aac_realsense/board_layout.py) | el tablero y su vocabulario |
| [display.py](aac_realsense/display.py), [ui.py](aac_realsense/ui.py), [tts.py](aac_realsense/tts.py) | pantallas, dibujo, voz (portados sin cambios) |

`tracker.py` expone deliberadamente la **misma API** que
[../Eyetrax/aac_eyetrax/tracker.py](../Eyetrax/aac_eyetrax/tracker.py) —
`GazeSample`, `GazeTracker.process`, `DwellSelector`— así que el tablero, el
arnés de precisión y la vista en vivo son los mismos programas contra otro motor
de mirada. Eso es lo que hace que la comparación en este repositorio sea una
comparación y no dos proyectos distintos.

`GazeSample` gana cuatro campos que la webcam no podía producir, todos del mismo
origen: `confidence`, `distance_cm`, `iris_px` y los ángulos reales de cabeza.

## Notas para el proyecto

- **Hay dos estimadores, a propósito.** `--estimator geometric` es el modelo 3D;
  `--estimator hybrid` es una regresión sobre 16 rasgos **métricos** (posición 3D
  del ojo, ángulos de mirada, pose de cabeza) en vez de 486 coordenadas sin
  escala. Comparten plantilla, pose y modelo de ojo: difieren **solo** en cómo un
  rayo se convierte en píxel. Eso separa dos afirmaciones que se confunden — que
  los rasgos métricos ayudan, y que el modelo geométrico ayuda — y permite
  atribuir la mejora a cada una.
- **El modelo se guarda en JSON, no en pickle.** La primera iteración serializaba
  un estimador de sklearn, lo que ata una calibración a las versiones de librería
  que la produjeron. Acá son once parámetros físicos, una pose de pantalla, una
  plantilla craneal y unos coeficientes: números con significado, que van a un
  apéndice de tesis y que van a seguir cargando en dos años.
- **Ahora hay confianza por cuadro.** La regresión de la webcam siempre devolvía
  un punto, por más equivocado que fuera, así que un cuadro malo era
  indistinguible de uno bueno. Acá la confianza tiene causas nombrables —se
  perdió la profundidad, el residuo del ajuste rígido es alto, el iris es chico,
  la mirada es muy excéntrica— y la permanencia puede sostener en vez de
  comprometerse.
- **PnP es el camino normal, no una excepción.** La profundidad sobre una cara se
  cae constantemente: pelo, cejas, piel especular, el ojo mismo. Como la
  plantilla es métrica, PnP recupera traslación métrica y no solo una dirección,
  así que perder profundidad en un cuadro cuesta precisión en vez de causar
  fallo. Está medido: 0.00 mm de error en el sintético.
- **El IMU va en su propio pipeline.** Combinar flujos de movimiento con
  profundidad y color en un solo pipeline de librealsense es una fuente
  documentada de timeouts en `wait_for_frames` y de IMU que se cortan a los
  pocos minutos. Nada acá necesita sincronizar el IMU con el video: se lee para
  medir la inclinación de la cámara, que cambia cuando alguien golpea el
  soporte, no cuadro a cuadro.
- **La banda de rechazo de profundidad se corrigió a 100 mm.** Empezó en 60 mm y
  una prueba mostró que rechazaba puntos reales de la sien a 18° de giro. Son los
  peores que se pueden perder: las sienes son los puntos más separados del
  conjunto y por lo tanto los que fijan el giro, así que descartarlas sesga la
  pose justo cuando la cabeza está girada.
- **El ajuste trabaja en radianes, no en píxeles.** En píxeles el residuo no está
  acotado: si el centro del ojo se desplaza cerca de su propio radio, el rayo roza
  la esfera en tangente y la intersección se va a decenas de miles de píxeles.
  Con `soft_l1` esos residuos saturan la pérdida, el gradiente se anula y el
  optimizador no se mueve — es exactamente lo que pasaba antes del cambio. El
  ángulo está acotado por π, es suave a través del caso tangente, y es la
  cantidad sobre la que el modelo realmente tiene incertidumbre.
- **El ajuste tarda unos segundos.** Con ~500 muestras y 17 parámetros, el
  jacobiano numérico de scipy pide unas 18 evaluaciones por iteración; contar
  entre 15 y 60 s. Es una vez por calibración.
- **Los dos modelos de distorsión Brown-Conrady se tratan igual**, con la
  undistorsión iterativa, porque es lo que hace el SDK actual — determinado
  midiendo, no leyendo los nombres. Se conserva la distinción en el JSON del
  modelo a título informativo. En una D435 real los coeficientes de color son
  chicos o cero, así que en la práctica importa poco; eso no es razón para
  estar equivocado.
- **La corrección polinómica está separada y reportada.** `FitReport` informa el
  error *solo geométrico* junto al final, y avisa si el polinomio está haciendo
  la mayor parte del trabajo — porque en ese caso es un ajuste de curva
  disfrazado de modelo, y no va a extrapolar a posturas nuevas.
- **La pantalla de colocación cambió de propósito.** Ya no verifica que la
  persona reproduzca la postura de la calibración, porque el modelo no lo
  necesita. Verifica lo que el **sensor** sí necesita: rango de profundidad
  (35-85 cm), píxeles de iris, cobertura de profundidad sobre la cara, y ángulo
  de cabeza no extremo. Todos son límites físicos, no preferencias.
- **La suposición de pantalla vertical.** `mount` deriva inclinación y rotación
  de la gravedad suponiendo que la pantalla es vertical. Cierto para un monitor
  en soporte o brazo; **falso para la tapa de una notebook**. En ese caso hay que
  medir el ángulo de la tapa y sumarlo a mano.
- **Límites del modelo de esfera.** La córnea deja de ser esférica más allá de
  unos 25° de excentricidad de mirada, así que las celdas de los extremos en una
  pantalla grande y cercana van a ser peores. `eye_weight` baja la confianza allí
  en vez de ignorarlo. Si aparece como problema, la respuesta es una pantalla más
  chica o más lejos, no más parámetros.
- **Lo que sigue sin resolverse:** la calibración necesita que la persona pueda
  fijar nueve puntos y sostener la mirada. Para alguien en una etapa avanzada eso
  puede no ser posible. Un camino es calibración implícita a partir del uso del
  tablero (las selecciones acertadas son objetivos conocidos a posteriori), pero
  no está implementado.
