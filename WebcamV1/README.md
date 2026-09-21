# WebcamV1 — GazeFollower sobre la webcam del portátil

Mismo tablero, misma calibración por puntos, misma medición de precisión que
[Eyetrax](../Eyetrax). **Lo único distinto es el estimador de mirada.**

Esa es la condición para que la comparación signifique algo. Si cada módulo
midiera con su propia cuadrícula, su propio suavizado y su propia rutina, la
diferencia entre los dos sería inatribuible: no se sabría cuánto es el modelo y
cuánto el instrumento. Acá el arnés de Eyetrax se **importa**, no se copia.

## Cómo se logró un solo arnés

`aac_eyetrax.calibrate/accuracy/aac_board/live_gaze/pipeline` recibieron un
parámetro `make_tracker`, con el comportamiento anterior por defecto. WebcamV1
pasa el suyo:

```python
from aac_webcam import tracker
accuracy.run(settings, 3, 3, make_tracker=tracker.make)
```

`WebcamTracker` expone exactamente lo que el arnés usa —`process`, `calibrate`,
`save`, `load`, `face_pose`, `open_window`, `reset_smoother`— con GazeFollower
adentro. El chequeo `nuestro tracker cumple la interfaz` falla si el arnés
empieza a pedir algo que no tenemos, para que eso se sepa antes de sentar a
alguien a calibrar y no a mitad de la sesión.

## La cadena, por cuadro

```
cuadro BGR
  -> TasksFaceAlignment   recortes de cara y ojos + 478 puntos
  -> MGazeNet (MNN)       embedding de 258 valores; los dos primeros son
                          la mirada CRUDA en cm respecto de la cámara
  -> SVRCalibration       embedding -> cm, aprendido de esta persona
  -> cm2px                cm -> píxel, con la geometría física medida
  -> suavizado de Eyetrax el mismo filtro que el otro estimador
```

Detalle que sorprende: una vez calibrada, **la SVR predice solo desde el
embedding** e ignora la mirada cruda. Por eso sin calibrar el cursor es
inservible aunque el modelo cargue bien.

## Hay que medir la pantalla con una regla

MGazeNet no predice píxeles sino **centímetros respecto de la cámara**. Para
llegar a una celda hacen falta dos medidas que el software no puede adivinar:

- **`--screen-cm`**: ancho y alto del área visible, en cm.
- **`--camera-cm`**: dónde está la cámara desde la esquina superior izquierda
  de esa área. La `y` es negativa si está por encima del borde, que es el caso
  de casi todo portátil.

```
run.cmd geometry
```

imprime lo que está asumiendo. Si no coincide con la regla, el error es
sistemático y **la calibración no lo arregla**: el modelo entrega cm
verdaderos y las etiquetas estarían en cm escalados, así que el ajuste empeora
en vez de fallar limpio. Los valores por defecto (34.4 × 19.3 cm, cámara en
17.2 / −0.7) son los del portátil de desarrollo, no una constante.

## Dos bugs de GazeFollower que hubo que rodear

**No puede releer su propia calibración.** Su constructor hace
`self.svr_x.load(ruta)`, pero `load` en OpenCV es un método *estático*:
devuelve un modelo nuevo y deja la instancia intacta. Verificado: tras
`instance.load(p)` queda `isTrained() False` con `varCount 0`, mientras
`cv2.ml.SVM.load(p)` sí devuelve uno entrenado.

Lo peligroso es lo que hacen después: marcan `has_calibrated = True`. O sea que
**la segunda sesión y todas las siguientes arrancan diciendo que están
calibradas y prediciendo con un modelo vacío.** Sin excepción y sin aviso, con
la mirada sin sentido — indistinguible de «este estimador no sirve». Se corrige
en `tracker._new_calibration`.

**`import gazefollower` falla entero en Python 3.12**, porque
`MediaPipeFaceAlignment()` y `SVRCalibration()` están como valores por defecto
de argumentos y se evalúan al importar. `gf_compat.patch()` instala un stub que
falla ruidosamente si alguien lo usa de verdad.

## Rendimiento, medido

| | por cuadro | fps |
| --- | --- | --- |
| **EyeTrax** | 39 ms | **26** |
| **GazeFollower** | 72 ms | **14** |

Desglose de los 72 ms: alineación MediaPipe ~33 ms, inferencia MNN ~40 ms. El
preprocesado (recortes, resize) es despreciable: 0.8 ms.

Bajar la resolución de captura **no ayuda** (101 ms a 640×480 contra 108 a
1280×720): la CNN recibe recortes de tamaño fijo. Lo que sí ayuda es el reparto
de hilos de MNN, que GazeFollower fija en 4 y no expone. Medido sobre 16
núcleos:

| hilos | process | fps |
| --- | --- | --- |
| 4 (el suyo) | 99 ms | 10.1 |
| **8 (por defecto acá)** | **72 ms** | **13.8** |
| 12 | 84 ms | 11.9 |

**GazeFollower cuesta el doble de tiempo por cuadro.** A 800 ms de permanencia
son ~11 cuadros por selección: alcanza, pero el cursor se siente más pesado.
Si la precisión no mejora lo suficiente, este costo no se justifica.

## Uso

```
run.cmd smoke        # 9 chequeos, sin cámara
run.cmd geometry     # MIRALO ANTES DE CALIBRAR
run.cmd pipeline     # colocar, calibrar, medir, tablero
run.cmd calibrate --method 9point
run.cmd live         # cursor, para ver si quedó bien
run.cmd accuracy     # error en grados, comparable con Eyetrax
run.cmd board --dwell 800
```

La comparación que importa:

```
cd ../Eyetrax  && run.cmd accuracy    # el número que ya conocés (5.2°)
cd ../WebcamV1 && run.cmd accuracy    # el mismo arnés, otro estimador
```

Cada uno con su propia calibración, hecha con su propia tubería: la SVR absorbe
un desvío sistemático de los puntos faciales **solo si es el mismo en
calibración y en uso**. Calibrar con uno y correr con el otro es el error a no
cometer.

## Lo que todavía no se sabe

**Si mejora los 5.2°.** Todo lo de acá dice que el camino funciona y cuánto
cuesta; ninguna línea dice que sea más preciso. Eso sale de correr `accuracy`
en los dos, con la misma persona, la misma luz y el mismo día.

**Si la licencia sirve.** GazeFollower es CC BY-NC-SA. Si eso bloquea el
dispositivo, hay que cambiar el estimador por uno permisivo — y gracias a la
costura de `make_tracker`, eso ahora es un archivo, no un módulo.

## Entornos

`.conda` (Python 3.12) tiene todo junto: mediapipe 0.10.35, MNN, gazefollower,
eyetrax 0.4 y OpenCV 4.11. `eyetrax` está instalado acá porque el arnés lo
importa, no porque se use su estimador.

**OpenCV tiene que ser 4.x.** La calibración de GazeFollower usa
`cv2.ml.SVM`, que OpenCV 5 removió. El chequeo `cv2.ml.SVM existe` lo verifica
antes de que cueste una sesión entera.

`.conda311` (Python 3.11, mediapipe 0.10.10) conserva la configuración
publicada del paper, con la API legacy que `tasks_alignment.py` reemplaza. Se
mantiene como referencia para comparar el port contra el original.
