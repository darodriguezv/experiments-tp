# Estudio de las tres condiciones

Mide cuánto cuesta decir algo con el tablero en tres configuraciones, con los
mismos participantes en las tres.

    python main.py servir      # el conductor, en el navegador
    python main.py plan --n 12 # el contrabalanceo y la asignación
    python main.py protocolo   # condiciones, caso y conjuntos de frases
    python main.py informe     # el ahorro, por persona y agregado
    python main.py smoke       # chequeos del diseño

## Las tres condiciones

| | tablero | predicción | SLM | contexto del caso |
|-|-|-|-|-|
| Letra a letra | deletreo | no | no | no |
| Sistema sin contexto | palabras + deletreo | no | sí | no |
| Sistema contextualizado | palabras + deletreo | sí | sí | sí |

La línea de base es el deletreo porque es lo que hoy usa mucha gente cuando el
tablero no tiene nada más. Comparar contra una versión degradada de nuestro
propio sistema mediría qué tan bien apagamos una función.

## Lo que el diseño protege

**El orden.** Para la tercera condición la persona lleva veinte minutos
manejando un cursor con los ojos y es bastante más diestra que al empezar. Si
todos hicieran 1-2-3 y la tercera ganara, no habría forma de separar el
contexto de la práctica. Cada participante hace un orden distinto, con las seis
permutaciones: equilibran la posición *y* el arrastre.

Con diez participantes el equilibrio **no es exacto** -seis secuencias piden un
múltiplo de seis- y hay que declararlo en la tesis. Con doce es exacto: son dos
sesiones más.

**Las frases.** Tres conjuntos emparejados en caracteres (108/108/107) que
rotan contra las condiciones, para que nadie llegue a la segunda sabiendo dónde
está cada letra. La primera versión los rotaba por posición en la secuencia y
medido daba que `letra` recibía F0 en 8 de cada 12 casos mientras
`sin_contexto` no lo recibía nunca; ahora rotan por condición.

**Las consignas.** Se pide una intención ("Tiene sed"), nunca la frase. Si se
dictara, el estudio mediría transcripción y la condición con modelo perdería
sentido: no quedaría nada que interpretar.

## Lo que el estudio NO puede afirmar

Los participantes son voluntarios sanos haciendo un rol. Eso **sí** mide el
costo de interacción -permanencias y segundos por enunciado- y **no** mide la
experiencia de quien depende del dispositivo. Un voluntario que se cansa deja
el experimento; alguien con ELA se queda sin voz.

El caso es el mismo para todos, así que los resultados valen **para ese caso**.
Con diez participantes el caso no puede ser una variable.

No se calculan pruebas de significancia: con n=10 cualquier p se leería como
más de lo que es. El dato queda por participante y por tarea para que el
contraste emparejado lo haga quien corresponda.

## Datos

`datos/` está excluido del repositorio. No se guardan nombres -el participante
es un número- pero sí el recorrido completo de una sesión, que con una fecha es
reidentificable para cualquiera que haya estado en la sala. El consentimiento
se firma en papel y se guarda aparte.
