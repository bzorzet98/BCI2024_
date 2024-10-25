-------------- board_config -------------------
Boards IDs:
SYNTHETIC_BOARD: -1
CYTON_BOARD: 0
GANGLION_BOARD: 1
CYTON_DAISY_BOARD: 2
UNICORN_BOARD: 8

port: chequear el nombre correspondiente en la computadora que se use para adquirir

-------------- experiment_config --------------
session_ID: es un ID diferente por cada día de registro diferente. Podemos usar el código 0 para la sesión de calibración y de 1 en adelante para las diferentes sesiones de lazo cerrado que hagamos (o definir otro criterio para estos IDs)
run_ID: es un ID diferente por cada vez que se llama al protocolo de estimulación. En este caso, cada ronda tiene 20 trials, 10 de cada clase. Solemos hacer 5 o 6 rondas de calibración para tener una mínima cantidad de datos para entrenar el modelo.
task: es el nombre del proyecto, que se va a usar para generar una carpeta raíz donde luego se guarden todos los datos generados. No puede tener los símbolos `-`, `_`, o `/` en el nombre.



