@echo off
rem El tablero por mirada necesita el entorno de Eyetrax (mediapipe + eyetrax),
rem no el de RealsenseV1 que usa run.cmd para el resto del modulo.
"%~dp0..\Eyetrax\.conda\python.exe" "%~dp0main.py" %*
