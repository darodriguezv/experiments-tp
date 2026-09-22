@echo off
rem El tablero por mirada necesita el entorno de Eyetrax (mediapipe + eyetrax).
"%~dp0..yetrax\.conda\python.exe" "%~dp0main.py" %*
