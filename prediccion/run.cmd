@echo off
rem Reusa el interprete de RealsenseV1: mismas versiones, sin duplicar 2 GB.
"%~dp0..\RealsenseV1\.conda\python.exe" "%~dp0main.py" %*
