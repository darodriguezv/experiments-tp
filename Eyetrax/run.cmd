@echo off
rem Forward to the project env interpreter so no activation is needed.
"%~dp0.conda\python.exe" "%~dp0main.py" %*
