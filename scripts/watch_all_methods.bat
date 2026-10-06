@echo off
setlocal
set "PYTHON=%USERPROFILE%\.conda\envs\fleetrl-env\python.exe"
if not exist "%PYTHON%" set "PYTHON=%~dp0..\.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo Cannot find fleetrl-env or the repository virtual environment.
    pause
    exit /b 1
)

cd /d "%~dp0.."
"%PYTHON%" "%~dp0study_status.py" --interval 20 %*
