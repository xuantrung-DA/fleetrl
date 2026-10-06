@echo off
title FleetRL cleanup monitor
setlocal
cd /d "%~dp0.."
set "PYTHON=%USERPROFILE%\.conda\envs\fleetrl-env\python.exe"
if not exist "%PYTHON%" set "PYTHON=%~dp0..\.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo Cannot find fleetrl-env or the repository virtual environment.
  pause
  exit /b 1
)
echo FleetRL cleanup monitor is visible in this window.
echo Completed jobs: remove unreferenced latest checkpoints only.
echo Best and final checkpoints stay on disk.
echo.
if "%~1"=="" (
  "%PYTHON%" "%~dp0study_cleanup.py" --output "%CD%\runs\study14" --watch --interval 60
) else (
  "%PYTHON%" "%~dp0study_cleanup.py" --output "%CD%\runs\study14" --watch --interval 60 --parent-pid %~1
)
set "RESULT=%ERRORLEVEL%"
echo.
echo Cleanup monitor stopped with code %RESULT%.
pause
exit /b %RESULT%
