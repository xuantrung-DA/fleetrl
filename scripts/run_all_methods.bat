@echo off
setlocal
cd /d "%~dp0.."
start "FleetRL method status" "%~dp0watch_all_methods.bat"
start "FleetRL cleanup" "%~dp0watch_cleanup.bat"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0train_study.ps1"
set "RESULT=%ERRORLEVEL%"
echo.
if "%RESULT%"=="0" (echo Full study finished.) else (echo Study stopped with code %RESULT%. Check the job manifests above.)
pause
exit /b %RESULT%
