@echo off
setlocal

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0watch_training.ps1"

echo.
echo Training monitor stopped. Press any key to close this window.
pause >nul
