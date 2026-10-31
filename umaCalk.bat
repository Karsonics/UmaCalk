@echo off
rem umaCalk launcher - keeps the console open on crash so errors are readable.
setlocal
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
py -3.13 -m umacalk.ui %*
if errorlevel 1 (
    echo.
    echo umaCalk exited with an error. Press any key to close.
    pause >nul
)
