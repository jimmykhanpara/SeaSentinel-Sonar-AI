@echo off
title SeaSentinel AI Sonar Dashboard
color 0A
echo ===================================================================
echo   Starting SeaSentinel AI Marine Debris Mission Dashboard...
echo ===================================================================
cd /d "%~dp0"

REM 1. Try Virtual Environment Python
if exist ".\venv\Scripts\python.exe" (
    echo [*] Using project virtual environment: .\venv\Scripts\python.exe
    ".\venv\Scripts\python.exe" run_dashboard.py
    goto end
)

REM 2. Fallback to System Python in PATH
where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [*] Virtualenv not found. Using system Python...
    python run_dashboard.py
    goto end
)

REM 3. Fallback to Windows Python Launcher
where py >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [*] Using Windows Python Launcher (py)...
    py run_dashboard.py
    goto end
)

echo.
echo [ERROR] Python was not found on your system!
echo Please install Python 3.10+ or make sure .\venv is present.
echo.

:end
echo.
echo ===================================================================
echo   Server stopped or exited. Press any key to close this window.
echo ===================================================================
pause
