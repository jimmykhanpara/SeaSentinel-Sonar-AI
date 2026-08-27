@echo off
title AEGIS-SSS Marine Sonar Dashboard
echo ========================================================
echo   Starting AEGIS-SSS Marine Sonar Dashboard Server...
echo ========================================================
cd /d "%~dp0"
.\venv\Scripts\python.exe run_dashboard.py
pause
