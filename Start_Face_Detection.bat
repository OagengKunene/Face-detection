@echo off
setlocal
cd /d "%~dp0"
call setup_windows.bat
if errorlevel 1 goto failed
".venv\Scripts\python.exe" run.py models
if errorlevel 1 goto failed
".venv\Scripts\python.exe" run.py gui
if errorlevel 1 goto failed
exit /b 0
:failed
echo.
echo Setup or startup failed. Read the message above.
pause
exit /b 1
