@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto install
where py >nul 2>nul
if not errorlevel 1 (
    py -3 -c "import sys; assert (3,12) <= sys.version_info[:2] < (3,15), 'Install Python 3.12, 3.13 or 3.14 (64-bit)'"
    if errorlevel 1 exit /b 1
    py -3 -m venv .venv
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo Install 64-bit Python 3.12, 3.13 or 3.14 from https://www.python.org/downloads/windows/
        echo Include Tcl/Tk and pip in the Python installer.
        exit /b 1
    )
    python -c "import sys; assert (3,12) <= sys.version_info[:2] < (3,15), 'Python 3.12-3.14 required'"
    if errorlevel 1 exit /b 1
    python -m venv .venv
)
if errorlevel 1 exit /b 1
:install
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 exit /b 1
exit /b 0
