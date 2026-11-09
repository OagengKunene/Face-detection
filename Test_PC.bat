@echo off
setlocal
cd /d "%~dp0"
set "BIO_CAMERA=0"
if not "%~1"=="" set "BIO_CAMERA=%~1"
call setup_windows.bat
if errorlevel 1 goto failed
".venv\Scripts\python.exe" run.py models
if errorlevel 1 goto failed
".venv\Scripts\python.exe" run.py self-test
if errorlevel 1 goto failed
".venv\Scripts\python.exe" run.py doctor --camera "%BIO_CAMERA%" --output pc_test_report.json
if errorlevel 1 goto failed
echo.
echo PC checks passed. Open Start_Face_Detection.bat to enrol and verify your face.
echo Diagnostic report: pc_test_report.json
pause
exit /b 0
:failed
echo.
echo A check failed. Read the message above and check pc_test_report.json if created.
echo If the webcam is camera 1, run: Test_PC.bat 1
pause
exit /b 1
