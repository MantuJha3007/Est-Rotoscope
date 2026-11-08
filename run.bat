@echo off
cd /d "%~dp0"

if not exist "venv\Lib\site-packages\fastapi" (
    echo [INFO] First time setup or missing dependencies detected. Running setup.bat...
    call setup.bat
)

echo Starting SAM 2 Unlimited Rotoscoping Studio (NVIDIA RTX GPU)...
call venv\Scripts\activate.bat

:: Launch default browser after 2 seconds
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8000"

"%~dp0venv\Scripts\python.exe" app.py
pause

