@echo off
setlocal enabledelayedexpansion

echo ======================================================
echo   SAM 2 UNLIMITED ROTOSCOPING STUDIO SETUP
echo   Target: NVIDIA RTX 4060 GPU with CUDA Acceleration
echo ======================================================
echo.

cd /d "%~dp0"

:: Check Python
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python was not found in your PATH!
    echo Please install Python 3.10-3.12 or 3.13 from python.org and check "Add Python to PATH".
    pause
    exit /b 1
)

echo [1/5] Creating dedicated Python virtual environment (venv)...
if not exist "venv" (
    python -m venv venv
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create venv!
        pause
        exit /b 1
    )
)
echo Virtual environment ready.
echo.

echo [2/5] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/5] Installing PyTorch with CUDA acceleration (NVIDIA RTX)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
if %ERRORLEVEL% neq 0 (
    echo Retrying with standard PyTorch wheels...
    pip install torch torchvision
)
echo.

echo [4/5] Installing application requirements and SAM 2...
pip install -r requirements.txt
set SAM2_BUILD_CUDA=0
pip install git+https://github.com/facebookresearch/sam2.git
if %ERRORLEVEL% neq 0 (
    echo Fallback: installing with no build isolation...
    pip install git+https://github.com/facebookresearch/sam2.git --no-build-isolation
)
echo.

echo [5/5] Checking SAM 2.1 Model Checkpoint...
if not exist "models" mkdir "models"
if exist "C:\Users\mantu\AppData\Local\kvn_rotoscope\weights\sam2.1_hiera_base_plus.pt" (
    echo Copying existing SAM 2.1 weights from system cache...
    copy "C:\Users\mantu\AppData\Local\kvn_rotoscope\weights\sam2.1_hiera_base_plus.pt" "models\sam2.1_hiera_base_plus.pt" >nul
) else (
    if not exist "models\sam2.1_hiera_base_plus.pt" (
        echo Downloading SAM 2.1 Base+ checkpoint (~320MB)...
        curl -L -o "models\sam2.1_hiera_base_plus.pt" "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
    )
)

echo.
echo ======================================================
echo   SETUP COMPLETE!
echo   You can now start the studio by running run.bat
echo ======================================================
pause

