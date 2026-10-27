@echo off
setlocal enabledelayedexpansion

title ClassPulse AI - Multimodal Attendance & Engagement Agent

echo =====================================================================
echo           ClassPulse AI - Attendance & Engagement Intelligence
echo    Smart classroom analytics powered by computer vision & AI insights
echo =====================================================================
echo.

:: Check Python installation
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found in your system PATH!
    echo Please install Python 3.10+ from https://www.python.org/ and check "Add to PATH".
    pause
    exit /b 1
)

:: Check for existing virtual environment
if exist .venv\Scripts\activate.bat (
    echo [INFO] Activating virtual environment (.venv)...
    call .venv\Scripts\activate.bat
) else if exist venv\Scripts\activate.bat (
    echo [INFO] Activating virtual environment (venv)...
    call venv\Scripts\activate.bat
) else (
    echo [INFO] No dedicated virtual environment detected. Running with system Python environment.
)

:: Ensure core dependencies
echo [INFO] Verifying core packages...
python -c "import streamlit, pandas, numpy, matplotlib, cv2, scipy" >nul 2>nul
if %errorlevel% neq 0 (
    echo [INFO] Installing required dependencies from requirements.txt...
    python -m pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [WARNING] Failed to install some dependencies. Attempting to start regardless...
    )
) else (
    echo [INFO] All core dependencies verified.
)

:: Set temporary Matplotlib cache directory to avoid permission issues
set MPLCONFIGDIR=%TEMP%\classpulse_mpl

echo.
echo =====================================================================
echo Launching ClassPulse AI Dashboard...
echo Streamlit will open automatically in your browser (http://localhost:8501)
echo Press Ctrl+C in this console window anytime to stop the server.
echo =====================================================================
echo.

python -m streamlit run app.py --server.headless=false

pause
