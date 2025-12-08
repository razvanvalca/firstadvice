@echo off
REM Swiss Life Voice Insurance Advisor - Windows Startup Script
REM ============================================================

echo.
echo ========================================
echo  Swiss Life Voice Insurance Advisor
echo  Starting application...
echo ========================================
echo.

REM Check if .env file exists
if not exist ".env" (
    echo [WARNING] .env file not found!
    echo [INFO] Creating .env from .env.example...
    if exist ".env.example" (
        copy .env.example .env
        echo [INFO] Please edit .env and add your API keys.
        echo.
        pause
        exit /b 1
    ) else (
        echo [ERROR] .env.example not found!
        pause
        exit /b 1
    )
)

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo [INFO] No virtual environment found.
    echo [INFO] Consider creating one with: python -m venv venv
    echo.
)

REM Check Python version
python --version 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found in PATH!
    echo [INFO] Please install Python 3.10 or higher.
    pause
    exit /b 1
)

REM Install dependencies if needed
if not exist "venv\Lib\site-packages\aiohttp" (
    echo [INFO] Installing dependencies...
    pip install -r requirements.txt
    echo.
)

REM Start the application
echo [INFO] Starting server...
echo [INFO] Press Ctrl+C to stop
echo.

python main.py

pause
