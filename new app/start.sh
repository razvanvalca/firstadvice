#!/bin/bash
# Swiss Life Voice Insurance Advisor - Unix/Linux/macOS Startup Script
# =====================================================================

set -e

echo ""
echo "========================================"
echo "  Swiss Life Voice Insurance Advisor"
echo "  Starting application..."
echo "========================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "[WARNING] .env file not found!"
    echo "[INFO] Creating .env from .env.example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "[INFO] Please edit .env and add your API keys."
        echo ""
        exit 1
    else
        echo "[ERROR] .env.example not found!"
        exit 1
    fi
fi

# Check if virtual environment exists
if [ -f "venv/bin/activate" ]; then
    echo "[INFO] Activating virtual environment..."
    source venv/bin/activate
else
    echo "[INFO] No virtual environment found."
    echo "[INFO] Consider creating one with: python3 -m venv venv"
    echo ""
fi

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 not found!"
    echo "[INFO] Please install Python 3.10 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "[INFO] Python version: $PYTHON_VERSION"

# Install dependencies if needed
if ! python3 -c "import aiohttp" &> /dev/null; then
    echo "[INFO] Installing dependencies..."
    pip install -r requirements.txt
    echo ""
fi

# Start the application
echo "[INFO] Starting server..."
echo "[INFO] Press Ctrl+C to stop"
echo ""

python3 main.py
