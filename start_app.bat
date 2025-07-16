@echo off
echo 🚗 YourCabs Prediction Application
echo ================================
echo.

echo 📦 Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo ✅ Python found
echo.

echo 📦 Installing/Updating requirements...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Failed to install requirements
    pause
    exit /b 1
)

echo ✅ Requirements installed
echo.

echo 🚀 Starting YourCabs Prediction Application...
echo 🌐 Application will open in your default browser
echo 📱 Access URL: http://localhost:8501
echo.
echo ⏹️  Press Ctrl+C to stop the application
echo.

streamlit run src/app_simple.py

echo.
echo 👋 Application stopped. Press any key to exit.
pause >nul
