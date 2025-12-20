@echo off
echo.
echo ========================================
echo   CryptoForecast - Starting Backend
echo ========================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo Setting up environment...
    python -m venv .venv
    .venv\Scripts\python.exe -m pip install -q --upgrade pip
    .venv\Scripts\python.exe -m pip install -q -r requirements.txt
    echo Setup complete!
    echo.
)

echo Server accessible at: http://localhost:8000
echo Frontend: Open index.html in browser
echo.

.venv\Scripts\python.exe -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload