@echo off
cd /d "%~dp0"
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo Starting Kronos Server...
python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload
pause
