@echo off
echo Starting ToxPredict Backend Server...
echo.

cd /d "%~dp0"

echo Backend Directory: %CD%
echo.
echo Starting server on http://0.0.0.0:8000
echo API Documentation: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

C:\Users\geeta\Anaconda3\envs\sslgcn_new\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000

pause
