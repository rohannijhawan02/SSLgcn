@echo off
title ToxPredict - Complete Launcher
color 0B

echo ========================================
echo   ToxPredict Web Application Launcher
echo ========================================
echo.
echo This will start BOTH backend and frontend servers.
echo.
echo WARNING: You need to keep BOTH windows open!
echo.
pause

echo.
echo [1/2] Starting Backend Server...
echo.

start "ToxPredict Backend" cmd /k "cd /d "%~dp0backend" && start.bat"

timeout /t 5 /nobreak >nul

echo.
echo [2/2] Starting Frontend Server...
echo.

start "ToxPredict Frontend" cmd /k "cd /d "%~dp0frontend" && npm run dev"

echo.
echo ========================================
echo   Both servers are starting!
echo ========================================
echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:3000
echo.
echo Two new windows have opened:
echo   - Backend Server (keep open)
echo   - Frontend Server (keep open)
echo.
echo Your browser should open automatically.
echo If not, go to: http://localhost:3000
echo.
echo To stop: Close the Backend and Frontend windows
echo          or press Ctrl+C in each window
echo.
echo ========================================
echo.

timeout /t 3 /nobreak >nul

echo Opening browser...
start http://localhost:3000

echo.
echo Done! You can minimize this window.
echo.
pause
