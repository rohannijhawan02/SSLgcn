# ToxPredict Web Application - Quick Start Guide
# Run this script to set up and start both backend and frontend

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ToxPredict Web Application Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$webappRoot = $PSScriptRoot
$backendDir = Join-Path $webappRoot "backend"
$frontendDir = Join-Path $webappRoot "frontend"

# Check if Python is installed
Write-Host "[1/6] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found! Please install Python 3.8 or higher." -ForegroundColor Red
    exit 1
}

# Check if Node.js is installed
Write-Host "[2/6] Checking Node.js installation..." -ForegroundColor Yellow
try {
    $nodeVersion = node --version 2>&1
    Write-Host "✓ Node.js found: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Node.js not found! Please install Node.js 16 or higher." -ForegroundColor Red
    exit 1
}

# Backend Setup
Write-Host "[3/6] Setting up backend..." -ForegroundColor Yellow
Set-Location $backendDir

if (-not (Test-Path "venv")) {
    Write-Host "Creating Python virtual environment..." -ForegroundColor Cyan
    python -m venv venv
}

Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& ".\venv\Scripts\Activate.ps1"

Write-Host "Installing backend dependencies..." -ForegroundColor Cyan
pip install -r requirements.txt --quiet

Write-Host "✓ Backend setup complete!" -ForegroundColor Green

# Frontend Setup
Write-Host "[4/6] Setting up frontend..." -ForegroundColor Yellow
Set-Location $frontendDir

if (-not (Test-Path "node_modules")) {
    Write-Host "Installing frontend dependencies..." -ForegroundColor Cyan
    npm install
} else {
    Write-Host "✓ Frontend dependencies already installed" -ForegroundColor Green
}

Write-Host "✓ Frontend setup complete!" -ForegroundColor Green

# Ready to start
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To start the application:" -ForegroundColor White
Write-Host ""
Write-Host "1. Start Backend (in this terminal):" -ForegroundColor Yellow
Write-Host "   cd $backendDir" -ForegroundColor White
Write-Host "   .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "   uvicorn app:app --reload --host 0.0.0.0 --port 8000" -ForegroundColor White
Write-Host ""
Write-Host "2. Start Frontend (in a NEW terminal):" -ForegroundColor Yellow
Write-Host "   cd $frontendDir" -ForegroundColor White
Write-Host "   npm run dev" -ForegroundColor White
Write-Host ""
Write-Host "Then open your browser to: http://localhost:5173" -ForegroundColor Cyan
Write-Host ""
Write-Host "API Documentation: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""

# Ask if user wants to start backend now
Write-Host "Would you like to start the backend now? (Y/N): " -ForegroundColor Yellow -NoNewline
$response = Read-Host

if ($response -eq "Y" -or $response -eq "y") {
    Write-Host ""
    Write-Host "Starting backend server..." -ForegroundColor Green
    Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
    Write-Host ""
    Set-Location $backendDir
    & ".\venv\Scripts\Activate.ps1"
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
} else {
    Write-Host ""
    Write-Host "Setup complete! Follow the instructions above to start manually." -ForegroundColor Green
}
