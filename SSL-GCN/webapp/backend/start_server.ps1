# Start Backend Server
# Simple startup script

Write-Host "Starting ToxPredict Backend..." -ForegroundColor Cyan

# Change to backend directory
$backendDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $backendDir

Write-Host "Backend directory: $backendDir" -ForegroundColor Yellow
Write-Host "Starting uvicorn..." -ForegroundColor Green
Write-Host "Backend will be available at: http://127.0.0.1:8000" -ForegroundColor Cyan
Write-Host "API Docs at: http://127.0.0.1:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Start uvicorn directly with the sslgcn_new Python
try {
    & "C:\Users\geeta\Anaconda3\envs\sslgcn_new\python.exe" -m uvicorn app:app --host 0.0.0.0 --port 8000
} catch {
    Write-Host "Error starting server: $_" -ForegroundColor Red
}

Write-Host "Server stopped." -ForegroundColor Red
