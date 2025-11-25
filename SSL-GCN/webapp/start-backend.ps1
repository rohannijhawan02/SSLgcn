# Start Backend Server
# Uses sslgcn_new conda environment

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Starting ToxPredict Backend" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$backendDir = Join-Path $PSScriptRoot "backend"

Write-Host "Backend Directory: $backendDir" -ForegroundColor Yellow
Write-Host ""
Write-Host "Starting FastAPI server..." -ForegroundColor Green
Write-Host ""
Write-Host "Backend API: http://localhost:8000" -ForegroundColor Cyan
Write-Host "API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host "KEEP THIS WINDOW OPEN while using the app!" -ForegroundColor Yellow
Write-Host ""

# Start using batch file for reliability
Start-Process -FilePath "$backendDir\start.bat" -WorkingDirectory $backendDir -Wait

Write-Host ""
Write-Host "Backend server stopped." -ForegroundColor Red
