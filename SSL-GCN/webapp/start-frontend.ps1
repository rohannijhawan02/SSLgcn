# Start Frontend Development Server
# Make sure you've run setup.ps1 first!

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Starting ToxPredict Frontend" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$frontendDir = Join-Path $PSScriptRoot "frontend"
Set-Location $frontendDir

# Check if node_modules exists
if (Test-Path "node_modules") {
    Write-Host "Starting Vite development server..." -ForegroundColor Green
    Write-Host ""
    Write-Host "Frontend App: http://localhost:5173" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
    Write-Host ""
    
    npm run dev
} else {
    Write-Host "✗ node_modules not found!" -ForegroundColor Red
    Write-Host "Please run setup.ps1 first" -ForegroundColor Yellow
    exit 1
}
