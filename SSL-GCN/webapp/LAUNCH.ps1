# Complete Launch Script - Starts Both Backend and Frontend
# Uses existing sslgcn conda environment and node_modules

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ToxPredict - Quick Launch" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "This will start both backend and frontend servers." -ForegroundColor White
Write-Host ""
Write-Host "You'll need TWO terminal windows:" -ForegroundColor Yellow
Write-Host ""
Write-Host "Terminal 1 (Backend):" -ForegroundColor Cyan
Write-Host "  cd C:\Users\geeta\OneDrive\Attachments\Desktop\SSL-GCN\webapp" -ForegroundColor White
Write-Host "  .\START-BACKEND.ps1" -ForegroundColor White
Write-Host ""
Write-Host "Terminal 2 (Frontend):" -ForegroundColor Cyan
Write-Host "  cd C:\Users\geeta\OneDrive\Attachments\Desktop\SSL-GCN\webapp" -ForegroundColor White
Write-Host "  .\START-FRONTEND.ps1" -ForegroundColor White
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Would you like to start the BACKEND now? (Y/N): " -ForegroundColor Yellow -NoNewline
$response = Read-Host

if ($response -eq "Y" -or $response -eq "y") {
    Write-Host ""
    Write-Host "Starting backend... Open a NEW terminal for frontend!" -ForegroundColor Green
    Write-Host ""
    & "$PSScriptRoot\START-BACKEND.ps1"
} else {
    Write-Host ""
    Write-Host "To start manually:" -ForegroundColor White
    Write-Host "1. Open terminal → Run: .\START-BACKEND.ps1" -ForegroundColor Cyan
    Write-Host "2. Open NEW terminal → Run: .\START-FRONTEND.ps1" -ForegroundColor Cyan
    Write-Host ""
}
