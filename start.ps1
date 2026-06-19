$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Preparando o ambiente ChessLab..." -ForegroundColor Green
    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
}

Write-Host "ChessLab AI disponível em http://127.0.0.1:5000" -ForegroundColor Green
Start-Process "http://127.0.0.1:5000"
& .\.venv\Scripts\python.exe app.py

