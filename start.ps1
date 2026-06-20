$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]*)=(.*)$') {
            $Name = $Matches[1].Trim()
            $Value = $Matches[2].Trim().Trim('"').Trim("'")
            if (-not [Environment]::GetEnvironmentVariable($Name, "Process")) {
                [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
            }
        }
    }
}

$HostName = if ($env:CHESSLAB_HOST) { $env:CHESSLAB_HOST } else { "127.0.0.1" }
$Port = if ($env:CHESSLAB_PORT) { [int]$env:CHESSLAB_PORT } else { 5000 }
$Url = "http://${HostName}:${Port}"

$Running = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($Running) {
    Write-Host "ChessLab já está rodando em $Url" -ForegroundColor Green
    Start-Process $Url
    exit 0
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Preparando o ambiente ChessLab..." -ForegroundColor Green
    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
}

Write-Host "ChessLab AI disponível em $Url" -ForegroundColor Green
Start-Process $Url
& .\.venv\Scripts\python.exe app.py
