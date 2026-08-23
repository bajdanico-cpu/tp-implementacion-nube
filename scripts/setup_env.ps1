<#
.SYNOPSIS
    Crea el entorno virtual del TP y verifica que quede usable.

.DESCRIPTION
    El venv se crea FUERA de la carpeta del proyecto a propósito: el proyecto vive en
    OneDrive, y un venv adentro son ~30.000 archivos que OneDrive intentaría sincronizar.

    Requiere Python 3.14 (todo el requirements.txt está pineado y verificado con 3.14.3).

.EXAMPLE
    .\scripts\setup_env.ps1
    .\scripts\setup_env.ps1 -Recrear
#>
[CmdletBinding()]
param(
    [string]$Ruta = "$env:USERPROFILE\.venvs\tp-premier-ml",
    [switch]$Recrear
)

$ErrorActionPreference = "Stop"
$raiz = Split-Path -Parent $PSScriptRoot

Write-Host "== Entorno del TP Premier ML ==" -ForegroundColor Cyan
Write-Host "  proyecto : $raiz"
Write-Host "  venv     : $Ruta"

if ($Recrear -and (Test-Path $Ruta)) {
    Write-Host "`nBorrando el venv anterior..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $Ruta
}

if (-not (Test-Path $Ruta)) {
    Write-Host "`nCreando el venv con Python 3.14..." -ForegroundColor Cyan
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) { & py -3.14 -m venv $Ruta } else { & python -m venv $Ruta }
} else {
    Write-Host "`nEl venv ya existe; se reutiliza (usá -Recrear para rehacerlo)."
}

$python = Join-Path $Ruta "Scripts\python.exe"
if (-not (Test-Path $python)) { throw "No se creó el venv en $Ruta" }

Write-Host "`nInstalando dependencias..." -ForegroundColor Cyan
& $python -m pip install --upgrade pip --quiet
& $python -m pip install -r (Join-Path $raiz "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Falló la instalación de requirements.txt" }

Write-Host "`n== Verificación ==" -ForegroundColor Cyan
Push-Location $raiz
try {
    & $python -m common.config
    Write-Host ""
    & $python -m training.device
} finally {
    Pop-Location
}

Write-Host "`n== Listo ==" -ForegroundColor Green
Write-Host "Activá el entorno con:"
Write-Host "    $Ruta\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "`nY después, desde cero:"
Write-Host "    python -m ingestion.run        # baja Bronze (~27 MB, sin credenciales)"
Write-Host "    python -m transform.silver     # arma las 4 tablas Silver"
Write-Host "    python -m features.gold_tp     # arma Gold (1.520 x 165)"
Write-Host "    python -m training.run --todos # entrena y evalúa"
Write-Host "    pytest                         # la suite completa"
