# Crée l'environnement virtuel .venv à la racine du dépôt et installe les dépendances (Windows).
# Usage : powershell -ExecutionPolicy Bypass -File scripts\setup_venv.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Création de l'environnement virtuel .venv ..."
    if (Get-Command py -ErrorAction SilentlyContinue) { py -3 -m venv .venv } else { python -m venv .venv }
    if ($LASTEXITCODE -ne 0) { throw "Échec de la création du venv (Python 3.10+ requis)" }
}

$py = ".venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Échec de la mise à jour de pip" }
& $py -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Échec de l'installation des dépendances" }

& $py -c "import torch, onnx, onnxruntime; print('OK : torch', torch.__version__, '| CUDA :', torch.cuda.is_available())"
Write-Host ""
Write-Host "Environnement prêt. Pour lancer JupyterLab :"
Write-Host "  .venv\Scripts\jupyter lab"
