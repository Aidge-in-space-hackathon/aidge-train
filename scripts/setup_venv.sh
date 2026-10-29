#!/usr/bin/env sh
# Crée l'environnement virtuel .venv à la racine du dépôt et installe les dépendances (Linux / macOS).
# Usage : sh scripts/setup_venv.sh
set -e
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
if [ ! -x .venv/bin/python ]; then
    echo "Création de l'environnement virtuel .venv ..."
    "$PYTHON" -m venv .venv || { echo "Échec : Python 3.10+ et le module venv sont requis (Debian/Ubuntu : apt install python3-venv)"; exit 1; }
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

.venv/bin/python -c "import torch, onnx, onnxruntime; print('OK : torch', torch.__version__, '| CUDA :', torch.cuda.is_available())"
echo ""
echo "Environnement prêt. Pour lancer JupyterLab :"
echo "  .venv/bin/jupyter lab"
