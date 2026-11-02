"""! @brief Chemins du projet, tous relatifs à la racine du dépôt."""
import os
from pathlib import Path

## Racine du dépôt.
ROOT = Path(__file__).resolve().parent.parent
## Données (ignorées par git) : un sous-dossier par jeu d'images.
DATA_DIR = ROOT / "data"
## Images Kodak, téléchargées automatiquement.
KODAK_DIR = DATA_DIR / "kodak"
## Résultats (ignorés par git).
OUT_DIR = ROOT / "outputs"
## Modèles entraînés par le notebook.
CKPT_DIR = OUT_DIR / "checkpoints"
## Poids pré-entraînés téléchargés.
WEIGHTS_DIR = OUT_DIR / "weights"
## Encodeurs exportés pour ONNX Runtime et Aidge.
ONNX_DIR = OUT_DIR / "onnx"
## Résultats de benchmark, un CSV par campagne.
BENCH_DIR = OUT_DIR / "bench"
## Base de la file d'entraînement (runs.sqlite) et dossiers des jobs ; surchargeable par AIDGE_RUNS_DIR.
RUNS_DIR = Path(os.environ.get("AIDGE_RUNS_DIR", OUT_DIR / "runs"))
## Jeux d'images d'entraînement, un sous-dossier par jeu (exclus des jeux du banc par images.datasets()).
TRAIN_DIR = DATA_DIR / "train"
## Résultats des optimisations Aidge, un sous-dossier par job.
OPTIM_DIR = OUT_DIR / "optim"
## Capacité BRAM du Zynq-7020 retenue : 140 BRAM36, environ 630 Ko, soit environ 615 Kio.
BRAM_BYTES = 630_000
