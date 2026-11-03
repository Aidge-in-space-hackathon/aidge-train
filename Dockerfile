# Image de la plateforme aidge-train (PyTorch CPU par défaut) : banc d'essai, entraînement, worker
# Plateforme (app + worker, image CUDA aidge-in-space:gpu) : docker compose up --build   puis http://localhost:7860
# PC sans carte NVIDIA : utiliser le venv (scripts/setup_venv.ps1), le worker de docker-compose.yml réserve le GPU
# Serveur : images publiées par la CI sur GHCR, voir docs/deploiement.md
FROM python:3.11-slim

# Index PyTorch : version CPU par défaut du Dockerfile (image ~4 fois plus légère que la version CUDA).
# docker-compose.yml le fixe à https://download.pytorch.org/whl/cu126 (image CUDA) pour la plateforme ;
# la CI construit l'image CPU (tests) et les deux variantes publiées sur GHCR.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

# NUMBA_CACHE_DIR : noyaux compilés du codeur arithmétique (aidge_in_space/entropy.py), dans ~/.cache
# (inscriptible par aidge, volume nommé en production) plutôt qu'à côté du code, monté depuis l'hôte en local.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NUMBA_CACHE_DIR=/home/aidge/.cache/numba

# g++ compile et vérifie les exports C de l'onglet Optimisation Aidge (sans CMake : voir aidge_in_space/optim/export.py)
RUN apt-get update \
 && apt-get install -y --no-install-recommends g++ \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install torch --index-url "${TORCH_INDEX}" \
 && pip install -r /tmp/requirements.txt

# Utilisateur non root. Un volume nommé neuf hérite du propriétaire du dossier qu'il recouvre, d'où
# les dossiers créés et attribués ici : /runs (base SQLite et journaux des entraînements), data/ et
# outputs/ (volumes du déploiement serveur), ~/.cache (poids pré-entraînés téléchargés).
RUN useradd --create-home --uid 1000 aidge \
 && mkdir -p /runs /workspace/data /workspace/outputs /home/aidge/.cache \
 && chown -R aidge:aidge /runs /workspace /home/aidge/.cache

# Code copié dans l'image, pour que l'image publiée par la CI se suffise à elle-même.
# En local, les fichiers compose montent le dépôt sur /workspace par-dessus cette copie.
# Placé après l'installation des dépendances : modifier le code ne réinstalle pas PyTorch.
COPY --chown=aidge:aidge . /workspace

USER aidge
WORKDIR /workspace

EXPOSE 7860
# Par défaut : l'interface (les fichiers compose précisent toujours leur commande)
CMD ["python", "-m", "aidge_in_space.app", "--host", "0.0.0.0", "--port", "7860"]
