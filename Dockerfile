# Image de la plateforme aidge-train (PyTorch CPU par défaut) : banc d'essai, entraînement, worker
# Plateforme (app + worker, image CUDA aidge-in-space:gpu) : docker compose up --build   puis http://localhost:7860
# Plateforme sans carte NVIDIA (image CPU) : docker compose -f docker-compose.yml -f docker-compose.cpu.yml up --build
# Serveur : images publiées par la CI sur GHCR, voir docs/deploiement.md
FROM python:3.11-slim

# Index PyTorch : version CPU par défaut du Dockerfile (image ~4 fois plus légère que la version CUDA).
# docker-compose.yml le fixe à https://download.pytorch.org/whl/cu126 (image CUDA) pour la plateforme ;
# docker-compose.cpu.yml utilise l'index CPU.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

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
