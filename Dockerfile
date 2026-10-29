# Image du projet AIDGE IN SPACE (PyTorch CPU par défaut) : JupyterLab, banc d'essai, worker
# Plateforme (app + worker, image CUDA aidge-in-space:gpu) : docker compose up --build   puis http://localhost:7860
# Plateforme sans carte NVIDIA (image CPU) : docker compose -f docker-compose.yml -f docker-compose.cpu.yml up --build
# JupyterLab (image CPU) : docker compose -f docker-compose.notebook.yml up --build   puis http://localhost:8888
FROM python:3.11-slim

# Index PyTorch : version CPU par défaut du Dockerfile (image ~4 fois plus légère que la version CUDA).
# docker-compose.yml le fixe à https://download.pytorch.org/whl/cu126 (image CUDA) pour la plateforme ;
# docker-compose.cpu.yml et docker-compose.notebook.yml utilisent l'index CPU.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt /tmp/requirements.txt
RUN pip install torch --index-url "${TORCH_INDEX}" \
 && pip install -r /tmp/requirements.txt

# Utilisateur non root ; le dépôt est monté dans /workspace par les fichiers compose.
# /runs (base SQLite et journaux des entraînements) est un volume nommé : un volume neuf hérite
# du propriétaire de ce dossier, d'où le chown (sinon /runs appartient à root et reste inscriptible
# par root seul).
RUN useradd --create-home --uid 1000 aidge \
 && mkdir -p /runs && chown aidge:aidge /runs
USER aidge
WORKDIR /workspace

EXPOSE 8888 7860
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", \
     "--IdentityProvider.token=", "--ServerApp.root_dir=/workspace"]
