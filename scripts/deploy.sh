#!/usr/bin/env bash
# Déploie une version de la plateforme sur le serveur avec docker compose, et revient seul à la
# version précédente si la nouvelle ne démarre pas.
# Appelé par le job « deploy » de .github/workflows/ci-cd.yml. À la main, depuis le serveur, pour
# revenir à une version donnée (son image est conservée localement) :
#   AIDGE_IMAGE=ghcr.io/aidge-in-space-hackathon/aidge-train AIDGE_TAG=gpu-<sha> /opt/aidge-train/deploy.sh
# Variables : AIDGE_IMAGE et AIDGE_TAG (obligatoires) ; DEPLOY_DIR (par défaut /opt/aidge-train) ;
# AIDGE_VARIANT (gpu ou cpu, déduite par défaut du préfixe du tag).
set -euo pipefail

: "${AIDGE_IMAGE:?AIDGE_IMAGE manquante}"
: "${AIDGE_TAG:?AIDGE_TAG manquante}"
DEPLOY_DIR="${DEPLOY_DIR:-/opt/aidge-train}"
VARIANT="${AIDGE_VARIANT:-${AIDGE_TAG%%-*}}"
WAIT_S="${AIDGE_WAIT_S:-300}"

# Lancé depuis le dépôt (CI) : copie des fichiers vers le dossier de déploiement.
# Lancé depuis le dossier de déploiement (retour arrière manuel) : rien à copier.
script_dir="$(cd "$(dirname "$0")" && pwd)"
install -d "$DEPLOY_DIR"
if [ ! -f "$script_dir/docker-compose.prod.yml" ]; then
  repo_dir="$(dirname "$script_dir")"
  cp "$repo_dir/docker-compose.prod.yml" "$repo_dir/docker-compose.prod.gpu.yml" "$DEPLOY_DIR/"
  install -m 755 "$repo_dir/scripts/deploy.sh" "$DEPLOY_DIR/deploy.sh"
fi
cd "$DEPLOY_DIR"

# Réglages propres au serveur (AIDGE_UI_BIND, AIDGE_UI_USERS...), créés à la main, jamais versionnés
touch .env

files=(-f docker-compose.prod.yml)
case "$VARIANT" in
  gpu) files+=(-f docker-compose.prod.gpu.yml) ;;
  cpu) ;;
  *) echo "Variante inconnue : $VARIANT (attendu : gpu ou cpu)" >&2; exit 1 ;;
esac

prev_tag=""
if [ -f release.env ]; then
  prev_tag="$(sed -n 's/^AIDGE_TAG=//p' release.env)"
  cp release.env release.env.prev
fi
printf 'AIDGE_IMAGE=%s\nAIDGE_TAG=%s\n' "$AIDGE_IMAGE" "$AIDGE_TAG" > release.env

compose() { docker compose "${files[@]}" --env-file .env --env-file release.env "$@"; }

echo "Déploiement de $AIDGE_IMAGE:$AIDGE_TAG (variante $VARIANT) dans $DEPLOY_DIR"
compose config -q

# Tags <variante>-<sha> immuables : on ne tire l'image que si elle manque.
# --wait attend que l'interface réponde à son contrôle de santé.
if ! compose up -d --pull missing --remove-orphans --wait --wait-timeout "$WAIT_S"; then
  echo "La version $AIDGE_TAG ne démarre pas correctement" >&2
  compose logs --tail 200 >&2 || true
  if [ -n "$prev_tag" ] && [ "$prev_tag" != "$AIDGE_TAG" ]; then
    echo "Retour à la version précédente : $prev_tag" >&2
    mv release.env.prev release.env
    compose up -d --pull missing --remove-orphans --wait --wait-timeout "$WAIT_S" || true
  fi
  exit 1
fi
compose ps

# Libère le disque : ne garde que la version en service et la précédente (retour arrière rapide)
docker image ls "$AIDGE_IMAGE" --format '{{.Tag}}' | while read -r tag; do
  case "$tag" in
    "$AIDGE_TAG" | "<none>") continue ;;
  esac
  [ -n "$prev_tag" ] && [ "$tag" = "$prev_tag" ] && continue
  docker image rm "$AIDGE_IMAGE:$tag" > /dev/null 2>&1 || true
done

echo "Version $AIDGE_TAG en service"
