# Déploiement continu

Chaque pull request construit l'image CPU, valide les fichiers compose, lance les tests puis vérifie que l'interface et le worker démarrent. Chaque push sur `main` fait la même chose, publie ensuite deux images sur le registre GitHub (`ghcr.io/aidge-in-space-hackathon/aidge-train`, tags `cpu-<sha>` et `gpu-<sha>`, plus `latest` et `gpu`) et les déploie sur le serveur. Le tout est décrit dans [`.github/workflows/ci-cd.yml`](../.github/workflows/ci-cd.yml).

Le serveur cible est la VM `aidge` du homelab (Debian, Docker Compose, RTX 2060 SUPER en passthrough), décrite dans le [cahier des charges](cahier_des_charges.md). Le déploiement passe par un **runner auto-hébergé** installé dans cette VM : c'est le serveur qui va chercher le travail chez GitHub, si bien qu'un serveur derrière une box ou accessible seulement en VPN n'a aucun port à ouvrir. Le runner lance [`scripts/deploy.sh`](../scripts/deploy.sh), qui met les conteneurs à jour avec [`docker-compose.prod.yml`](../docker-compose.prod.yml) (et [`docker-compose.prod.gpu.yml`](../docker-compose.prod.gpu.yml) pour la carte NVIDIA), attend que l'interface réponde, et revient tout seul à la version précédente si la nouvelle ne démarre pas.

Sur le serveur, le code est dans l'image : rien n'est monté depuis un dépôt. La base des entraînements, les données, les résultats et les poids téléchargés vivent dans des volumes nommés (`aidge-train-prod_runs`, `_data`, `_outputs`, `_cache`), conservés d'une version à l'autre.

## Mise en place, une seule fois

### Sur le serveur

Il faut Docker Engine avec le plugin `docker compose` et, pour la variante GPU, le pilote NVIDIA et le NVIDIA Container Toolkit. On crée ensuite un utilisateur dédié au runner, membre du groupe `docker`, et le dossier de déploiement :

```bash
sudo useradd --create-home --groups docker aidge-runner
sudo install -d -o aidge-runner -g aidge-runner /opt/aidge-train
```

Les réglages propres au serveur vont dans `/opt/aidge-train/.env`, jamais dans le dépôt. Par défaut l'interface n'écoute que sur `127.0.0.1`. Pour que Caddy la joigne, on indique l'adresse de la VM sur le réseau local ; l'interface exige alors des comptes (tant que l'identité n'est pas lue depuis Authelia, voir la feuille de route du cahier des charges) :

```bash
AIDGE_UI_BIND=192.168.0.70
AIDGE_UI_USERS=...
AIDGE_MAX_UPLOAD_MB=500
```

Le pare-feu Proxmox de la VM n'accepte le port 7860 que depuis l'adresse de Caddy.

Le runner s'installe depuis GitHub, dans *Settings, Actions, Runners, New self-hosted runner* : la page donne les commandes à exécuter en tant que `aidge-runner`. Au moment du `config.sh`, ajouter l'étiquette `aidge-deploy`, puis l'installer comme service pour qu'il redémarre avec la machine (`sudo ./svc.sh install aidge-runner`, puis `sudo ./svc.sh start`).

### Sur GitHub

Dans *Settings, Secrets and variables, Actions, Variables*, créer :

| Variable | Valeur |
| :- | :- |
| `DEPLOY_ENABLED` | `true` (sans elle, la CI publie les images mais ne déploie pas) |
| `AIDGE_VARIANT` | `gpu` (par défaut) ou `cpu` si le serveur n'a pas de carte NVIDIA |
| `DEPLOY_DIR` | facultatif, `/opt/aidge-train` par défaut |

Le job de déploiement utilise l'environnement `production`, créé automatiquement au premier passage. Dans *Settings, Environments, production*, on peut exiger une validation manuelle avant chaque déploiement et le limiter à la branche `main`.

Aucun secret à créer : le jeton fourni par GitHub à chaque job suffit pour publier puis tirer les images, même si le paquet reste privé.

**Le dépôt est public** : une pull request venue d'un fork peut modifier le workflow pour viser le runner du homelab, qui a les droits Docker sur la VM. Dans les réglages de l'organisation (*Settings, Actions, General*), exiger une approbation pour les workflows de **tous** les contributeurs externes, et n'approuver qu'après avoir lu les modifications de `.github/`.

## Au quotidien

Un merge sur `main` suffit. Le suivi se fait dans l'onglet *Actions* du dépôt ; sur le serveur, `docker compose -p aidge-train-prod ps` et `docker compose -p aidge-train-prod logs -f` montrent l'état des conteneurs.

Un déploiement redémarre le worker : un entraînement en cours est arrêté proprement (90 s de délai pour sauvegarder) et passe au statut « interrompu ». Pour l'instant il faut le relancer depuis l'onglet Suivi ; la reprise automatique au démarrage du worker est prévue (feuille de route du cahier des charges). D'ici là, mieux vaut éviter de merger pendant un long entraînement, ou exiger une validation manuelle sur l'environnement `production`.

Pour revenir à une version précédente sans repasser par la CI, depuis le serveur (l'image de la version précédente est conservée localement) :

```bash
AIDGE_IMAGE=ghcr.io/aidge-in-space-hackathon/aidge-train AIDGE_TAG=gpu-<sha> /opt/aidge-train/deploy.sh
```

Le plus propre reste un `git revert` sur `main`, qui redéploie la version corrigée par le même chemin.
