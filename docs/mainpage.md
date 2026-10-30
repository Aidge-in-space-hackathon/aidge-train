# Banc d'essai AIDGE IN SPACE

Documentation du paquet Python `aidge_in_space`, le banc d'essai qui compare des algorithmes et des
modèles de compression d'image pour choisir ce qui sera porté sur la carte (chaîne ONNX → Aidge → LoCod).
Le contexte du projet et l'installation sont décrits dans le README du dépôt.

## Lancer

| Usage | Commande |
| :- | :- |
| Interface (http://127.0.0.1:7860) | `python -m aidge_in_space.app` |
| Liste des codecs, jeux d'images et moteurs | `python -m aidge_in_space.bench --list` |
| Campagne sans interface | `python -m aidge_in_space.bench --codecs jpeg,bmshj2018-hyperprior --images 24 --mode rgb` |
| Tests | `python -m pytest tests` |
| Plateforme d'entraînement : worker | `python -m aidge_in_space.worker` |
| Entraînement en ligne de commande | `python -m aidge_in_space.train --config mon_essai.yaml` |
| Cette documentation | `doxygen Doxyfile` puis `docs/html/index.html` |

## Organisation

| Module | Rôle |
| :- | :- |
| `aidge_in_space.codecs` | Registre des codecs : tout module du dossier qui définit `codecs()` est découvert automatiquement |
| `aidge_in_space.codecs.base` | `Codec`, `Level`, `Result` et `LearnedCodec` (padding, moteur de l'encodeur, débit réel ou estimé, décodage) |
| `aidge_in_space.codecs.classical` | JPEG, JPEG 2000 et WebP via Pillow |
| `aidge_in_space.codecs.balle2017` | Modèles Ballé 2017 entraînés par le notebook (`outputs/checkpoints`) |
| `aidge_in_space.codecs.bmshj2018` | Ballé 2018 factorisé et hyperprior, poids pré-entraînés de CompressAI |
| `aidge_in_space.coding` | Codage entropique des latents, une classe par modèle d'entropie (`LatentCoding`) |
| `aidge_in_space.entropy` | Codeur arithmétique et tables de fréquences |
| `aidge_in_space.nn` | Architectures, GDN, modèles d'entropie |
| `aidge_in_space.backends` | Exécution de l'encodeur sur PyTorch, ONNX Runtime ou Aidge |
| `aidge_in_space.profiling` | Calcul, paramètres et mémoire de la partie embarquée, couche par couche |
| `aidge_in_space.metrics` | PSNR, MS-SSIM, BD-rate |
| `aidge_in_space.bench` | Campagnes de mesures (CSV dans `outputs/bench`), agrégation, figures |
| `aidge_in_space.app` | Interface Gradio |
| `aidge_in_space.ui_training` | Onglets Données, Entraînement et Suivi : fonctions pures (jeux d'images, jobs, courbes) et adaptateurs Gradio |
| `aidge_in_space.jobs` | `JobStore`, la file d'attente SQLite : jobs, métriques, battements de cœur des workers, statuts |
| `aidge_in_space.worker` | Prend les jobs un par un et lance chacun dans un sous-processus `train` |
| `aidge_in_space.train` | Ligne de commande d'entraînement (`--config` ou `--job-id`) et codes de retour |
| `aidge_in_space.training` | Paquet d'entraînement : `config` (`TrainConfig`, préréglages), `loop` (boucle, checkpoints, reprise), `imaging` (conversion 8 bits) |

## Plateforme d'entraînement

L'interface n'entraîne jamais elle-même. Elle écrit des jobs dans la file (`aidge_in_space.jobs.JobStore`,
base `outputs/runs/runs.sqlite`, ou le volume `runs` sous Docker). Le worker (`aidge_in_space.worker`) prend
les jobs un par un et lance `python -m aidge_in_space.train --job-id N` dans un sous-processus. Ce processus
écrit les métriques dans la base et un checkpoint `bls2017_c*_N*_lambda*_run<N>.pt` dans `outputs/checkpoints`.
Le statut final du job est déduit par le worker du code de retour (0 terminé, 1 échec, 2 configuration invalide,
3 annulé, 4 arrêt demandé donc interrompu et reprenable). Au rechargement, `aidge_in_space.codecs.balle2017`
regroupe les checkpoints en une série par architecture et par nom de run.

Mode d'emploi pour l'équipe : `docs/guide_interface.md`. Conception détaillée : `docs/conception_entrainement.md`.

## Ajouter un algorithme

1. Créer un fichier dans `aidge_in_space/codecs/` avec une fonction `codecs()` qui renvoie des instances de `Codec`.
2. Implémenter `levels(mode)` (du plus bas au plus haut débit) et `run(img, level, mode, backend, real_rate)`.
3. Pour un réseau, hériter de `LearnedCodec` et fournir `make_model()` et `load_weights()` ; un nouveau modèle
   d'entropie se branche par l'attribut `coding` (sous-classe de `LatentCoding`).

Le codec apparaît alors dans l'interface, la ligne de commande et les tests.

## Conventions de documentation

Docstrings Doxygen en Python : elles commencent par `"""!` et utilisent `@brief`, `@param`, `@return`,
`@throws`. Les constantes et attributs se documentent par un commentaire `##` sur la ligne précédente.
