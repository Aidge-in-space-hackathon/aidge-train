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
| Optimisation Aidge en ligne de commande | `python -m aidge_in_space.optimize --config configs/optim/mixte_kodak.yaml` |
| Cette documentation | `doxygen Doxyfile` puis `docs/html/index.html` |

## Organisation

| Module | Rôle |
| :- | :- |
| `aidge_in_space.codecs` | Registre des codecs : tout module du dossier qui définit `codecs()` est découvert automatiquement |
| `aidge_in_space.codecs.base` | `Codec`, `Level`, `Result` et `LearnedCodec` (padding, moteur de l'encodeur, débit réel ou estimé, décodage) |
| `aidge_in_space.codecs.classical` | JPEG, JPEG 2000 et WebP via Pillow |
| `aidge_in_space.codecs.balle2017` | Modèles Ballé 2017 entraînés par le notebook (`outputs/checkpoints`) |
| `aidge_in_space.codecs.bmshj2018` | Ballé 2018 factorisé et hyperprior, poids pré-entraînés de CompressAI |
| `aidge_in_space.codecs.aidge_variants` | Variantes optimisées par Aidge : encodeur Aidge par tuiles, décodeur du checkpoint source (`outputs/optim`) |
| `aidge_in_space.coding` | Codage entropique des latents, une classe par modèle d'entropie (`LatentCoding`) |
| `aidge_in_space.entropy` | Codeur arithmétique et tables de fréquences |
| `aidge_in_space.nn` | Architectures, GDN, modèles d'entropie |
| `aidge_in_space.backends` | Exécution de l'encodeur sur PyTorch, ONNX Runtime ou Aidge |
| `aidge_in_space.profiling` | Calcul, paramètres et mémoire de la partie embarquée, couche par couche |
| `aidge_in_space.metrics` | PSNR, MS-SSIM, BD-rate |
| `aidge_in_space.bench` | Campagnes de mesures (CSV dans `outputs/bench`), agrégation, figures |
| `aidge_in_space.app` | Interface Gradio |
| `aidge_in_space.ui_training` | Onglets 1. Entraîner et 2. Suivre : fonctions pures (jeux d'images, jobs, courbes) et adaptateurs Gradio |
| `aidge_in_space.dataset_catalog` | Catalogue des jeux d'images téléchargeables : adresses, empreintes, licences |
| `aidge_in_space.jobs` | `JobStore`, la file d'attente SQLite : jobs, métriques, battements de cœur des workers, statuts |
| `aidge_in_space.worker` | Prend les jobs un par un et lance chacun dans un sous-processus `train` |
| `aidge_in_space.train` | Ligne de commande d'entraînement (`--config` ou `--job-id`) et codes de retour |
| `aidge_in_space.training` | Paquet d'entraînement : `config` (`TrainConfig`, préréglages), `loop` (boucle, checkpoints, reprise), `imaging` (conversion 8 bits) |
| `aidge_in_space.ui_optim` | Onglet 5. Optimisation Aidge |
| `aidge_in_space.optimize` | Ligne de commande d'optimisation (`--config`, `--job-id`, `--analyze`), mêmes codes de retour que `train` |
| `aidge_in_space.optim` | Paquet d'optimisation : `config`, `graph`, `ptq`, `memory`, `tiling`, `evaluate`, `export` (C++ d'Aidge), `export_c` (C pur pour LoCod), `pipeline`, `results` |

## Plateforme d'entraînement

L'interface n'entraîne jamais elle-même. Elle écrit des jobs dans la file (`aidge_in_space.jobs.JobStore`,
base `outputs/runs/runs.sqlite`, ou le volume `runs` sous Docker). Le worker (`aidge_in_space.worker`) prend
les jobs un par un et lance `python -m aidge_in_space.train --job-id N` dans un sous-processus. Ce processus
écrit les métriques dans la base et un checkpoint `bls2017_c*_N*_lambda*_run<N>.pt` dans `outputs/checkpoints`.
Le statut final du job est déduit par le worker du code de retour (0 terminé, 1 échec, 2 configuration invalide,
3 annulé, 4 arrêt demandé donc interrompu et reprenable). Au rechargement, `aidge_in_space.codecs.balle2017`
regroupe les checkpoints en une série par architecture et par nom de run.

Mode d'emploi pour l'équipe : `docs/guide_interface.md`. Conception détaillée : `docs/conception_entrainement.md`.

## Optimisation Aidge

Un job de type `optim` passe par la même file et le même worker, qui lance `python -m aidge_in_space.optimize --job-id N`.
Trois recettes : `fp32` (référence), `int8` (tentative sur le graphe entier, détectée sans effet en Aidge 0.10.1 car `Sqrt`
n'est pas quantifiable) et `mixte` (convolutions en INT8, GDN en flottant). Le pipeline écrit `outputs/optim/<id>/report.json`,
deux exports vérifiés (C++ d'Aidge, et un `main.c` unique pour LoCod), puis `aidge_in_space.codecs.aidge_variants` expose les
variantes comme des codecs. Une variante se reconstruit depuis sa recette (checkpoint, configuration, `calib.npz`).

Conception, résultats du spike et axe de performance : `docs/conception_optimisation.md`. Spike LoCod et Bambu : `locod/spike/`.

## Ajouter un algorithme

1. Créer un fichier dans `aidge_in_space/codecs/` avec une fonction `codecs()` qui renvoie des instances de `Codec`.
2. Implémenter `levels(mode)` (du plus bas au plus haut débit) et `run(img, level, mode, backend, real_rate)`.
3. Pour un réseau, hériter de `LearnedCodec` et fournir `make_model()` et `load_weights()` ; un nouveau modèle
   d'entropie se branche par l'attribut `coding` (sous-classe de `LatentCoding`).

Le codec apparaît alors dans l'interface, la ligne de commande et les tests.

## Conventions de documentation

Docstrings Doxygen en Python : elles commencent par `"""!` et utilisent `@brief`, `@param`, `@return`,
`@throws`. Les constantes et attributs se documentent par un commentaire `##` sur la ligne précédente.
