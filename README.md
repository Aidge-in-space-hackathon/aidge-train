# AIDGE IN SPACE 

**Compression d'images par réseau de neurones à bord d'un rover lunaire, déployée sur une cible spatialisable CPU + FPGA avec Aidge et LoCod.**

![Challenge Aidge 2026](https://img.shields.io/badge/Challenge%20Aidge-2026-2ec4b6)
![Cas d'usage](https://img.shields.io/badge/cas%20d%27usage-Viveris%20%C3%97%20CNES-5b8def)
![Cible](https://img.shields.io/badge/cible-Zynq%20PS%20%2B%20PL-ff9f43)
![Statut](https://img.shields.io/badge/statut-d%C3%A9marrage-yellow)
![Licence](https://img.shields.io/badge/licence-MIT-lightgrey)

![Vue d'ensemble : le rover compresse l'image avec un encodeur neuronal, transmet un train binaire réduit, la plateforme reconstruit l'image](Ressources/img/mission_overview.svg)

Un rover lunaire produit beaucoup plus d'images que sa liaison radio ne peut en transmettre. Ce projet embarque à bord un **encodeur neuronal** (le modèle de compression de J. Ballé et al., ICLR 2017) qui transforme chaque image en un petit train binaire, et laisse la plateforme spatiale reconstruire l'image à l'arrivée. Le défi n'est pas seulement de compresser : il faut **prouver** que ça tient sur une carte CPU + FPGA réaliste, **mesurer** ce que ça coûte (mémoire, charge CPU, ressources FPGA) et **optimiser** le modèle comme le découpage entre processeur et logique programmable.


> Ce README est une première version, rédigée au lancement du challenge (octobre 2026). Les sections marquées *proposition* sont des pistes de travail à valider en équipe, pas des décisions prises.

## Sommaire

1. [Démarrage rapide](#démarrage-rapide)
    1. [Option 1 : environnement virtuel Python (venv)](#option-1--environnement-virtuel-python-venv)
    2. [Option 2 : Docker](#option-2--docker)
    3. [Données et résultats](#données-et-résultats)
    4. [Banc d'essai de compression](#banc-dessai-de-compression)
    5. [Plateforme d'entraînement](#plateforme-dentraînement)
2. [Le défi](#le-défi)
3. [Pourquoi un réseau de neurones pour compresser ?](#pourquoi-un-réseau-de-neurones-pour-compresser-)
4. [La chaîne d'outils, d'Aidge à LoCod](#la-chaîne-doutils-daidge-à-locod)
    1. [Aidge](#aidge)
    2. [LoCod](#locod)
5. [Cible matérielle et répartition entre PS et PL](#cible-matérielle-et-répartition-entre-ps-et-pl)
6. [Calendrier officiel du challenge](#calendrier-officiel-du-challenge)
7. [Ressources](#ressources)
    1. [Documents du dépôt](#documents-du-dépôt)
    2. [Liens Aidge](#liens-aidge)
    3. [Liens LoCod](#liens-locod)
    4. [Compression apprise](#compression-apprise)
    5. [Carte](#carte)
8. [Licence](#licence)

## Démarrage rapide

Le point d'entrée du projet est le notebook [`balle2017_end2end.ipynb`](balle2017_end2end.ipynb) : il réimplémente le modèle de Ballé et al. en PyTorch, l'entraîne, l'évalue sur Kodak, prouve la compression avec un vrai codeur arithmétique et exporte l'encodeur en ONNX pour Aidge.

Deux façons de l'exécuter, au choix :

| | venv | Docker |
| :- | :- | :- |
| Prérequis | Python 3.10 ou plus récent | Docker Desktop (Windows, macOS) ou Docker Engine avec `docker compose` (Linux) |
| Installation | une commande, environ 1 Go dans `.venv/` | une commande, image d'environ 2 Go |
| GPU NVIDIA | possible (voir ci-dessous) | non, CPU uniquement |
| Conseillé pour | entraîner le modèle, travailler dans VS Code | lancer le notebook sans rien installer d'autre que Docker |

### Option 1 : environnement virtuel Python (venv)

Depuis la racine du dépôt, le script crée `.venv/` et installe les dépendances de [`requirements.txt`](requirements.txt).

**Windows (PowerShell)**

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_venv.ps1
.venv\Scripts\jupyter lab
```

**Linux / macOS**

```bash
sh scripts/setup_venv.sh
.venv/bin/jupyter lab
```

JupyterLab s'ouvre dans le navigateur. Dans **VS Code**, il suffit d'ouvrir le notebook et de choisir l'interpréteur `.venv` comme noyau.

Relancer le script plus tard ne recrée pas le venv : il met simplement les dépendances à jour.

**GPU NVIDIA (optionnel).** Sous Windows, la version de PyTorch installée par défaut ne sait utiliser que le CPU. Pour entraîner sur GPU, remplacez-la par une version CUDA après l'installation :

```powershell
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
```

Sous Linux, la version par défaut gère déjà CUDA. La première cellule du notebook affiche `device : cuda` quand le GPU est bien détecté.

### Option 2 : Docker

Depuis la racine du dépôt :

```bash
docker compose up --build
```

Ouvrez ensuite <http://localhost:8888> : JupyterLab s'affiche avec le dépôt entier. Le premier build télécharge PyTorch et prend quelques minutes, les lancements suivants sont immédiats. Pour arrêter : `Ctrl+C`, puis `docker compose down`.

Pour exécuter tout le notebook sans interface, par exemple sur une machine distante :

```bash
docker compose run --rm notebook jupyter nbconvert --to notebook --execute balle2017_end2end.ipynb --output-dir outputs
```

Le notebook exécuté, avec ses figures, est écrit dans `outputs/balle2017_end2end.ipynb`.

Bon à savoir :

- le dépôt est **monté** dans le conteneur (`/workspace`) : les notebooks modifiés, `data/` et `outputs/` restent sur votre machine, rien n'est perdu à l'arrêt du conteneur ;
- JupyterLab tourne **sans mot de passe**, mais il n'est joignable que depuis votre propre machine (`127.0.0.1`) ;
- l'image utilise PyTorch **CPU** : parfait pour l'évaluation, l'export ONNX et le profilage, mais un vrai entraînement sera bien plus rapide dans un venv avec GPU ;
- sous Linux, le conteneur écrit avec l'identifiant utilisateur 1000 : si le vôtre est différent (`id -u`), donnez-lui les droits d'écriture sur `data/` et `outputs/`.

### Données et résultats

- `data/kodak/` : les 24 images de test Kodak, **téléchargées automatiquement** par le notebook au premier lancement ;
- `data/train/` : vos images d'entraînement (png, jpg, tif, sous-dossiers acceptés). Si le dossier est vide, le notebook s'entraîne sur Kodak, ce qui suffit à valider la chaîne mais pas à obtenir un bon modèle ;
- `outputs/` : checkpoints du modèle (`outputs/checkpoints/`), fichiers compressés et export ONNX avec ses vecteurs de test (`outputs/export/`).

Ces deux dossiers sont ignorés par Git. Pour un premier essai rapide, passez `MODE_RAPIDE = True` dans la cellule de configuration du notebook.

### Banc d'essai de compression

Le paquet [`aidge_in_space/`](aidge_in_space) compare plusieurs algorithmes et modèles de compression pour justifier ce qui ira sur la carte. Le notebook reste le support pédagogique ; le banc d'essai réutilise son modèle et ses checkpoints.

```powershell
.venv\Scripts\python -m aidge_in_space.app        # interface sur http://127.0.0.1:7860
```

Avec Docker : `docker compose up app`, puis <http://localhost:7860>.

| Onglet | Rôle |
| :- | :- |
| Comparer | une image, plusieurs codecs au même débit visé : reconstructions, zoom, débit, PSNR, MS-SSIM, temps |
| Explorer un codec | un codec à un niveau donné, avec la carte d'erreur |
| Benchmark | campagnes sur un jeu d'images : courbes débit/distorsion, BD-rate, temps d'encodage, CSV |
| Encodeur embarqué | calcul, paramètres et mémoire de la partie embarquée, couche par couche ; vérification PyTorch / ONNX Runtime / Aidge |

Codecs disponibles : JPEG, JPEG 2000 et WebP (références), les modèles Ballé 2017 entraînés par le notebook (détectés dans `outputs/checkpoints/`), Ballé 2018 factorisé et hyperprior avec les poids pré-entraînés de [CompressAI](https://github.com/InterDigitalInc/CompressAI) (téléchargés au premier usage). L'encodeur des modèles appris peut tourner sur PyTorch, ONNX Runtime ou le backend CPU d'Aidge.

**Ajouter un algorithme** : un fichier dans [`aidge_in_space/codecs/`](aidge_in_space/codecs) avec une fonction `codecs()` ; il apparaît seul dans l'interface, la ligne de commande et les tests (exemple dans l'onglet « Codecs et aide »).

Campagnes sans interface, par exemple sur la machine d'entraînement (résultats dans `outputs/bench/`, relus par l'interface) :

```bash
python -m aidge_in_space.bench --list
python -m aidge_in_space.bench --codecs jpeg,jpeg2000,bmshj2018-hyperprior --images 24 --mode rgb --points 6
python -m pytest tests                              # tests de tous les codecs enregistrés
```

**Documentation du code** : générée par [Doxygen](https://www.doxygen.nl) (`doxygen Doxyfile`, puis `docs/html/index.html`) et publiée sur GitHub Pages à chaque push sur `main` ([workflow](.github/workflows/docs.yml)). Les docstrings commencent par `"""!` et utilisent `@brief`, `@param`, `@return`, `@throws` ; les attributs se documentent par un commentaire `##` sur la ligne précédente. Les tests vérifient que chaque fonction et classe en a une.

### Plateforme d'entraînement

Pour entraîner des modèles sans toucher au notebook ni à un terminal, le banc d'essai contient trois onglets supplémentaires : **Données** (déposer des images ou un zip), **Entraînement** (régler et lancer, un job par valeur de λ) et **Suivi** (statuts, courbes, annuler, relancer). Les entraînements passent par une file d'attente et sont exécutés par un programme séparé, le *worker* ; les modèles terminés apparaissent dans le banc d'essai, à côté de JPEG.

Sous Windows, après l'installation du venv, double-cliquez sur `scripts\lancer_plateforme.bat` : il ouvre le worker et l'interface, puis le navigateur sur <http://127.0.0.1:7860>. Avec Docker : `docker compose up app worker` (variante GPU : `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build worker`, pas encore testée). Un entraînement en cours ralentit la machine et fausse les temps mesurés par le banc d'essai.

Mode d'emploi pas à pas : [`docs/guide_interface.md`](docs/guide_interface.md). Conception détaillée : [`docs/conception_entrainement.md`](docs/conception_entrainement.md).

## Le défi

Le projet répond au cas d'usage **AIDGE IN SPACE** proposé par **Viveris** et le **CNES** dans le cadre du [Challenge Aidge 2026](https://eclipse.dev/aidge/), un challenge étudiant sur l'IA embarquée qui se déroule du 1er octobre au 31 décembre 2026.

![Fiche officielle du cas d'usage AIDGE IN SPACE présentée au lancement du challenge](Ressources/img/defi_aidge_in_space.png)

*Fiche du cas d'usage, présentation de lancement du Challenge Aidge (1er octobre 2026).*


Les objectifs fixés par Viveris et le CNES sont les suivants :

| Objectif | Ce que cela implique pour nous |
| :- | :- |
| Mesurer les performances de compression | débit en bit/pixel, qualité PSNR et MS-SSIM, comparaison à JPEG et JPEG 2000 |
| Mesurer l'encombrement mémoire et la charge CPU | instrumentation du code embarqué, rapports mémoire d'Aidge, temps par couche |
| Optimiser le modèle | quantification, réduction du nombre de canaux, tuilage, élagage |
| Optimiser les interconnexions PL / PS du code autogénéré | choix des couches accélérées, volume et fréquence des transferts AXI |
| **Livrable attendu** | **preuve de compression, mesures réalisées, objectif d'optimisation** |



## Pourquoi un réseau de neurones pour compresser ?

Les codecs classiques (JPEG, JPEG 2000) suivent tous le même schéma de *codage par transformée* : une transformée linéaire fixe (DCT, ondelettes), une quantification, puis un codage entropique, chaque étage étant réglé séparément. Dans le papier [END-TO-END OPTIMIZED IMAGE COMPRESSION](https://arxiv.org/pdf/1611.01704), Ballé, Laparra et Simoncelli remplacent la transformée linéaire par une **transformée non linéaire apprise** et optimisent les trois étages **ensemble**, de bout en bout, en minimisant un compromis débit / distorsion `R + λ·D`.

![Comparaison à débit quasi identique : JPEG à 0,121 bit/px, Ballé et al. à 0,113 bit/px, JPEG 2000 à 0,113 bit/px](Ressources/img/balle2017_comparaison_jpeg.png)

*À débit quasi identique (environ 0,11 à 0,12 bit/pixel, soit un taux de compression d'environ 200 pour 1 sur cette image couleur), le modèle appris évite les blocs de JPEG et le flou de JPEG 2000. PSNR mesuré sur la luminance. Source : Ballé et al., 2017, figure 5.*


## La chaîne d'outils, d'Aidge à LoCod

![Chaîne d'outils : entraînement, Aidge, LoCod, carte PYNQ-Z2, mesures, avec une boucle d'optimisation](Ressources/img/toolchain.svg)

Le flot suit celui présenté dans la fiche du défi : un modèle ONNX entre dans Aidge, qui le transforme en code C ; LoCod convertit ensuite les fonctions choisies en accélérateurs matériels et génère toute l'architecture qui les relie au processeur. Les mesures sur carte alimentent une boucle d'optimisation qui remonte jusqu'au choix du modèle.

### Aidge

[Eclipse Aidge](https://eclipse.dev/aidge/) est un framework open source développé par le CEA LIST et hébergé par la fondation Eclipse. Plutôt que d'embarquer un interpréteur générique, il génère du **code C/C++ statique, spécialisé pour le modèle**, sans dépendance à un runtime. Chaque transformation du graphe reste visible et traçable, ce qui correspond bien aux exigences du spatial.

![Flot de déploiement Aidge : interopérabilité ONNX, inférence et optimisation avec un backend, système d'export, code C/C++ autonome, déploiement sur cible](Ressources/img/aidge_workflow_erts2026.png)

*Flot de déploiement Aidge. Source : Moineau et al., ERTS 2026, figure 1 .*



### LoCod

[LoCod](https://github.com/viveris/LoCod) est un outil open source de co-conception matériel / logiciel développé par **Viveris** et le **CNES** pour les SoC qui combinent CPU et FPGA. On écrit l'application en C, on désigne les fonctions à accélérer avec la macro `FPGA()`, et LoCod se charge du reste : conversion C vers VHDL avec [Panda Bambu](https://panda.deib.polimi.it/), génération de l'architecture HDL qui relie les accélérateurs au processeur, puis production du bitstream et de l'exécutable.


## Cible matérielle et répartition entre PS et PL

La carte prêtée par les partenaires n'est pas encore arrivée. D'après la photo de la fiche du défi et les cibles supportées par LoCod, il devrait s'agir d'une **TUL PYNQ-Z2** (à confirmer à réception).


## Calendrier officiel du challenge

| Date | Étape |
| :- | :- |
| 1er octobre 2026 | Réunion de lancement, présentation des cas d'usage et d'Aidge |
| 8 octobre 2026, 14 h | Présentation détaillée du cas d'usage par Viveris |
| 15 octobre et 5 novembre 2026, 14 h | Sessions de questions / réponses sur Aidge |
| Octobre à décembre 2026 | Réalisation du projet |
| **31 décembre 2026** | **Remise des livrables : présentation, vidéo, code** |
| 14 janvier 2027 | Soutenance devant le jury |
| 26 janvier 2027 | Remise des prix aux Assises de l'embarqué, à Bercy |


## Ressources

### Documents du dépôt

Dossier [`Ressources/docPDF`](Ressources/docPDF) :

- J. Ballé, V. Laparra, E. P. Simoncelli, *End-to-end Optimized Image Compression*, ICLR 2017 ([arXiv:1611.01704](https://arxiv.org/abs/1611.01704))
- C. Moineau et al., *Aidge: A comprehensive and traceable library for deploying standalone embedded deep neural networks*, ERTS 2026 ([HAL](https://hal.science/hal-05513843v1/document))
- Plaquette et présentation de lancement du Challenge Aidge

### Liens Aidge

[documentation](https://eclipse.dev/aidge/), [tutoriels](https://eclipse.dev/aidge/source/Tutorial/index.html), [exemples](https://gitlab.eclipse.org/eclipse/aidge/aidge/-/tree/main/examples), [code source](https://gitlab.eclipse.org/eclipse/aidge), [wiki des contributeurs](https://gitlab.eclipse.org/groups/eclipse/aidge/-/wikis/), [tchat Matrix](https://chat.eclipse.org/#/room/#aidge:matrix.eclipse.org)

### Liens LoCod

[dépôt Viveris](https://github.com/viveris/LoCod), [miroir CNES](https://github.com/CNES/LoCod), [présentation au FPGA Developers' Forum du CERN](https://indico.cern.ch/event/1381060/contributions/5923282/)

### Compression apprise

[TensorFlow Compression](https://github.com/tensorflow/compression), [CompressAI](https://github.com/InterDigitalInc/CompressAI)

### Carte

[PYNQ-Z2 (programme universitaire AMD)](https://www.amd.com/en/corporate/university-program/aup-boards/pynq-z2.html)


## Licence

Le code de ce dépôt est publié sous [licence MIT](LICENSE). LoCod est distribué sous GPL-3.0 : on l'utilise comme outil externe, sans copier son code ici, pour ne pas changer la licence du dépôt. Les figures reprises de publications sont créditées sous chacune d'elles ; la fiche du défi provient de la présentation de lancement du Challenge Aidge (CEA, Embedded France, Viveris, CNES, DeepGreen).
