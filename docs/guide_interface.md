# Guide de l'interface d'entraînement

Ce guide s'adresse à toute personne de l'équipe, même sans avoir jamais ouvert de terminal. L'interface suit cinq étapes, une par onglet : **1. Entraîner**, **2. Suivre**, **3. Tester**, **4. Valider** et **5. Optimisation Aidge**.

## Sommaire

1. [Démarrer la plateforme](#1-démarrer-la-plateforme)
2. [Ajouter des images](#2-ajouter-des-images)
3. [Lancer un entraînement (onglet 1. Entraîner)](#3-lancer-un-entraînement-onglet-1-entraîner)
4. [Suivre l'entraînement (onglet 2. Suivre)](#4-suivre-lentraînement-onglet-2-suivre)
5. [Annuler, relancer, dupliquer](#5-annuler-relancer-dupliquer)
6. [Tester et valider le modèle (onglets 3 et 4)](#6-tester-et-valider-le-modèle-onglets-3-et-4)
7. [Préparer le modèle pour la carte (onglet 5. Optimisation Aidge)](#7-préparer-le-modèle-pour-la-carte-onglet-5-optimisation-aidge)
8. [Quand quelque chose ne va pas](#8-quand-quelque-chose-ne-va-pas)
9. [Bon à savoir](#9-bon-à-savoir)

## 1. Démarrer la plateforme

La plateforme se compose de deux programmes qui tournent en même temps :

- l'**interface**, la page web que vous utilisez ;
- le **worker**, le programme qui fait réellement les calculs d'entraînement. L'interface ne calcule rien : elle dépose les demandes dans une file d'attente, et le worker les prend une par une.

### Sous Windows (cas courant)

1. Une seule fois sur votre ordinateur : si le dossier `.venv` n'existe pas à la racine du projet, faites un clic droit sur `scripts\setup_venv.ps1`, puis « Exécuter avec PowerShell », et attendez la fin. Si vous oubliez cette étape, le lanceur vous le rappellera.
2. Double-cliquez sur `scripts\lancer_plateforme.bat`.
3. Deux fenêtres noires s'ouvrent : « AIDGE - worker » et « AIDGE - interface ». **Laissez-les ouvertes** : les fermer arrête la plateforme.
4. Le lanceur attend que l'interface réponde (jusqu'à 2 minutes), puis ouvre le navigateur sur <http://127.0.0.1:7860>. Si vous voyez la page avec les onglets, c'est prêt.

Pour arrêter la plateforme, fermez les deux fenêtres noires. Un entraînement en cours est alors sauvegardé puis interrompu ; vous pourrez le relancer plus tard (voir [section 5](#5-annuler-relancer-dupliquer)).

**Si la page ne s'ouvre pas, ou affiche autre chose que l'interface AIDGE.** Un autre programme occupe peut-être déjà le port 7860. Dans ce cas, la fenêtre « AIDGE - interface » affiche une erreur de port, et le navigateur peut ouvrir l'autre programme. Fermez cet autre programme, ou l'ancienne fenêtre de la plateforme restée ouverte, puis relancez `lancer_plateforme.bat`.

### Avec Docker

Demandez à la personne qui gère Docker, ou lancez dans un terminal, depuis la racine du projet :

```
docker compose up --build app worker
```

puis ouvrez <http://localhost:7860>. Ne mélangez pas les deux méthodes : l'interface lancée avec le `.bat` et le worker lancé avec Docker n'utilisent pas la même file d'attente, et les entraînements resteraient en attente pour toujours.

### Se connecter

Par défaut, l'interface n'est accessible que depuis votre propre ordinateur et ne demande aucun identifiant. Si l'équipe a ouvert la plateforme sur le réseau, la page demande un identifiant et un mot de passe : ils vous sont donnés par la personne qui administre la machine.

## 2. Ajouter des images

Les images servent d'exemples au modèle pour apprendre à compresser. Plus elles ressemblent à ce que le rover verra, meilleur sera le modèle. Tout se passe dans l'onglet **1. Entraîner**, dans la section **Ajouter des images**, sous le menu « Jeu d'images ». Cette section est déjà ouverte tant qu'aucun jeu n'existe.

### Télécharger un jeu prêt à l'emploi

Pas d'images sous la main ? Choisissez un jeu dans « Télécharger un jeu prêt à l'emploi » puis cliquez sur **Télécharger**. Il est ensuite sélectionné dans le menu « Jeu d'images ».

| Jeu | Contenu | Images | Téléchargement | Licence |
| :- | :- | -: | -: | :- |
| `uc-merced` | vues aériennes proches d'images satellite, 256 px | 2 100 | 332 Mo | usage recherche, citer Yang et Newsam (2010) |
| `mars-curiosity` | sol martien vu par le rover Curiosity, environ 256 px | 6 737 | 61 Mo | CC BY-SA 4.0 |
| `lune-analogue-dfki` | terrain d'essai qui imite le sol lunaire (pas de vraies images de la Lune), 640 px | 3 198 | 72 Mo | CC BY 4.0 |
| `imagewoof` | chiens de 10 races, 320 px sur le petit côté | 12 954 | 328 Mo | conditions d'ImageNet : recherche non commerciale |
| `div2k-valid` | photos variées en haute définition, environ 2040 px | 100 | 449 Mo | recherche académique uniquement |

Ce qu'il faut savoir :

- **Durée** : de quelques minutes à plus selon la connexion. La barre de progression indique l'étape en cours. Un seul téléchargement se fait à la fois : les suivants attendent leur tour.
- **Déjà présent** : si le dossier du jeu contient déjà des images, rien n'est téléchargé.
- **`mars-curiosity`** : la plupart des images font moins de 256 px. Réglez la « Taille des morceaux » sur 128 pour ne pas les agrandir.
- **Vérification** : chaque archive est comparée à une empreinte enregistrée. Si la source change son fichier, le téléchargement est refusé avec un message, et le catalogue devra être mis à jour (`aidge_in_space/dataset_catalog.py`).

### Déposer vos propres images

1. Donnez un nom au jeu, court, par exemple `lune-lro`. Si ce jeu existe déjà, vos images y sont ajoutées. Aucun fichier existant n'est remplacé : en cas de doublon, le nouveau fichier reçoit un suffixe `_2`.
2. Glissez vos fichiers dans « Images ou archive zip », ou cliquez pour les choisir.
3. Cliquez sur **Créer le jeu**.

Ce qu'il faut savoir :

- **Formats acceptés** : les images courantes (PNG, JPEG, TIFF, etc.) et les archives **zip** contenant des images. Les fichiers qui ne sont pas des images sont ignorés, avec un message.
- **Taille conseillée** : au moins 256 pixels de côté. Une image plus petite est gardée, mais elle sera agrandie pendant l'entraînement, et vous recevez un avertissement.
- **Images 16 bits** (fréquentes en imagerie spatiale) : elles sont converties en 8 bits en étirant leurs valeurs jusqu'à leur maximum, et un message vous le dit.
- **Limites** : l'envoi total est limité à 500 Mo par défaut. Un zip peut contenir au plus 10 000 fichiers, et sa taille une fois décompressé est limitée au double de la taille maximale d'envoi.
- **Tout ou rien** : si l'envoi échoue (archive illisible, trop gros, aucune image valide), rien n'est ajouté au jeu. Corrigez et recommencez.
- Les images illisibles sont écartées avec un message, et le reste de l'envoi est conservé.

Le menu « Jeu d'images » liste vos jeux avec leur nombre d'images. L'aperçu situé dessous affiche quelques images du jeu choisi, pour vérifier que le dépôt est correct.

## 3. Lancer un entraînement (onglet 1. Entraîner)

Faites toujours un test rapide avant un long entraînement : il vérifie que tout fonctionne.

1. Ouvrez l'onglet **1. Entraîner**.
2. Choisissez le préréglage **Test rapide (quelques minutes)**. Il remplit tous les champs. L'autre préréglage, **Standard (plusieurs heures, GPU conseillé)**, sert aux vrais entraînements. Vous pouvez ensuite modifier les champs. Le choix **Personnalisé** ne remplit rien : il laisse les champs tels quels.
3. Remplissez **Nom du run** (obligatoire) : tous les modèles d'un même run sont regroupés. **Auteur** est facultatif et sert à retrouver vos entraînements.
4. Choisissez le **Jeu d'images** (voir la [section 2](#2-ajouter-des-images)).
5. Modifiez les réglages si besoin :
   - **Mode** : niveaux de gris ou couleur.
   - **Valeurs de λ** : cochez une ou plusieurs valeurs. Un entraînement séparé est créé pour chaque valeur cochée.
   - **Nombre de pas** : la durée de l'entraînement.
   - **Largeur du modèle (N)** : plus N est grand, meilleur est le modèle, mais plus il est lourd à calculer et à stocker.
   - **Réglages avancés** : GDN finale, taille du lot, pas d'apprentissage, taille des morceaux d'image, nombre de morceaux gardés en mémoire, graine aléatoire. Les valeurs des préréglages conviennent en général.
6. Lisez l'encart sous les réglages : il estime le poids de la partie du modèle qui tournera sur la carte (paramètres, calculs par pixel, taille en INT8) et dit si elle tient dans la mémoire BRAM du FPGA.
7. Cliquez sur **Lancer l'entraînement**. L'interface passe toute seule dans l'onglet **2. Suivre**, sur le premier entraînement créé.

### Combien de temps ça prend ?

- Le **premier** entraînement télécharge les 24 images Kodak, qui servent à mesurer la qualité du modèle. Il faut une connexion internet. Sur le PC de l'équipe, le premier test rapide a duré environ 4 min 30 à cause de ce téléchargement.
- Ensuite, un test rapide dure de 30 secondes à 1 minute 15 environ (sur le processeur, sans GPU).
- Sans connexion et sans Kodak déjà téléchargé, la mesure de qualité se fait sur des images de votre jeu d'entraînement, avec un avertissement. Les chiffres sont alors moins comparables.
- Un entraînement **Standard** dure plusieurs heures, et un GPU est conseillé.

## 4. Suivre l'entraînement (onglet 2. Suivre)

L'onglet **2. Suivre** se rafraîchit tout seul toutes les 5 secondes.

- **Le bandeau en haut** dit si un worker est actif. Si non, voir [section 8](#8-quand-quelque-chose-ne-va-pas).
- **Le tableau « Entraînements »** liste les jobs (un job = un entraînement pour une valeur de λ), du plus récent au plus ancien, avec leur statut, leur progression en pourcentage et le temps restant estimé.
- **Le menu « Entraînement à examiner »** affiche en dessous un résumé des réglages et les courbes. La section **Journal** contient les 50 dernières lignes du journal.

### Les statuts

| Statut affiché | Signification |
| :- | :- |
| En attente | Le job attend que le worker soit libre. |
| En cours | Le worker l'exécute. |
| Annulation demandée | Vous avez demandé l'arrêt ; il s'arrête dans quelques secondes. |
| Terminé | L'entraînement est fini, le modèle est sauvegardé. |
| En échec | Une erreur est survenue ; son message est dans le résumé et le journal. |
| Annulé | Vous l'avez annulé. |
| Interrompu | Le worker s'est arrêté ou a disparu en cours de route. |

### Lire les courbes

Trois graphiques s'affichent :

- **Perte** : le score que l'entraînement cherche à réduire. Elle doit globalement baisser. Elle mélange la qualité et le débit (voir λ plus bas).
- **Débit (bpp)** : le nombre de **bits par pixel** nécessaires pour stocker l'image compressée. Plus il est bas, plus la compression est forte. Une image 8 bits non compressée vaut 8 bpp ; un bon compresseur descend sous 1 bpp.
- **PSNR (dB)** : une mesure de la ressemblance entre l'image reconstruite et l'originale. Plus il est haut, plus l'image est fidèle (autour de 30 dB, la différence est déjà peu visible ; plus de 40 dB, elle est excellente).

Sur ces graphiques, deux courbes se côtoient :

- **entraînement (bruit)** : mesurée à chaque étape pendant l'apprentissage. Pour que le modèle puisse apprendre, on y ajoute volontairement un petit bruit à la place de l'arrondi réel. C'est une estimation, elle bouge beaucoup.
- **Kodak (arrondi)** : mesurée de temps en temps sur les images de test Kodak, avec le **vrai arrondi** utilisé à la compression. C'est la mesure qui compte pour juger le modèle. Elle porte un autre nom (« images d'évaluation » ou « jeu d'entraînement ») quand Kodak n'a pas pu être utilisé.

Les deux courbes ne coïncident pas exactement : c'est normal.

### Ce que fait λ

λ (lambda) règle le compromis entre qualité et poids de l'image compressée.

- **Grand λ** : on privilégie la qualité, l'image compressée contient plus de bits, le PSNR est plus haut.
- **Petit λ** : on privilégie la compression, l'image est plus petite mais plus dégradée.

C'est pourquoi un même run peut comporter plusieurs λ : chacun produit un modèle à un niveau de compression différent, et l'ensemble dessine une courbe débit-qualité comparable à celle de JPEG.

## 5. Annuler, relancer, dupliquer

Sélectionnez d'abord le job dans le menu « Entraînement à examiner ».

- **Télécharger le modèle** : sous les courbes, la zone « Modèle entraîné » propose le fichier `.pt` du job choisi dès que son premier point de sauvegarde existe. Pour un job en cours, c'est le dernier point de sauvegarde, pas le modèle final : attendez le statut Terminé pour le modèle complet.
- **Annuler** : ouvrez la section « Annuler cet entraînement », cochez « Je confirme l'annulation », puis cliquez sur **Annuler**. Un job en attente est annulé aussitôt. Un job en cours sauvegarde son avancement puis s'arrête dans les secondes qui suivent.
- **Relancer** : disponible pour un job En échec, Annulé ou Interrompu. Le job repart dans la file et **reprend à son dernier point de sauvegarde**, sans refaire ce qui est déjà fait. Une optimisation Aidge, elle, recommence depuis le début. Si le job échoue de nouveau, lisez le message d'erreur dans le résumé.
- **Dupliquer la configuration** : copie les réglages du job dans l'onglet 1. Entraîner (le nom reçoit « (copie) »), pour en lancer un nouveau en changeant une valeur. Si le jeu d'images d'origine n'existe plus, un avertissement vous demande d'en choisir un autre.

## 6. Tester et valider le modèle (onglets 3 et 4)

**Tester : une image, pour voir le résultat.** Dans l'onglet 2. Suivre, cliquez sur **Tester ce modèle** : l'onglet **3. Tester** s'ouvre sur le modèle et le niveau de cet entraînement. Vous pouvez aussi les choisir vous-même dans le menu. Choisissez une image ou importez la vôtre, puis cliquez sur **Tester**. Vous obtenez l'original, l'image reconstruite, l'erreur amplifiée quatre fois, un zoom et les mesures. Les codecs cochés dans la section repliable « Comparer à » (JPEG et JPEG 2000 par défaut) sont pris au débit le plus proche de celui du modèle. Le mode d'image est celui du modèle : il n'y a rien à régler.

**Valider : un jeu d'images, pour conclure.** Dans l'onglet **4. Valider**, cochez les modèles et les références, choisissez le jeu et le nombre d'images, puis cliquez sur **Lancer la campagne**. Vous obtenez les courbes débit-qualité et le BD-rate : un BD-rate négatif signifie que le modèle utilise moins de bits que la référence à qualité égale. Il faut au moins 4 points par courbe, donc au moins 4 valeurs de λ pour un modèle entraîné ici. La section « Détails de la campagne » contient la référence du BD-rate, les temps d'encodage et le CSV.

**Le coût sur la carte.** La section « Coût sur la carte » de l'onglet 4. Valider donne le calcul et la mémoire de l'encodeur, couche par couche. Elle vérifie aussi que PyTorch, ONNX Runtime et Aidge donnent les mêmes latents.

Un modèle terminé apparaît tout seul dans les menus. Sinon, ouvrez la section « Aide » en bas de page et cliquez sur **Recharger les codecs**.

Le banc d'essai est décrit dans le [README](../README.md#banc-dessai-de-compression).

## 7. Préparer le modèle pour la carte (onglet 5. Optimisation Aidge)

Sur la carte, la mémoire et le calcul sont comptés. Un modèle entraîné utilise des nombres à virgule, coûteux et volumineux. Cet onglet demande à l'outil **Aidge** de le préparer : il remplace ces nombres par de petits entiers là où c'est possible (on dit « quantifier »), vérifie ce que cela coûte en qualité, et produit le code à mettre sur la carte.

### Les recettes

| Recette | Ce qu'elle fait | Résultat attendu |
| :- | :- | :- |
| **mixte** (conseillée, cochée par défaut) | Quantifie les grandes opérations de filtrage (les convolutions), et garde en nombres à virgule la partie de normalisation (la GDN) | Beaucoup moins de mémoire pour un coût de qualité faible |
| **fp32** | Ne quantifie rien | Sert de point de comparaison : elle est toujours calculée |
| **int8** | Essaie de quantifier tout le modèle | Aujourd'hui, l'alerte **SANS EFFET** s'affiche : Aidge ne sait pas encore quantifier la GDN. Ce n'est pas une panne de votre part |

### Pas à pas

1. Donnez un nom à l'optimisation et cochez un ou plusieurs modèles. Cochez plusieurs niveaux de λ pour obtenir une courbe.
2. Choisissez les **images de calibration** : Aidge s'en sert pour régler les entiers. Prenez des images proches de ce que verra le rover.
3. Gardez la **taille de tuile** de 64 si vous n'avez pas de raison d'en changer. L'image est découpée en carrés de cette taille, traités un par un : c'est ce qui limite la mémoire.
4. Cliquez sur **Analyser** : en quelques secondes, vous voyez si le modèle tiendra en mémoire.
5. Cliquez sur **Lancer** : un job part dans la file, comme un entraînement. Suivez-le dans l'onglet **2. Suivre**. Comptez plusieurs minutes par modèle, davantage avec la compilation et les 24 images d'évaluation.

Une fois le job terminé, la variante apparaît dans les onglets **3. Tester** et **4. Valider**, sous un nom qui commence par « Aidge ». Dans ces onglets, elle s'exécute toujours avec Aidge, quel que soit le moteur choisi : c'est plus lent, et c'est voulu.

### Lire les résultats

| Colonne | Signification |
| :- | :- |
| **pic / BRAM** | Mémoire de travail nécessaire, comparée à la mémoire rapide du FPGA (la BRAM, 615 Kio sur notre carte). Au-delà de 100 %, cela ne tient pas |
| **poids conv / BRAM** | Place des poids qui restent dans le FPGA, la GDN étant calculée par le processeur |
| **Δbpp** | Écart de débit face à fp32. +3 % : le fichier compressé est 3 % plus gros |
| **ΔPSNR** | Écart de qualité face à fp32. -0,05 dB : une perte très faible |
| **identiques Aidge** | Le code généré donne-t-il les mêmes résultats qu'Aidge ? **En dessous de 99 %, la variante passe en ÉCHEC** : ce code ne doit pas partir sur la carte |
| **identiques PyTorch** | À titre d'information : l'écart avec le modèle d'origine, normal pour une variante quantifiée. C'est Δbpp et ΔPSNR qui disent si cet écart est gênant |

**BD-rate** : l'écart de débit à qualité égale sur toute la courbe, face à fp32. Il faut au moins 4 modèles de λ différents, et des courbes régulières, sinon la raison s'affiche à la place du chiffre. Les écarts point par point restent toujours affichés. La mention **indicative** rappelle que les modèles n'ont pas eu le même entraînement.

### Récupérer le code

Le menu **Code C à télécharger** propose :

- **export C++ (zip)** : le code d'Aidge, à compiler pour le processeur de la carte ;
- **fichier C pour LoCod** : un seul fichier `main.c`, pour la chaîne FPGA. Les convolutions y sont des accélérateurs, et la GDN reste sur le processeur.

Sans compilateur sur la machine, le code est quand même produit, mais non vérifié : la colonne **export** indique `non_compile`.

## 8. Quand quelque chose ne va pas

| Situation | Que faire |
| :- | :- |
| Le bandeau dit « Aucun worker actif : les entraînements restent en attente » | Le worker ne tourne pas. Double-cliquez sur `scripts\lancer_plateforme.bat` (ou lancez `docker compose up worker`). Les jobs en attente démarreront alors seuls. |
| Un job reste « En attente » alors que le bandeau dit « Worker actif » | Un autre job est probablement en cours : le worker les traite un par un. Patientez. |
| Un job est « Interrompu » | Le worker s'est arrêté (fenêtre fermée, ordinateur éteint, mise à jour). Remettez la plateforme en route, sélectionnez le job, cliquez sur **Relancer**. |
| Après un redémarrage, un job reste « En cours » sans avancer | Le worker précédent s'est arrêté brutalement. Le nouveau worker s'en aperçoit au bout d'environ une minute et passe le job en « Interrompu ». Attendez cette minute, puis cliquez sur **Relancer**. |
| Un job est « En échec » | Lisez le message d'erreur dans le résumé et la fin du journal. Cause fréquente : pas d'internet au premier lancement (Kodak), jeu d'images vide ou supprimé, mémoire insuffisante (diminuez le nombre de morceaux gardés en mémoire). Corrigez, puis **Relancer** ou **Dupliquer la configuration**. |
| Le navigateur n'affiche pas l'interface AIDGE | Voir [section 1](#1-démarrer-la-plateforme) : le port 7860 est occupé par un autre programme. |
| Un message d'erreur s'affiche dans une fenêtre | Il explique en français ce qui manque (nom du run vide, aucun λ coché, jeu d'images absent, etc.). |
| Une variante affiche SANS EFFET | C'est normal pour `int8` tant qu'Aidge ne sait pas quantifier la GDN : utilisez `mixte`. |
| Une variante affiche ÉCHEC | Lisez la raison au-dessus du tableau. Si le code généré diffère d'Aidge, ne l'utilisez pas et transmettez le numéro du job à l'équipe. |

Si le problème persiste, notez le numéro du job et le message d'erreur et transmettez-les à l'équipe.

## 9. Bon à savoir

- **Un entraînement en cours ralentit l'ordinateur** et fausse les temps de calcul mesurés dans le banc d'essai. Ne lancez pas de campagne de mesure de temps pendant un entraînement : l'interface vous le rappelle dans le bandeau.
- **Un GPU est conseillé** pour les entraînements standard (plusieurs heures). Le PC de l'équipe a une carte NVIDIA, mais l'installation par défaut utilise le processeur seul ; voir le [README](../README.md) pour installer la version GPU de PyTorch.
- **Où sont les fichiers ?** Les modèles entraînés sont dans `outputs/checkpoints/`, avec le numéro du job dans le nom (`..._run12.pt`). Les journaux et la file d'attente sont dans `outputs/runs/`. Les variantes Aidge et leur code sont dans `outputs/optim/`. Ces dossiers ne sont pas versionnés.
- **Ne supprimez pas** `outputs/runs/runs.sqlite` : c'est la mémoire de la file d'attente.
- Les jeux d'images sont dans `data/train/`, un sous-dossier par jeu.
