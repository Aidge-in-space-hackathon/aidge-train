# Guide de l'interface d'entraînement

Ce guide s'adresse à toute personne de l'équipe, même sans avoir jamais ouvert de terminal. Il explique comment entraîner un modèle de compression d'image depuis le navigateur, suivre l'entraînement et comparer le résultat avec JPEG.

## Sommaire

1. [Démarrer la plateforme](#1-démarrer-la-plateforme)
2. [Déposer des images (onglet Données)](#2-déposer-des-images-onglet-données)
3. [Lancer un test rapide (onglet Entraînement)](#3-lancer-un-test-rapide-onglet-entraînement)
4. [Suivre l'entraînement (onglet Suivi)](#4-suivre-lentraînement-onglet-suivi)
5. [Annuler, relancer, dupliquer](#5-annuler-relancer-dupliquer)
6. [Comparer avec JPEG](#6-comparer-avec-jpeg)
7. [Quand quelque chose ne va pas](#7-quand-quelque-chose-ne-va-pas)
8. [Bon à savoir](#8-bon-à-savoir)

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

## 2. Déposer des images (onglet Données)

Les images servent d'exemples au modèle pour apprendre à compresser. Plus elles ressemblent à ce que le rover verra, meilleur sera le modèle.

1. Ouvrez l'onglet **Données**.
2. Donnez un **Nom du jeu**, court, par exemple `lune-lro`. Si ce jeu existe déjà, vos images y sont ajoutées. Aucun fichier existant n'est remplacé : en cas de doublon, le nouveau fichier reçoit un suffixe `_2`.
3. Glissez vos fichiers dans « Images ou archive zip », ou cliquez pour les choisir.
4. Cliquez sur **Créer le jeu**.

Ce qu'il faut savoir :

- **Formats acceptés** : les images courantes (PNG, JPEG, TIFF, etc.) et les archives **zip** contenant des images. Les fichiers qui ne sont pas des images sont ignorés, avec un message.
- **Taille conseillée** : au moins 256 pixels de côté. Une image plus petite est gardée, mais elle sera agrandie pendant l'entraînement, et vous recevez un avertissement.
- **Images 16 bits** (fréquentes en imagerie spatiale) : elles sont converties en 8 bits en étirant leurs valeurs jusqu'à leur maximum, et un message vous le dit.
- **Limites** : l'envoi total est limité à 500 Mo par défaut. Un zip peut contenir au plus 10 000 fichiers, et sa taille une fois décompressé est limitée au double de la taille maximale d'envoi.
- **Tout ou rien** : si l'envoi échoue (archive illisible, trop gros, aucune image valide), rien n'est ajouté au jeu. Corrigez et recommencez.
- Les images illisibles sont écartées avec un message, et le reste de l'envoi est conservé.

Le tableau « Jeux d'images disponibles » liste vos jeux et leur nombre d'images. Le menu « Voir les vignettes d'un jeu » affiche quelques images pour vérifier que le dépôt est correct.

## 3. Lancer un test rapide (onglet Entraînement)

Faites toujours un test rapide avant un long entraînement : il vérifie que tout fonctionne.

1. Ouvrez l'onglet **Entraînement**.
2. Choisissez le préréglage **Test rapide (quelques minutes)**. Il remplit tous les champs. L'autre préréglage, **Standard (plusieurs heures, GPU conseillé)**, sert aux vrais entraînements. Vous pouvez ensuite modifier les champs. Le choix **Personnalisé** ne remplit rien : il laisse les champs tels quels.
3. Remplissez **Nom du run** (obligatoire) : tous les modèles d'un même run seront regroupés dans le banc d'essai. Ajoutez votre prénom dans **Auteur** pour retrouver vos entraînements.
4. Choisissez le **Jeu d'images** créé à l'étape précédente.
5. Choisissez le **Mode** : niveaux de gris ou couleur.
6. Réglez les valeurs :
   - **Largeur du modèle (N)** : plus N est grand, meilleur est le modèle, mais plus il est lourd à calculer et à stocker.
   - **GDN finale** : laissez cochée, comme dans l'article de référence.
   - **Valeurs de λ** : cochez une ou plusieurs valeurs. Un entraînement séparé est créé pour chaque valeur cochée.
   - **Nombre de pas** : la durée de l'entraînement.
   - **Réglages avancés** : taille du lot, pas d'apprentissage, taille des morceaux d'image, nombre de morceaux gardés en mémoire, graine aléatoire. Les valeurs des préréglages conviennent en général.
7. Lisez l'encart sous les champs : il estime le poids de la partie du modèle qui tournera sur la carte (paramètres, calculs par pixel, taille en INT8) et dit si elle tient dans la mémoire BRAM du FPGA.
8. Cliquez sur **Lancer**. L'interface passe toute seule dans l'onglet **Suivi**.

### Combien de temps ça prend ?

- Le **premier** entraînement télécharge les 24 images Kodak, qui servent à mesurer la qualité du modèle. Il faut une connexion internet. Sur le PC de l'équipe, le premier test rapide a duré environ 4 min 30 à cause de ce téléchargement.
- Ensuite, un test rapide dure de 30 secondes à 1 minute 15 environ (sur le processeur, sans GPU).
- Sans connexion et sans Kodak déjà téléchargé, la mesure de qualité se fait sur des images de votre jeu d'entraînement, avec un avertissement. Les chiffres sont alors moins comparables.
- Un entraînement **Standard** dure plusieurs heures, et un GPU est conseillé.

## 4. Suivre l'entraînement (onglet Suivi)

L'onglet **Suivi** se rafraîchit tout seul toutes les 5 secondes.

- **Le bandeau en haut** dit si un worker est actif. Si non, voir [section 7](#7-quand-quelque-chose-ne-va-pas).
- **Le tableau « Entraînements »** liste les jobs (un job = un entraînement pour une valeur de λ), du plus récent au plus ancien, avec leur statut, leur progression en pourcentage et le temps restant estimé.
- **Le menu « Job à examiner »** affiche en dessous les courbes, la fin du journal (50 dernières lignes) et un résumé des réglages.

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

Sélectionnez d'abord le job dans le menu « Job à examiner ».

- **Annuler** : cochez « Je confirme l'annulation », puis cliquez sur **Annuler**. Un job en attente est annulé aussitôt. Un job en cours sauvegarde son avancement puis s'arrête dans les secondes qui suivent.
- **Relancer** : disponible pour un job En échec, Annulé ou Interrompu. Le job repart dans la file et **reprend à son dernier point de sauvegarde**, sans refaire ce qui est déjà fait. Si le job échoue de nouveau, lisez le message d'erreur dans le résumé.
- **Dupliquer la configuration** : copie les réglages du job dans l'onglet Entraînement (le nom reçoit « (copie) »), pour en lancer un nouveau en changeant une valeur. Si le jeu d'images d'origine n'existe plus, un avertissement vous demande d'en choisir un autre.

## 6. Comparer avec JPEG

Quand un job passe à **Terminé**, ses modèles apparaissent tout seuls dans le banc d'essai (une série par run). Ouvrez l'onglet **Comparer** pour voir une image compressée par plusieurs méthodes au même débit, ou l'onglet **Benchmark** pour lancer une campagne sur plusieurs images et obtenir les courbes débit-qualité face à JPEG, JPEG 2000 et WebP.

Si les modèles n'apparaissent pas, cliquez sur le bouton « Recharger les codecs » du banc d'essai.

Le banc d'essai est décrit dans le [README](../README.md#banc-dessai-de-compression).

## 7. Quand quelque chose ne va pas

| Situation | Que faire |
| :- | :- |
| Le bandeau dit « Aucun worker actif : les entraînements restent en attente » | Le worker ne tourne pas. Double-cliquez sur `scripts\lancer_plateforme.bat` (ou lancez `docker compose up worker`). Les jobs en attente démarreront alors seuls. |
| Un job reste « En attente » alors que le bandeau dit « Worker actif » | Un autre job est probablement en cours : le worker les traite un par un. Patientez. |
| Un job est « Interrompu » | Le worker s'est arrêté (fenêtre fermée, ordinateur éteint, mise à jour). Remettez la plateforme en route, sélectionnez le job, cliquez sur **Relancer**. |
| Après un redémarrage, un job reste « En cours » sans avancer | Le worker précédent s'est arrêté brutalement. Le nouveau worker s'en aperçoit au bout d'environ une minute et passe le job en « Interrompu ». Attendez cette minute, puis cliquez sur **Relancer**. |
| Un job est « En échec » | Lisez le message d'erreur dans le résumé et la fin du journal. Cause fréquente : pas d'internet au premier lancement (Kodak), jeu d'images vide ou supprimé, mémoire insuffisante (diminuez le nombre de morceaux gardés en mémoire). Corrigez, puis **Relancer** ou **Dupliquer la configuration**. |
| Le navigateur n'affiche pas l'interface AIDGE | Voir [section 1](#1-démarrer-la-plateforme) : le port 7860 est occupé par un autre programme. |
| Un message d'erreur s'affiche dans une fenêtre | Il explique en français ce qui manque (nom du run vide, aucun λ coché, jeu d'images absent, etc.). |

Si le problème persiste, notez le numéro du job et le message d'erreur et transmettez-les à l'équipe.

## 8. Bon à savoir

- **Un entraînement en cours ralentit l'ordinateur** et fausse les temps de calcul mesurés dans le banc d'essai. Ne lancez pas de campagne de mesure de temps pendant un entraînement : l'interface vous le rappelle dans le bandeau.
- **Un GPU est conseillé** pour les entraînements standard (plusieurs heures). Le PC de l'équipe a une carte NVIDIA, mais l'installation par défaut utilise le processeur seul ; voir le [README](../README.md) pour installer la version GPU de PyTorch.
- **Où sont les fichiers ?** Les modèles entraînés sont dans `outputs/checkpoints/`, avec le numéro du job dans le nom (`..._run12.pt`). Les journaux et la file d'attente sont dans `outputs/runs/`. Ces dossiers ne sont pas versionnés.
- **Ne supprimez pas** `outputs/runs/runs.sqlite` : c'est la mémoire de la file d'attente.
- Les jeux d'images sont dans `data/train/`, un sous-dossier par jeu.
