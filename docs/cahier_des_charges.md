# Cahier des charges de la plateforme aidge-train

Ce document fixe ce que la plateforme doit faire, pour qui, et sur quelle infrastructure. Il a été validé en équipe le 29 octobre 2026. La conception détaillée de la file d'entraînement reste dans [`conception_entrainement.md`](conception_entrainement.md), le mode d'emploi dans [`guide_interface.md`](guide_interface.md) et la mise en production dans [`deploiement.md`](deploiement.md).

## Pourquoi cette plateforme

Le cas d'usage AIDGE IN SPACE demande de choisir un modèle de compression qui tienne sur une carte CPU + FPGA. Ce choix se justifie par des mesures : il faut entraîner beaucoup de variantes (architecture, nombre de canaux, valeur de λ) et les comparer entre elles et aux codecs classiques. `aidge-train` permet à chaque membre de l'équipe de lancer ces entraînements depuis son navigateur, sur une carte graphique partagée, sans installer quoi que ce soit, et de comparer les résultats dans le banc d'essai.

Le dépôt couvre la plateforme d'entraînement et le banc d'essai de compression. Le notebook de référence, le portage Aidge et LoCod et le travail sur la carte vivent dans d'autres dépôts.

## Utilisateurs et droits

Deux populations utilisent la plateforme, depuis internet.

L'**équipe** a accès à tout : jeux d'images, entraînement, suivi des jobs et banc d'essai. Chaque membre est identifié par Authelia, et chaque job porte automatiquement le nom de son auteur. Il n'y a pas de quota : la file est commune et on se fait confiance.

Le **jury et les partenaires** (Viveris, CNES) n'ont accès qu'au banc d'essai : ils compressent leurs propres images avec nos modèles et les comparent à JPEG et JPEG 2000. Ils ne voient ni les jeux d'images ni les entraînements, et ne peuvent pas en lancer.

Ces droits ne reposent pas sur l'interface (un onglet masqué ne protège rien) mais sur deux adresses distinctes filtrées par Authelia :

| Adresse | Instance | Groupes Authelia autorisés |
| :- | :- | :- |
| `aidge-train.gvrt.fr` | app complète | équipe |
| `aidge-bench.gvrt.fr` | app en mode banc d'essai seul | équipe, jury |

## Ce que l'équipe doit pouvoir faire

Depuis l'interface Gradio, un membre de l'équipe peut partir d'un préréglage (test rapide, standard en gris ou en couleur) ou régler lui-même les hyperparamètres (N, λ, taille des patchs, batch, nombre de pas, taux d'apprentissage) et enregistrer sa configuration. Il dépose ses propres jeux d'images d'entraînement, en couleur. Il suit ses jobs, les annule ou les relance, puis compare les modèles obtenus dans le banc d'essai (courbes débit/distorsion, BD-rate, temps d'encodage) et télécharge les checkpoints et les exports ONNX.

Trois familles de modèles doivent pouvoir être entraînées :

- **Ballé 2017 en couleur** : le modèle du cas d'usage, déjà entraînable en gris et en couleur ;
- **Ballé 2018 hyperprior** : meilleure compression, mais plus de calcul à bord ; il n'existe aujourd'hui qu'en inférence, avec les poids de CompressAI ;
- **des variantes allégées** pensées pour le FPGA : moins de canaux, sans GDN finale, et d'autres pistes à définir.

## Parallélisme et mémoire GPU

Deux entraînements tournent en même temps sur la RTX 2060 SUPER (8 Go). Ils se partagent la puissance de calcul, donc chacun va environ deux fois moins vite qu'en étant seul ; l'intérêt est qu'un long entraînement ne bloque pas les tests rapides des autres.

Le worker estime la mémoire de chaque job à partir de sa configuration et ne lance le suivant que s'il tient dans la place restante. Ordres de grandeur estimés par le calcul (patchs 256 × 256, batch 16, 8000 patchs gardés sur le GPU), à confirmer par la mesure :

| Configuration | Banque de patchs | Activations et pic | Contexte CUDA | Total |
| :- | -: | -: | -: | -: |
| Ballé 2017 gris, N64 | 0,5 Go | 0,35 Go | 0,4 Go | environ 1,3 Go |
| Ballé 2017 gris, N128 | 0,5 Go | 0,7 Go | 0,4 Go | environ 1,6 Go |
| Ballé 2017 couleur, N128 | 1,5 Go | 0,75 Go | 0,4 Go | environ 2,7 Go |
| Ballé 2017 couleur, N192 | 1,5 Go | 1,1 Go | 0,4 Go | environ 3 Go |

Deux Ballé 2017 en couleur tiennent donc ensemble ; deux hyperpriors en N192 probablement pas, et le second attendra. Le journal de chaque job affiche la mémoire GPU réellement utilisée, ce qui permet d'affiner l'estimation.

## Infrastructure

La plateforme tourne sur le homelab, dans une **VM Debian avec Docker Compose**, à laquelle la RTX 2060 SUPER est attribuée en passthrough. La VM isole mieux qu'un conteneur LXC un service exposé sur internet, et la carte dédiée évite tout conflit de mémoire avec les autres services du homelab (llama.cpp et Ollama gardent la GT 1030).

Dans la VM, un même fichier compose lance :

- `app-train`, l'app complète ;
- `app-bench`, la même image en mode banc d'essai seul ;
- `worker`, qui exécute jusqu'à deux jobs à la fois sur la carte ;
- des volumes partagés : la file SQLite, les jeux d'images, les modèles et les résultats. Un modèle entraîné apparaît donc aussitôt dans le banc d'essai du jury.

L'accès depuis internet passe uniquement par le reverse proxy du homelab (Caddy) : certificat wildcard `*.gvrt.fr` obtenu en DNS-01 chez OVH, authentification Authelia avec deuxième facteur, bannissement des scanners par CrowdSec. La VM n'accepte le trafic de l'interface que depuis Caddy, et son administration ne passe que par le VPN.

Le banc d'essai mesure ses temps d'encodage sur un seul cœur. Pendant un entraînement, ces mesures deviennent bruitées : la VM reçoit assez de vCPU, et les mesures de référence se refont carte au repos.

## Intégration et déploiement continus

Personne ne déploie à la main. Un membre pousse sa branche et ouvre une pull request ; la CI vérifie les tests, les fichiers compose, la construction de l'image et le démarrage de l'app et du worker. On n'entre sur `main` que par pull request, avec la CI au vert. Au merge, GitHub publie les images sur GHCR ; le runner installé dans la VM les récupère (connexion sortante, aucun port ouvert), redéploie le compose, vérifie que l'interface répond et revient seul à la version précédente sinon.

Un déploiement redémarre le worker : les jobs en cours sauvegardent leur checkpoint, et le nouveau worker les remet en file automatiquement. Un merge ne coûte ainsi que les quelques minutes écoulées depuis le dernier checkpoint. L'interface affiche le commit déployé. Dependabot propose chaque semaine les mises à jour des dépendances, qui passent par la même CI.

Le dépôt étant public, une pull request venue d'un fork pourrait modifier le workflow pour exécuter du code sur le runner du homelab. Les workflows des contributeurs externes doivent donc attendre une approbation (réglage *Actions, General* de l'organisation), et le déploiement ne part que de `main`.

## Feuille de route

1. **Remise à plat du dépôt** : configurations et documentation restaurées, préréglages couleur, CI avec les tests sur chaque pull request, image qui embarque le code, chaîne de déploiement.
2. **Infrastructure** : VM et passthrough de la carte, runner, Authelia avec les groupes équipe et jury, les deux sous-domaines, CrowdSec.
3. **Identité et rôles** : auteur des jobs lu depuis Authelia, mode banc d'essai seul, compose de production à deux instances.
4. **Worker parallèle** : deux emplacements, réservation de mémoire, réservation des jobs sûre entre processus, reprise automatique après redémarrage, mémoire GPU réelle dans le journal.
5. **Entraînement de l'hyperprior** : débit sur `y` et `z`, perte auxiliaire du modèle d'entropie, checkpoints lisibles par le banc d'essai.
6. **Variantes allégées** pour le FPGA.

Le calendrier du challenge fixe la limite : livrables le 31 décembre 2026, soutenance le 14 janvier 2027.

## Questions ouvertes

- Quel jeu d'images d'entraînement commun en couleur (par exemple CLIC ou DIV2K), et qui le dépose sur le serveur ?
- Quelles variantes allégées précises : nombre de canaux, quantification, suppression des GDN ?
