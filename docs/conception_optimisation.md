# Conception de l'optimisation Aidge et de l'export C

Document de référence de l'onglet « Optimisation Aidge ». **Il fait foi** pour l'implémentation : contrat CLI,
format de `report.json`, schéma de la table `jobs`, cycle de vie d'une variante. Tout écart se signale dans le
compte rendu au lieu de changer le contrat en silence. Conventions : voir `CLAUDE.md` et le préambule de
`docs/conception_entrainement.md`.

Statut : **validée le 2 novembre 2026**, implémentation en cours. Le spike LoCod et Bambu est dans `locod/spike/`, avec ses résultats.

---

## 1. Résultats du spike

Conditions : Aidge 0.10.1, Linux (image `aidge-in-space:verif`, plus g++ 14 et CMake dans une image jetable), un
seul thread, tuile 3 x 64 x 64 sauf mention contraire. Modèle de référence :
`bls2017_c3_N128_lambda1024_run9.pt` (N=128, 62 995 pas), le seul checkpoint RGB entraîné longtemps. Les scripts
sont dans `outputs/spike/` (non versionné).

### 1.1 Ce qui marche et ce qui ne marche pas

| Sujet | Résultat | Preuve |
| :- | :- | :- |
| PTQ du graphe entier | **Impossible** | `quantize_network` ne fait rien : sept variantes testées, graphe et sorties inchangés ou plantage (§ 1.2) |
| Convolutions seules en INT8 | **Marche** | poids `int8`, biais `int32`, sortie `Conv2D` `int32`, Quantizer `int8` : le calcul est vraiment entier |
| Précision mixte dans Aidge (conv INT8, GDN flottante) | **Marche** | graphe unique, 95,65 % de symboles identiques à PyTorch |
| Export C++ FP32 | **Marche, avec un réglage obligatoire** | 100,000 % de symboles identiques à PyTorch sur 4 tuiles, erreur max 3,3e-5 |
| Export C++ mixte | **Marche, avec trois réglages obligatoires** | binaire strictement identique au graphe Aidge (erreur 0) sur 4 tuiles |
| Persistance `save_model_folder` / `load_model_folder` | **FP32 oui, mixte non** | le graphe mixte rechargé ne s'exécute plus (type de tenseur perdu sur la convolution) |
| Reconstruction déterministe du mixte | **Marche** | écart 0,0, **dans un même processus** (à revérifier entre deux processus) |
| `generate_optimized_memory_info` | **Marche** | le pic annoncé est la taille de `mem_0` dans le code généré |
| Tuilage interne `get_conv_horizontal_tiling` | **Inutilisable** | `GraphView.replace` échoue ou renvoie `False` dans 6 configurations ; un `True` isolé non reproduit ; gain mémoire jamais mesuré |
| INT16 | **Impossible** | quantifie, puis pas de noyau CPU `Pad` en `int16` à l'exécution |
| `cross_layer_equalization` | **Plante** | `getRawInput(): Mul operator has only 2 input(s), trying to get inpput#2` |
| Compilation | **Marche sans CMake** | `g++ -O2 -std=c++17 -Idnn/include -I.` : 65 à 146 s pour N=128 |

### 1.2 Pourquoi la PTQ du graphe entier échoue

`check_architecture` refuse `Sqrt`, la racine de la GDN gelée. `quantize_network` journalise alors une erreur et
rend la main **sans exception** : une application dont les journaux sont coupés croit avoir réussi.

```
[ERROR] - GraphView can't be quantized : node type Sqrt is not supported !
```

| Variante essayée | Résultat |
| :- | :- |
| nue ; `auto_assign_node_precision(g, int8, int8, int32)` ; `fake_quantize` ; `single_shift` ; `no_quant` ; `channel_wise` + MSE sans égalisation | graphe et sorties inchangés, au bit près |
| `prepare_network` puis `quantize_network` | `check_architecture` passe à `True` (cause inconnue, `Sqrt` est toujours là), puis `RuntimeError` dans l'égalisation |
| `prepare_network`, `equalize=False` | `IndexError: vector::_M_range_check` |
| Enchaînement manuel sans le garde-fou (`insert_scaling_nodes`, `normalize_*`, `quantize_normalized_network`) | 33 Quantizer insérés, sorties **NaN** |

Le dernier essai laisse penser que le problème dépasse l'absence d'un opérateur. La GDN est `x / sqrt(conv(x²) + β²)` :
un facteur d'échelle `s` sur `x` devient `s²` sur `x²` et `s` après la racine, alors que la normalisation semble
traiter chaque nœud comme linéaire. C'est une hypothèse, pas un diagnostic établi.

### 1.3 Coût mesuré

Débit estimé par le modèle d'entropie, **4 images Kodak**, un seul point débit-distorsion. Ordres de grandeur :
le BD-rate n'est pas encore mesuré.

| Encodeur | bpp | PSNR (dB) | Écart face au FP32 de même tuile |
| :- | -: | -: | :- |
| FP32, image entière | 0,6821 | 31,011 | |
| FP32, tuiles 64 | 0,6931 | 31,029 | |
| FP32, tuiles 128 | 0,6866 | 31,050 | |
| FP32, tuiles 256 | 0,6837 | 31,038 | |
| Mixte, tuiles 64, MAX, calibration lune | 0,7144 | 30,987 | +3,07 % ; -0,042 dB |
| Mixte, tuiles 64, MSE, calibration Kodak | 0,6964 | 30,905 | +0,47 % ; -0,124 dB |
| Mixte, tuiles 256, MAX, calibration lune | 0,7026 | 31,013 | +2,76 % ; -0,025 dB |
| Mixte, tuiles 256, MSE, calibration lune | 0,6834 | 30,917 | -0,05 % ; -0,121 dB |

Les autres combinaisons (calibration DIV2K) restent entre +0,3 et +3,2 % et -0,03 et -0,13 dB. Le jeu de
calibration pèse peu ici. MAX et MSE échangent du débit contre de la qualité sans vainqueur net.

Fidélité des latents arrondis face au FP32, **sur le modèle entraîné** :

| Tuile, clipping | Symboles identiques | Identiques parmi les non nuls | Erreur arrondie max |
| :- | -: | -: | -: |
| 64, MAX | 95,93 % | 91,86 % | 6 |
| 64, MAX par canal | 96,17 % | 92,34 % | 6 |
| 64, MSE | 96,24 % | 93,73 % | 9 |
| 256, MAX | 94,43 % | 91,74 % | 1 |
| 256, MSE | 95,74 % | 92,21 % | 4 |

Sur un modèle peu entraîné (1 960 pas) le même calcul donnait 99 %, car presque tous les symboles valent 0. **Le
pourcentage brut est donc trompeur** : `report.json` donne toujours aussi la part de symboles non nuls et la
fidélité sur ceux-ci.

### 1.4 Mémoire face à la BRAM du Zynq-7020 (630 000 o, 615 Kio)

| N=128, tuile | Pic d'activations | Poids FP32 | Poids mixtes |
| :- | -: | -: | -: |
| 64 | 393 216 o (**62 %**) | 3 600 896 o (572 %) | 1 049 984 o (167 %) |
| 128 | 1 572 864 o (250 %) | idem | idem |
| 256 | 6 291 456 o (999 %) | idem | idem |
| 512 | 25 165 824 o (3 995 %) | idem | idem |

- **Seule la tuile 64 tient en BRAM à N=128**, et seulement pour les activations.
- **Le mixte réduit les poids de 71 % mais pas le pic d'activations** : il reste à 393 216 o, parce que la GDN
  flottante travaille sur 4 octets par valeur.
- À N=128, les poids ne tiennent en BRAM dans aucune variante : ils devront vivre en DDR.
- Binaire : `.text` de 3,66 Mo en FP32, 1,39 Mo en mixte. Il contient les poids.

**Selon N et la tuile, recette mixte** (en % des 630 000 o) :

| N | Tuile | Pic d'activations | Poids totaux | Pic + poids | Poids des convolutions seules |
| -: | -: | -: | -: | -: | -: |
| 32 | 64 | 106 496 o (17 %) | 72 032 o (11 %) | **28 %** | 59 360 o (9 %) |
| 32 | 128 | 425 984 o (68 %) | 72 032 o (11 %) | **79 %** | 59 360 o (9 %) |
| 64 | 32 | 49 152 o (8 %) | 271 040 o (43 %) | **51 %** | 221 120 o (35 %) |
| **64** | **64** | **196 608 o (31 %)** | **271 040 o (43 %)** | **74 %** | **221 120 o (35 %)** |
| 64 | 128 | 786 432 o (125 %) | 271 040 o (43 %) | 168 % | 221 120 o (35 %) |
| 96 | 64 | 294 912 o (47 %) | 597 024 o (95 %) | 142 % | 485 280 o (77 %) |
| 128 | 64 | 393 216 o (62 %) | 1 049 984 o (167 %) | 229 % | 851 840 o (135 %) |

- **À N=64, tuile 64, mixte : poids et activations tiennent ensemble en BRAM, à 74 %.** C'est la cible retenue pour la
  série défendable.
- En FP32 à N=64, tuile 64, le même total atteint 179 % : c'est la quantification des convolutions qui fait tenir l'ensemble.
- Avec la GDN sur le PS (§ 10), le PL ne garde en BRAM que les poids des convolutions : **35 %** à N=64, 135 % à N=128.
- Mesures : plan mémoire d'Aidge sur le graphe mixte. Le contrôle sur le vrai checkpoint N=128 redonne exactement les
  valeurs du tableau précédent. La qualité d'image à N=64 reste à mesurer sur la série en cours d'entraînement.

Deux pièges de mesure, à reprendre dans `memory.py` et ses tests :

- **Un modèle neuf fausse les chiffres.** Ses GDN sont identiques et l'export ONNX fusionne les initialiseurs identiques
  (première mesure : 525 312 o de pic au lieu de 393 216 o, et deux GDN disparues des poids). Les tests doivent perturber
  chaque paramètre avant l'export.
- **Sur un graphe mixte, le plan mémoire exige une inférence préalable**, sinon `Cannot generate memory with token-based
  producer-consumer model for node ... (of type Dequantizer)`.

### 1.5 Extraits d'API exacts

**Quantifier une convolution.** Le graphe doit être compilé et ordonnancé avant.

```python
g = aidge_onnx.load_onnx(path_du_segment)
g.set_dataformat(aidge_core.dformat.nchw)
g.compile("cpu", aidge_core.dtype.float32, dims=[list(shape)])
aidge_core.SequentialScheduler(g).generate_scheduling()
q.auto_assign_node_precision(g, int8, int8, int32)
q.quantize_network(g, calibration_set=[aidge_core.Tensor(t) for t in activations_entree],
                   clipping_mode=q.Clipping.MAX, equalize=False, rescale_inputs=True, dequantize=True)
```

**Les trois réglages sans lesquels l'export mixte est faux ou plante :**

| Réglage | Sans lui |
| :- | :- |
| `g.set_dataformat(nchw)` **avant** `compile`, et `dformat="nchw"` à l'export | le binaire compile et tourne, mais seuls 43 à 47 % des symboles sont justes (`NB_CHANNELS 64, IN_HEIGHT 3`) |
| Préfixer les nœuds de chaque segment (`n.set_name(f"{segment}_{n.name()}")`) | `No node named bias_ProducerScaling_CastOut` : les trois segments ont les mêmes noms |
| Remplacer chaque `Dequantizer` par `Cast(float32)` puis `Mul(échelle)` | `Select input out of range` dans `constant_folding` (échelle : le `Producer` du `Mul` de son micro-graphe) |

**Assembler** : `aidge_core.GraphView.replace(old, new)`, avec `old` (`GraphView` : la convolution et ses deux
`Producer`) et `new` (`GraphView` du segment quantifié).

**Exporter** : `dtype="float32"` même pour le mixte, car le type de chaque nœud est déjà fixé.

```python
cfg = aidge_core.export_utils.config.ExportConfig(target="host", export_folder=str(dossier), compile=False, cuda=False,
                                                  dtype="float32", legacy=False, dformat="nchw")
artefact = aidge_core.export_utils.export(graph=g, config=cfg, input_tensors=[t], labels=None, nb_test=0)
# artefact.manifest.inputs / .outputs : noms, types et dimensions ; t porte set_data_format(nchw)
```

`export` modifie le graphe : il faut lui en passer un reconstruit. Voies écartées : `model_prequantized=True`
compile mais le binaire est faux (49 % sur le modèle réel, cause non établie) ; `legacy=True` plante sur `Select`.

**Mémoire** : `peak, info = aidge_core.generate_optimized_memory_info(scheduler, wrapping=False, display_names=False)`.
`peak` est une **liste** (une entrée par mémoire), en octets. `wrapping=True` n'apporte rien ici.

**Rapport de quantifiabilité** : `quantizability_report` écrit sur le descripteur 1, que seul `os.dup2` capture.
Lignes à analyser : `- [(\S+)] (\S+) x(\d+)`.

### 1.6 Pistes de contribution à Aidge (prix dédié)

1. `quantize_network` : lever une exception au lieu de rendre la main sur un graphe refusé.
2. PTQ : prendre en charge `Sqrt`, et la loi d'échelle de la GDN (couche standard en compression apprise).
3. Export : refuser un `dformat` indéterminé. Aujourd'hui, le binaire produit est faux sans erreur à l'exécution.
4. `Dequantizer` : plantage de `constant_folding` (`Select input out of range`).
5. `prepare_network` : `check_architecture` devient vrai alors que `Sqrt` reste, puis plantage. Égalisation : plantage sur `Mul` et `Div`.
6. `save_model_folder` : perte du type des tenseurs d'un graphe quantifié.
7. `ExportConfig.clone_model` : la valeur par défaut est un tuple, à cause d'une virgule en trop dans le code source, donc toujours vraie.
8. `Pad` en `int16` absent du backend CPU, alors que `Conv` l'a. `model_prequantized=True` : binaire faux (à isoler).

Chaque piste demande un cas minimal reproductible avant d'être déposée : ils n'ont pas été recoupés avec les
tickets existants ni avec une version plus récente d'Aidge.

---

## 2. Décisions

1. **Trois recettes de précision.** `fp32` (référence), `int8` (tentative sur le graphe entier, qui finit en
   `sans_effet` en 0.10.1) et `mixte` (convolutions INT8, GDN flottante). La première et la dernière donnent des
   variantes utilisables.
2. **Persistance par recette, pas par graphe.** Une variante se reconstruit depuis le checkpoint, la config et
   `calib.npz` (les tuiles de calibration elles-mêmes, pour ne pas dépendre d'un jeu qui changerait). On n'utilise
   pas `load_model_folder`, qui ne marche pas sur le mixte et charge un fichier pickle.
3. **Détection du no-op obligatoire**, sur trois signaux : `check_architecture` faux, types de nœuds inchangés,
   sorties identiques. Un seul suffit pour le statut `sans_effet`, jamais présenté comme un succès.
4. **La vérification face à PyTorch est obligatoire avant de déclarer un export valide.** Le spike montre qu'un
   binaire peut compiler et tourner en étant faux.
5. **Compilation directe par g++**, sans CMake : seul g++ existe sur le poste Windows actuel.
6. **« Analyser » tourne dans un sous-processus court**, écart assumé par rapport à « exécuté directement » : la
   capture par `dup2` agit sur tout le processus, et un plantage C++ d'Aidge tuerait l'interface. Pour l'utilisateur,
   rien ne change : le résultat s'affiche directement, sans passer par la file.
7. **Pas de tuilage interne des convolutions** (§ 1.1). Seule la tuile d'image fixe, qui a fait ses preuves, est
   proposée. La config refuse tout autre réglage.
8. **Une config = un checkpoint = un job.** Comme à l'entraînement, cocher plusieurs niveaux dépose plusieurs jobs
   de même nom ; le plugin les regroupe en une courbe.
9. **Le BD-rate se calcule par paires**, chaque niveau de la variante face au FP32 du même checkpoint. Il faut au
   moins 4 niveaux (`metrics.bd_rate`), sinon il vaut `null` et l'interface dit « 2 niveaux sur 4 ». **Le calcul est
   refusé si une courbe n'est pas monotone**, avec cette raison affichée. Les écarts Δbpp et ΔPSNR restent affichés point
   par point. La série actuelle (λ=1024, 2048, 4096, 8192, entraînements hétérogènes) est **signalée comme indicative**.
10. **Mixte est la recette par défaut** dans l'interface.
11. **La vérification de l'export est bloquante.** Moins de 99 % de symboles identiques au graphe Aidge de la même
    recette : la variante passe en `echec`, avec le taux affiché. Le taux face à PyTorch est affiché aussi, à titre
    d'information : l'impact réel du mixte se lit dans les écarts de débit-distorsion.
12. **Deux exports.** L'export C++ d'Aidge reste la référence : vérification sur PC, zip, et plus tard compilation ARM pour
    mesurer le processeur seul. Le générateur C pur (§ 10) produit le fichier pour LoCod. Il arrive dans son propre commit,
    après `export.py` et avant l'intégration des jobs.
13. **La GDN reste sur le PS** dans la première version : pas de racine en PL.
14. **g++ dans l'image du worker, sans CMake.**

---

## 3. Architecture

```
aidge_in_space/optim/
  config.py     OptimConfig, YAML, validation ; sans torch ni Aidge
  graph.py      ONNX -> GraphView, rapport de quantifiabilité, compte des nœuds, recettes
  ptq.py        calibration, tentative INT8 avec détection du no-op, recette mixte
  memory.py     plan mémoire, pic, poids, comparaison à la BRAM
  evaluate.py   fidélité des latents, débit-distorsion sur Kodak, BD-rate
  export.py     export C++ d'Aidge, compilation si g++ est là, vérification, taille, zip
  export_c.py   générateur C pur pour LoCod, depuis le graphe Aidge quantifié (§ 10)
  pipeline.py   enchaînement des étapes, écriture de report.json
aidge_in_space/optimize.py          CLI
aidge_in_space/codecs/aidge_variants.py   plugin auto-découvert
aidge_in_space/ui_optim.py          onglet « 5. Optimisation Aidge »
configs/optim/*.yaml                exemples
```

Modules modifiés : `jobs.py`, `worker.py`, `app.py`, `ui_training.py` (colonne `type`, jobs sans courbes),
`codecs/base.py` (deux points d'extension, § 8), `config.py` (`OPTIM_DIR`, et `BRAM_BYTES` déplacé depuis
`ui_training.py`, qui le réimporte), `Dockerfile` (g++, selon la décision en attente).

Correspondance des noms : « Comparer » et « Benchmark » dans la demande initiale sont les onglets actuels
**3. Tester** et **4. Valider** ; « Suivi » est **2. Suivre**. `CLAUDE.md` indique que le dossier `configs/` a été
retiré : `configs/optim/` le recrée à la demande, et `CLAUDE.md` sera mis à jour.

---

## 4. `OptimConfig`

Mêmes règles que `TrainConfig` : `ConfigError(ValueError)`, `from_dict`, `from_yaml`, `to_dict`, `to_yaml`, clé
inconnue refusée, messages `"<champ> : <problème> (reçu <valeur>)"`.

```yaml
name: lune-mixte          # obligatoire pour un job ; regroupe les niveaux en une courbe
author: ""
source:
  codec: balle2017-c3-n128-tkt
  checkpoint: outputs/checkpoints/bls2017_c3_N128_lambda1024_run9.pt
tile: [3, 64, 64]         # C, H, W
recipes: [fp32, int8, mixte]
ptq:
  bits: 8                 # seule valeur acceptée : INT16 n'a pas de noyau Pad en 0.10.1
  clipping: MAX           # MAX, MSE, KL
  per_channel: false
  n_calibration: 32
  calibration_set: lune-analogue-dfki   # sous-dossier de data/train
  seed: 0
  float_ops: [Sqrt, Div, Mul]   # documentaire et contrôlé : la GDN reste en flottant
evaluate:
  eval_dir: data/kodak
  n_images: 24
export:
  generate: true
  compile: true
```

| Champ | Règle |
| :- | :- |
| `tile` | C égal à `in_ch` du checkpoint, H et W multiples de `pad_multiple` (16), 16 ≤ côté ≤ 1024 |
| `recipes` | sous-ensemble non vide de `fp32`, `int8`, `mixte` ; `fp32` est ajouté s'il manque (il sert de référence) |
| `ptq.bits` | 8 uniquement, avec un message qui donne la raison |
| `ptq.clipping` | `MAX`, `MSE` ou `KL` (`KL` a donné de moins bons résultats et reste proposé) |
| `ptq.n_calibration` | 1 à 512 |
| `evaluate.n_images` | 1 à 24 |
| `export.compile` | sans effet si `generate` est faux |

L'existence des fichiers et du jeu de calibration se vérifie au démarrage du pipeline (code 2), pas dans le module
de config, qui reste sans dépendance lourde.

---

## 5. CLI

```
python -m aidge_in_space.optimize --config FICHIER.yaml [--out DOSSIER]
python -m aidge_in_space.optimize --job-id N [--db CHEMIN] [--watch-stdin]
python -m aidge_in_space.optimize --analyze FICHIER.yaml      # import, rapport, mémoire ; JSON sur stdout
```

Point d'entrée testable : `main(argv=None) -> int`. Sortie par `os._exit` après vidage de stdout, comme `train`.

| Code | Constante | Cas |
| :- | :- | :- |
| 0 | `EXIT_DONE` | `report.json` écrit et référence FP32 évaluée, **même si `int8` est `sans_effet`** |
| 1 | `EXIT_FAILED` | exception, ou échec de la référence FP32 |
| 2 | `EXIT_CONFIG` | config invalide, checkpoint ou jeu absent, job non `running` |
| 3 | `EXIT_CANCELLED` | annulation dans la base |
| 4 | `EXIT_STOPPED` | fin de stdin ou signal |

L'arrêt se vérifie entre les étapes, avec le `StopWatcher` de `train.py`. Une compilation en cours est interrompue
en tuant g++. Il n'y a pas de reprise à mi-chemin : `requeue` recommence le job dans un dossier temporaire, renommé
seulement à la fin.

### Étapes et progression

`steps` du job = nombre d'étapes prévues. Chaque étape appelle `log_metrics(id, k, "eval", {"kind": "eval", "stage": ...})`.

```
import -> rapport -> mémoire -> calibration -> [recette : quantification -> fidélité -> débit-distorsion -> export -> compilation -> vérification]
```

Une recette en échec n'arrête pas les suivantes : l'échec va dans `report.json`.

### Fichiers

```
outputs/optim/<id>/         <id> = n° du job, ou cfg-<nom>-<date> en mode --config
  config.yaml
  report.json               écrit de façon atomique, à chaque étape
  calib.npz
  run.log                   copie du journal à la fin (le journal vivant : RUNS_DIR/<id>/train.log)
  fp32/export/ , fp32/export.zip
  mixte/export/ , mixte/export.zip
  mixte/locod/main.c        fichier unique pour LoCod, recette mixte seulement
```

---

## 6. `report.json`

```json
{
  "schema": 1,
  "id": "42",
  "name": "lune-mixte",
  "status": "done",
  "aidge": {"core": "0.10.1", "quantization": "0.10.1", "export_cpp": "0.10.1"},
  "source": {"codec": "balle2017-c3-n128-tkt", "checkpoint": "outputs/checkpoints/...run9.pt",
             "fingerprint": "e27b02aed812", "in_ch": 3, "N": 128, "lmbda": 1024, "step": 62995},
  "tile": [3, 64, 64],
  "bram_bytes": 630000,
  "quantizability": {"quantizable": false,
                     "operators": [{"type": "Conv2D", "count": 3, "state": "ok"},
                                   {"type": "Sqrt", "count": 3, "state": "inconnu"}]},
  "variants": {
    "fp32": {
      "status": "ok",
      "reason": null,
      "nodes_before": {"Conv2D": 3, "PaddedConv2D": 3, "Mul": 3, "Div": 3, "Sqrt": 3, "Producer": 12},
      "nodes_after": {"Conv2D": 3, "PaddedConv2D": 3, "Mul": 3, "Div": 3, "Sqrt": 3, "Producer": 12},
      "memory": {"peak_bytes": 393216, "weights_bytes": 3600896, "bram_ratio": 0.624, "fits_bram": true, "weights_fit_bram": false},
      "fidelity": {"symbols_identical_pct": 100.0, "nonzero_share_pct": 36.7, "nonzero_identical_pct": 100.0,
                   "max_rounded_error": 0, "max_raw_error": 3.3e-05, "n_tiles": 32},
      "rd": {"images": 24, "rate_kind": "réel", "bpp": 0.0, "psnr": 0.0, "ms_ssim": 0.0,
             "delta_bpp_pct": null, "delta_psnr_db": null, "delta_ms_ssim_db": null},
      "export": {"status": "verifie", "path": "fp32/export", "zip": "fp32/export.zip", "compiler": "g++ 14.2.0",
                 "text_bytes": 3658507, "bss_bytes": 491552, "verified_against": "aidge",
                 "symbols_identical_pct": 100.0, "symbols_identical_vs_pytorch_pct": 100.0, "message": null},
      "locod": null
    },
    "int8": {
      "status": "sans_effet",
      "reason": "Aidge 0.10.1 ne sait pas quantifier Sqrt, la racine de la GDN : quantize_network n'a rien changé.",
      "signals": {"check_architecture": false, "nodes_unchanged": true, "outputs_identical": true},
      "nodes_before": {}, "nodes_after": {}, "memory": null, "fidelity": null, "rd": null, "export": null
    },
    "mixte": {"status": "ok", "reason": null,
              "locod": {"status": "verifie", "path": "mixte/locod/main.c", "accelerators": 3,
                        "pl_weights_bytes": 851840, "transfers_per_tile_bytes": 96256,
                        "layers_identical_pct": {"conv1": 100.0, "conv2": 100.0, "conv3": 100.0}}}
  },
  "series": {"indicative": true, "reason": "entraînements hétérogènes (1 960 à 62 995 pas)"}
}
```

Les valeurs numériques `0.0` ci-dessus sont des emplacements, pas des mesures. Les écarts `delta_*` sont pris face à
la variante `fp32` de ce rapport, donc à même checkpoint et même tuile. Le BD-rate n'est pas dans ce fichier : il
porte sur plusieurs jobs et se calcule à la lecture, avec les écarts point par point et le refus sur courbe non monotone.
`series.indicative` vient de la config (`evaluate.indicative_series`, faux par défaut).

| `status` d'une variante | Sens | Dans Tester et Valider |
| :- | :- | :- |
| `ok` | graphe produit et évalué | oui |
| `sans_effet` | no-op détecté | non, avec alerte dans l'onglet |
| `echec` | exception, message dans `reason` | non, avec le message |

| `export.status` | Sens |
| :- | :- |
| `verifie` | binaire exécuté, latents comparés |
| `non_compile` | pas de compilateur : export seul, message clair |
| `echec_compilation` | g++ a échoué, fin de stderr dans `message` |
| `divergent` | moins de 99 % de symboles identiques au graphe Aidge : **la variante passe en `echec`, avec le taux affiché** |

Seuil unique : `EXPORT_MIN_IDENTICAL_PCT = 99`, face au graphe Aidge de la même recette, en FP32 comme en mixte. Le taux
face à PyTorch (`symbols_identical_vs_pytorch_pct`) est donné pour information : pour le mixte, il mesure le coût de la
quantification, pas un défaut de l'export.

---

## 7. Table `jobs` et worker

`SCHEMA_VERSION = 2`.

```sql
ALTER TABLE jobs ADD COLUMN kind TEXT NOT NULL DEFAULT 'train' CHECK (kind IN ('train', 'optim'));
```

Migration dans `_init_schema`, dans une transaction `BEGIN IMMEDIATE`, en relisant `user_version` à l'intérieur pour
le cas de deux processus simultanés. Une base en version 1 garde ses jobs, tous de type `train`.

**Pas de retour arrière, assumé.** Une image antérieure refuse une base en version 2 (`RuntimeError`, `user_version` trop
élevé) : après cette migration, le rollback automatique de `scripts/deploy.sh` ne peut pas relancer l'ancienne image sur
cette base. On le documente sans le contourner, dans `docs/deploiement.md`.

- `Job` gagne `kind: str`. `submit(config, kind="train")` : les colonnes `in_ch`, `n`, `lmbda`, `steps` restent
  obligatoires et valent, pour un job d'optimisation, les valeurs du checkpoint source et le nombre d'étapes.
- `worker.py` : `JOB_CMDS = {"train": TRAIN_CMD, "optim": OPTIM_CMD}`. Le paramètre `cmd` de `run_worker` reste
  prioritaire pour les tests. Le journal garde le nom `train.log`.
- Onglet Suivre : colonne `type` dans `jobs_table`. `_curves` renvoie `None` pour un job `optim`, dont le détail
  affiche le résumé de `report.json`. `duplicate_config` refuse un job `optim`.

---

## 8. Cycle de vie d'une variante et plugin

```
(formulaire) -> Analyser        sous-processus court, rien n'est écrit dans la base
             -> Lancer          submit(kind="optim") -> queued
queued -> running (worker)      optimize --job-id N
running -> done                 report.json final ; rechargement des codecs par new_done_jobs
                                variantes ok -> visibles dans Tester et Valider
running -> failed / cancelled / interrupted -> requeue : recommence le job
```

`codecs/aidge_variants.py` lit `outputs/optim/*/report.json` sans importer Aidge, et :

- regroupe par (nom du job, architecture, recette, tuile) : une série, un `Level` par job, triés par λ ;
- ignore un rapport illisible (message, comme `balle2017.py`) ;
- marque indisponible (`available()` faux, avec la raison) une variante dont le checkpoint a disparu ou changé
  d'empreinte, ou si Aidge manque sur la machine ;
- hérite de `LearnedCodec` : décodeur et modèle d'entropie du checkpoint source, encodeur reconstruit à la première
  utilisation et gardé dans le cache LRU, `backends = ("aidge",)`.

`LearnedCodec.run` gagne deux points d'extension, au comportement inchangé pour les autres codecs :
`_analyze(level, xp, backend)` et `_synthesize(model, y_hat)`. La variante découpe l'image complétée en tuiles,
encode chacune, assemble les latents **en un seul flux** (les temps et le débit restent comparables), puis décode
tuile par tuile.

Clés de codec : `aidge-<slug du nom>-c<canaux>-n<N>-<recette>-t<H>` (l'architecture évite qu'un même nom donne deux fois la même clé). `LearnedCodec` gagne aussi `_pad_multiple(model)` : l'image est complétée à un multiple de la tuile. Les tests paramétrés de `test_bench.py` les couvrent
automatiquement.

---

## 9. Interface `ui_optim.py`

Fonctions pures et adaptateurs Gradio fins, comme `ui_training.py`, avec les aides `ui_style` (`intro`, `card`, `hint`).

- **Formulaire** : codec et niveaux, tuile, recettes (**mixte coché par défaut**), clipping, par canal, nombre d'images et jeu de calibration, export et compilation.
- **Analyser** : opérateurs (dont `Sqrt` non pris en charge), nœuds, pic mémoire face à la BRAM, et phrase d'explication.
- **Lancer** : un job par niveau coché.
- **Tableau des variantes** : recette, statut, nœuds, pic face à la BRAM, poids, Δbpp, ΔPSNR, BD-rate (ou « n niveaux sur 4 », ou « courbe non monotone »), mention « série indicative », taille du binaire, état de l'export, taux de symboles identiques face à Aidge et, pour information, face à PyTorch.
- **Alerte** quand `int8` est `sans_effet` : « Aidge 0.10.1 ne sait pas quantifier la GDN. Le graphe n'a pas changé. La variante mixte garde la GDN en flottant et quantifie les convolutions. »
- **Téléchargement** du zip de l'export C.

---

## 10. Générateur C pur pour LoCod (`optim/export_c.py`)

### 10.1 Pourquoi

`locod.sh` copie **un seul fichier** vers `main.c`, le passe à Bambu sans `-I`, et le compile en C côté CPU. Le C++ à gabarits
de l'export Aidge s'arrête donc sur `'cstdint' file not found`, alors que Bambu seul l'accepte. Une convolution réécrite en
C pur passe sans modifier LoCod : identique à Aidge et juste en simulation Verilog (`locod/spike/README.md`).

### 10.2 Contrat

```python
def export_locod(graph, out_dir, *, tile, name="model") -> LocodExport
```

- **Il lit le graphe Aidge après quantification, jamais PyTorch** : poids et biais (`Producer`), échelles
  (`aidge_quantization.get_scaling_factor`, et le `Producer` du `Mul` de déquantification), attributs de convolution
  (micro-graphe de `PaddedConv2D`), ordre des couches (`get_ordered_nodes`).
- **Réutilisable, pensé comme un export Aidge** (piste de contribution) : le module n'importe que `numpy`, `aidge_core` et
  `aidge_quantization`, rien d'`aidge_in_space`, ni torch. Il suit la forme de `aidge_export_cpp` : un registre associe chaque
  motif de nœuds à un émetteur de code, et tout motif inconnu est refusé par `ExportCError`, jamais ignoré.
- **Sortie : un seul `main.c`**, poids compris, sans autre inclusion locale, avec `#ifndef LOCOD_FPGA` autour de ce qui
  concerne le CPU.
- **Convolutions en `FPGA()`** : `void conv_k(struct param_conv_k *param, struct result_conv_k *result)`, tableaux de
  taille fixe, aucun pointeur interne, poids en `static const` (donc en BRAM avec la politique `LSS`), sortie en virgule
  fixe. Toute fonction appelée depuis un accélérateur est `static inline`, à cause de `-O0` (`doc/locod_bram.md` de LoCod).
- **Côté CPU** : quantification de l'entrée, déquantification, GDN flottante, requantification, et enchaînement des
  couches par `CPU()` ou `FPGA()` selon une macro de compilation.
- **Virgule fixe** : l'échelle `s` du Quantizer devient `(M, n)`, avec `M` sur 31 bits et `s ≈ M / 2^n`. Sur les deux couches
  du spike : 100 % de valeurs identiques à Aidge sur une tuile. Le pipeline mesure ce taux sur toutes les tuiles de contrôle.

### 10.3 Vérification couche par couche

Face à l'**export C++** (`-DSAVE_OUTPUTS` écrit la sortie de chaque couche) et au graphe Aidge :

1. Compilation native de `main.c` avec un banc qui écrit la sortie de chaque couche.
2. Comparaison couche par couche, après conversion NHWC vers NCHW, sur les tuiles de contrôle.
3. Même seuil que l'export C++ : moins de 99 % de symboles identiques en sortie finale, la variante passe en `echec`. Le
   taux par couche va dans `report.json`.

Sans g++, `locod.status` vaut `non_compile`. Bambu n'est pas requis par le pipeline : `locod/verifier_bambu.sh` fait la
preuve manuelle (HDL par accélérateur, puis simulation d'une couche réduite, en lisant `return_port`).

### 10.4 Volume de transferts PS/PL par tuile

La GDN reste sur le PS : chaque convolution reçoit des `int8` du CPU et lui rend des `int8`.

| Couche | N=128, PS vers PL | N=128, PL vers PS | N=64, PS vers PL | N=64, PL vers PS |
| :- | -: | -: | -: | -: |
| `conv1` | 12 288 o | 32 768 o | 12 288 o | 16 384 o |
| `conv2` | 32 768 o | 8 192 o | 16 384 o | 4 096 o |
| `conv3` | 8 192 o | 2 048 o | 4 096 o | 1 024 o |
| **Total par tuile 64** | **53 248 o** | **43 008 o** | **32 768 o** | **21 504 o** |

Soit **96 256 o par tuile à N=128 et 54 272 o à N=64**. Pour une image Kodak (96 tuiles) : 9,2 Mo et 5,2 Mo. Les poids restent
en BRAM : ils ne sont pas transférés à chaque tuile. Ces volumes sont calculés depuis les dimensions, pas mesurés. Le coût
réel d'un échange dépend de `cp_param_and_result_to_accel_memory` de LoCod et du bus, à mesurer sur carte.

### 10.5 Limites

- `locod.sh` de bout en bout n'a pas été exécuté. Les commits épinglés de `locod-fpga` et de `LoCod-docker-nanoxplore`
  n'existent plus sur GitHub. Leur branche par défaut existe, sa compatibilité n'est pas testée.
- L'interface réelle `param` et `result` a été synthétisée, pas simulée.
- `dformat="nchw"` ajoute des `Transpose` à l'export C++ : ils n'existent pas dans le générateur C.

---

## 11. Performance PL

**Axe d'optimisation du livrable, à mesurer : rien n'est implémenté ici.** Toutes les durées ci-dessous sont des
**estimations**, pas des mesures sur carte.

### 11.1 Point de départ

Une simulation Verilog sur la couche réduite donne 55 619 cycles pour 11 532 multiplications, soit environ 4,8 cycles en
comptant les copies du banc, de l'ordre de **4,5 cycles par multiplication** pour la convolution seule. Le circuit généré
avec `-O0` n'effectue qu'**une multiplication à la fois**, à environ 100 MHz d'après Bambu.

| Couche | Fenêtres (bords compris) | Multiplications N=128 | Multiplications N=64 |
| :- | :- | -: | -: |
| `conv1` | 139 x 139 x 3 x N | 7,42 M | 3,71 M |
| `conv2` | 37 x 37 x N x N | 22,43 M | 5,61 M |
| `conv3` | 17 x 17 x N x N | 4,73 M | 1,18 M |
| **Total par tuile 64** | | **34,6 M** | **10,5 M** |

### 11.2 Débit estimé

Convolutions seules. Sont exclus la GDN sur le PS, les transferts et le codage entropique, donc ce débit est un majorant.

| Cible | N=128 : par tuile | N=128 : image Kodak | N=64 : par tuile | N=64 : image Kodak |
| :- | -: | -: | -: | -: |
| PL, 4,5 cycles par multiplication, 100 MHz | 1,56 s | 2,5 min | 0,47 s | 45 s |
| Cortex-A9, 650 MHz, C scalaire, environ 4 cycles par multiplication | 0,21 s | 20 s | 0,065 s | 6 s |
| Cortex-A9, 650 MHz, NEON, 0,5 à 1 cycle par multiplication | 0,03 à 0,05 s | 3 à 5 s | 0,008 à 0,016 s | 1 à 2 s |

Les chiffres du Cortex-A9 sont des ordres de grandeur tirés de sa fréquence et d'hypothèses de cycles par opération (celle
de la carte PYNQ-Z2), **sans aucune mesure** : caches et accès mémoire peuvent les dégrader fortement.

**Lecture honnête : tel quel, le PL serait environ 7 fois plus lent que le processeur scalaire.** Pour le rattraper à
100 MHz, il faudrait environ 1,6 multiplication par cycle, contre 0,22 aujourd'hui. C'est
l'enjeu de cet axe : un accélérateur séquentiel n'accélère rien. La compilation ARM de l'export C++ doit fournir la vraie
référence.

### 11.3 Leviers à mesurer ensuite

| Levier | Idée | Contrainte connue |
| :- | :- | :- |
| **N=64** | 3,3 fois moins de multiplications, poids à 35 % de la BRAM | qualité à mesurer sur la série en cours |
| **Déroulage et pipeline** | plusieurs multiplications par cycle (`--pipelining`, déroulage à la source) | LoCod impose `-O0` pour la BRAM et les sous-fonctions : à revalider si l'on change d'option |
| **Nombre de DSP** | le spike en utilise 0 à 3 sur les 220 d'un XC7Z020 ; les multiplications 8 x 8 passent en logique | marge considérable, à exploiter avec le déroulage |
| **Partition des poids** | une BRAM à simple accès limite à une lecture par cycle : répartir les poids par filtre | augmente le nombre de blocs BRAM |
| **Accélérateurs parallèles** | LoCod lance plusieurs `FPGA()` en parallèle : découper une couche par filtres ou par lignes | surface multipliée d'autant, bus partagé |
| **Fréquence** | 100 à 118 MHz estimés sans placement ni routage | à confirmer avec Vivado |
| **Volume de transferts** | enchaîner `conv` en PL réduirait les échanges, mais la GDN l'interdit dans cette version | GDN simplifiée, hors périmètre |
| **`single_shift`** | Quantizer en décalage seul, plus simple que la multiplication par `M` | fidélité à mesurer |

Mesures à consigner pour chaque essai : cycles par simulation Verilog, surface, DSP, BRAM, fréquence, et l'écart débit-distorsion
si la structure change.

---

## 12. Limites connues et points non vérifiés

- **Toutes les mesures viennent de Linux dans Docker.** Le poste actuel n'a pas de `.venv`, et ni Aidge ni torch dans
  son Python. La compilation par le g++ MSYS2 sous Windows n'est pas essayée.
- **Coût débit-distorsion : 4 images, un seul checkpoint, débit estimé.** Les chiffres réels sortiront du pipeline.
- **BD-rate** : la série actuelle n'a pas le même entraînement (1 960 à 62 995 pas). Le chiffre sera indicatif, signalé comme
  tel, en attendant la série N=64.
- **Déterminisme** vérifié dans un même processus seulement. Le pipeline le contrôlera entre deux processus, et
  l'empreinte du checkpoint sera vérifiée au rechargement.
- **Temps** : les mesures du binaire (70 à 230 ms par tuile 64) ont été prises sur un PC chargé. Elles sont
  indicatives, et un job en cours fausse aussi le banc d'essai.
- **Calibration** : avec 16 tuiles 64 x 64, l'erreur brute atteint 5 à 9 sur quelques valeurs. D'où le défaut de 32
  tuiles, sans garantie.
- **Hyperprior** : hors périmètre, seul Ballé 2017 factorisé est pris en charge.

Hors périmètre : Vivado, mesures sur carte, GDN simplifiée (Johnston et al. 2019), racine en PL. Le notebook n'est pas modifié.
