# Conception de l'optimisation Aidge et de l'export C

Document de référence de l'onglet « Optimisation Aidge ». **Il fait foi** pour l'implémentation : contrat CLI,
format de `report.json`, schéma de la table `jobs`, cycle de vie d'une variante. Tout écart se signale dans le
compte rendu au lieu de changer le contrat en silence. Conventions : voir `CLAUDE.md` et le préambule de
`docs/conception_entrainement.md`.

Statut : **proposition, en attente de validation**. Aucun code n'est écrit.

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

- **Seule la tuile 64 tient en BRAM**, et seulement pour les activations.
- **Le mixte réduit les poids de 71 % mais pas le pic d'activations** : il reste à 393 216 o, parce que la GDN
  flottante travaille sur 4 octets par valeur.
- À N=128, les poids ne tiennent en BRAM dans aucune variante : ils devront vivre en DDR.
- Binaire : `.text` de 3,66 Mo en FP32, 1,39 Mo en mixte. Il contient les poids.

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
   moins 4 niveaux (`metrics.bd_rate`), sinon il vaut `null` et l'interface dit « 2 niveaux sur 4 ».

---

## 3. Architecture

```
aidge_in_space/optim/
  config.py     OptimConfig, YAML, validation ; sans torch ni Aidge
  graph.py      ONNX -> GraphView, rapport de quantifiabilité, compte des nœuds, recettes
  ptq.py        calibration, tentative INT8 avec détection du no-op, recette mixte
  memory.py     plan mémoire, pic, poids, comparaison à la BRAM
  evaluate.py   fidélité des latents, débit-distorsion sur Kodak, BD-rate
  export.py     export, compilation si g++ est là, vérification, taille, zip
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
                 "text_bytes": 3658507, "bss_bytes": 491552, "verified_against": "pytorch",
                 "symbols_identical_pct": 100.0, "message": null}
    },
    "int8": {
      "status": "sans_effet",
      "reason": "Aidge 0.10.1 ne sait pas quantifier Sqrt, la racine de la GDN : quantize_network n'a rien changé.",
      "signals": {"check_architecture": false, "nodes_unchanged": true, "outputs_identical": true},
      "nodes_before": {}, "nodes_after": {}, "memory": null, "fidelity": null, "rd": null, "export": null
    },
    "mixte": {"status": "ok", "reason": null}
  }
}
```

Les valeurs numériques `0.0` ci-dessus sont des emplacements, pas des mesures. Les écarts `delta_*` sont pris face à
la variante `fp32` de ce rapport, donc à même checkpoint et même tuile. Le BD-rate n'est pas dans ce fichier : il
porte sur plusieurs jobs et se calcule à la lecture.

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
| `divergent` | le binaire diffère du graphe Aidge au-delà de la tolérance : **variante en alerte, jamais présentée comme valide** |

Tolérance : FP32, symboles identiques à 100 % face à PyTorch ; mixte, identiques à 100 % face au graphe Aidge de la
même recette, puisque l'écart à PyTorch est le coût mesuré de la quantification.

---

## 7. Table `jobs` et worker

`SCHEMA_VERSION = 2`.

```sql
ALTER TABLE jobs ADD COLUMN kind TEXT NOT NULL DEFAULT 'train' CHECK (kind IN ('train', 'optim'));
```

Migration dans `_init_schema`, dans une transaction `BEGIN IMMEDIATE`, en relisant `user_version` à l'intérieur pour
le cas de deux processus simultanés. Une base en version 1 garde ses jobs, tous de type `train`.

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

Clés de codec : `aidge-<slug du nom>-<recette>-t<H>`. Les tests paramétrés de `test_bench.py` les couvrent
automatiquement.

---

## 9. Interface `ui_optim.py`

Fonctions pures et adaptateurs Gradio fins, comme `ui_training.py`, avec les aides `ui_style` (`intro`, `card`, `hint`).

- **Formulaire** : codec et niveaux, tuile, recettes, clipping, par canal, nombre d'images et jeu de calibration, export et compilation.
- **Analyser** : opérateurs (dont `Sqrt` non pris en charge), nœuds, pic mémoire face à la BRAM, et phrase d'explication.
- **Lancer** : un job par niveau coché.
- **Tableau des variantes** : recette, statut, nœuds, pic face à la BRAM, poids, Δbpp, ΔPSNR, BD-rate (ou « n niveaux sur 4 »), taille du binaire, état de l'export.
- **Alerte** quand `int8` est `sans_effet` : « Aidge 0.10.1 ne sait pas quantifier la GDN. Le graphe n'a pas changé. La variante mixte garde la GDN en flottant et quantifie les convolutions. »
- **Téléchargement** du zip de l'export C.

---

## 10. Contraintes pour isoler une couche par `FPGA()` (hors périmètre, à ne pas fermer)

Ce que l'export fournit déjà :

- tailles entièrement fixées à la compilation, par tuile (`#define` par couche) ;
- mémoire statique unique et calculée (`static unsigned char mem_0[393216]`), avec un décalage connu par couche ;
- poids dans des fichiers séparés par couche ;
- découpage naturel : convolutions entières d'un côté, GDN flottante de l'autre.

Ce qui reste à traiter :

- **Le code généré est du C++ à gabarits** (`export_cpp::convolution_forward<...>`), pas du C. L'énoncé prévoit
  « ONNX vers C » : **la compatibilité avec LoCod n'est pas vérifiée**, et c'est le premier risque de la suite.
- Les Quantizer calculent encore en flottant (`Mul`, `Round`, `Clip`). Pour une couche purement entière, il faudra
  `single_shift=True`, pas encore essayé à l'export.
- La convolution produit de l'`int32` : accumulateurs à dimensionner côté PL.
- `dformat="nchw"` ajoute des `Transpose` : le cœur calcule en NHWC, ce qui coûte de la mémoire (`.bss` de 491 552 o
  au lieu de 393 216 o). À revoir pour la carte.
- La GDN flottante (`Sqrt`, `Div`) suggère un partage convolutions en PL, GDN en PS. C'est une piste : son coût
  d'échange PS/PL n'est pas mesuré.
- `export.py` range le code par couche dans `layers/`, et n'édite jamais les fichiers générés : un futur outil pourra
  les envelopper.

---

## 11. Limites connues et points non vérifiés

- **Toutes les mesures viennent de Linux dans Docker.** Le poste actuel n'a pas de `.venv`, et ni Aidge ni torch dans
  son Python. La compilation par le g++ MSYS2 sous Windows n'est pas essayée.
- **Coût débit-distorsion : 4 images, un seul checkpoint, débit estimé.** Les chiffres réels sortiront du pipeline.
- **BD-rate non mesuré.** Une seule série ne compte pas 4 niveaux aujourd'hui. Les checkpoints utilisables
  (λ=1024 run9, 2048 run4, 4096 run7, 8192 run6) n'ont pas le même entraînement (1 960 à 62 995 pas) : le chiffre
  sera indicatif, à refaire avec une vraie série.
- **Déterminisme** vérifié dans un même processus seulement. Le pipeline le contrôlera entre deux processus, et
  l'empreinte du checkpoint sera vérifiée au rechargement.
- **Retour arrière du déploiement** : une ancienne image refuse une base en version 2 (`user_version` trop élevé). Le
  rollback automatique de `scripts/deploy.sh` échouerait après la migration. À documenter dans `docs/deploiement.md`.
- **Temps** : les mesures du binaire (70 à 230 ms par tuile 64) ont été prises sur un PC chargé. Elles sont
  indicatives, et un job en cours fausse aussi le banc d'essai.
- **Calibration** : avec 16 tuiles 64 x 64, l'erreur brute atteint 5 à 9 sur quelques valeurs. D'où le défaut de 32
  tuiles, sans garantie.
- **Hyperprior** : hors périmètre, seul Ballé 2017 factorisé est pris en charge.

Hors périmètre : LoCod, Vivado, mesures sur carte, GDN simplifiée (Johnston et al. 2019). Le notebook n'est pas modifié.
