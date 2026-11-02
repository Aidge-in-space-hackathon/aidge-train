# Spike LoCod et Bambu : une convolution de l'encodeur vers le matériel

Question posée : une convolution de l'export mixte Aidge passe-t-elle de C/C++ à HDL par la chaîne LoCod, et qu'est-ce
qui bloque ? Date : 2 novembre 2026. Vivado n'a pas été utilisé : la conversion vers HDL suffisait à trancher.

Ce dossier ne contient que du code écrit pour ce spike et des poids issus de nos modèles. **Rien ne vient de LoCod ni de
ses sous-modules (GPL-3.0)**, qu'il faut cloner à part, hors de ce dépôt. Le modèle est décrit dans
`docs/conception_optimisation.md`.

## Conclusion

- **La chaîne est viable, à condition de produire un fichier C unique.** `locod.sh` copie un seul fichier vers `main.c`,
  le passe à Bambu sans `-I` et le compile en C côté CPU. Le C++ de l'export Aidge s'arrête sur `'cstdint' file not found`.
- **Bambu seul accepte pourtant le C++ à gabarits** et produit un circuit juste, avec les bons compilateurs. L'adapter à
  LoCod demanderait quatre changements dans cet outil tiers : écarté.
- **Une convolution réécrite en C pur passe sans modifier LoCod.** Elle est identique à Aidge et juste en simulation
  Verilog, y compris avec un Quantizer en virgule fixe, donc sans aucun flottant.
- **Le Quantizer flottant est à proscrire** : surface multipliée par 10.
- **La racine de la GDN** n'a pas d'unité fonctionnelle avec les options de LoCod.

## Outil et options

- Bambu **PandA 0.9.8, révision `eda4c22d5adaec44fd8489ae49b854b244d2cf70`**, soit la révision citée par le README de
  LoCod. AppImage officiel `https://release.bambuhls.eu/appimage/bambu-0.9.8.AppImage`, extrait dans un conteneur, et
  non l'image Docker `LoCod-docker-PandA`, dont le build dure plusieurs heures.
- Options exactes de `locod.sh` (ligne 47), utilisées pour tous les tests :

```
--writer=V --generate-interface=MINIMAL --memory-allocation-policy=LSS --channels-type=MEM_ACC_11
--memory-ctrl-type=D21 --addr-bus-bitsize=32 --data-bus-bitsize=32 -DLOCOD_FPGA -O0
```

Aucune option `--device-name` : la cible est celle de Bambu par défaut, comme dans `locod.sh`.

## Résultats

Modèle : `bls2017_c3_N128_lambda1024_run9.pt` (N=128, 62 995 pas), tuile 64 x 64. Surface et fréquence sont **des
estimations de Bambu**, sans placement ni routage. « BRAM » est la mémoire que Bambu alloue pour les tableaux statiques.

### Convolutions et GDN en C

| Test | Contenu | Surface | DSP | Bascules | MHz | BRAM (o) |
| :- | :- | -: | -: | -: | -: | -: |
| C1 | `conv1`, sortie int32, poids statiques | 1 210 | 0 | 1 201 | 118,6 | 31 616 |
| C2 | `conv1`, virgule fixe, poids statiques | 1 174 | 3 | 1 264 | 111,8 | 31 616 |
| C3 | `conv1`, Quantizer flottant, poids statiques | 12 257 | 3 | 2 201 | 101,8 | 31 616 |
| C4 | `conv1`, virgule fixe, poids en paramètre | 923 | 3 | 1 200 | 111,8 | 0 |
| D1 | `conv2`, virgule fixe, poids statiques | 1 155 | 3 | 1 234 | 111,8 | 410 112 |
| D2 | `conv2`, virgule fixe, poids en paramètre | 906 | 3 | 1 170 | 111,8 | 0 |
| C5 | GDN, `sqrtf` : **échec**, « does not exist a functional unit » | | | | | |
| C5b | GDN, `sqrtf` avec `-lm` (absent des options de LoCod) | 4 237 | 0 | 416 | 100,0 | non relevée |
| C6 | division flottante seule | 2 856 | 0 | 862 | 102,8 | non relevée |
| C7 | GDN, racine de Newton en C, sans libm | 8 382 | 2 | 986 | 100,0 | non relevée |

Sur la GDN : 64 valeurs par appel, boucle sans déroulage. La précision de la racine de Newton n'a pas été vérifiée. Les
fichiers `.mem` et le Verilog ont bien été produits pour chaque ligne sans échec.

### C++ à gabarits de l'export Aidge

| Compilateur | Surface | DSP | MHz | Simulation de la version réduite |
| :- | -: | -: | -: | :- |
| par défaut | 2 074 | 0 | 106,8 | **échec** : `Unrecognized keyword decltype_type` |
| `I386_GCC7 --std=c++14` | 2 254 | 0 | 127,4 | juste, 46 725 cycles |
| `I386_CLANG12 --std=c++14` | 10 243 | 0 | 100,3 | juste, 47 726 cycles |
| `I386_CLANG6 --std=c++14` | 8 147 | 0 | 102,1 | juste, 47 726 cycles |

Le même fichier, nommé `main.c` comme le fait `locod.sh`, s'arrête dès la première ligne, avec ou sans `-I`.

### Justesse

| Vérification | Résultat |
| :- | :- |
| C natif face à Aidge, `conv1` et `conv2`, une tuile de test, sortie int32 | 100 % de valeurs identiques |
| Idem, Quantizer flottant et virgule fixe, sortie int8 | 100 %, écart max 0 |
| Poids en paramètre face à poids statiques | sorties identiques octet par octet |
| Verilog simulé, couche réduite (4 filtres de `conv1`, tuile 16 x 16) : int32, virgule fixe, flottant | `return_port = 0` ; 55 619, 55 827 et 56 339 cycles |
| Contre-exemple : une valeur attendue faussée | `return_port = 1` |

**Dans un rapport de simulation Bambu, seul `return_port` compte** : « Simulation completed with SUCCESS » s'affiche
aussi avec la valeur faussée, car Bambu compare le circuit au logiciel, qui compte la même erreur. Le banc `check()`
renvoie le nombre de valeurs fausses.

### Cycles par multiplication

La couche réduite compte 31 x 31 x 3 x 4 = 11 532 multiplications pour 55 619 cycles, copies comprises : environ 4,8
cycles au total, de l'ordre de 4,5 hors copie de l'entrée et comparaison finale. C'est un ordre de grandeur, tiré d'une
seule simulation sur poids en BRAM.

## Non vérifié

- `locod.sh` de bout en bout, et donc l'intégration dans le design FPGA. Les commits épinglés de `locod-fpga` et de
  `LoCod-docker-nanoxplore` n'existent plus sur GitHub : `git submodule update --recursive` échoue. Leur branche par
  défaut existe, sa compatibilité n'est pas testée.
- L'interface réelle `param` et `result` en simulation : elle est synthétisée, mais le banc utilise des structures internes.
- La simulation de la couche complète, ainsi que `conv2`.
- L'image Docker officielle : même révision, mais pas le même binaire.

## Contenu

| Dossier | Fichiers |
| :- | :- |
| `couches/` | `conv_c.c` (la couche, modes `MODE_INT32`, `MODE_FLOATQ`, `MODE_FIXQ`, poids statiques ou `WEIGHTS_IN_PARAM`), `conv1_dims.h`, `conv1_weights.h`, `conv2_dims.h`, `gdn.c`, `gdn_div.c`, `gdn_newton.c` |
| `reduite/` | banc auto-vérifiant : `check_MODE_*.c`, `check_cpp.cpp`, poids, entrée, valeurs attendues, `tb.xml` |
| `scripts/` | `gen_c.py` (extrait les couches d'Aidge), `gen_cpp.py`, `small_gen.py`, `native.sh`, `campaign.sh`, `sim.sh`, `sim2.sh`, `sim3.sh`, `asc.sh`, `gdn2.sh` |

Les poids de `conv2` (1,4 Mo de texte) ne sont pas versionnés : `gen_c.py` les régénère. `gen_cpp.py` copie des en-têtes
de l'export Aidge, qui ne sont pas versionnés non plus.

## Reproduire

Les scripts attendent `/w` (volume contenant LoCod et Bambu extrait dans `/w/bambu/squashfs-root`), `/out` (dossier de
travail, avec le contenu à plat de `couches/` et de `scripts/`, et `reduite/` sous le nom `small/`) et `/workspace` (ce dépôt,
avec le checkpoint). Ils ont tourné dans deux images locales :

- une image Aidge (`aidge-in-space`, plus `g++`), pour `gen_c.py`, `gen_cpp.py`, `small_gen.py` et `native.sh` ;
- `ubuntu:22.04` avec `verilator iverilog make g++ gcc-multilib g++-multilib perl perl-modules-5.34`, pour Bambu.

```
python gen_c.py && python gen_cpp.py && python small_gen.py && sh native.sh    # image Aidge
bash campaign.sh base|cpp|c|big ; bash gdn2.sh ; bash asc.sh ; bash sim.sh ; bash sim3.sh    # image Bambu
```

Trois pièges rencontrés avec l'AppImage :

- `AppRun` change de dossier : les fichiers générés arrivent dans `squashfs-root/usr`. Il faut appeler `usr/bin/bambu` en
  définissant `APPDIR`, `PATH` et `LD_LIBRARY_PATH`.
- Bambu exige un simulateur présent, même sans simulation, et les en-têtes 32 bits (`gcc-multilib`).
- `verilator` a besoin des modules Perl, absents avec `--no-install-recommends`.
