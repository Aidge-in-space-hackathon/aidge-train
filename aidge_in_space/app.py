"""! @brief Interface Gradio du banc d'essai.

Lancement : @c python -m aidge_in_space.app puis http://127.0.0.1:7860. Chaque action est aussi
exposée par l'API Gradio (api_name : compare, explore, benchmark, profile, compare_encoders,
check_backends), utilisable avec gradio_client.
"""
import argparse
import ipaddress
import os
import sys
import time
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from PIL import Image

from . import backends, bench, profiling, ui_training
from .codecs import LearnedCodec, all_codecs, closest_level, discover, get
from .images import MODE_LABELS, convert, datasets, download_kodak, load_image
from .metrics import MS_SSIM_MIN_SIDE

## Choix du mode d'image : (libellé, identifiant).
MODE_CHOICES = [(label, key) for key, label in MODE_LABELS.items()]
## Côté de la zone zoomée, en pixels de l'image.
ZOOM_SIZE = 128
## Agrandissement de la zone zoomée.
ZOOM_FACTOR = 3
## Tailles d'image proposées pour le profilage : libellé -> (H, W).
PROFILE_PRESETS = {"512 x 512": (512, 512), "Kodak 768 x 512": (512, 768), "1024 x 1024": (1024, 1024),
                   "2048 x 2048": (2048, 2048)}


# ---------------------------------------------------------------- choix proposés

def codec_choices(mode, learned_only=False):
    """! @brief Codecs disponibles dans un mode, pour un menu Gradio.

    @param mode          @c "gris" ou @c "rgb".
    @param learned_only  True pour ne garder que les codecs appris.
    @return Liste de couples (libellé, identifiant).
    """
    out = []
    for c in all_codecs():
        if mode in c.modes and c.available()[0] and (not learned_only or isinstance(c, LearnedCodec)):
            out.append((c.label, c.key))
    return out


def backend_choices():
    """! @brief Moteurs disponibles : liste de couples (libellé, identifiant)."""
    return [(backends.BACKENDS[k], k) for k in backends.available_backends()]


def level_choices(key, mode):
    """! @brief Niveaux d'un codec : liste de couples (libellé, indice dans levels(mode))."""
    if not key:
        return []
    return [(lvl.label, i) for i, lvl in enumerate(get(key).levels(mode))]


def dataset_choices():
    """! @brief Noms des jeux d'images disponibles."""
    return list(datasets())


def image_choices(dataset):
    """! @brief Noms de fichier des images d'un jeu."""
    return [p.name for p in datasets().get(dataset, [])]


def codecs_markdown():
    """! @brief Tableau Markdown de tous les codecs enregistrés (onglet « Codecs et aide »)."""
    lines =["| Codec | Famille | Modes | Description |", "| :- | :- | :- | :- |"]
    for c in all_codecs():
        ok, why = c.available()
        desc = c.description + ("" if ok else f" **Indisponible : {why}**")
        lines.append(f"| {c.label} (`{c.key}`) | {c.family} | {', '.join(c.modes)} | {desc} |")
    return "\n".join(lines)


def _image(dataset, name, upload, mode):
    """! @brief Image à traiter : l'image importée si elle existe, sinon celle choisie dans le jeu.

    @return Couple (image uint8, titre).
    @throws gr.Error si aucune image n'est choisie.
    """
    if upload is not None:
        return convert(upload, mode), "image importée"
    paths = {p.name: p for p in datasets().get(dataset, [])}
    if name not in paths:
        raise gr.Error("Choisissez une image (ou importez-en une).")
    return load_image(paths[name], mode), name


def _selected(key, level_idx, mode):
    """! @brief Codec et niveau choisis dans un onglet.

    @param key        Identifiant du codec.
    @param level_idx  Indice du niveau dans levels(mode) (valeur du menu, éventuellement texte).
    @param mode       @c "gris" ou @c "rgb".
    @return Couple (Codec, Level).
    @throws gr.Error si le codec ou le niveau est absent ou invalide.
    """
    if not key or level_idx is None:
        raise gr.Error("Choisissez un codec et un niveau.")
    codec = get(key)
    levels = codec.levels(mode)
    try:
        return codec, levels[int(level_idx)]
    except (ValueError, IndexError):
        raise gr.Error(f"Niveau invalide : {level_idx}.") from None


def _zoom(img, zx, zy):
    """! @brief Zone carrée agrandie ZOOM_FACTOR fois (plus proche voisin), pour comparer les détails.

    @param img  Image uint8.
    @param zx   Position horizontale de la zone, en % de la marge disponible.
    @param zy   Position verticale de la zone, en %.
    @return Image PIL.
    """
    H, W = img.shape[:2]
    s = min(ZOOM_SIZE, H, W)
    x0 = int(round((W - s) * zx / 100))
    y0 = int(round((H - s) * zy / 100))
    crop = Image.fromarray(img[y0:y0 + s, x0:x0 + s])
    return crop.resize((s * ZOOM_FACTOR, s * ZOOM_FACTOR), Image.NEAREST)


def _row(row, res):
    """! @brief Ligne de tableau affichée pour une mesure de bench.evaluate()."""
    out = {
        "codec": row["codec_label"], "niveau": row["niveau_label"], "moteur": row["moteur"],
        "débit": row["débit"], "bpp": row["bpp"], "taux": row["taux"],
        "PSNR (dB)": row["psnr"], "MS-SSIM": row["ms_ssim"],
        "encodage (ms)": row["t_encodage_ms"], "dont analyse (ms)": row["t_analyse_ms"],
        "décodage (ms)": row["t_decodage_ms"],
    }
    info = {k: v for k, v in res.info.items() if k != "latent"}
    out["latent"] = res.info.get("latent", "-")
    out["détails"] = ", ".join(f"{k} : {v}" for k, v in info.items())
    return out


def _round(df, digits=3):
    """! @brief Arrondit un DataFrame pour l'affichage (laisse un DataFrame vide tel quel)."""
    return df.round(digits) if not df.empty else df


# ---------------------------------------------------------------- onglet Comparer

def compare(dataset, name, upload, mode, keys, target, backend, real_rate, zx, zy, progress=gr.Progress()):
    """! @brief Onglet « Comparer » : une image, plusieurs codecs au niveau le plus proche d'un débit visé.

    Un codec en erreur est signalé (gr.Warning) et sauté.
    @param dataset    Jeu d'images.
    @param name       Image du jeu.
    @param upload     Image importée (PIL) ou None.
    @param mode       @c "gris" ou @c "rgb".
    @param keys       Identifiants des codecs.
    @param target     Débit visé, en bpp.
    @param backend    Moteur de l'encodeur des codecs appris.
    @param real_rate  Débit réel ou estimé.
    @param zx         Position horizontale du zoom, en %.
    @param zy         Position verticale du zoom, en %.
    @param progress   Barre de progression Gradio.
    @return Triplet (galerie des reconstructions, galerie des zooms, tableau des mesures).
    @throws gr.Error si aucun codec n'est coché.
    """
    if not keys:
        raise gr.Error("Cochez au moins un codec.")
    img, title = _image(dataset, name, upload, mode)
    H, W = img.shape[:2]
    gallery = [(Image.fromarray(img), f"Original · {title} · {W}x{H}")]
    zooms = [(_zoom(img, zx, zy), "Original")]
    rows = []
    for i, key in enumerate(keys):
        codec = get(key)
        progress(i / len(keys), desc=f"{codec.label} : recherche du niveau")
        try:
            level = closest_level(codec, img, mode, target)
            progress((i + 0.5) / len(keys), desc=f"{codec.label} · {level.label}")
            row, res = bench.evaluate(codec, img, level, mode, backend, real_rate)
        except Exception as e:
            gr.Warning(f"{codec.label} : {e}")
            continue
        caption = f"{codec.label} · {level.label} · {row['bpp']:.3f} bpp · {row['psnr']:.2f} dB"
        gallery.append((Image.fromarray(res.recon), caption))
        zooms.append((_zoom(res.recon, zx, zy), f"{codec.label} · {row['bpp']:.3f} bpp"))
        rows.append(_row(row, res))
    return gallery, zooms, _round(pd.DataFrame(rows))


# ---------------------------------------------------------------- onglet Explorer

def explore(dataset, name, upload, mode, key, level_idx, backend, real_rate):
    """! @brief Onglet « Explorer un codec » : un codec à un niveau précis, avec la carte d'erreur.

    Les paramètres sont ceux de compare(), plus le codec (@p key) et l'indice du niveau (@p level_idx).
    @return Couple (galerie original / reconstruction / erreur x 4, tableau des mesures).
    """
    codec, level = _selected(key, level_idx, mode)
    img, _ = _image(dataset, name, upload, mode)
    row, res = bench.evaluate(codec, img, level, mode, backend, real_rate)
    diff = np.abs(img.astype(np.int16) - res.recon.astype(np.int16))
    diff = (np.clip(diff * 4, 0, 255)).astype(np.uint8)          # erreur amplifiée x4
    gallery = [(Image.fromarray(img), "Original"),
               (Image.fromarray(res.recon), f"{level.label} · {row['bpp']:.3f} bpp · {row['psnr']:.2f} dB"),
               (Image.fromarray(diff), "|erreur| x 4")]
    return gallery, _round(pd.DataFrame([_row(row, res)]))


# ---------------------------------------------------------------- onglet Benchmark

def image_count_slider(n):
    """! @brief Bornes du curseur « Nombre d'images » pour un jeu de @p n images.

    Gradio refuse un curseur dont le minimum égale le maximum : le maximum vaut au moins 2, même pour
    un jeu vide ou d'une seule image (le benchmark prend alors les images qui existent).
    @param n  Nombre d'images du jeu.
    @return Dictionnaire @c maximum et @c value, à passer à gr.Slider ou à gr.update.
    """
    return {"maximum": max(2, n), "value": min(4, max(1, n))}


def run_benchmark(dataset, n_images, mode, keys, points, backend, real_rate, name, ref, metric,
                  progress=gr.Progress()):
    """! @brief Onglet « Benchmark » : lance une campagne, l'enregistre et l'affiche.

    @param dataset    Jeu d'images.
    @param n_images   Nombre d'images du jeu à utiliser.
    @param mode       @c "gris" ou @c "rgb".
    @param keys       Identifiants des codecs.
    @param points     Niveaux par codec.
    @param backend    Moteur de l'encodeur des codecs appris.
    @param real_rate  Débit réel ou estimé.
    @param name       Nom de la campagne (jamais écrasée, voir bench.campaign_name()).
    @param ref        Courbe de référence du BD-rate.
    @param metric     Qualité du BD-rate (@c psnr ou @c ms_ssim_db).
    @param progress   Barre de progression Gradio.
    @return Statut (Markdown), mise à jour de la liste des campagnes, puis les sorties de show_runs().
    @throws gr.Error si aucun codec n'est coché ou si le jeu d'images est vide.
    """
    if not keys:
        raise gr.Error("Cochez au moins un codec.")
    paths = datasets().get(dataset, [])[:int(n_images)]
    if not paths:
        raise gr.Error("Jeu d'images vide.")
    t0 = time.time()
    df, path, errors = bench.run_benchmark(keys, paths, mode, int(points), backend, real_rate, name,
                                           progress=lambda f, d: progress(f, desc=d))
    if path is None:
        status = "Aucune mesure réussie : rien n'a été enregistré."
        selected = []
    else:
        status = (f"Campagne **{path.stem}** : {len(df)} mesures en {time.time() - t0:.0f} s, "
                  f"enregistrée dans `{path}`.")
        selected = [path.stem]
    if errors:
        status += f"\n\nErreurs ({len(errors)}) :\n" + "\n".join(f"- {e}" for e in errors[:10])
    return (status, gr.update(choices=bench.list_runs(), value=selected)) + show_runs(selected, ref, metric)


def show_runs(names, ref, metric):
    """! @brief Affiche une ou plusieurs campagnes enregistrées.

    @param names   Noms des campagnes.
    @param ref     Courbe de référence du BD-rate ; remplacée par JPEG (ou la première courbe) si absente.
    @param metric  Qualité du BD-rate.
    @return Figures PSNR, MS-SSIM et temps, table des BD-rates, table des points, menu de référence
            mis à jour, et chemin du CSV si une seule campagne est affichée.
    """
    df =bench.load_runs(names or [])
    agg = bench.aggregate(df)
    curves = list(dict.fromkeys(agg["courbe"])) if not agg.empty else []
    if ref not in curves:
        ref = next((c for c in curves if c.startswith("JPEG")), curves[0] if curves else None)
    bd = bench.bd_table(agg, ref, metric) if ref else pd.DataFrame()
    cols = ["courbe", "niveau_label", "images", "bpp", "psnr", "ms_ssim", "t_encodage_ms", "t_analyse_ms",
            "t_decodage_ms"]
    table = agg[cols] if not agg.empty else agg
    return (bench.rd_figure(agg, "psnr"), bench.rd_figure(agg, "ms_ssim_db"), bench.time_figure(agg),
            _round(bd, 2), _round(table, 4), gr.update(choices=curves, value=ref),
            str(bench.BENCH_DIR / f"{names[0]}.csv") if names and len(names) == 1 else None)


# ---------------------------------------------------------------- onglet Encodeur embarqué

def _input_shape(codec, preset):
    """! @brief Forme d'entrée (1, C, H, W) de l'encodeur d'un codec appris pour une taille prédéfinie."""
    H, W = PROFILE_PRESETS[preset]
    return 1, codec.in_ch, H, W


def profile(key, level_idx, preset, mode):
    """! @brief Onglet « Encodeur embarqué » : profil couche par couche de la partie embarquée.

    Calculé sur l'architecture non entraînée : aucun téléchargement de poids.
    @param key        Identifiant du codec appris.
    @param level_idx  Indice du niveau.
    @param preset     Clé de PROFILE_PRESETS.
    @param mode       @c "gris" ou @c "rgb".
    @return Triplet (résumé Markdown, tableau des couches, figure de répartition des MAC).
    """
    codec, level = _selected(key, level_idx, mode)
    shape = _input_shape(codec, preset)
    df = profiling.profile_module(codec.onboard_module(codec.make_model(level)), shape)
    s = profiling.summarize(df)
    n_params = f"{s['paramètres']:,}".replace(",", " ")
    md = (f"**{codec.label}** ({codec.architecture(level)}), entrée {'x'.join(map(str, shape[1:]))} : "
          f"**{s['GMAC']:.2f} GMAC**, {s['sqrt+div (M)']:.1f} M racines/divisions, "
          f"{n_params} paramètres ({s['poids fp32 (Mo)']:.2f} Mo en fp32, {s['poids int8 (Mo)']:.2f} Mo en int8), "
          f"plus grande activation {s['plus grande activation fp32 (Mo)']:.1f} Mo, "
          f"pic entrée + sortie d'une couche {s['pic entrée+sortie fp32 (Mo)']:.1f} Mo (fp32).")
    fig = _mac_figure(df)
    return md, _round(df, 3), fig


def _mac_figure(df):
    """! @brief Histogramme de la part des MAC par couche (convolutions en bleu, GDN en orange)."""
    fig =Figure(figsize=(8, 3.5))
    ax = fig.subplots()
    colors = ["tab:blue" if t.startswith("Conv") else "tab:orange" for t in df["type"]]
    ax.bar(df["couche"], df["% MAC"], color=colors)
    ax.set_ylabel("% des MAC")
    ax.set_title("Répartition du calcul (bleu : convolutions, orange : GDN)")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def compare_encoders(preset, mode):
    """! @brief Totaux du profil de chaque architecture de chaque codec appris (un niveau par architecture).

    @param preset  Clé de PROFILE_PRESETS.
    @param mode    @c "gris" ou @c "rgb".
    @return Tableau : codec, architecture, niveaux concernés et totaux de profiling.summarize().
    """
    rows = []
    for c in all_codecs():
        if not isinstance(c, LearnedCodec) or mode not in c.modes or not c.available()[0]:
            continue
        seen = {}
        for level in c.levels(mode):
            arch = c.architecture(level)
            seen.setdefault(arch, []).append(level)
        for arch, levels in seen.items():
            module = c.onboard_module(c.make_model(levels[0]))
            s = profiling.summarize(profiling.profile_module(module, _input_shape(c, preset)))
            rows.append({"codec": c.label, "architecture": arch,
                         "niveaux": f"{levels[0].label} → {levels[-1].label}" if len(levels) > 1 else levels[0].label,
                         **s})
    df = pd.DataFrame(rows)
    return _round(df, 2)


def check_backends(key, level_idx, preset, mode, progress=gr.Progress()):
    """! @brief Exécute l'encodeur entraîné sur chaque moteur et compare les sorties à PyTorch.

    Entrée aléatoire (graine fixe) ; le premier passage (export ONNX, compilation) est chronométré
    à part.
    @param key        Identifiant du codec appris.
    @param level_idx  Indice du niveau.
    @param preset     Clé de PROFILE_PRESETS.
    @param mode       @c "gris" ou @c "rgb".
    @param progress   Barre de progression Gradio.
    @return Tableau : moteur, temps de préparation, temps d'un passage, écart max et % de latents
            arrondis identiques à PyTorch.
    """
    codec, level = _selected(key, level_idx, mode)
    enc = codec.encoder(level)
    x = np.random.default_rng(0).random(_input_shape(codec, preset), dtype=np.float32)
    ref = None
    rows = []
    names = backends.available_backends()
    n = len(names)
    for i, b in enumerate(names):
        progress(i / n, desc=f"{backends.BACKENDS[b]} : préparation (export ONNX, compilation)")
        t0 = time.perf_counter()
        run = backends.get_runner(b, enc, x.shape)
        y = run(x)                               # premier passage (échauffement)
        t_prep = time.perf_counter() - t0
        progress((i + 0.5) / n, desc=f"{backends.BACKENDS[b]} : mesure")
        t0 = time.perf_counter()
        y = run(x)
        t_run = time.perf_counter() - t0
        if ref is None:
            ref = y
        rows.append({"moteur": backends.BACKENDS[b], "préparation + 1er passage (s)": t_prep,
                     "temps (ms)": 1e3 * t_run, "écart max vs PyTorch": float(np.abs(y - ref).max()),
                     "latents arrondis identiques (%)": 100 * float(np.mean(np.round(y) == np.round(ref)))})
    return _round(pd.DataFrame(rows), 6)


# ---------------------------------------------------------------- liens entre onglets

def mode_updates(m):
    """! @brief Mises à jour des menus de codecs quand le mode change.

    @param m  @c "gris" ou @c "rgb".
    @return Tuple de gr.update pour (keys, ex_codec, ex_level, b_keys, p_codec, p_level).
    """
    all_keys = codec_choices(m)
    learned = codec_choices(m, learned_only=True)
    return (gr.update(choices=all_keys, value=[v for _, v in all_keys]),
            gr.update(choices=all_keys, value=None), gr.update(choices=[], value=None),
            gr.update(choices=all_keys, value=[v for _, v in all_keys]),
            gr.update(choices=learned, value=None), gr.update(choices=[], value=None))


def codec_menus(m):
    """! @brief Relit le registre de codecs sans le redécouvrir (session neuve ou autre session ayant rechargé).

    @param m  Mode courant.
    @return Même forme que reload_codecs() : mode_updates(), puis le tableau des codecs.
    """
    return mode_updates(m) + (codecs_markdown(),)


def reload_codecs(m):
    """! @brief Redécouvre les codecs (bouton « Recharger les codecs », ou fin d'un entraînement).

    @param m  Mode courant.
    @return Les mises à jour de mode_updates(), puis le tableau des codecs (onglet « Codecs et aide »).
    """
    discover()
    backends.clear_runners()
    return mode_updates(m) + (codecs_markdown(),)


# ---------------------------------------------------------------- accès

def parse_users(spec):
    """! @brief Comptes de l'interface depuis AIDGE_UI_USERS (@c login:motdepasse,login2:motdepasse2).

    Chaque entrée est coupée au premier @c :. Les virgules dans un mot de passe ne sont pas prises en charge.
    @param spec  Texte de la variable, ou None.
    @return Liste de couples (login, mot de passe) ; vide si @p spec est vide.
    @throws ValueError si une entrée est mal formée (le mot de passe n'apparaît jamais dans le message).
    """
    if spec is None or not spec.strip():
        return []
    users = []
    for k, entry in enumerate(spec.split(","), start=1):
        login, sep, password = entry.strip().partition(":")
        login, password = login.strip(), password.strip()
        if not sep or not login or not password:
            raise ValueError(f"AIDGE_UI_USERS mal formé : entrée n°{k}")
        users.append((login, password))
    return users


def is_loopback(host):
    """! @brief Indique si une adresse d'écoute n'est joignable que depuis cette machine.

    @param host  Nom ou adresse (@c localhost, 127.0.0.1, ::1…).
    @return Faux pour une adresse vide, @c 0.0.0.0, @c :: ou toute autre adresse.
    """
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def check_exposure(host, users, bind, in_docker):
    """! @brief Refuse d'exposer l'interface sur le réseau sans authentification.

    @param host       Adresse d'écoute demandée.
    @param users      Comptes (parse_users()).
    @param bind       AIDGE_UI_BIND : adresse de l'hôte où Docker publie le port.
    @param in_docker  Vrai si le programme tourne dans un conteneur (alors @p bind prime sur @p host).
    @throws ValueError si l'adresse effective n'est pas locale et qu'aucun compte n'est défini.
    """
    effective = bind if in_docker and bind else host
    if not is_loopback(effective) and not users:
        raise ValueError(f"Exposition réseau refusée : l'interface écouterait sur {effective} sans "
                         "authentification. Définissez AIDGE_UI_USERS (login:motdepasse) ou gardez 127.0.0.1.")


# ---------------------------------------------------------------- interface

## Texte de l'onglet « Codecs et aide » (Markdown).
HELP = """
### Mesures
- **bpp** : bits par pixel du flux compressé. **Réel** : taille du fichier produit par le codeur arithmétique
  (avec en-tête), décodé et vérifié. **Estimé** : entropie prédite par le modèle (plus rapide, écart de l'ordre de 0,5 %).
- **PSNR** et **MS-SSIM** sur l'image 8 bits reconstruite (MS-SSIM à 5 échelles, image d'au moins {MS_SSIM_MIN_SIDE} px de côté).
  MS-SSIM (dB) = -10 log10(1 - MS-SSIM).
- **BD-rate** : écart de débit moyen à qualité égale par rapport à la courbe de référence (Bjøntegaard).
  Négatif = économie de débit. Il faut au moins 4 points par courbe et une plage de qualité commune.
- **Temps** : sur un thread de ce PC, donc indicatifs. L'**encodage** est ce qui tournerait à bord
  (transformée d'analyse + codage entropique en Python pur), l'**analyse** la seule transformée
  (exécutée par le moteur choisi). Le backend CPU d'Aidge est une implémentation de référence, lente sur PC.

### Ajouter un algorithme ou un modèle
Créer un fichier dans `aidge_in_space/codecs/` avec une fonction `codecs()` qui renvoie des objets `Codec` :
il apparaît automatiquement ici et dans la ligne de commande (bouton « Recharger les codecs » si l'interface tourne).

```python
from .base import Codec, Level, Result

class MonCodec(Codec):
    key, label, family = "mon-codec", "Mon codec", "Classique"
    modes = ("gris", "rgb")
    def levels(self, mode):                # du plus bas au plus haut débit
        return [Level(q, f"qualité {q}") for q in (10, 50, 90)]
    def run(self, img, level, mode, backend=None, real_rate=True):
        ...                                # compresser puis décompresser img (uint8)
        return Result(recon, n_bits, "réel", {"encodage": t_enc, "décodage": t_dec})

def codecs():
    return [MonCodec()]
```
Pour un réseau, hériter de `LearnedCodec` (voir `bmshj2018.py`) : il suffit de fournir l'architecture et
le chargement des poids ; padding, moteurs (PyTorch, ONNX Runtime, Aidge), codage entropique et profilage sont communs.
Un nouveau modèle d'entropie s'ajoute en écrivant une sous-classe de `LatentCoding` (`aidge_in_space/coding.py`),
affectée à l'attribut `coding` du codec.
Les nouveaux modèles Ballé 2017 entraînés par le notebook (`outputs/checkpoints/bls2017_*.pt`) sont détectés seuls.

### Ligne de commande (machine d'entraînement, sans interface)
```
python -m aidge_in_space.bench --list
python -m aidge_in_space.bench --codecs jpeg,jpeg2000,bmshj2018-hyperprior --images 24 --mode rgb --points 6
```
Les campagnes sont écrites dans `outputs/bench/*.csv` (jamais écrasées) et relues par l'onglet Benchmark.
""".replace("{MS_SSIM_MIN_SIDE}", str(MS_SSIM_MIN_SIDE))


def build():
    """! @brief Construit l'interface (sans la lancer).

    @return gr.Blocks.
    """
    ds = dataset_choices()
    ds0 = ds[0] if ds else None
    imgs0 = image_choices(ds0)
    bk = backend_choices()

    with gr.Blocks(title="AIDGE IN SPACE · banc d'essai") as demo:
        gr.Markdown("# AIDGE IN SPACE · banc d'essai de compression d'image\n"
                    "Comparer des algorithmes et des modèles de compression pour choisir ce qui ira sur la carte.")

        with gr.Row():
            mode = gr.Radio(MODE_CHOICES, value="gris", label="Mode", scale=1)
            backend = gr.Dropdown(bk, value="torch", label="Moteur de l'encodeur (codecs appris)", scale=1)
            real_rate = gr.Checkbox(value=True, label="Débit réel (codeur arithmétique, plus lent)", scale=1)
            reload_btn = gr.Button("Recharger les codecs", scale=0)

        with gr.Tabs() as tabs:
            # ------------------------------------------------ Comparer
            with gr.Tab("Comparer"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=280):
                        dataset = gr.Dropdown(ds, value=ds0, label="Jeu d'images")
                        image_name = gr.Dropdown(imgs0, value=imgs0[0] if imgs0 else None, label="Image")
                        upload = gr.Image(type="pil", label="… ou importer une image", sources=["upload", "clipboard"],
                                          height=160)
                        keys = gr.CheckboxGroup(codec_choices("gris"), value=[v for _, v in codec_choices("gris")],
                                                label="Codecs")
                        target = gr.Slider(0.05, 2.0, value=0.25, step=0.01, label="Débit visé (bpp)")
                        with gr.Row():
                            zx = gr.Slider(0, 100, value=50, step=1, label="Zoom : position x (%)")
                            zy = gr.Slider(0, 100, value=50, step=1, label="Zoom : position y (%)")
                        go = gr.Button("Comparer", variant="primary")
                    with gr.Column(scale=3):
                        gallery = gr.Gallery(label="Reconstructions (niveau le plus proche du débit visé)",
                                             columns=3, height=520, object_fit="contain")
                        zoom = gr.Gallery(label=f"Zoom x{ZOOM_FACTOR}", columns=6, height=230, object_fit="contain")
                        table = gr.Dataframe(label="Mesures", wrap=True)
                go.click(compare, [dataset, image_name, upload, mode, keys, target, backend, real_rate, zx, zy],
                         [gallery, zoom, table], api_name="compare")

            # ------------------------------------------------ Explorer
            with gr.Tab("Explorer un codec"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=280):
                        ex_codec = gr.Dropdown(codec_choices("gris"), label="Codec")
                        ex_level = gr.Dropdown([], label="Niveau", allow_custom_value=True)
                        ex_go = gr.Button("Compresser", variant="primary")
                        gr.Markdown("L'image est celle choisie dans l'onglet Comparer.")
                    with gr.Column(scale=3):
                        ex_gallery = gr.Gallery(columns=3, height=420, object_fit="contain", label="Résultat")
                        ex_table = gr.Dataframe(label="Mesures", wrap=True)
                ex_codec.change(lambda k, m: gr.update(choices=level_choices(k, m), value=0 if k else None),
                                [ex_codec, mode], ex_level)
                ex_go.click(explore, [dataset, image_name, upload, mode, ex_codec, ex_level, backend, real_rate],
                            [ex_gallery, ex_table], api_name="explore")

            # ------------------------------------------------ Benchmark
            with gr.Tab("Benchmark"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=280):
                        b_dataset = gr.Dropdown(ds, value=ds0, label="Jeu d'images")
                        b_n = gr.Slider(1, step=1, label="Nombre d'images", **image_count_slider(len(imgs0)))
                        b_keys = gr.CheckboxGroup(codec_choices("gris"), value=[v for _, v in codec_choices("gris")],
                                                  label="Codecs")
                        b_points = gr.Slider(2, 12, value=6, step=1, label="Points par courbe")
                        b_name = gr.Textbox(label="Nom de la campagne (facultatif)")
                        b_go = gr.Button("Lancer la campagne", variant="primary")
                        b_status = gr.Markdown()
                        gr.Markdown("Débit estimé conseillé pour les grosses campagnes : décochez « Débit réel ».")
                    with gr.Column(scale=3):
                        with gr.Row():
                            runs = gr.Dropdown(bench.list_runs(), multiselect=True, label="Campagnes affichées")
                            ref = gr.Dropdown([], label="Référence du BD-rate")
                            metric = gr.Radio([("PSNR", "psnr"), ("MS-SSIM (dB)", "ms_ssim_db")], value="psnr",
                                              label="BD-rate sur")
                        with gr.Row():
                            fig_psnr = gr.Plot(label="PSNR")
                            fig_ssim = gr.Plot(label="MS-SSIM")
                        bd = gr.Dataframe(label="BD-rate (négatif = moins de bits que la référence à qualité égale)")
                        fig_time = gr.Plot(label="Temps d'encodage")
                        agg_table = gr.Dataframe(label="Points des courbes (moyennes sur les images)")
                        csv_file = gr.File(label="CSV de la campagne")
                b_dataset.change(lambda d: gr.update(**image_count_slider(len(image_choices(d)))), b_dataset, b_n)
                show_outputs = [fig_psnr, fig_ssim, fig_time, bd, agg_table, ref, csv_file]
                b_go.click(run_benchmark, [b_dataset, b_n, mode, b_keys, b_points, backend, real_rate, b_name, ref,
                                           metric], [b_status, runs] + show_outputs, api_name="benchmark")
                for comp in (runs, ref, metric):      # .input : seulement les actions de l'utilisateur
                    comp.input(show_runs, [runs, ref, metric], show_outputs)

            # ------------------------------------------------ Encodeur embarqué
            with gr.Tab("Encodeur embarqué"):
                gr.Markdown("Coût de la partie qui tourne **à bord** (transformée d'analyse, et pour l'hyperprior "
                            "h_a + h_s), calculé sur l'architecture : aucun poids n'est nécessaire.")
                with gr.Row():
                    p_codec = gr.Dropdown(codec_choices("gris", learned_only=True), label="Codec appris")
                    p_level = gr.Dropdown([], label="Niveau", allow_custom_value=True)
                    p_size = gr.Dropdown(list(PROFILE_PRESETS), value="512 x 512", label="Taille de l'image")
                with gr.Row():
                    p_go = gr.Button("Profiler couche par couche", variant="primary")
                    p_all = gr.Button("Comparer tous les encodeurs")
                    p_check = gr.Button("Vérifier les moteurs (PyTorch / ONNX Runtime / Aidge)")
                p_md = gr.Markdown()
                p_fig = gr.Plot(label="Répartition du calcul")
                p_table = gr.Dataframe(label="Couches", wrap=True)
                p_check_table = gr.Dataframe(label="Moteurs : mêmes latents ? (entrée aléatoire, 1 thread)")
                p_codec.change(lambda k, m: gr.update(choices=level_choices(k, m), value=0 if k else None),
                               [p_codec, mode], p_level)
                p_go.click(profile, [p_codec, p_level, p_size, mode], [p_md, p_table, p_fig], api_name="profile")
                p_all.click(compare_encoders, [p_size, mode], p_table, api_name="compare_encoders")
                p_check.click(check_backends, [p_codec, p_level, p_size, mode], p_check_table,
                              api_name="check_backends")

            # ------------------------------------------------ Aide
            with gr.Tab("Codecs et aide"):
                codecs_md = gr.Markdown(codecs_markdown())
                gr.Markdown(HELP)

            # ------------------------------------------------ Données, Entraînement, Suivi
            mode_outputs = [keys, ex_codec, ex_level, b_keys, p_codec, p_level]
            ui_training.build_tabs(mode, mode_outputs + [codecs_md], reload_codecs, tabs=tabs,
                                  menus_fn=codec_menus)

        # ------------------------------------------------ liens entre onglets
        dataset.change(lambda d: gr.update(choices=image_choices(d), value=(image_choices(d) or [None])[0]),
                       dataset, image_name)

        mode.change(mode_updates, mode, mode_outputs)
        reload_btn.click(reload_codecs, mode, mode_outputs + [codecs_md])
    return demo


def main(argv=None):
    """! @brief Point d'entrée : télécharge Kodak si besoin, découvre les codecs et lance le serveur.

    @param argv  Arguments (--host, --port, --inbrowser) ; ceux de la ligne de commande si None.

    Variables d'environnement : AIDGE_UI_HOST, AIDGE_UI_USERS, AIDGE_UI_BIND, AIDGE_MAX_UPLOAD_MB.
    @throws SystemExit (code 2) si la configuration d'accès est invalide ou expose le réseau sans authentification.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Interface du banc d'essai AIDGE IN SPACE")
    ap.add_argument("--host", default=None, help="adresse d'écoute (défaut : AIDGE_UI_HOST, sinon 127.0.0.1)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--inbrowser", action="store_true")
    args = ap.parse_args(argv)
    host = args.host or os.environ.get("AIDGE_UI_HOST") or "127.0.0.1"
    try:
        users = parse_users(os.environ.get("AIDGE_UI_USERS"))
        check_exposure(host, users, os.environ.get("AIDGE_UI_BIND"), Path("/.dockerenv").exists())
        upload_mb = ui_training.max_upload_mb()
    except ValueError as e:
        print(e, file=sys.stderr)
        raise SystemExit(2) from None
    download_kodak()
    discover()
    build().queue().launch(server_name=host, server_port=args.port, inbrowser=args.inbrowser,
                           auth=users or None, max_file_size=f"{upload_mb}mb", theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
