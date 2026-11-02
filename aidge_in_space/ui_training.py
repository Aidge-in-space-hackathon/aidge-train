"""! @brief Onglets « Entraîner » (avec l'ajout d'images) et « Suivre » de l'interface Gradio.

L'interface n'entraîne jamais elle-même : elle écrit des jobs dans la file (aidge_in_space.jobs.JobStore)
et lit leurs statuts, leurs métriques et leurs journaux. Toute la logique est dans des fonctions pures,
testables sans lancer Gradio ; les callbacks de build_tabs() n'en sont que de minces adaptateurs qui
convertissent les erreurs en messages simples (gr.Error, gr.Warning). Contrat : docs/conception_entrainement.md, § 2.7.
"""
import functools
import http.client
import os
import re
import shutil
import tarfile
import threading
import time
import unicodedata
import uuid
import zipfile
from pathlib import Path

import gradio as gr
import pandas as pd
from matplotlib.figure import Figure
from PIL import Image

from . import config, ui_style
from .dataset_catalog import CATALOG, download_archive
from .images import IMAGE_EXTS, MODE_LABELS, list_images
from .jobs import STATUS_LABELS, JobStore
from .training.imaging import to_8bit
from .training.config import PRESETS, TrainConfig, schedule_for

## Choix de N proposés.
N_CHOICES = [32, 64, 96, 128, 192]
## Choix de λ proposés.
LAMBDA_CHOICES = [256, 512, 1024, 2048, 4096, 8192]
## Libellé du préréglage libre.
CUSTOM_PRESET = "Personnalisé"
## Préréglages affichés -> (clé de training.config.PRESETS ou None, λ cochés par défaut).
PRESET_CHOICES = {"Test rapide (quelques minutes)": ("test_rapide", [2048]),
                  "Standard (plusieurs heures, GPU conseillé)": ("standard", [512, 1024, 2048, 4096]),
                  CUSTOM_PRESET: (None, None)}
## Capacité BRAM du Zynq-7020 retenue : 140 BRAM36, environ 630 Ko, soit environ 615 Kio.
BRAM_BYTES = 630_000
## Taille d'image pour l'estimation des MAC par pixel.
ESTIMATE_SHAPE = (256, 256)
## Côté minimal conseillé d'une image d'entraînement (avertissement en dessous).
MIN_SIDE = 256
## Taille maximale par défaut d'un envoi, en Mo (surchargée par AIDGE_MAX_UPLOAD_MB).
DEFAULT_MAX_UPLOAD_MB = 500
## Taille décompressée maximale d'une archive : ce facteur fois la limite d'envoi.
UNZIP_FACTOR = 2
## Nombre maximal de fichiers dans une archive.
MAX_ZIP_FILES = 10_000
## Noms réservés de Windows, refusés tels quels.
WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                    *(f"LPT{i}" for i in range(1, 10))}
## Période de rafraîchissement de l'onglet Suivre, en secondes.
REFRESH_S = 5
## Nombre de vignettes montrées par jeu d'images.
N_SAMPLES = 6
## Colonnes du tableau des jobs.
TABLE_COLUMNS = ["n°", "nom", "auteur", "statut", "λ", "N", "mode", "progression (%)", "temps restant", "créé le"]
## Octets lus en fin de train.log pour en extraire les dernières lignes.
LOG_TAIL_BYTES = 65536
## Nombre maximal de messages d'avertissement affichés en fenêtre après un envoi.
MAX_WARNINGS = 5
## Copie par blocs de cette taille (octets).
_CHUNK = 1 << 20


# ---------------------------------------------------------------- jeux d'images

def max_upload_mb():
    """! @brief Taille maximale d'un envoi, en Mo.

    @return AIDGE_MAX_UPLOAD_MB (entier > 0) ou DEFAULT_MAX_UPLOAD_MB.
    @throws ValueError si la variable est mal formée.
    """
    raw = os.environ.get("AIDGE_MAX_UPLOAD_MB")
    if raw is None or not raw.strip():
        return DEFAULT_MAX_UPLOAD_MB
    try:
        value = int(raw.strip())
    except ValueError:
        value = 0
    if value <= 0:
        raise ValueError(f"AIDGE_MAX_UPLOAD_MB mal formé : un entier strictement positif est attendu (reçu {raw!r})")
    return value


def safe_name(name, max_len=60):
    """! @brief Nom de dossier ou de fichier assaini (ASCII, sans séparateur ni point de tête).

    Exemples : @c "Lune LRO (été)" donne @c "Lune_LRO_ete", @c "../../etc" donne @c "etc", @c "con" donne @c "_con".
    @param name     Nom d'origine.
    @param max_len  Longueur maximale du résultat.
    @return Le nom assaini.
    @throws ValueError si le résultat est vide.
    """
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._-")
    s = s[:max_len].rstrip("._-")
    if not s:
        raise ValueError(f"nom invalide : {name!r}")
    if s.split(".")[0].upper() in WINDOWS_RESERVED:
        s = "_" + s
    return s


class _Names:
    """! @brief Attribution de noms de fichiers libres dans un dossier, en temps amorti constant.

    Le contenu du dossier est lu une seule fois ; ensuite chaque nom de base garde en mémoire son prochain
    suffixe à essayer (@c stem, @c stem_2, @c stem_3…), si bien que n fichiers de même nom coûtent O(n).
    La comparaison ignore la casse (Windows). Aucun nom déjà présent n'est jamais rendu.
    """

    def __init__(self, folder):
        """! @brief Lit le contenu du dossier (vide s'il n'existe pas).
        @param folder  Dossier dont les noms sont réservés.
        """
        ## Dossier géré.
        self.folder = Path(folder)
        ## Noms (en minuscules) déjà pris.
        self._taken = {p.name.lower() for p in self.folder.iterdir()} if self.folder.is_dir() else set()
        ## Prochain indice à essayer, par nom complet en minuscules.
        self._next = {}

    def free(self, stem, suffix):
        """! @brief Réserve et renvoie le premier chemin libre folder/stem+suffix, puis stem_2, stem_3…
        @param stem    Nom sans extension.
        @param suffix  Extension avec le point.
        @return Le chemin réservé (le fichier n'est pas créé).
        """
        key = f"{stem}{suffix}".lower()
        k = self._next.get(key, 1)
        while True:
            name = f"{stem}{suffix}" if k == 1 else f"{stem}_{k}{suffix}"
            k += 1
            if name.lower() not in self._taken:
                break
        self._taken.add(name.lower())
        self._next[key] = k
        return self.folder / name


## Verrou qui sérialise le déplacement des fichiers préparés vers un jeu (deux envois simultanés).
_MOVE_LOCK = threading.Lock()
## Préfixe des dossiers de préparation d'un envoi (nom caché, ignoré par list_datasets).
STAGING_PREFIX = ".envoi-"


def _dangerous_member(info):
    """! @brief Indique si un membre d'archive est dangereux (chemin absolu, « .. », lien, chiffré).

    @param info  zipfile.ZipInfo du membre.
    @return Vrai si l'archive entière doit être refusée.
    """
    name = info.filename
    if name.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", name):
        return True
    if ".." in re.split(r"[/\\]", name):
        return True
    if ((info.external_attr >> 16) & 0o170000) == 0o120000:
        return True
    return bool(info.flag_bits & 0x1)


def _member_target(filename, names):
    """! @brief Choisit où écrire un membre d'archive, à plat, sous un nom assaini.

    @param filename  Nom du membre dans l'archive.
    @param names     _Names du dossier de destination.
    @return Couple (chemin réservé, None) pour une image, (None, message) pour un membre à ignorer.
    """
    parts = [p for p in re.split(r"[/\\]", filename) if p]
    base = parts[-1] if parts else ""
    if "__MACOSX" in parts or base.startswith("."):
        return None, f"« {filename} » ignoré (fichier système ou caché)."
    suffix = Path(base).suffix.lower()
    if suffix not in IMAGE_EXTS:
        return None, f"« {filename} » ignoré (ce n'est pas une image)."
    try:
        stem = safe_name(Path(base).stem)
    except ValueError:
        stem = "image"
    return names.free(stem, suffix), None


def _copy_member(src, target, total, max_total_bytes):
    """! @brief Copie en flux un membre d'archive, en comptant les octets réellement écrits (bombe d'archive).

    @param src              Flux binaire du membre.
    @param target           Fichier à écrire.
    @param total            Octets déjà extraits de l'archive.
    @param max_total_bytes  Taille décompressée maximale cumulée.
    @return Le nouveau total.
    @throws ValueError si la limite est dépassée.
    """
    with open(target, "wb") as out:
        while True:
            block = src.read(_CHUNK)
            if not block:
                return total
            total += len(block)
            if total > max_total_bytes:
                raise ValueError(f"archive refusée : taille décompressée supérieure à "
                                 f"{max_total_bytes / 2 ** 20:.0f} Mo")
            out.write(block)


def extract_zip(zip_path, dest, max_total_bytes, max_files=MAX_ZIP_FILES, names=None):
    """! @brief Extrait les images d'une archive zip, à plat, après contrôle de tous les membres.

    Aucun fichier n'est écrit avant que tous les noms aient été contrôlés (zip slip : chemins absolus,
    « .. », liens symboliques, archive chiffrée). La taille décompressée est bornée deux fois : sur les
    tailles déclarées, puis sur les octets réellement copiés (bombe zip).
    @param zip_path         Archive.
    @param dest             Dossier de destination (créé si besoin).
    @param max_total_bytes  Taille décompressée maximale cumulée.
    @param max_files        Nombre maximal de membres.
    @param names            _Names partagé entre plusieurs appels ; None : un nouveau, lu dans @c dest.
    @return Couple (images écrites, messages).
    @throws ValueError si l'archive est illisible, dangereuse ou trop grosse.
    """
    try:
        zf = zipfile.ZipFile(zip_path)
    except (zipfile.BadZipFile, OSError):
        raise ValueError("archive zip illisible") from None
    written, messages = [], []
    with zf:
        infos = zf.infolist()
        if len(infos) > max_files:
            raise ValueError(f"archive refusée : {len(infos)} fichiers, au plus {max_files} autorisés")
        for info in infos:
            if _dangerous_member(info):
                raise ValueError(f"archive refusée : chemin dangereux « {info.filename} »")
        declared = sum(i.file_size for i in infos)
        if declared > max_total_bytes:
            raise ValueError(f"archive refusée : taille décompressée {declared / 2 ** 20:.0f} Mo "
                             f"supérieure à {max_total_bytes / 2 ** 20:.0f} Mo")
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        names = names or _Names(dest)
        total = 0
        try:
            for info in infos:
                if info.is_dir():
                    continue
                target, message = _member_target(info.filename, names)
                if target is None:
                    messages.append(message)
                    continue
                written.append(target)             # enregistré avant l'écriture : nettoyé en cas d'échec
                with zf.open(info) as src:
                    total = _copy_member(src, target, total, max_total_bytes)
        except BaseException as e:
            for p in written:
                p.unlink(missing_ok=True)
            if isinstance(e, (zipfile.BadZipFile, RuntimeError, OSError, EOFError)):
                raise ValueError("archive zip illisible") from None
            raise
    return written, messages


def extract_tar(tar_path, dest, max_total_bytes, max_files=MAX_ZIP_FILES):
    """! @brief Extrait les images d'une archive tar (compressée ou non), à plat.

    Le contenu est lu en flux et écrit sous un nom assaini : ni les chemins de l'archive ni ses liens, ignorés,
    ne peuvent faire écrire hors de @p dest.
    @param tar_path         Archive.
    @param dest             Dossier de destination (créé si besoin).
    @param max_total_bytes  Taille décompressée maximale cumulée.
    @param max_files        Nombre maximal de membres.
    @return Couple (images écrites, messages).
    @throws ValueError si l'archive est illisible ou trop grosse.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    names, written, messages = _Names(dest), [], []
    total = 0
    try:
        with tarfile.open(tar_path) as tf:
            for i, info in enumerate(tf, start=1):
                if i > max_files:
                    raise ValueError(f"archive refusée : plus de {max_files} fichiers")
                if not info.isfile():
                    continue
                target, message = _member_target(info.name, names)
                if target is None:
                    messages.append(message)
                    continue
                written.append(target)             # enregistré avant l'écriture : nettoyé en cas d'échec
                total = _copy_member(tf.extractfile(info), target, total, max_total_bytes)
    except BaseException as e:
        for p in written:
            p.unlink(missing_ok=True)
        if isinstance(e, (tarfile.TarError, OSError, EOFError)):
            raise ValueError("archive tar illisible") from None
        raise
    return written, messages


def _check_image(path):
    """! @brief Vérifie qu'une image est lisible.

    @param path  Fichier image.
    @return (largeur, hauteur).
    @throws Exception si PIL ne sait pas la lire (y compris DecompressionBombError).
    """
    with Image.open(path) as im:
        im.verify()
    with Image.open(path) as im:
        return im.size


def _to_8bit_file(path):
    """! @brief Lit une image et la ramène à 8 bits si elle ne l'est pas.

    L'image ouverte est libérée avant le retour : Pillow projette parfois le fichier en mémoire (TIFF non
    compressé), ce qui empêcherait de le réécrire sous Windows.
    @param path  Fichier image.
    @return Couple (image 8 bits, format du fichier), ou (None, None) si l'image n'a pas besoin de conversion.
    """
    with Image.open(path) as im:
        fmt = im.format
        im.load()
        img8, changed = to_8bit(im)
    return (img8, fmt) if changed else (None, None)


def create_dataset(name, files, train_dir=None, max_upload_mb=None):
    """! @brief Crée (ou complète) un jeu d'images d'entraînement à partir de fichiers et d'archives zip.

    Tout est extrait et vérifié dans un dossier de préparation unique à l'envoi (@c .envoi-xxxx dans le dossier
    des jeux). Si quoi que ce soit échoue, seul ce dossier est supprimé : un jeu existant n'est pas modifié et
    deux envois simultanés ne s'effacent pas. En cas de succès, les images sont déplacées dans le jeu
    (créé si besoin) sans écraser aucun fichier (suffixe @c _2, @c _3…). Les images illisibles sont écartées,
    les images 16 bits ou flottantes sont converties en 8 bits (mise à l'échelle sur leur valeur maximale)
    et une image de moins de MIN_SIDE pixels de côté est gardée avec un avertissement.
    @param name           Nom du jeu (assaini).
    @param files          Chemins des fichiers envoyés (images ou zip).
    @param train_dir      Dossier des jeux ; None : config.TRAIN_DIR lu à l'appel.
    @param max_upload_mb  Taille maximale de l'envoi en Mo ; None : max_upload_mb().
    @return Couple (dossier du jeu, messages).
    @throws ValueError si le nom est invalide, s'il n'y a aucun fichier, si l'envoi est trop gros,
            si une archive est refusée ou si l'envoi ne contient aucune image valide.
    """
    folder_name = safe_name(name or "")
    paths = [Path(f) for f in (files or [])]
    if not paths:
        raise ValueError("Déposez au moins une image ou une archive zip.")
    limit_mb = globals()["max_upload_mb"]() if max_upload_mb is None else max_upload_mb
    sent = sum(p.stat().st_size for p in paths if p.exists())
    if sent > limit_mb * 2 ** 20:
        raise ValueError(f"Envoi trop gros : {sent / 2 ** 20:.0f} Mo, la limite est de {limit_mb} Mo.")
    root = Path(train_dir) if train_dir is not None else Path(config.TRAIN_DIR)
    dest = root / folder_name
    stage = root / f"{STAGING_PREFIX}{uuid.uuid4().hex}"
    stage.mkdir(parents=True)
    messages, images = [], []
    unzip_budget = UNZIP_FACTOR * limit_mb * 2 ** 20
    try:
        names = _Names(stage)
        for p in paths:
            ext = p.suffix.lower()
            if ext == ".zip":
                got, msgs = extract_zip(p, stage, unzip_budget, names=names)
                unzip_budget -= sum(g.stat().st_size for g in got)
                images += got
                messages += msgs
            elif ext in IMAGE_EXTS:
                try:
                    stem = safe_name(p.stem)
                except ValueError:
                    stem = "image"
                target = names.free(stem, ext)
                shutil.copyfile(p, target)
                images.append(target)
            else:
                messages.append(f"« {p.name} » ignoré (ni image, ni archive zip).")
        valid, converted = [], 0
        for img in images:
            try:
                w, h = _check_image(img)
                img8, fmt = _to_8bit_file(img)
                if img8 is not None:
                    img8.save(img, format=fmt)
                    converted += 1
            except Exception:
                img.unlink(missing_ok=True)
                messages.append(f"« {img.name} » supprimé : image illisible.")
                continue
            valid.append(img)
            if min(w, h) < MIN_SIDE:
                messages.append(f"« {img.name} » fait {w} x {h} px (moins de {MIN_SIDE}) : elle sera agrandie.")
        if converted:
            messages.append(f"{converted} image{'s' if converted > 1 else ''} en 16 bits "
                            f"{'ont été converties' if converted > 1 else 'a été convertie'} en 8 bits "
                            f"(mise à l'échelle sur leur valeur maximale).")
        if not valid:
            raise ValueError("Aucune image valide dans les fichiers envoyés.")
        with _MOVE_LOCK:
            dest.mkdir(parents=True, exist_ok=True)
            final = _Names(dest)
            for img in valid:
                os.replace(img, final.free(img.stem, img.suffix))
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return dest, messages


def install_catalog_dataset(key, train_dir=None, progress=None):
    """! @brief Télécharge un jeu du catalogue et l'installe dans le dossier des jeux.

    Un jeu déjà présent n'est pas retéléchargé. Sinon, l'archive est vérifiée, extraite dans un dossier de
    préparation, puis ses images passent par create_dataset (mêmes contrôles et même tout ou rien qu'un envoi).
    Le dossier de préparation porte un nom fixe : s'il reste d'un arrêt brutal, le prochain essai le supprime.
    Appels à sérialiser pour un même jeu (concurrency_limit=1 côté Gradio).
    @param key        Clé de dataset_catalog.CATALOG.
    @param train_dir  Dossier des jeux ; None : config.TRAIN_DIR lu à l'appel.
    @param progress   Fonction appelée avec (fraction ou None, description), ou None.
    @return Triplet (dossier du jeu, messages, vrai si le jeu vient d'être installé).
    @throws ValueError si le jeu est inconnu, si le disque manque de place ou si le téléchargement échoue.
    """
    entry = CATALOG.get(key)
    if entry is None:
        raise ValueError("Choisissez un jeu à télécharger.")
    root = Path(train_dir) if train_dir is not None else Path(config.TRAIN_DIR)
    dest = root / entry.key
    if dest.is_dir() and list_images(dest):
        return dest, [], False
    root.mkdir(parents=True, exist_ok=True)
    needed = entry.size_bytes + 2 * entry.unpacked_mb * 2 ** 20    # archive, images extraites, copie dans le jeu
    free = shutil.disk_usage(root).free
    if free < needed:
        raise ValueError(f"Pas assez de place sur le disque : {needed / 2 ** 30:.1f} Gio nécessaires, "
                         f"{free / 2 ** 30:.1f} Gio libres.")
    report = progress or (lambda *_: None)
    stage = root / f"{STAGING_PREFIX}telechargement-{entry.key}"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir()
    try:
        archive = stage / "archive"
        download_archive(entry, archive, lambda f: report(f, f"Téléchargement de {entry.key}"))
        report(None, "Extraction et vérification des images")
        budget = (entry.unpacked_mb + 1) * 2 ** 20
        extract = extract_zip if zipfile.is_zipfile(archive) else extract_tar
        images, _ = extract(archive, stage / "images", budget, max_files=2 * entry.n_images + 100)
        archive.unlink()
        folder, messages = create_dataset(entry.key, images, train_dir=root, max_upload_mb=entry.unpacked_mb + 1)
    except (OSError, http.client.HTTPException) as e:
        raise ValueError(f"Échec du téléchargement de « {entry.key} » : {e}. Réessayez.") from None
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return folder, messages, True


def catalog_choices():
    """! @brief Jeux du catalogue pour un menu : couples (libellé, clé)."""
    return [(e.label, e.key) for e in CATALOG.values()]


def catalog_details(key):
    """! @brief Fiche d'un jeu du catalogue, en Markdown (vide si le jeu est inconnu)."""
    e = CATALOG.get(key)
    if e is None:
        return ""
    text = (f"{e.description}\n\n{e.n_images} images, téléchargement de {e.size_bytes / 10 ** 6:.0f} Mo. "
            f"Licence : {e.license}. [Source]({e.source})")
    return f"{text}\n\n**Conseil :** {e.advice}" if e.advice else text


def list_datasets(train_dir=None):
    """! @brief Jeux d'images d'entraînement existants.

    @param train_dir  Dossier des jeux ; None : config.TRAIN_DIR.
    @return Liste triée par nom de dictionnaires @c name, @c path, @c n_images et @c samples (6 vignettes au plus).
    """
    root = Path(train_dir) if train_dir is not None else Path(config.TRAIN_DIR)
    out = []
    if not root.is_dir():
        return out
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        imgs = list_images(d)
        if imgs:
            out.append({"name": d.name, "path": d, "n_images": len(imgs), "samples": imgs[:N_SAMPLES]})
    return out


def dataset_choices(train_dir=None):
    """! @brief Jeux d'entraînement pour un menu : couples (« nom (k images) », nom)."""
    return [(f"{d['name']} ({d['n_images']} images)", d["name"]) for d in list_datasets(train_dir)]


def dataset_samples(name, train_dir=None):
    """! @brief Vignettes d'un jeu : chemins d'images (liste vide si le jeu est absent)."""
    for d in list_datasets(train_dir):
        if d["name"] == name:
            return [str(p) for p in d["samples"]]
    return []


# ---------------------------------------------------------------- onglet Entraîner

@functools.lru_cache(maxsize=64)
def _estimate(mode, N, gdn_last):
    """! @brief Calcul (mis en cache) de estimate()."""
    from .nn import Balle2017
    from .profiling import profile_module, summarize
    in_ch = 1 if mode == "gris" else 3
    model = Balle2017(in_ch, N, gdn_last)
    s = summarize(profile_module(model.g_a, (1, in_ch) + ESTIMATE_SHAPE))
    params = int(s["paramètres"])
    mac_px = s["GMAC"] * 1e9 / (ESTIMATE_SHAPE[0] * ESTIMATE_SHAPE[1])
    fits = params <= BRAM_BYTES
    kib = params / 1024
    # Nombre avec une espace pour les milliers.
    def sp(v):
        return f"{v:,.0f}".replace(",", " ")

    md = (f"**Encodeur embarqué** (`g_a`, {MODE_LABELS[mode].lower()}, N={N}) : **{sp(params)}** paramètres, "
          f"**{sp(mac_px)}** MAC par pixel, poids en INT8 : **{sp(kib)} Kio**.\n\n"
          f"Tient dans la mémoire BRAM du Zynq-7020 (environ {BRAM_BYTES / 1024:.0f} Kio) : "
          f"**{'oui' if fits else 'non'}**.")
    return {"params": params, "mac_per_pixel": float(mac_px), "int8_bytes": params, "int8_kib": kib,
            "bram_bytes": BRAM_BYTES, "fits_bram": fits, "markdown": md}


def estimate(mode, N, gdn_last):
    """! @brief Coût de l'encodeur embarqué d'un réglage (paramètres, MAC par pixel, poids INT8, tient dans la BRAM).

    @param mode      @c "gris" ou @c "rgb".
    @param N         Largeur du modèle (entre 8 et 512).
    @param gdn_last  Présence de la GDN finale.
    @return Dictionnaire @c params, @c mac_per_pixel, @c int8_bytes, @c int8_kib, @c bram_bytes, @c fits_bram, @c markdown.
    @throws ValueError si le mode ou N est invalide.
    """
    if mode not in MODE_LABELS:
        raise ValueError(f"mode invalide : {mode!r}")
    if isinstance(N, bool) or not isinstance(N, (int, float)) or N != int(N) or not 8 <= N <= 512:
        raise ValueError(f"N doit être un entier entre 8 et 512 (reçu {N!r})")
    return dict(_estimate(mode, int(N), bool(gdn_last)))


def preset_values(preset):
    """! @brief Valeurs des champs du formulaire pour un préréglage.

    @param preset  Libellé d'un préréglage de PRESET_CHOICES.
    @return Dictionnaire @c N, @c gdn_last, @c lambdas, @c steps, @c batch_size, @c lr, @c crop, @c n_patches, @c seed ;
            vide pour « Personnalisé ».
    @throws ValueError si le préréglage est inconnu.
    """
    if preset not in PRESET_CHOICES:
        raise ValueError(f"préréglage inconnu : {preset!r}")
    key, lambdas = PRESET_CHOICES[preset]
    if key is None:
        return {}
    p = PRESETS[key]
    out = {k: p[k] for k in ("N", "gdn_last", "steps", "batch_size", "lr", "crop", "n_patches", "seed")}
    out["lambdas"] = list(lambdas)
    return out


def _portable_path(path):
    """! @brief Chemin relatif à config.ROOT en notation POSIX si possible, sinon absolu."""
    p = Path(path).resolve()
    try:
        return p.relative_to(Path(config.ROOT).resolve()).as_posix()
    except ValueError:
        return str(p)


def _num(field, v):
    """! @brief Nombre d'un champ de formulaire (un flottant entier devient entier).

    @throws ValueError si la valeur est vide ou n'est pas un nombre.
    """
    if v is None or isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{field} : un nombre est attendu.")
    return int(v) if float(v) == int(v) else v


def launch_jobs(store, name, author, dataset, mode, N, gdn_last, lambdas, steps, batch_size, lr, crop,
                n_patches, seed, train_dir=None):
    """! @brief Crée un job d'entraînement par valeur de λ (tout ou rien).

    Toutes les configurations sont validées avant la première soumission.
    @param store       JobStore.
    @param name        Nom du run (obligatoire).
    @param author      Auteur (facultatif).
    @param dataset     Nom du jeu d'images sous @p train_dir.
    @param mode        @c "gris" ou @c "rgb".
    @param N           Largeur du modèle.
    @param gdn_last    Présence de la GDN finale.
    @param lambdas     Valeurs de λ (dédupliquées, triées).
    @param steps       Nombre de pas.
    @param batch_size  Patchs par lot.
    @param lr          Pas d'apprentissage.
    @param crop        Côté des patchs.
    @param n_patches   Patchs pré-extraits.
    @param seed        Graine.
    @param train_dir   Dossier des jeux ; None : config.TRAIN_DIR.
    @return Identifiants des jobs créés, dans l'ordre croissant des λ.
    @throws ValueError (dont ConfigError) si le nom est vide, si aucun λ n'est choisi, si le jeu est absent
            ou vide, ou si un réglage est invalide.
    """
    name = (name or "").strip()
    if not name:
        raise ValueError("Donnez un nom au run.")
    if mode not in MODE_LABELS:
        raise ValueError("Choisissez le mode (gris ou couleur).")
    lams = sorted({float(_num("λ", v)) for v in (lambdas or [])})
    if not lams:
        raise ValueError("Cochez au moins une valeur de λ.")
    root = Path(train_dir) if train_dir is not None else Path(config.TRAIN_DIR)
    if not dataset or not any(d["name"] == dataset for d in list_datasets(root)):
        raise ValueError("Choisissez un jeu d'images existant (ou ajoutez-en un avec « Ajouter des images »).")
    steps = _num("nombre de pas", steps)
    base = {"name": name, "author": (author or "").strip(), "train_dir": _portable_path(root / dataset),
            "in_ch": 1 if mode == "gris" else 3, "N": _num("N", N), "gdn_last": bool(gdn_last),
            "steps": steps, "batch_size": _num("batch", batch_size), "lr": lr, "crop": _num("crop", crop),
            "n_patches": _num("patchs", n_patches), "seed": _num("graine", seed), **schedule_for(steps)}
    configs = [TrainConfig.from_dict({**base, "lmbda": lam}) for lam in lams]
    return [store.submit(c.to_dict()) for c in configs]


def format_duration(seconds):
    """! @brief Durée lisible : @c "-" (inconnue), @c "< 1 min", @c "12 min" ou @c "2 h 05 min"."""
    if seconds is None:
        return "-"
    if seconds < 60:
        return "< 1 min"
    if seconds < 3600:
        return f"{int(seconds // 60)} min"
    return f"{int(seconds // 3600)} h {int(seconds % 3600 // 60):02d} min"


def job_progress(job, metrics_train, now):
    """! @brief Progression d'un job et temps restant estimé.

    @param job            Le job.
    @param metrics_train  Métriques @c train du job (JobStore.metrics).
    @param now            Heure courante (secondes Unix).
    @return Couple (progression en %, secondes restantes ou None). Le temps restant n'existe que pour un job
            en cours qui a déjà écrit une métrique.
    """
    pct = 100.0 if job.status == "done" else round(100 * job.last_step / max(1, job.steps), 1)
    eta = None
    if job.status == "running" and metrics_train:
        last = metrics_train[-1]
        if "eta_s" in last:
            eta = max(0.0, float(last["eta_s"]) - (now - float(last["created_at"])))
    return pct, eta


def _status_label(job):
    """! @brief Statut affiché d'un job (avec la demande d'annulation en cours)."""
    if job.status == "running" and job.cancel_requested:
        return "Annulation demandée"
    return STATUS_LABELS[job.status]


def _lmbda_text(v):
    """! @brief λ affiché sans zéros inutiles."""
    return f"{v:g}"


## Statuts d'un job non terminé : toujours visibles, quel que soit leur âge.
OPEN_STATUSES = ("queued", "running", "failed", "cancelled", "interrupted")
## Nombre de jobs récents toujours listés.
RECENT_JOBS = 200


def visible_jobs(store):
    """! @brief Jobs à afficher : les RECENT_JOBS plus récents et tous les jobs non terminés, sans doublon.

    @param store  JobStore.
    @return Liste de Job, id décroissant.
    """
    jobs = {j.id: j for j in store.list_jobs(limit=RECENT_JOBS)}
    jobs.update({j.id: j for j in store.list_jobs(limit=None, statuses=OPEN_STATUSES)})
    return [jobs[k] for k in sorted(jobs, reverse=True)]


def jobs_table(store, now=None):
    """! @brief Tableau des jobs, du plus récent au plus ancien.

    @param store  JobStore.
    @param now    Heure courante ; None : time.time().
    @return DataFrame aux colonnes TABLE_COLUMNS.
    """
    now = time.time() if now is None else now
    rows = []
    for job in visible_jobs(store):
        train = store.metrics(job.id, "train") if job.status == "running" else []
        pct, eta = job_progress(job, train, now)
        rows.append({"n°": job.id, "nom": job.name, "auteur": job.author, "statut": _status_label(job),
                     "λ": _lmbda_text(job.lmbda), "N": job.N, "mode": "gris" if job.in_ch == 1 else "rgb",
                     "progression (%)": pct, "temps restant": format_duration(eta),
                     "créé le": time.strftime("%Y-%m-%d %H:%M", time.localtime(job.created_at))})
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def job_choices(store):
    """! @brief Jobs pour un menu : couples (« n°12 · nom · λ=2048 · En cours », identifiant)."""
    return [(f"n°{j.id} · {j.name} · λ={_lmbda_text(j.lmbda)} · {_status_label(j)}", j.id)
            for j in visible_jobs(store)]


def _log_tail(path, n_lines):
    """! @brief Dernières lignes d'un journal ; « (journal vide) » s'il est absent ou vide."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - LOG_TAIL_BYTES))
            data = f.read()
    except OSError:
        return "(journal vide)"
    text = data.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    return "\n".join(lines[-n_lines:]) if lines else "(journal vide)"


def _curves(metrics):
    """! @brief Figure à trois axes (perte, débit, PSNR) à partir des métriques d'un job."""
    train = [m for m in metrics if m["kind"] == "train"]
    ev = [m for m in metrics if m["kind"] == "eval"]
    source = ev[-1].get("source") if ev else None
    ev_label = {"kodak": "Kodak (arrondi)", "eval": "images d'évaluation (arrondi)"}.get(
        source, "jeu d'entraînement (arrondi)")
    fig = Figure(figsize=(12, 3.4))
    ax_l, ax_b, ax_p = fig.subplots(1, 3)
    steps = [m["step"] for m in train]
    ax_l.plot(steps, [m["loss"] for m in train], color="tab:blue")
    ax_l.set_title("Perte")
    ax_b.plot(steps, [m["bpp"] for m in train], color="tab:blue", label="entraînement (bruit)")
    ax_p.plot(steps, [m["psnr"] for m in train], color="tab:blue", label="entraînement (bruit)")
    if ev:
        ax_b.plot([m["step"] for m in ev], [m["bpp"] for m in ev], "o-", color="tab:orange", label=ev_label)
        ax_p.plot([m["step"] for m in ev], [m["psnr"] for m in ev], "o-", color="tab:orange", label=ev_label)
    ax_b.set_title("Débit (bpp)")
    ax_p.set_title("PSNR (dB)")
    for ax in (ax_l, ax_b, ax_p):
        ax.set_xlabel("pas")
        ax.grid(alpha=0.3)
    if train or ev:
        ax_b.legend(fontsize=8)
        ax_p.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _summary(job):
    """! @brief Synthèse Markdown d'un job : statut, réglages, checkpoint, erreur."""
    c = job.config
    lines = [f"**n°{job.id} · {job.name}** ({job.author or 'auteur inconnu'}) : **{_status_label(job)}**, "
             f"pas {job.last_step} sur {job.steps}.",
             f"- Réglages : {'gris' if job.in_ch == 1 else 'RGB'}, N={job.N}, λ={_lmbda_text(job.lmbda)}, "
             f"batch {c.get('batch_size', '?')}, lr {c.get('lr', '?')}, patchs {c.get('crop', '?')} px, "
             f"graine {c.get('seed', '?')}.",
             f"- Images d'entraînement : `{c.get('train_dir', '?')}`."]
    if job.ckpt_path:
        lines.append(f"- Checkpoint : `{job.ckpt_path}`.")
    if job.error:
        lines.append(f"- Message d'erreur :\n\n```\n{job.error}\n```")
    return "\n".join(lines)


def job_details(store, job_id, n_log_lines=50):
    """! @brief Détail d'un job : courbes, fin du journal, synthèse.

    @param store        JobStore.
    @param job_id       Identifiant, ou None.
    @param n_log_lines  Nombre de lignes de train.log à renvoyer.
    @return Triplet (figure matplotlib, texte du journal, Markdown de synthèse) ; (None, "", invitation)
            si @p job_id est None.
    @throws KeyError si le job est inconnu.
    """
    if job_id is None:
        return None, "", "Sélectionnez un job."
    job = store.get(int(job_id))
    metrics = store.metrics(job.id)
    return _curves(metrics), _log_tail(store.job_dir(job.id) / "train.log", n_log_lines), _summary(job)


def job_model_file(store, job_id):
    """! @brief Checkpoint à télécharger pour un job.

    @param store   JobStore.
    @param job_id  Identifiant du job, ou None.
    @return Chemin du checkpoint s'il existe sur le disque, sinon None (job inconnu, pas encore de checkpoint).
    """
    if job_id is None:
        return None
    try:
        path = store.get(int(job_id)).ckpt_path
    except KeyError:
        return None
    return str(path) if path and Path(path).is_file() else None


def cancel_job(store, job_id, confirmed):
    """! @brief Annule un job après confirmation.

    @param store      JobStore.
    @param job_id     Identifiant du job.
    @param confirmed  Case de confirmation cochée.
    @return Message de confirmation.
    @throws ValueError si la confirmation manque ou si le job est déjà terminé.
    @throws KeyError si le job est inconnu.
    """
    if not confirmed:
        raise ValueError("Cochez la case de confirmation pour annuler.")
    result = store.cancel(int(job_id))
    if result == "cancelled":
        return f"Le job n°{job_id} était en attente : il est annulé."
    return f"Annulation demandée pour le job n°{job_id} : il s'arrêtera dans quelques secondes."


def requeue_job(store, job_id):
    """! @brief Remet en file un job en échec, annulé ou interrompu (il reprend à son dernier checkpoint).

    @return Message de confirmation.
    @throws ValueError si le job n'est pas relançable.
    @throws KeyError si le job est inconnu.
    """
    store.requeue(int(job_id))
    return f"Le job n°{job_id} est de nouveau en attente : il reprendra à son dernier point de sauvegarde."


def duplicate_config(store, job_id):
    """! @brief Valeurs du formulaire d'entraînement pour refaire un job avec les mêmes réglages.

    @return Dictionnaire @c name (« nom (copie) »), @c author, @c dataset, @c mode, @c preset, @c lambdas,
            @c N, @c gdn_last, @c steps, @c batch_size, @c lr, @c crop, @c n_patches, @c seed.
    @throws KeyError si le job est inconnu.
    """
    job = store.get(int(job_id))
    c = job.config
    lam = job.lmbda
    return {"name": f"{job.name} (copie)", "author": job.author, "dataset": Path(c.get("train_dir", "")).name,
            "mode": "gris" if job.in_ch == 1 else "rgb", "preset": CUSTOM_PRESET,
            "lambdas": [int(lam) if lam == int(lam) else lam], "N": job.N, "gdn_last": c.get("gdn_last", True),
            "steps": job.steps, "batch_size": c.get("batch_size", 16), "lr": c.get("lr", 1e-4),
            "crop": c.get("crop", 256), "n_patches": c.get("n_patches", 8000), "seed": c.get("seed", 1234)}


def worker_banner(store):
    """! @brief Message sur la présence d'un worker, avec la marche à suivre s'il n'y en a pas."""
    if store.worker_alive():
        msg = "🟢 **Worker actif.** Les entraînements en file sont pris en charge."
    else:
        msg = ("🔴 **Aucun worker actif : les entraînements restent en attente.** Double-cliquez sur "
               "`scripts\\lancer_plateforme.bat`, ou lancez `docker compose up worker`.")
    if any(j.status == "running" for j in visible_jobs(store)):
        msg += ("\n\n⚠️ Un entraînement est en cours sur cette machine : les temps mesurés dans les onglets Tester "
                "et Valider seront faussés tant qu'il tourne.")
    return msg


def new_done_jobs(store, seen):
    """! @brief Détecte les jobs qui viennent de se terminer.

    @param store  JobStore.
    @param seen   Identifiants des jobs terminés déjà connus ; None au premier appel.
    @return Couple (identifiants des jobs terminés, vrai s'il y en a de nouveaux). Jamais vrai au premier appel.
    """
    done = {j.id for j in visible_jobs(store) if j.status == "done"}
    return done, (seen is not None and bool(done - seen))


## Jobs terminés pris en compte par le dernier rechargement des codecs (None : jamais). État du registre de codecs,
## partagé par toutes les sessions du serveur.
_codecs_loaded = None
## Verrou : un seul rechargement à la fois pour toute l'app.
_codecs_lock = threading.Lock()


def mark_codecs_loaded(done):
    """! @brief Enregistre l'ensemble des jobs terminés couverts par le registre de codecs actuel.

    @param done  Identifiants des jobs terminés (ensemble).
    """
    global _codecs_loaded
    with _codecs_lock:
        _codecs_loaded = frozenset(done)


def codecs_stale(done, loaded):
    """! @brief Vrai si le registre de codecs ne couvre pas l'ensemble courant des jobs terminés.

    @param done    Identifiants des jobs terminés à l'instant.
    @param loaded  Identifiants pris en compte par le dernier rechargement (None : jamais rechargé).
    @return Vrai si @p loaded est None ou diffère de @p done.
    """
    return loaded is None or frozenset(done) != loaded


def codec_updates(done, mode_value, reload_fn, menus_fn, n_outputs):
    """! @brief Mises à jour des menus de codecs d'une session, avec au plus un rechargement par changement.

    Si le registre est périmé (voir codecs_stale), @p reload_fn (qui relance discover()) est appelée une seule
    fois pour toute l'app, sous verrou ; les autres sessions n'ont qu'à relire le registre via @p menus_fn.
    @param done        Identifiants des jobs terminés à l'instant.
    @param mode_value  Mode courant de la session.
    @param reload_fn   Rechargement complet des codecs (renvoie les mises à jour des menus).
    @param menus_fn    Relecture du registre sans redécouverte ; None : rien à mettre à jour hors rechargement.
    @param n_outputs   Nombre de composants de sortie (pour les gr.update vides).
    @return Tuple de @p n_outputs gr.update.
    """
    global _codecs_loaded
    with _codecs_lock:
        if codecs_stale(done, _codecs_loaded):
            out = tuple(reload_fn(mode_value))
            _codecs_loaded = frozenset(done)
            return out
    if menus_fn is None:
        return tuple(gr.update() for _ in range(n_outputs))
    return tuple(menus_fn(mode_value))


def choice_update(choices, current):
    """! @brief gr.update d'un menu : nouveaux choix, valeur courante gardée si elle existe encore.

    @param choices  Liste de couples (libellé, valeur).
    @param current  Valeur actuellement choisie dans la session.
    @return gr.update(choices, value) ; sans valeur courante valide, la première valeur (ou None).
    """
    values = [v for _, v in choices]
    return gr.update(choices=choices, value=current if current in values else (values[0] if values else None))


# ---------------------------------------------------------------- assemblage Gradio

def _form_updates(vals, store_choices_ds):
    """! @brief gr.update pour chaque champ du formulaire à partir d'un dictionnaire de valeurs.

    @param vals              Dictionnaire de valeurs (clés absentes : champ inchangé).
    @param store_choices_ds  Noms de jeux existants (un jeu absent laisse le champ inchangé).
    @return Liste de gr.update dans l'ordre FORM_KEYS.
    """
    out = []
    for k in FORM_KEYS:
        if k not in vals or (k == "dataset" and vals[k] not in store_choices_ds):
            out.append(gr.update())
        elif k == "N":
            out.append(gr.update(choices=sorted(set(N_CHOICES) | {vals[k]}), value=vals[k]))
        elif k == "lambdas":
            out.append(gr.update(choices=sorted(set(LAMBDA_CHOICES) | set(vals[k])), value=vals[k]))
        else:
            out.append(gr.update(value=vals[k]))
    return out


## Champs du formulaire d'entraînement, dans l'ordre des composants.
FORM_KEYS = ["preset", "name", "author", "dataset", "mode", "N", "gdn_last", "lambdas", "steps", "batch_size",
             "lr", "crop", "n_patches", "seed"]


def _initial_estimate(vals):
    """! @brief Encart d'estimation initial du formulaire (texte vide si le calcul échoue)."""
    try:
        return estimate("gris", vals["N"], vals["gdn_last"])["markdown"]
    except Exception:
        return ""


def build_tabs(store=None, tabs=None):
    """! @brief Crée les onglets « 1. Entraîner » et « 2. Suivre » (à appeler dans le gr.Tabs de l'app).

    Les onglets suivants, créés après ceux-ci par l'app, sont reliés ensuite par la fonction renvoyée.
    @param store  JobStore ; None : JobStore() créé ici.
    @param tabs   gr.Tabs de l'app ; s'il est fourni, « Lancer » ouvre l'onglet Suivre, « Dupliquer » l'onglet
                  Entraîner et « Tester ce modèle » l'onglet Tester.
    @return Fonction @c connect(mode, reload_outputs, reload_fn, menus_fn, open_model_fn, open_model_outputs), à
            appeler une fois tous les onglets créés. @c mode est le gr.Radio d'entrée de @c reload_fn ; @c reload_fn
            (rechargement des codecs : reçoit le mode, renvoie un tuple de gr.update pour @c reload_outputs) et
            @c menus_fn (idem sans redécouvrir les codecs) rafraîchissent les menus ; @c open_model_fn reçoit le
            chemin du checkpoint d'un job et renvoie les gr.update de @c open_model_outputs.

    Les choix calculés à la construction sont rafraîchis à chaque chargement de session (événement @c load du
    gr.Blocks courant) et, pour les jeux, à l'ouverture de l'onglet Entraîner.
    """
    store = store or JobStore()
    mark_codecs_loaded({j.id for j in visible_jobs(store) if j.status == "done"})
    ds_choices = dataset_choices()
    ds_names = [v for _, v in ds_choices]
    first = preset_values(next(iter(PRESET_CHOICES)))
    nav = [tabs] if tabs is not None else []
    jobs0 = job_choices(store)

    # Mise à jour de la sélection d'onglet (liste vide sans gr.Tabs).
    def go(tab_id):
        return [gr.Tabs(selected=tab_id)] if tabs is not None else []

    # ------------------------------------------------ Entraîner
    with gr.Tab("1. Entraîner", id="entrainement") as tab_train:
        ui_style.intro("**Apprenez à un modèle à compresser des images.** Partez d'un préréglage, choisissez vos images, "
                       "ajustez si besoin, puis lancez. Un modèle est entraîné pour chaque valeur de λ cochée.",
                       "l'onglet **2. Suivre** s'ouvre tout seul pour montrer la progression.")
        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                with ui_style.card("Point de départ"):
                    t_preset = gr.Radio(list(PRESET_CHOICES), value=next(iter(PRESET_CHOICES)), label="Préréglage",
                                        info="Remplit les réglages ; vous pouvez ensuite les modifier.")
                    with gr.Row():
                        t_name = gr.Textbox(label="Nom du run", placeholder="lune-essai-1",
                                            info="Regroupe les modèles de cet entraînement.")
                        t_author = gr.Textbox(label="Auteur", info="Facultatif : votre prénom.")
                with ui_style.card("Images d'entraînement", "Les images avec lesquelles le modèle apprend."):
                    t_dataset = gr.Dropdown(ds_choices, value=ds_names[0] if ds_names else None, label="Jeu d'images")
                    d_gallery = gr.Gallery(dataset_samples(ds_names[0]) if ds_names else [], label="Aperçu du jeu",
                                           show_label=False, columns=6, height=130, object_fit="contain")
                    with gr.Accordion("Ajouter des images", open=not ds_names):
                        c_pick = gr.Dropdown(catalog_choices(), value=next(iter(CATALOG)),
                                             label="Télécharger un jeu prêt à l'emploi")
                        c_details = gr.Markdown(catalog_details(next(iter(CATALOG))))
                        c_go = gr.Button("Télécharger")
                        c_status = gr.Markdown()
                        d_name = gr.Textbox(label="… ou créer un jeu : nom",
                                            info="Un nom court, par exemple « lune-lro ». "
                                            "Si le jeu existe déjà, les images y sont ajoutées.")
                        d_files = gr.File(label="Images ou archive zip", file_count="multiple",
                                          file_types=sorted(IMAGE_EXTS) + [".zip"])
                        d_go = gr.Button("Créer le jeu")
                        d_status = gr.Markdown()
            with gr.Column(scale=1, min_width=320):
                with ui_style.card("Réglages du modèle"):
                    t_mode = gr.Radio([(v, k) for k, v in MODE_LABELS.items()], value="gris", label="Mode",
                                      info="Le type d'images que le modèle saura compresser.")
                    t_lambdas = gr.CheckboxGroup(LAMBDA_CHOICES, value=first["lambdas"], label="Valeurs de λ",
                                                 info="Le compromis qualité/taille : grand λ donne une meilleure "
                                                 "qualité, mais plus de bits.")
                    t_steps = gr.Number(value=first["steps"], precision=0, minimum=1, label="Nombre de pas",
                                        info="Durée de l'entraînement : plus de pas, plus de temps de calcul.")
                    t_N = gr.Dropdown(N_CHOICES, value=first["N"], label="Largeur du modèle (N)",
                                      info="Plus N est grand, meilleur est le modèle, mais plus il est lourd.")
                    with gr.Accordion("Réglages avancés", open=False):
                        t_gdn = gr.Checkbox(value=first["gdn_last"], label="GDN finale",
                                            info="Comme dans l'article ; décochez pour une variante plus légère.")
                        t_batch = gr.Number(value=first["batch_size"], precision=0, minimum=1, label="Taille du lot",
                                            info="Nombre de morceaux d'image traités ensemble.")
                        t_lr = gr.Number(value=first["lr"], minimum=1e-7, maximum=0.01, label="Pas d'apprentissage",
                                         info="Vitesse d'apprentissage (1e-4 convient en général).")
                        t_crop = gr.Number(value=first["crop"], precision=0, minimum=32, label="Taille des morceaux",
                                           info="Côté des morceaux d'image, multiple de 16.")
                        t_patches = gr.Number(value=first["n_patches"], precision=0, minimum=1,
                                              label="Nombre de morceaux gardés en mémoire",
                                              info="Plus il est grand, plus la mémoire utilisée augmente.")
                        t_seed = gr.Number(value=first["seed"], precision=0, minimum=0, label="Graine aléatoire",
                                           info="Pour refaire exactement le même entraînement.")
                with ui_style.card("Coût estimé à bord", "Ce que l'encodeur demandera à la carte. MAC : opérations de "
                                   "calcul, moins il y en a, plus il est léger."):
                    t_estimate = gr.Markdown(_initial_estimate(first))
                t_go = gr.Button("Lancer l'entraînement", variant="primary", elem_classes="main-action")
                t_result = gr.Markdown()
    form = [t_preset, t_name, t_author, t_dataset, t_mode, t_N, t_gdn, t_lambdas, t_steps, t_batch, t_lr,
            t_crop, t_patches, t_seed]

    # ------------------------------------------------ Suivre
    with gr.Tab("2. Suivre", id="suivi"):
        ui_style.intro("**Regardez où en sont les entraînements.** Ils tournent en arrière-plan : vous pouvez fermer "
                       "cette page sans les interrompre. Elle se met à jour toute seule.",
                       "une fois un entraînement terminé, cliquez sur **Tester ce modèle**.")
        s_banner = gr.Markdown(worker_banner(store), elem_classes="status-banner")
        s_table = gr.Dataframe(jobs_table(store), label="Entraînements", interactive=False, wrap=True)
        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                with ui_style.card("Entraînement choisi"):
                    s_pick = gr.Dropdown(jobs0, value=jobs0[0][1] if jobs0 else None, label="Entraînement à examiner")
                    s_summary = gr.Markdown("Sélectionnez un entraînement.")
                    s_test = gr.Button("Tester ce modèle", variant="primary")
                    with gr.Row():
                        s_requeue = gr.Button("Relancer")
                        s_dup = gr.Button("Dupliquer la configuration")
                    ui_style.hint("**Relancer** reprend à la dernière sauvegarde un entraînement échoué, annulé ou "
                                  "interrompu. **Dupliquer** recopie ses réglages dans l'onglet Entraîner.")
                    s_msg = gr.Markdown()
                    ui_style.hint("**Modèle entraîné** (checkpoint .pt, à télécharger) :")
                    s_file = gr.File(label="Modèle entraîné (checkpoint .pt, à télécharger)", show_label=False,
                                     interactive=False, elem_classes="compact-file",
                                     value=job_model_file(store, jobs0[0][1] if jobs0 else None))
                    with gr.Accordion("Annuler cet entraînement", open=False):
                        with gr.Row():
                            s_confirm = gr.Checkbox(label="Je confirme l'annulation")
                            s_cancel = gr.Button("Annuler", variant="stop")
            with gr.Column(scale=2, min_width=320):
                with ui_style.card("Courbes", "**Perte** : doit descendre. **Débit** : bits par pixel, plus bas signifie "
                                   "plus compressé. **PSNR** : qualité de l'image, plus haut est meilleur."):
                    s_plot = gr.Plot(label="Courbes", show_label=False)
                with gr.Accordion("Journal", open=False):
                    s_log = gr.Textbox(label="Fin du journal (50 dernières lignes)", lines=12, max_lines=12,
                                       interactive=False)
        s_seen = gr.State(None)
        s_choices = gr.State(None)
        timer = gr.Timer(REFRESH_S)

    # ---- callbacks des jeux d'images
    # Adaptateur de create_dataset.
    def on_create(name, files):
        try:
            dest, msgs = create_dataset(name, files)
        except ValueError as e:
            raise gr.Error(str(e)) from None
        for m in msgs[:MAX_WARNINGS]:
            gr.Warning(m)
        return dataset_ready(dest, msgs, 20)

    # État du formulaire une fois un jeu prêt : au plus max_msgs remarques affichées.
    def dataset_ready(dest, msgs, max_msgs):
        n = next((d["n_images"] for d in list_datasets() if d["name"] == dest.name), 0)
        status = f"Jeu **{dest.name}** prêt : {n} images."
        if msgs:
            status += "\n\n" + "\n".join(f"- {m}" for m in msgs[:max_msgs])
            if len(msgs) > max_msgs:
                status += f"\n- … et {len(msgs) - max_msgs} autres remarques."
        return status, dataset_samples(dest.name), gr.update(choices=dataset_choices(), value=dest.name)

    # Adaptateur de install_catalog_dataset. Pas de fenêtres d'avertissement : les remarques sont connues d'avance.
    def on_download(key, progress=gr.Progress()):
        try:
            dest, msgs, installed = install_catalog_dataset(key, progress=lambda f, desc: progress(f, desc=desc))
        except ValueError as e:
            raise gr.Error(str(e)) from None
        if not installed:
            gr.Info(f"Le jeu « {dest.name} » est déjà présent : rien à télécharger.")
        return dataset_ready(dest, msgs, 3)

    d_go.click(on_create, [d_name, d_files], [d_status, d_gallery, t_dataset])
    t_dataset.change(lambda n: dataset_samples(n), t_dataset, d_gallery, show_progress="hidden")
    c_pick.input(catalog_details, c_pick, c_details)
    # Un téléchargement à la fois : un second clic attend, puis constate que le jeu est présent.
    c_go.click(on_download, c_pick, [c_status, d_gallery, t_dataset], concurrency_limit=1)

    # ---- callbacks Entraîner
    # Remplit le formulaire avec un préréglage.
    def on_preset(preset):
        try:
            vals = preset_values(preset)
        except ValueError as e:
            raise gr.Error(str(e)) from None
        vals.pop("preset", None)
        return _form_updates(vals, ds_names_now())[1:]

    # Noms des jeux existants à l'instant.
    def ds_names_now():
        return [v for _, v in dataset_choices()]

    # Met à jour l'encart d'estimation.
    def on_estimate(m, n, g):
        try:
            return estimate(m, n, g)["markdown"]
        except (ValueError, TypeError):
            return "Estimation indisponible pour ces réglages."

    t_preset.input(on_preset, t_preset, form[1:])
    for comp in (t_mode, t_N, t_gdn):
        comp.change(on_estimate, [t_mode, t_N, t_gdn], t_estimate, show_progress="hidden")

    # Adaptateur de launch_jobs : ouvre le suivi sur le premier entraînement créé.
    def on_launch(*vals):
        v = dict(zip(FORM_KEYS, vals))
        try:
            ids = launch_jobs(store, v["name"], v["author"], v["dataset"], v["mode"], v["N"], v["gdn_last"],
                              v["lambdas"], v["steps"], v["batch_size"], v["lr"], v["crop"], v["n_patches"],
                              v["seed"])
        except ValueError as e:
            raise gr.Error(str(e)) from None
        txt = "Entraînements créés : " + ", ".join(f"n°{i}" for i in ids) + "."
        gr.Info(txt)
        return [txt, jobs_table(store), gr.update(choices=job_choices(store), value=ids[0])] + go("suivi")

    t_go.click(on_launch, form, [t_result, s_table, s_pick] + nav)

    # ---- callbacks Suivre
    # Détail d'un job pour les trois zones d'affichage.
    def show(job_id):
        try:
            return job_details(store, job_id)
        except KeyError:
            return None, "", "Ce job n'existe plus."

    s_pick.change(show, s_pick, [s_plot, s_log, s_summary], show_progress="hidden")
    s_pick.change(lambda j: job_model_file(store, j), s_pick, s_file, show_progress="hidden")

    # Adaptateur de cancel_job.
    def on_cancel(job_id, ok):
        if job_id is None:
            raise gr.Error("Sélectionnez d'abord un entraînement.")
        try:
            msg = cancel_job(store, job_id, ok)
        except (ValueError, KeyError) as e:
            raise gr.Error(str(e.args[0]) if isinstance(e, ValueError) else "Ce job n'existe plus.") from None
        return msg, False

    # Adaptateur de requeue_job.
    def on_requeue(job_id):
        if job_id is None:
            raise gr.Error("Sélectionnez d'abord un entraînement.")
        try:
            return requeue_job(store, job_id)
        except (ValueError, KeyError) as e:
            raise gr.Error(str(e.args[0]) if isinstance(e, ValueError) else "Ce job n'existe plus.") from None

    # Adaptateur de duplicate_config.
    def on_dup(job_id):
        if job_id is None:
            raise gr.Error("Sélectionnez d'abord un entraînement.")
        try:
            vals = duplicate_config(store, job_id)
        except KeyError:
            raise gr.Error("Ce job n'existe plus.") from None
        names = ds_names_now()
        if vals["dataset"] not in names:
            gr.Warning(f"Le jeu d'images « {vals['dataset']} » n'existe plus : choisissez-en un autre.")
        gr.Info("Configuration copiée dans l'onglet Entraîner.")
        return _form_updates(vals, names) + go("entrainement")

    s_cancel.click(on_cancel, [s_pick, s_confirm], [s_msg, s_confirm])
    s_requeue.click(on_requeue, s_pick, s_msg)
    s_dup.click(on_dup, s_pick, form + nav)

    # Ouverture de l'onglet Entraîner : jeux créés depuis une autre session.
    def on_train_tab(current):
        return choice_update(dataset_choices(), current)

    tab_train.select(on_train_tab, t_dataset, t_dataset, show_progress="hidden")

    # Relie ce qui dépend des onglets créés ensuite par l'app (voir la description de la valeur renvoyée).
    def connect(mode, reload_outputs, reload_fn, menus_fn=None, open_model_fn=None, open_model_outputs=()):
        reload_outputs = list(reload_outputs)

        # Rafraîchissement périodique : tableau, bannière, détail du job, rechargement des codecs.
        def on_tick(m, job_id, seen, last_choices):
            done, fresh = new_done_jobs(store, seen)
            reload = (codec_updates(done, m, reload_fn, menus_fn, len(reload_outputs)) if fresh
                      else tuple(gr.update() for _ in reload_outputs))
            choices = job_choices(store)
            pick = gr.update(choices=choices) if choices != last_choices else gr.update()
            fig, log, summary = show(job_id) if job_id is not None else (gr.update(), gr.update(), gr.update())
            model = job_model_file(store, job_id) if job_id is not None else None
            return (jobs_table(store), worker_banner(store), pick, fig, log, summary, done, choices,
                    model) + reload

        timer.tick(on_tick, [mode, s_pick, s_seen, s_choices],
                   [s_table, s_banner, s_pick, s_plot, s_log, s_summary, s_seen, s_choices, s_file]
                   + reload_outputs,
                   show_progress="hidden")

        # Chargement d'une session : jeux, jobs et codecs d'après l'état actuel du serveur.
        def on_load(m, train_ds, job_id):
            ds = choice_update(dataset_choices(), train_ds)
            jobs = job_choices(store)
            done = {j.id for j in visible_jobs(store) if j.status == "done"}
            reload = codec_updates(done, m, reload_fn, menus_fn, len(reload_outputs))
            return (ds, dataset_samples(ds["value"]) if ds["value"] else [], choice_update(jobs, job_id),
                    jobs_table(store), jobs, done) + reload

        gr.context.Context.root_block.load(
            on_load, [mode, t_dataset, s_pick],
            [t_dataset, d_gallery, s_pick, s_table, s_choices, s_seen] + reload_outputs,
            show_progress="hidden")

        # « Tester ce modèle » : ouvre l'onglet Tester sur le modèle du job choisi.
        def on_test(job_id):
            if job_id is None:
                raise gr.Error("Sélectionnez d'abord un entraînement.")
            try:
                ckpt = store.get(int(job_id)).ckpt_path
            except KeyError:
                raise gr.Error("Ce job n'existe plus.") from None
            return list(open_model_fn(ckpt)) + go("tester")

        if open_model_fn is not None:
            s_test.click(on_test, s_pick, list(open_model_outputs) + nav)

    return connect
