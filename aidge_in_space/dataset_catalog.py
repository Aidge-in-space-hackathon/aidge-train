"""! @brief Catalogue de jeux d'images publics, téléchargeables depuis l'onglet Entraîner.

Chaque archive est identifiée par son empreinte SHA-256 : un fichier modifié à la source est refusé au lieu d'être
installé. Les images ne sont pas redistribuées par ce dépôt : chaque jeu garde la licence de sa source.
"""
import hashlib
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class CatalogEntry:
    """! @brief Un jeu d'images téléchargeable."""

    ## Identifiant, qui sert aussi de nom de dossier dans data/train/.
    key: str
    ## Libellé affiché dans le menu.
    label: str
    ## Contenu du jeu, en une phrase.
    description: str
    ## Conditions d'utilisation de la source.
    license: str
    ## Page de présentation de la source.
    source: str
    ## Adresse de l'archive (zip ou tar compressé).
    url: str
    ## Empreinte SHA-256 de l'archive.
    sha256: str
    ## Taille de l'archive, en octets.
    size_bytes: int
    ## Taille une fois décompressée, en Mio (arrondie au supérieur).
    unpacked_mb: int
    ## Nombre d'images de l'archive.
    n_images: int
    ## Conseil d'utilisation (vide si aucun).
    advice: str = ""


## Jeux proposés, par identifiant. Tailles et nombres d'images relevés sur les archives elles-mêmes.
CATALOG = {e.key: e for e in (
    CatalogEntry(
        key="uc-merced",
        label="Vues aériennes : UC Merced Land Use",
        description="Orthophotos aériennes de l'USGS, proches d'images satellite : 21 types de terrain "
                    "(champs, forêts, ports, quartiers…), en couleur, 256 x 256 px.",
        license="Usage recherche, citer Yang et Newsam (2010)",
        source="http://weegee.vision.ucmerced.edu/datasets/landuse.html",
        url="https://huggingface.co/datasets/torchgeo/ucmerced/resolve/main/UCMerced_LandUse.zip",
        sha256="06c539ef28703a58fb07bd2837991ac7c48b813b00bb12ac197efd813a18daeb",
        size_bytes=332_468_434, unpacked_mb=420, n_images=2100,
        advice="Quelques images font un peu moins de 256 px : elles seront légèrement agrandies."),
    CatalogEntry(
        key="mars-curiosity",
        label="Sol martien : rover Curiosity",
        description="Vues du rover Curiosity (Mastcam, MAHLI), en version allégée de la NASA, d'environ 256 px de "
                    "côté : le plus proche du sujet parmi les vraies images planétaires.",
        license="CC BY-SA 4.0",
        source="https://zenodo.org/records/1049137",
        url="https://zenodo.org/api/records/1049137/files/msl-images.zip/content",
        sha256="53dde067668ed5d0e751bf28adf55d62dee0170090cd77a5e7002942a71f146e",
        size_bytes=60_635_475, unpacked_mb=59, n_images=6737,
        advice="La plupart des images font moins de 256 px : réglez la « Taille des morceaux » sur 128 pour "
               "éviter de les agrandir."),
    CatalogEntry(
        key="lune-analogue-dfki",
        label="Terrain lunaire analogue : roches et cratères (DFKI)",
        description="Terrain d'essai du DFKI qui imite le sol lunaire : ce ne sont pas de vraies images de la "
                    "Lune. En couleur, 640 x 640 px.",
        license="CC BY 4.0",
        source="https://zenodo.org/records/14673237",
        url="https://zenodo.org/api/records/14673237/files/Dataset_coco.zip/content",
        sha256="d3581f6c7b0118cad9e70307cf8778d3a62923c4c82c10d90e69b4c2dcf13678",
        size_bytes=72_191_694, unpacked_mb=75, n_images=3198),
    CatalogEntry(
        key="imagewoof",
        label="Animaux : Imagewoof (10 races de chiens)",
        description="Photos de chiens extraites d'ImageNet, en couleur, 320 px sur le petit côté.",
        license="Conditions d'ImageNet : recherche non commerciale",
        source="https://github.com/fastai/imagenette",
        url="https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-320.tgz",
        sha256="7db6120fdb9ae079e26346f89e7b00d7f184f8137791609b97fd0405d3f92305",
        size_bytes=328_387_740, unpacked_mb=321, n_images=12954),
    CatalogEntry(
        key="div2k-valid",
        label="Photos variées, haute définition : DIV2K (validation)",
        description="Photos de scènes très diverses, en couleur, environ 2040 x 1400 px : une base généraliste de "
                    "référence, utile pour comparer les modèles.",
        license="Recherche académique uniquement",
        source="https://data.vision.ee.ethz.ch/cvl/DIV2K/",
        url="https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
        sha256="20dd31fd84d777bc1cf5d6b7654a3f569c0aec74458ae094122ad1d0489900fc",
        size_bytes=448_993_893, unpacked_mb=429, n_images=100),
)}
## En-tête User-Agent des téléchargements : Zenodo répond 403 à un « Mozilla/5.0 » seul.
USER_AGENT = "aidge-train/1.0 (+https://github.com/Aidge-in-space-hackathon/aidge-train)"
## Délai d'attente réseau par opération, en secondes.
TIMEOUT_S = 60
## Téléchargement par blocs de cette taille (octets).
_CHUNK = 1 << 20


def download_archive(entry, dest, progress=None):
    """! @brief Télécharge l'archive d'un jeu et vérifie sa taille et son empreinte.

    @param entry     CatalogEntry à télécharger.
    @param dest      Fichier de destination, laissé à l'appelant en cas d'échec.
    @param progress  Fonction appelée avec la fraction déjà reçue (entre 0 et 1), ou None.
    @throws ValueError si le fichier reçu n'est pas celui attendu.
    @throws OSError si le réseau ou le disque échoue.
    """
    digest, received = hashlib.sha256(), 0
    req = urllib.request.Request(entry.url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as src, open(dest, "wb") as out:
        while True:
            block = src.read(_CHUNK)
            if not block:
                break
            received += len(block)
            if received > entry.size_bytes:
                break
            digest.update(block)
            out.write(block)
            if progress is not None:
                progress(received / entry.size_bytes)
    if received != entry.size_bytes or digest.hexdigest() != entry.sha256:
        raise ValueError(f"Le fichier reçu pour « {entry.key} » n'est pas celui attendu : "
                         "téléchargement interrompu ou archive modifiée à la source.")
