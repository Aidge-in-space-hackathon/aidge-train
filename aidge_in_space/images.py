"""! @brief Jeux d'images et conversions entre tableaux uint8 et tenseurs."""
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .config import DATA_DIR, KODAK_DIR

## Modes d'image : identifiant -> libellé affiché.
MODE_LABELS = {"gris": "Niveaux de gris", "rgb": "Couleur (RGB)"}
## Extensions reconnues comme images.
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def download_kodak(dest=KODAK_DIR):
    """! @brief Télécharge les 24 images Kodak, en sautant celles déjà présentes.

    Un échec (hors ligne) est signalé sans lever d'exception.
    @param dest  Dossier de destination.
    @return Chemins des images présentes, triés.
    """
    dest.mkdir(parents=True, exist_ok=True)
    base = "https://r0k.us/graphics/kodak/kodak/kodim{:02d}.png"
    for i in range(1, 25):
        f = dest / f"kodim{i:02d}.png"
        if f.exists():
            continue
        try:
            req = urllib.request.Request(base.format(i), headers={"User-Agent": "Mozilla/5.0"})
            f.write_bytes(urllib.request.urlopen(req, timeout=30).read())
        except Exception as e:
            print(f"Échec du téléchargement de {f.name} : {e}")
            break
    return sorted(dest.glob("kodim*.png"))


def list_images(folder):
    """! @brief Images d'un dossier et de ses sous-dossiers, triées par chemin."""
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def datasets():
    """! @brief Jeux d'images disponibles : Kodak, puis chaque sous-dossier de data/ (sauf train/).

    @return Dictionnaire nom -> liste de chemins (seuls les jeux non vides sont listés).
    """
    out = {}
    kodak = list_images(KODAK_DIR)
    if kodak:
        out["kodak"] = kodak
    if DATA_DIR.exists():
        for d in sorted(p for p in DATA_DIR.iterdir() if p.is_dir() and p.name not in ("kodak", "train")):
            imgs = list_images(d)
            if imgs:
                out[d.name] = imgs
    return out


def convert(img, mode):
    """! @brief Convertit une image PIL ou un tableau dans le mode demandé.

    @param img   Image PIL ou tableau numpy.
    @param mode  @c "gris" ou @c "rgb".
    @return Tableau uint8, H x W (gris) ou H x W x 3 (rgb).
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    return np.asarray(img.convert("L" if mode == "gris" else "RGB"), dtype=np.uint8).copy()


def load_image(path, mode):
    """! @brief Charge une image depuis un fichier, convertie avec convert()."""
    return convert(Image.open(path), mode)


def to_tensor(img, channels=None):
    """! @brief Tableau uint8 -> tenseur 1 x C x H x W dans [0, 1].

    @param img       Image uint8, H x W ou H x W x 3.
    @param channels  Canaux attendus par le modèle : une image grise est répliquée si 3.
    @return Tenseur float32.
    """
    a = img.astype(np.float32) / 255.0
    a = a[None] if a.ndim == 2 else a.transpose(2, 0, 1)
    t = torch.from_numpy(np.ascontiguousarray(a))[None]
    if channels == 3 and t.shape[1] == 1:
        t = t.repeat(1, 3, 1, 1)
    return t


def from_tensor(t, mode):
    """! @brief Tenseur 1 x C x H x W dans [0, 1] -> tableau uint8 dans le mode demandé.

    @param t     Tenseur (valeurs hors de [0, 1] écrêtées).
    @param mode  @c "gris" (une sortie RGB est convertie en luminance) ou @c "rgb".
    @return Tableau uint8.
    """
    a = (t[0].clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    if mode == "gris":
        if a.shape[2] == 3:
            return np.asarray(Image.fromarray(a).convert("L"), dtype=np.uint8).copy()
        return a[:, :, 0].copy()
    return a
