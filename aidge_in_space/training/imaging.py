"""! @brief Conversion des images 16 bits vers 8 bits pour l'entraînement (module léger, sans torch).

Pillow écrête (au lieu de mettre à l'échelle) un convert("L") d'une image 16 bits : une image lunaire
de valeurs 0 à 4000 sortirait presque entièrement blanche. to_8bit() la ramène à 8 bits avant la conversion.
"""
import numpy as np
from PIL import Image

## Modes Pillow à plus de 8 bits par canal, convertis par mise à l'échelle linéaire.
HIGH_DEPTH_MODES = ("I;16", "I;16B", "I;16L", "I;16N", "I", "F")


def to_8bit(img: Image.Image) -> tuple[Image.Image, bool]:
    """! @brief Ramène une image à 8 bits par canal.

    Modes @c I;16, @c I;16B, @c I;16L, @c I;16N, @c I et @c F : mise à l'échelle linéaire de [0, max]
    vers [0, 255] (max = valeur maximale de l'image ; une image entièrement nulle ou négative donne du
    noir), résultat en mode @c L. Les autres modes sont renvoyés inchangés.
    @param img  Image PIL.
    @return Couple (image, converti) ; @c converti vaut True si une conversion a eu lieu.
    """
    if img.mode not in HIGH_DEPTH_MODES:
        return img, False
    arr = np.asarray(img, dtype=np.float64)
    top = float(arr.max()) if arr.size else 0.0
    if not top > 0:
        out = np.zeros(arr.shape, dtype=np.uint8)
    else:
        out = np.clip(np.rint(arr * (255.0 / top)), 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="L"), True
