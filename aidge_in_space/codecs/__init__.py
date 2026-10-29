"""! @brief Registre des codecs.

Chaque module de ce dossier qui définit une fonction @c codecs() (renvoyant une liste de Codec)
est découvert automatiquement : ajouter un algorithme revient à ajouter un fichier ici. Les
modules dont le nom commence par « _ » et le module @c base sont ignorés.
"""
import importlib
import pkgutil

from .base import Codec, LearnedCodec, Level, Result

__all__ = ["Codec", "LearnedCodec", "Level", "Result", "discover", "all_codecs", "get", "closest_level"]

## Codecs enregistrés : identifiant -> Codec.
_REGISTRY = {}


def discover():
    """! @brief (Re)construit le registre en important tous les modules de codecs.

    Utile après l'ajout d'un module ou d'un checkpoint dans outputs/checkpoints.
    @return Liste triée des codecs (voir all_codecs()).
    @throws ValueError si deux codecs ont le même identifiant.
    """
    _REGISTRY.clear()
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_") or info.name == "base":
            continue
        module = importlib.import_module(f"{__name__}.{info.name}")
        provide = getattr(module, "codecs", None)
        if provide is None:
            continue
        for codec in provide():
            if codec.key in _REGISTRY:
                raise ValueError(f"identifiant de codec en double : {codec.key}")
            _REGISTRY[codec.key] = codec
    return all_codecs()


def all_codecs():
    """! @brief Tous les codecs enregistrés : les classiques d'abord, puis par nom.

    @return Liste de Codec (le registre est construit au premier appel).
    """
    if not _REGISTRY:
        discover()
    return sorted(_REGISTRY.values(), key=lambda c: (c.family != "Classique", c.label))


def get(key):
    """! @brief Codec d'identifiant donné.

    @param key  Identifiant (Codec.key).
    @return Le codec.
    @throws KeyError si l'identifiant est inconnu.
    """
    if not _REGISTRY:
        discover()
    return _REGISTRY[key]


def closest_level(codec, img, mode, target_bpp):
    """! @brief Niveau dont le débit sur cette image est le plus proche d'une cible.

    Recherche par dichotomie (les niveaux vont du plus bas au plus haut débit), avec des débits
    estimés et le moteur PyTorch pour aller vite.
    @param codec       Codec.
    @param img         Image uint8.
    @param mode        @c "gris" ou @c "rgb".
    @param target_bpp  Débit visé, en bits par pixel.
    @return Level le plus proche.
    """
    levels = codec.levels(mode)
    n_px = img.shape[0] * img.shape[1]
    cache = {}

    def bpp(i):
        if i not in cache:
            cache[i] = codec.run(img, levels[i], mode, backend="torch", real_rate=False).bits / n_px
        return cache[i]

    lo, hi = 0, len(levels) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if bpp(mid) < target_bpp:
            lo = mid
        else:
            hi = mid
    return levels[min((lo, hi), key=lambda i: abs(bpp(i) - target_bpp))]
