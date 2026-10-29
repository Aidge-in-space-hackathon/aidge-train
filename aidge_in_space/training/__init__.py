"""! @brief Paquet d'entraînement de Ballé 2017 : configuration (sans torch) et boucle (chargée à la demande).

TrainConfig et ConfigError sont importés tout de suite ; train et TrainResult, qui dépendent de
torch, ne sont chargés qu'au premier accès pour que l'import de la configuration reste léger.
"""
from .config import ConfigError, TrainConfig

__all__ = ["TrainConfig", "ConfigError", "train", "TrainResult"]


def __getattr__(name):
    """! @brief Charge train et TrainResult (module loop, donc torch) au premier accès.

    @param name  Nom demandé.
    @return L'objet correspondant de aidge_in_space.training.loop.
    @throws AttributeError si le nom est inconnu.
    """
    if name in ("train", "TrainResult"):
        from . import loop
        return getattr(loop, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
