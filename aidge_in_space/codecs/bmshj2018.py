"""! @brief Ballé et al. 2018 avec les poids pré-entraînés de CompressAI (licence BSD 3-Clause Clear).

Les poids (entraînés en RGB pour la MSE, 8 niveaux de qualité) sont téléchargés au premier usage
dans outputs/weights. Les architectures sont réimplémentées dans aidge_in_space.nn.
"""
import torch

from ..coding import HyperpriorCoding
from ..config import WEIGHTS_DIR
from ..nn import FactorizedPrior2018, ScaleHyperprior2018, load_compatible
from .base import LearnedCodec, Level

## Racine des poids pré-entraînés de CompressAI.
ROOT_URL = "https://compressai.s3.amazonaws.com/models/v1"
## Fichier de poids par codec et par qualité (le suffixe est le début du SHA-256, vérifié au téléchargement).
URLS = {
    "bmshj2018-factorized": {
        1: "bmshj2018-factorized-prior-1-446d5c7f.pth.tar",
        2: "bmshj2018-factorized-prior-2-87279a02.pth.tar",
        3: "bmshj2018-factorized-prior-3-5c6f152b.pth.tar",
        4: "bmshj2018-factorized-prior-4-1ed4405a.pth.tar",
        5: "bmshj2018-factorized-prior-5-866ba797.pth.tar",
        6: "bmshj2018-factorized-prior-6-9b02ea3a.pth.tar",
        7: "bmshj2018-factorized-prior-7-6dfd6734.pth.tar",
        8: "bmshj2018-factorized-prior-8-5232faa3.pth.tar",
    },
    "bmshj2018-hyperprior": {
        1: "bmshj2018-hyperprior-1-7eb97409.pth.tar",
        2: "bmshj2018-hyperprior-2-93677231.pth.tar",
        3: "bmshj2018-hyperprior-3-6d87be32.pth.tar",
        4: "bmshj2018-hyperprior-4-de1b779c.pth.tar",
        5: "bmshj2018-hyperprior-5-f8b614e1.pth.tar",
        6: "bmshj2018-hyperprior-6-1ab9c41e.pth.tar",
        7: "bmshj2018-hyperprior-7-3804dcbd.pth.tar",
        8: "bmshj2018-hyperprior-8-a583f0cf.pth.tar",
    },
}


def channels(quality):
    """! @brief Canaux (N, M) des modèles pré-entraînés : plus de canaux pour les hauts débits.

    @param quality  Qualité CompressAI, de 1 à 8.
    @return Couple (N, M).
    """
    return (128, 192) if quality <= 5 else (192, 320)


class Pretrained2018(LearnedCodec):
    """! @brief Base des modèles 2018 pré-entraînés : 8 niveaux de qualité, poids téléchargés à la demande."""

    ## Modèles RGB ; en mode gris, l'image est répliquée sur 3 canaux.
    modes = ("rgb", "gris")
    ## Classe d'architecture de aidge_in_space.nn, construite avec channels(qualité).
    arch = None

    def levels(self, mode):
        """! @copydoc aidge_in_space.codecs.base.Codec.levels

        La valeur de chaque niveau est la qualité CompressAI (1 à 8).
        """
        return [Level(q, f"qualité {q}") for q in range(1, 9)]

    def make_model(self, level):
        """! @copydoc aidge_in_space.codecs.base.LearnedCodec.make_model"""
        return self.arch(*channels(level.value))

    def load_weights(self, model, level):
        """! @brief Télécharge (une fois, empreinte vérifiée) puis charge les poids CompressAI.

        @param model  Modèle renvoyé par make_model(level).
        @param level  Level du codec.
        @throws Exception si le téléchargement échoue ou si le checkpoint est incompatible.
        """
        url = f"{ROOT_URL}/{URLS[self.key][level.value]}"
        sd = torch.hub.load_state_dict_from_url(url, model_dir=str(WEIGHTS_DIR), map_location="cpu",
                                                check_hash=True, progress=False)
        load_compatible(model, sd)

    def architecture(self, level):
        """! @copydoc aidge_in_space.codecs.base.LearnedCodec.architecture"""
        N, M = channels(level.value)
        return f"N={N}, M={M}"


class Factorized2018(Pretrained2018):
    """! @brief bmshj2018-factorized pré-entraîné."""

    ## Identifiant (clé de URLS), nom affiché et architecture.
    key, label, arch = "bmshj2018-factorized", "Ballé 2018 factorisé (pré-entraîné)", FactorizedPrior2018
    ## Description affichée.
    description = ("Même principe que 2017 (GDN + prior factorisé) avec 4 conv 5x5 de pas 2. "
                   "Poids CompressAI, RGB, optimisés pour la MSE.")


class Hyperprior2018(Pretrained2018):
    """! @brief bmshj2018-hyperprior pré-entraîné."""

    ## Identifiant (clé de URLS), nom affiché et architecture.
    key, label, arch = "bmshj2018-hyperprior", "Ballé 2018 hyperprior (pré-entraîné)", ScaleHyperprior2018
    ## Description affichée.
    description = ("Ajoute un hyper-latent z qui prédit l'échelle de chaque latent : meilleur débit, "
                   "mais h_a et h_s s'exécutent aussi à bord. Poids CompressAI, RGB, MSE.")
    ## Codage en deux flux : hyper-latent z puis latents y.
    coding = HyperpriorCoding()


def codecs():
    """! @brief Les deux modèles 2018 pré-entraînés."""
    return [Factorized2018(), Hyperprior2018()]
