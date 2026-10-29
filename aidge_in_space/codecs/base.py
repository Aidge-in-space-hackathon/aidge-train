"""! @brief Interface commune des codecs : Codec, Level, Result et LearnedCodec.

Un codec expose des niveaux (du plus bas au plus haut débit) et une méthode run() qui compresse
puis reconstruit une image uint8. Les codecs appris héritent de LearnedCodec, qui gère le padding,
le moteur d'exécution de l'encodeur, le codage entropique réel (codeur arithmétique) ou estimé, et
le décodage ; le codage propre à chaque modèle d'entropie est délégué à un
aidge_in_space.coding.LatentCoding.
"""
import math
import struct
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .. import backends
from ..coding import FactorizedCoding
from ..images import from_tensor, to_tensor
from ..nn import freeze


@dataclass(frozen=True)
class Level:
    """! @brief Un réglage de débit d'un codec (qualité JPEG, checkpoint, débit cible...)."""

    ## Réglage interne, propre au codec (entier de qualité, chemin de checkpoint...). Sert de clé de cache.
    value: object
    ## Texte affiché dans l'interface et les CSV.
    label: str


@dataclass
class Result:
    """! @brief Résultat d'une compression suivie d'une décompression."""

    ## Image reconstruite : uint8, même forme que l'image d'entrée.
    recon: np.ndarray
    ## Taille du flux compressé, en bits (en-tête compris pour un débit réel).
    bits: float
    ## @c "réel" (taille du fichier produit) ou @c "estimé" (entropie prédite par le modèle).
    rate_kind: str
    ## Temps en secondes. Clés : @c "encodage" (tout ce qui tourne à bord), @c "analyse"
    ## (transformée d'analyse seule, codecs appris) et @c "décodage".
    timings: dict = field(default_factory=dict)
    ## Informations complémentaires affichées telles quelles (forme du latent, vérification du décodage...).
    info: dict = field(default_factory=dict)


class Codec:
    """! @brief Classe de base de tous les codecs.

    Pour ajouter un codec, créer un module dans aidge_in_space/codecs/ qui définit une fonction
    @c codecs() renvoyant des instances de sous-classes de Codec : il est découvert automatiquement
    par aidge_in_space.codecs.discover().
    """

    ## Identifiant unique, utilisé par la ligne de commande et dans les CSV.
    key = ""
    ## Nom affiché.
    label = ""
    ## @c "Classique" ou @c "Appris" ; les codecs classiques sont listés en premier.
    family = ""
    ## Description courte affichée dans l'onglet « Codecs et aide ».
    description = ""
    ## Modes d'image acceptés : @c "gris" et/ou @c "rgb".
    modes = ("gris", "rgb")
    ## Moteurs d'exécution possibles pour l'encodeur (voir aidge_in_space.backends) ; vide = sans objet.
    backends = ()

    def available(self):
        """! @brief Indique si le codec peut s'exécuter sur cette machine.

        @return Couple (disponible, raison de l'indisponibilité ou chaîne vide).
        """
        return True, ""

    def levels(self, mode):
        """! @brief Niveaux proposés, du plus bas au plus haut débit.

        @param mode  @c "gris" ou @c "rgb".
        @return Liste de Level.
        """
        raise NotImplementedError

    def run(self, img, level, mode, backend="torch", real_rate=True):
        """! @brief Compresse puis décompresse une image.

        @param img        Image uint8, H x W (gris) ou H x W x 3 (rgb).
        @param level      Level choisi parmi levels(mode).
        @param mode       @c "gris" ou @c "rgb".
        @param backend    Moteur de l'encodeur (codecs appris uniquement), clé de aidge_in_space.backends.BACKENDS.
        @param real_rate  True : flux réellement produit ; False : débit estimé si le codec le permet.
        @return Result dont l'image reconstruite a la même forme que @p img.
        """
        raise NotImplementedError

    def __repr__(self):
        """! @brief Représentation courte : classe et identifiant."""
        return f"<{type(self).__name__} {self.key}>"


## Signature des flux produits par LearnedCodec.
MAGIC = b"AIS"
## Version du format de flux.
VERSION = 1
## En-tête (format struct) : signature, version, hauteur, largeur, canaux du modèle (9 octets).
HEADER = ">3sBHHB"


class LearnedCodec(Codec):
    """! @brief Codec à base de réseau de neurones (transformées apprises + codage entropique).

    Une sous-classe fournit make_model() et load_weights() ; tout le reste est commun :
    - l'image est complétée (padding par réplication) à un multiple de @c model.pad_multiple ;
    - la transformée d'analyse g_a, aux GDN gelées, s'exécute sur le moteur choisi
      (PyTorch, ONNX Runtime ou Aidge, voir aidge_in_space.backends) ;
    - les latents sont codés par l'objet @c coding, puis décodés et vérifiés si le débit est réel ;
    - la transformée de synthèse g_s s'exécute toujours en PyTorch (elle tourne au sol).

    Le modèle renvoyé par make_model() doit exposer @c in_ch, @c N, @c M, @c pad_multiple,
    @c downsampling, @c g_a, @c g_s, plus ce que demande @c coding (par exemple @c prior).
    """

    ## Famille des codecs à base de réseau de neurones.
    family = "Appris"
    ## L'encodeur peut s'exécuter sur tous les moteurs de aidge_in_space.backends.BACKENDS.
    backends = tuple(backends.BACKENDS)
    ## Canaux d'entrée du modèle (3 pour les modèles RGB, qui reçoivent une image grise répliquée).
    in_ch = 3
    ## Codage entropique des latents (aidge_in_space.coding.LatentCoding).
    coding = FactorizedCoding()
    ## Nombre de modèles chargés gardés en mémoire (cache LRU).
    max_cached_models = 4

    def __init__(self):
        """! @brief Initialise le cache des modèles chargés."""
        ## Modèles chargés, indexés par Level.value, du moins au plus récemment utilisé.
        self._cache = OrderedDict()

    # --- à fournir par les sous-classes

    def make_model(self, level):
        """! @brief Architecture non entraînée du niveau (suffit au profilage, sans téléchargement).

        @param level  Level du codec.
        @return torch.nn.Module respectant le contrat décrit dans la classe.
        """
        raise NotImplementedError

    def load_weights(self, model, level):
        """! @brief Charge les poids du niveau dans le modèle.

        @param model  Modèle renvoyé par make_model(level).
        @param level  Level du codec.
        @throws Exception si les poids sont introuvables ou incompatibles.
        """
        raise NotImplementedError

    def architecture(self, level):
        """! @brief Texte court qui distingue deux architectures, par exemple « N=128, M=192 ».

        Deux niveaux de même architecture ont le même coût embarqué : l'onglet « Encodeur embarqué »
        ne profile qu'un niveau par architecture.
        @param level  Level du codec.
        @return Description de l'architecture.
        """
        m = self.make_model(level)
        return f"N={m.N}, M={m.M}"

    # --- commun

    def model(self, level):
        """! @brief Modèle entraîné du niveau, chargé au premier appel puis gardé en cache (LRU).

        @param level  Level du codec.
        @return Modèle en mode évaluation.
        """
        if level.value not in self._cache:
            m = self.make_model(level)
            self.load_weights(m, level)
            self._cache[level.value] = m.eval()
            while len(self._cache) > self.max_cached_models:
                self._cache.popitem(last=False)
        self._cache.move_to_end(level.value)
        return self._cache[level.value]

    def encoder(self, level):
        """! @brief Transformée d'analyse g_a aux GDN gelées : ce qui part dans ONNX, Aidge et sur la carte.

        @param level  Level du codec.
        @return Module gelé, mis en cache sur le modèle.
        """
        m = self.model(level)
        if not hasattr(m, "_ais_encoder"):
            m._ais_encoder = freeze(m.g_a)
        return m._ais_encoder

    def onboard_module(self, model):
        """! @brief Tout ce qui s'exécute à bord pour produire le flux (hors codage arithmétique).

        @param model  Modèle, entraîné ou non (le profilage n'a pas besoin des poids).
        @return Module gelé (g_a, plus h_a et h_s pour un hyperprior).
        """
        return self.coding.onboard(model)

    def run(self, img, level, mode, backend="torch", real_rate=True):
        """! @copydoc aidge_in_space.codecs.base.Codec.run

        Avec @p real_rate, le flux (en-tête HEADER compris) est décodé et comparé aux latents de
        l'encodeur : le résultat est dans @c info["décodage exact"].
        """
        model = self.model(level)
        x = to_tensor(img, channels=model.in_ch)
        H, W = x.shape[-2:]
        Hp, Wp = _padded(H, model.pad_multiple), _padded(W, model.pad_multiple)
        xp = F.pad(x, (0, Wp - W, 0, Hp - H), mode="replicate")
        runner = backends.get_runner(backend, self.encoder(level), tuple(xp.shape))
        if real_rate:
            self.coding.tables(model)                 # préparées hors chronométrage

        t0 = time.perf_counter()
        y = torch.from_numpy(runner(xp.numpy()))
        t_analysis = time.perf_counter() - t0

        with torch.no_grad(), backends.limited_threads():
            t0 = time.perf_counter()
            enc = self.coding.encode(model, y, real_rate)
            t_coding = time.perf_counter() - t0

            t0 = time.perf_counter()
            info = {"latent": "x".join(map(str, y.shape[1:]))}
            if real_rate:
                stream = struct.pack(HEADER, MAGIC, VERSION, H, W, model.in_ch) + enc["data"]
                bits = 8 * len(stream)
                y_hat = self.decode_latents(model, stream)
                info["décodage exact"] = bool(torch.equal(y_hat, enc["y_hat"]))
                info["échappements"] = enc["n_esc"]
                info["bits estimés"] = round(enc["est_bits"])
            else:
                bits = enc["est_bits"]
                y_hat = enc["y_hat"]
            x_hat = model.g_s(y_hat)[..., :H, :W]
            t_decoding = time.perf_counter() - t0

        if model.in_ch == 3 and img.ndim == 2:
            info["note"] = "image grise répliquée sur 3 canaux"
        timings = {"encodage": t_analysis + t_coding, "analyse": t_analysis, "décodage": t_decoding}
        return Result(from_tensor(x_hat, mode), bits, "réel" if real_rate else "estimé", timings, info)

    def decode_latents(self, model, stream):
        """! @brief Relit l'en-tête d'un flux de run() puis décode ses latents.

        @param model   Modèle qui a produit le flux.
        @param stream  Flux complet (en-tête HEADER + données).
        @return Latents quantifiés y_hat, tenseur 1 x M x h x w.
        @throws ValueError si la signature ou la version de l'en-tête ne correspond pas.
        """
        hs = struct.calcsize(HEADER)
        magic, version, H, W, _ = struct.unpack(HEADER, stream[:hs])
        if magic != MAGIC or version != VERSION:
            raise ValueError("flux non reconnu")
        m, d = model.pad_multiple, model.downsampling
        shape = (1, model.M, _padded(H, m) // d, _padded(W, m) // d)
        return self.coding.decode(model, stream[hs:], shape)


def _padded(size, multiple):
    """! @brief Plus petit multiple de @p multiple supérieur ou égal à @p size."""
    return math.ceil(size / multiple) * multiple
