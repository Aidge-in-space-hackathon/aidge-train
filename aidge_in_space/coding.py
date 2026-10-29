"""! @brief Codage entropique des latents, une classe par famille de modèle d'entropie.

Un codec appris (aidge_in_space.codecs.base.LearnedCodec) délègue à un objet LatentCoding tout ce
qui dépend du modèle d'entropie :
- la partie du réseau qui s'exécute à bord (onboard()) ;
- la préparation des tables de fréquences (prepare()) ;
- le codage des latents en un flux binaire, ou la seule estimation de son débit (encode()) ;
- le décodage du flux (decode()).

Pour ajouter un modèle d'entropie (mean-scale hyperprior, contexte autorégressif...), il suffit
d'écrire une nouvelle sous-classe de LatentCoding et de l'affecter à l'attribut @c coding du codec.
"""
import struct

import numpy as np
import torch

from .entropy import decode_symbols, encode_symbols, make_tables
from .nn import (HyperpriorOnboard, bits_from_likelihood, freeze, gaussian_likelihood, gaussian_pmfs,
                 scale_indexes)

## Demi-largeur (en pas de quantification) du support des tables d'un prior factorisé, autour de la médiane.
LIKELIHOOD_TABLE_RANGE = 255


class LatentCoding:
    """! @brief Interface du codage entropique des latents d'un modèle.

    Les tables de fréquences sont mises en cache sur le modèle lui-même (attribut @c _ais_tables),
    qui est déjà gardé en cache par le codec : elles ne sont calculées qu'une fois par niveau.
    """

    def onboard(self, model):
        """! @brief Module à exécuter à bord pour produire le flux (hors codage arithmétique).

        @param model  Modèle entraîné (FactorizedPrior2018, Balle2017...).
        @return Copie du module en mode évaluation, avec des GDN gelées (exportable en ONNX).
        """
        return freeze(model.g_a)

    def tables(self, model):
        """! @brief Tables de fréquences du modèle, calculées au premier appel puis mises en cache.

        @param model  Modèle entraîné.
        @return Tables utilisées par encode() et decode() (format propre à la sous-classe).
        """
        if not hasattr(model, "_ais_tables"):
            model._ais_tables = self._make_tables(model)
        return model._ais_tables

    def _make_tables(self, model):
        """! @brief Construit les tables de fréquences (à fournir par la sous-classe)."""
        raise NotImplementedError

    def encode(self, model, y, real):
        """! @brief Quantifie les latents et produit le flux binaire, ou estime seulement son débit.

        @param model  Modèle entraîné.
        @param y      Latents non quantifiés en sortie de g_a, tenseur 1 x M x h x w.
        @param real   True : codage arithmétique réel ; False : débit estimé par l'entropie du modèle.
        @return Dictionnaire avec les clés :
                - @c y_hat    : latents quantifiés, tels que le décodeur les reconstruira ;
                - @c est_bits : débit estimé en bits (toujours présent) ;
                - @c data     : flux binaire (bytes), si @p real ;
                - @c n_esc    : nombre de symboles échappés (hors des tables), si @p real.
        """
        raise NotImplementedError

    def decode(self, model, data, latent_shape):
        """! @brief Décode un flux produit par encode().

        @param model         Modèle entraîné (le même qu'à l'encodage).
        @param data          Flux binaire, sans l'en-tête du codec.
        @param latent_shape  Forme des latents y, (1, M, h, w).
        @return Latents quantifiés y_hat, identiques à ceux de encode().
        """
        raise NotImplementedError


class FactorizedCoding(LatentCoding):
    """! @brief Prior factorisé (Ballé 2017, bmshj2018-factorized) : une table par canal.

    Le modèle doit exposer @c prior, un aidge_in_space.nn.FactorizedDensity ou
    aidge_in_space.nn.EntropyBottleneck. Les latents sont quantifiés autour de la médiane de chaque
    canal, comme dans CompressAI.
    """

    def _make_tables(self, model):
        """! @brief Une table par canal, pour les symboles q = round(y - médiane)."""
        return factorized_tables(model.prior)

    def encode(self, model, y, real):
        """! @copydoc aidge_in_space.coding.LatentCoding.encode"""
        return factorized_encode(model.prior, self.tables(model) if real else None, y, real)

    def decode(self, model, data, latent_shape):
        """! @copydoc aidge_in_space.coding.LatentCoding.decode"""
        return factorized_decode(model.prior, self.tables(model), data, latent_shape)


class HyperpriorCoding(LatentCoding):
    """! @brief Scale hyperprior gaussien (bmshj2018-hyperprior).

    Le flux contient d'abord l'hyper-latent z (prior factorisé, une table par canal), précédé de sa
    longueur sur 4 octets, puis les latents y codés avec une loi gaussienne centrée dont l'échelle
    est prédite par h_s(z). Le modèle doit exposer @c h_a, @c h_s et @c entropy_bottleneck.
    """

    def onboard(self, model):
        """! @brief g_a, h_a et h_s s'exécutent à bord (h_s pour connaître les échelles de y).

        @param model  aidge_in_space.nn.ScaleHyperprior2018 entraîné.
        @return aidge_in_space.nn.HyperpriorOnboard aux GDN gelées.
        """
        return freeze(HyperpriorOnboard(model))

    def _make_tables(self, model):
        """! @brief Tables de z (une par canal) ; celles de y sont communes à tous les modèles."""
        gaussian_tables()
        return factorized_tables(model.entropy_bottleneck)

    def encode(self, model, y, real):
        """! @copydoc aidge_in_space.coding.LatentCoding.encode"""
        z = model.h_a(torch.abs(y))
        z_enc = factorized_encode(model.entropy_bottleneck, self.tables(model) if real else None, z, real)
        scales = model.h_s(z_enc["y_hat"])
        y_hat = torch.round(y)
        out = {"y_hat": y_hat,
               "est_bits": z_enc["est_bits"] + bits_from_likelihood(gaussian_likelihood(y_hat, scales))}
        if real:
            y_data, n_esc = encode_symbols(y_hat.to(torch.int64).numpy(), scale_indexes(scales).numpy(),
                                           gaussian_tables())
            out["data"] = struct.pack(">I", len(z_enc["data"])) + z_enc["data"] + y_data
            out["n_esc"] = n_esc + z_enc["n_esc"]
        return out

    def decode(self, model, data, latent_shape):
        """! @copydoc aidge_in_space.coding.LatentCoding.decode"""
        n_z = struct.unpack(">I", data[:4])[0]
        z_shape = (1, model.N, latent_shape[2] // 4, latent_shape[3] // 4)     # h_a divise par 4
        z_hat = factorized_decode(model.entropy_bottleneck, self.tables(model), data[4:4 + n_z], z_shape)
        scales = model.h_s(z_hat)
        q = decode_symbols(data[4 + n_z:], scale_indexes(scales).numpy(), gaussian_tables())
        return torch.from_numpy(q.reshape(latent_shape)).float()


# ---------------------------------------------------------------- prior factorisé

@torch.no_grad()
def factorized_tables(prior, R=LIKELIHOOD_TABLE_RANGE):
    """! @brief Tables de fréquences d'un prior factorisé, une par canal.

    @param prior  aidge_in_space.nn.FactorizedDensity ou aidge_in_space.nn.EntropyBottleneck.
    @param R      Demi-largeur du support autour de la médiane ; les valeurs plus éloignées sont échappées.
    @return aidge_in_space.entropy.CdfTables pour les symboles q = round(y - médiane).
    """
    k = torch.arange(-R, R + 1, dtype=torch.float32)
    v = k[None, :] + prior.medians()[:, None]
    pmf = prior.prob(v).double().numpy()
    return make_tables(list(pmf), [-R] * pmf.shape[0])


def channel_index(shape):
    """! @brief Indice de table de chaque symbole d'un tenseur 1 x C x h x w : son canal.

    @param shape  Forme (1, C, h, w).
    @return Tableau d'entiers de longueur C*h*w, dans l'ordre de parcours de numpy (canal par canal).
    """
    _, C, h, w = shape
    return np.repeat(np.arange(C), h * w)


def factorized_encode(prior, tables, y, real):
    """! @brief Quantifie y autour des médianes et le code avec un prior factorisé.

    @param prior   Prior factorisé du modèle.
    @param tables  Tables de factorized_tables(), ou None si @p real est faux.
    @param y       Tenseur 1 x C x h x w.
    @param real    True pour produire le flux, False pour la seule estimation du débit.
    @return Dictionnaire décrit dans LatentCoding.encode().
    """
    C = y.shape[1]
    med = prior.medians().view(1, C, 1, 1)
    q = torch.round(y - med)
    y_hat = q + med
    out = {"y_hat": y_hat, "est_bits": bits_from_likelihood(prior.likelihood(y_hat))}
    if real:
        out["data"], out["n_esc"] = encode_symbols(q.to(torch.int64).numpy(), channel_index(q.shape), tables)
    return out


def factorized_decode(prior, tables, data, shape):
    """! @brief Décode un flux de factorized_encode().

    @param prior   Prior factorisé du modèle.
    @param tables  Tables de factorized_tables().
    @param data    Flux binaire.
    @param shape   Forme (1, C, h, w) du tenseur codé.
    @return Tenseur quantifié y_hat (médianes rajoutées).
    """
    q = decode_symbols(data, channel_index(shape), tables)
    med = prior.medians().view(1, shape[1], 1, 1)
    return torch.from_numpy(q.reshape(shape)).float() + med


# ---------------------------------------------------------------- prior gaussien

## Cache de gaussian_tables().
_GAUSSIAN_TABLES = None


def gaussian_tables():
    """! @brief Tables des lois gaussiennes discrètes, une par niveau de aidge_in_space.nn.SCALE_TABLE.

    Ces tables ne dépendent pas des poids : elles sont calculées une fois pour tous les modèles.
    @return aidge_in_space.entropy.CdfTables indexées par aidge_in_space.nn.scale_indexes().
    """
    global _GAUSSIAN_TABLES
    if _GAUSSIAN_TABLES is None:
        _GAUSSIAN_TABLES = make_tables(*gaussian_pmfs())
    return _GAUSSIAN_TABLES
