"""! @brief Encodage et décodage par tuiles de taille fixe, sans dépendance à Aidge.

L'image est complétée (padding par réplication) à un multiple de la tuile, chaque tuile est encodée seule, et les latents
sont assemblés : le flux reste unique, seule la transformée voit des tuiles. Partagé par l'évaluation et le codec.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F


def padded_size(size, multiple):
    """! @brief Plus petit multiple de @p multiple supérieur ou égal à @p size."""
    return math.ceil(size / multiple) * multiple


def pad_to_tiles(x, tile_height, tile_width):
    """! @brief Complète une image pour la découper en tuiles entières.

    @param x            Tenseur 1 x C x H x W.
    @param tile_height  Hauteur de tuile.
    @param tile_width   Largeur de tuile.
    @return Tenseur complété par réplication.
    """
    H, W = x.shape[-2:]
    return F.pad(x, (0, padded_size(W, tile_width) - W, 0, padded_size(H, tile_height) - H), mode="replicate")


def encode_tiles(encode, xp, tile_height, tile_width):
    """! @brief Encode une image complétée, tuile par tuile, et assemble les latents.

    @param encode       Fonction tableau numpy 1 x C x h x w vers tableau numpy 1 x M x h' x w'.
    @param xp           Tenseur complété, 1 x C x H x W, aux côtés multiples de la tuile.
    @param tile_height  Hauteur de tuile.
    @param tile_width   Largeur de tuile.
    @return Tenseur des latents, 1 x M x H/d x W/d.
    """
    H, W = xp.shape[-2:]
    array = xp.numpy()
    rows = []
    for top in range(0, H, tile_height):
        row = [encode(np.ascontiguousarray(array[:, :, top:top + tile_height, left:left + tile_width]))
               for left in range(0, W, tile_width)]
        rows.append(np.concatenate(row, axis=3))
    return torch.from_numpy(np.concatenate(rows, axis=2))


def decode_tiles(decode, y_hat, latent_height, latent_width):
    """! @brief Décode des latents assemblés, tuile par tuile.

    @param decode         Fonction tenseur de latents d'une tuile vers tenseur image.
    @param y_hat          Latents, 1 x M x h x w, aux côtés multiples de la tuile latente.
    @param latent_height  Hauteur d'une tuile dans les latents.
    @param latent_width   Largeur d'une tuile dans les latents.
    @return Tenseur image reconstruit.
    """
    h, w = y_hat.shape[-2:]
    rows = [torch.cat([decode(y_hat[:, :, top:top + latent_height, left:left + latent_width])
                       for left in range(0, w, latent_width)], dim=3)
            for top in range(0, h, latent_height)]
    return torch.cat(rows, dim=2)
