"""! @brief Évaluation d'une variante : fidélité des latents arrondis, débit-distorsion, écarts et BD-rate.

Le pourcentage brut de symboles identiques est trompeur sur un modèle peu entraîné, dont presque tous les symboles valent
0 : fidelity() donne donc aussi la part de symboles non nuls et la fidélité sur ceux-ci.
"""
import struct

import numpy as np
import torch

from .. import backends, metrics
from ..codecs.base import HEADER, MAGIC, VERSION
from ..images import from_tensor, load_image, to_tensor
from .tiling import decode_tiles, encode_tiles, pad_to_tiles

## Nombre de niveaux nécessaires à metrics.bd_rate.
BD_RATE_MIN_LEVELS = 4


def fidelity(reference, candidate):
    """! @brief Fidélité de latents face à une référence, après arrondi (fonction pure).

    @param reference  Liste de tableaux de latents de référence.
    @param candidate  Liste de tableaux de latents à comparer, mêmes formes.
    @return Dictionnaire de la clé @c fidelity de report.json.
    """
    ref = np.concatenate([np.ravel(a) for a in reference])
    cand = np.concatenate([np.ravel(a) for a in candidate])
    r, c = np.round(ref), np.round(cand)
    nonzero = r != 0
    return {"symbols_identical_pct": round(float(np.mean(r == c)) * 100, 3),
            "nonzero_share_pct": round(float(np.mean(nonzero)) * 100, 3),
            "nonzero_identical_pct": round(float(np.mean(r[nonzero] == c[nonzero])) * 100, 3) if nonzero.any() else None,
            "max_rounded_error": int(np.abs(r - c).max()),
            "max_raw_error": float(np.abs(ref - cand).max()),
            "n_tiles": len(reference)}


def symbols_identical_pct(reference, candidate):
    """! @brief Pourcentage de symboles arrondis identiques entre deux tableaux (fonction pure)."""
    return float(np.mean(np.round(reference) == np.round(candidate))) * 100


def rd_image(model, encode, img, mode, tile):
    """! @brief Compresse et reconstruit une image par tuiles, avec le vrai codeur arithmétique.

    @param model   Modèle source : décodeur et modèle d'entropie.
    @param encode  Fonction tableau numpy de tuile vers tableau numpy de latents.
    @param img     Image uint8.
    @param mode    @c "gris" ou @c "rgb".
    @param tile    Taille de tuile (C, H, W).
    @return Dictionnaire @c bpp, @c psnr, @c ms_ssim.
    """
    from ..coding import FactorizedCoding
    coding = FactorizedCoding()
    x = to_tensor(img, channels=model.in_ch)
    H, W = x.shape[-2:]
    xp = pad_to_tiles(x, tile[1], tile[2])
    d = model.downsampling
    with torch.no_grad(), backends.limited_threads():
        coding.tables(model)
        y = encode_tiles(encode, xp, tile[1], tile[2])
        enc = coding.encode(model, y, True)
        x_hat = decode_tiles(model.g_s, enc["y_hat"], tile[1] // d, tile[2] // d)[..., :H, :W]
    stream = struct.pack(HEADER, MAGIC, VERSION, H, W, model.in_ch) + enc["data"]
    recon = from_tensor(x_hat, mode)
    return {"bpp": 8 * len(stream) / (H * W), "psnr": metrics.psnr(img, recon), "ms_ssim": metrics.ms_ssim_db(metrics.ms_ssim(img, recon))}


def rd_average(model, encode, paths, mode, tile, should_stop=None):
    """! @brief Débit-distorsion moyen sur des images.

    @param model        Modèle source.
    @param encode       Fonction tableau numpy de tuile vers tableau numpy de latents.
    @param paths        Chemins des images.
    @param mode         @c "gris" ou @c "rgb".
    @param tile         Taille de tuile.
    @param should_stop  Fonction appelée avant chaque image ; une valeur vraie interrompt le calcul.
    @return Dictionnaire de la clé @c rd de report.json (@c rate_kind vaut @c "réel"), ou None si interrompu.
    """
    rows = []
    for path in paths:
        if should_stop is not None and should_stop():
            return None
        rows.append(rd_image(model, encode, load_image(path, mode), mode, tile))
    return {"images": len(rows), "rate_kind": "réel",
            "bpp": float(np.mean([r["bpp"] for r in rows])), "psnr": float(np.mean([r["psnr"] for r in rows])),
            "ms_ssim": float(np.nanmean([r["ms_ssim"] for r in rows])) if any(r["ms_ssim"] == r["ms_ssim"] for r in rows) else None,
            "delta_bpp_pct": None, "delta_psnr_db": None, "delta_ms_ssim_db": None}


def add_deltas(rd, reference):
    """! @brief Complète un résultat débit-distorsion avec ses écarts face à une référence (fonction pure).

    @param rd         Résultat modifié en place.
    @param reference  Résultat de la référence, FP32 du même checkpoint et de la même tuile.
    @return @p rd.
    """
    rd["delta_bpp_pct"] = (rd["bpp"] / reference["bpp"] - 1) * 100
    rd["delta_psnr_db"] = rd["psnr"] - reference["psnr"]
    if rd.get("ms_ssim") is not None and reference.get("ms_ssim") is not None:
        rd["delta_ms_ssim_db"] = rd["ms_ssim"] - reference["ms_ssim"]
    return rd


def is_monotonic(points):
    """! @brief Vrai si le débit et la qualité croissent strictement avec λ (fonction pure).

    @param points  Liste de dictionnaires @c lmbda, @c bpp, @c psnr.
    @return Vrai si la courbe est monotone.
    """
    ordered = sorted(points, key=lambda p: p["lmbda"])
    return all(a["bpp"] < b["bpp"] and a["psnr"] < b["psnr"] for a, b in zip(ordered, ordered[1:]))


def bd_series(levels):
    """! @brief BD-rate d'une variante face au FP32, par paires de même checkpoint, et écarts point par point.

    @param levels  Liste de dictionnaires @c lmbda, @c variant et @c fp32, ces deux derniers ayant @c bpp et @c psnr.
    @return Dictionnaire @c bd_rate (None si le calcul est refusé), @c reason (pourquoi), @c n_levels, @c needed et
            @c points (écarts Δbpp et ΔPSNR par niveau, toujours donnés).
    """
    ordered = sorted(levels, key=lambda r: r["lmbda"])
    points = [{"lmbda": r["lmbda"], "delta_bpp_pct": (r["variant"]["bpp"] / r["fp32"]["bpp"] - 1) * 100,
               "delta_psnr_db": r["variant"]["psnr"] - r["fp32"]["psnr"]} for r in ordered]
    out = {"bd_rate": None, "reason": None, "n_levels": len(ordered), "needed": BD_RATE_MIN_LEVELS, "points": points}
    if len(ordered) < BD_RATE_MIN_LEVELS:
        out["reason"] = f"{len(ordered)} niveaux sur {BD_RATE_MIN_LEVELS}"
        return out
    for key, label in (("variant", "variante"), ("fp32", "FP32")):
        if not is_monotonic([{"lmbda": r["lmbda"], **r[key]} for r in ordered]):
            out["reason"] = f"courbe {label} non monotone"
            return out
    value = metrics.bd_rate([r["fp32"]["bpp"] for r in ordered], [r["fp32"]["psnr"] for r in ordered],
                            [r["variant"]["bpp"] for r in ordered], [r["variant"]["psnr"] for r in ordered])
    if value != value:
        out["reason"] = "pas de plage de qualité commune"
        return out
    out["bd_rate"] = value
    return out
