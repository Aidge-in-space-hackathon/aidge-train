"""! @brief Coût de la partie embarquée, couche par couche : calcul, paramètres, mémoire d'activation."""
import pandas as pd
import torch
import torch.nn as nn

from .nn import GDN, FrozenGDN, n_params


def profile_module(module, input_shape):
    """! @brief Profil couche par couche d'un module, obtenu par une passe sur une entrée nulle.

    Une ligne par convolution, convolution transposée et (I)GDN ; la conv 1x1 interne d'une
    FrozenGDN est comptée avec sa GDN.
    @param module       Module à profiler (les poids n'ont pas d'importance).
    @param input_shape  Forme de l'entrée, (1, C, H, W).
    @return DataFrame avec les colonnes couche, type, entrée, sortie, paramètres, MAC (M),
            sqrt+div (M), entrée fp32 (Mo), sortie fp32 (Mo) et % MAC.
    """
    rows, hooks = [], []

    def hook(m, inp, out):
        x, o = inp[0], out
        if isinstance(m, nn.Conv2d):
            kind = f"Conv {m.kernel_size[0]}x{m.kernel_size[1]} s{m.stride[0]}"
            macs = o[0].numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
            nl = 0
        elif isinstance(m, nn.ConvTranspose2d):
            kind = f"ConvT {m.kernel_size[0]}x{m.kernel_size[1]} s{m.stride[0]}"
            macs = x[0].numel() * (m.out_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
            nl = 0
        else:
            kind = "IGDN" if m.inverse else "GDN"
            C, h, w = o.shape[1:]
            macs = h * w * C * C + h * w * C          # conv 1x1 sur x^2 + calcul des carrés
            nl = 2 * h * w * C                        # une racine + une division (ou mult.) par valeur
        rows.append({
            "couche": names[m], "type": kind,
            "entrée": "x".join(map(str, x.shape[1:])), "sortie": "x".join(map(str, o.shape[1:])),
            "paramètres": n_params(m),
            "MAC (M)": macs / 1e6, "sqrt+div (M)": nl / 1e6,
            "entrée fp32 (Mo)": x[0].numel() * 4 / 2 ** 20,
            "sortie fp32 (Mo)": o[0].numel() * 4 / 2 ** 20,
        })

    names = {}
    for name, m in module.named_modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, GDN, FrozenGDN)):
            if isinstance(m, nn.Conv2d) and name.endswith(".norm"):
                continue
            names[m] = name
            hooks.append(m.register_forward_hook(hook))
    try:
        with torch.no_grad():
            module.eval()(torch.zeros(input_shape))
    finally:
        for h in hooks:
            h.remove()
    df = pd.DataFrame(rows)
    df["% MAC"] = 100 * df["MAC (M)"] / df["MAC (M)"].sum()
    return df


def summarize(df):
    """! @brief Totaux d'un profil de profile_module().

    @param df  Profil couche par couche.
    @return Dictionnaire : GMAC, sqrt+div (M), paramètres, poids fp32 (Mo), poids int8 (Mo),
            plus grande activation fp32 (Mo) et pic entrée+sortie fp32 (Mo). Ce dernier (entrée +
            sortie de la couche la plus gourmande) est une borne basse de la mémoire de travail.
    """
    params = int(df["paramètres"].sum())
    return {
        "GMAC": df["MAC (M)"].sum() / 1e3,
        "sqrt+div (M)": df["sqrt+div (M)"].sum(),
        "paramètres": params,
        "poids fp32 (Mo)": params * 4 / 2 ** 20,
        "poids int8 (Mo)": params / 2 ** 20,
        "plus grande activation fp32 (Mo)": df["sortie fp32 (Mo)"].max(),
        "pic entrée+sortie fp32 (Mo)": (df["entrée fp32 (Mo)"] + df["sortie fp32 (Mo)"]).max(),
    }
