"""! @brief Mesures de qualité (PSNR, MS-SSIM) et écart de débit de Bjøntegaard (BD-rate)."""
import math

import numpy as np
import torch
import torch.nn.functional as F

## Poids des 5 échelles de MS-SSIM (Wang et al. 2003).
MS_SSIM_WEIGHTS = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=torch.float64)
## Taille de la fenêtre gaussienne de SSIM.
SSIM_WINDOW = 11
## Plus petit côté d'image pour MS-SSIM : après 4 sous-échantillonnages par 2 (arrondis vers le bas),
## l'échelle la plus grossière doit encore contenir la fenêtre (176 // 16 = 11).
MS_SSIM_MIN_SIDE = SSIM_WINDOW * 2 ** (len(MS_SSIM_WEIGHTS) - 1)


def psnr(a, b):
    """! @brief Peak Signal-to-Noise Ratio entre deux images 8 bits.

    @param a  Image de référence (uint8).
    @param b  Image reconstruite, même forme.
    @return PSNR en dB (infini si les images sont identiques).
    """
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def _as_tensor(img):
    """! @brief Image uint8 -> tenseur float64 1 x C x H x W."""
    t = torch.from_numpy(img.astype(np.float64))
    return t[None, None] if t.ndim == 2 else t.permute(2, 0, 1)[None]


def _ssim_and_cs(x, y, win, data_range=255.0):
    """! @brief SSIM et terme de contraste-structure moyens par canal, fenêtre gaussienne séparable sans padding."""
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    ch = x.shape[1]
    wh = win.view(1, 1, 1, -1).repeat(ch, 1, 1, 1)
    wv = win.view(1, 1, -1, 1).repeat(ch, 1, 1, 1)
    blur = lambda t: F.conv2d(F.conv2d(t, wh, groups=ch), wv, groups=ch)
    mu1, mu2 = blur(x), blur(y)
    s11 = blur(x * x) - mu1 ** 2
    s22 = blur(y * y) - mu2 ** 2
    s12 = blur(x * y) - mu1 * mu2
    cs = (2 * s12 + c2) / (s11 + s22 + c2)
    ssim = ((2 * mu1 * mu2 + c1) / (mu1 ** 2 + mu2 ** 2 + c1)) * cs
    return ssim.mean(dim=(2, 3)), cs.mean(dim=(2, 3))


def ms_ssim(a, b):
    """! @brief MS-SSIM à 5 échelles (Wang et al. 2003), fenêtre gaussienne 11x11 de sigma 1,5.

    @param a  Image de référence (uint8).
    @param b  Image reconstruite, même forme.
    @return MS-SSIM dans [0, 1], moyenné sur les canaux ; NaN si un côté est inférieur à MS_SSIM_MIN_SIDE.
    """
    x, y = _as_tensor(a), _as_tensor(b)
    if min(x.shape[-2:]) < MS_SSIM_MIN_SIDE:
        return float("nan")
    c = torch.arange(SSIM_WINDOW, dtype=torch.float64) - SSIM_WINDOW // 2
    win = torch.exp(-c ** 2 / (2 * 1.5 ** 2))
    win /= win.sum()
    values = []
    n = len(MS_SSIM_WEIGHTS)
    for i in range(n):
        ssim, cs = _ssim_and_cs(x, y, win)
        if i < n - 1:
            values.append(torch.relu(cs))
            x, y = F.avg_pool2d(x, 2), F.avg_pool2d(y, 2)
    values.append(torch.relu(ssim))
    res = torch.prod(torch.stack(values) ** MS_SSIM_WEIGHTS.view(-1, 1, 1), dim=0)
    return float(res.mean())


def ms_ssim_db(v):
    """! @brief MS-SSIM en décibels : -10 log10(1 - MS-SSIM), plus lisible près de 1.

    @param v  MS-SSIM (NaN accepté).
    @return Valeur en dB, ou NaN.
    """
    return -10 * math.log10(max(1 - v, 1e-10)) if v == v else float("nan")


def bd_rate(rate_ref, quality_ref, rate_test, quality_test):
    """! @brief Écart de débit moyen de Bjøntegaard, à qualité égale.

    Interpolation cubique de log(débit) en fonction de la qualité, intégrée sur la plage de
    qualité commune aux deux courbes. Les points non finis ou de débit nul sont ignorés.
    @param rate_ref      Débits de la courbe de référence.
    @param quality_ref   Qualités (PSNR, MS-SSIM en dB...) de la référence.
    @param rate_test     Débits de la courbe testée.
    @param quality_test  Qualités de la courbe testée.
    @return Écart en % (négatif = le codec testé économise du débit), ou NaN s'il y a moins de
            4 points par courbe ou pas de plage de qualité commune.
    """
    r1, q1, r2, q2 = (np.asarray(v, dtype=np.float64) for v in (rate_ref, quality_ref, rate_test, quality_test))
    ok1, ok2 = np.isfinite(r1) & np.isfinite(q1) & (r1 > 0), np.isfinite(r2) & np.isfinite(q2) & (r2 > 0)
    r1, q1, r2, q2 = r1[ok1], q1[ok1], r2[ok2], q2[ok2]
    if len(r1) < 4 or len(r2) < 4:
        return float("nan")
    lo, hi = max(q1.min(), q2.min()), min(q1.max(), q2.max())
    if hi <= lo:
        return float("nan")
    p1, p2 = np.polyfit(q1, np.log(r1), 3), np.polyfit(q2, np.log(r2), 3)
    i1, i2 = np.polyint(p1), np.polyint(p2)
    avg = ((np.polyval(i2, hi) - np.polyval(i2, lo)) - (np.polyval(i1, hi) - np.polyval(i1, lo))) / (hi - lo)
    return float((math.exp(avg) - 1) * 100)
