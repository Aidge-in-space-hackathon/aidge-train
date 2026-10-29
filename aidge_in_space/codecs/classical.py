"""! @brief Codecs classiques de référence, via Pillow : JPEG, JPEG 2000, WebP."""
import io
import time

from PIL import Image, features

from ..images import convert
from .base import Codec, Level, Result

## Qualités JPEG proposées (1 à 95).
JPEG_QUALITIES = [1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 95]
## Qualités WebP proposées (0 à 100).
WEBP_QUALITIES = [0, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
## Débits cibles JPEG 2000 proposés, en bits par pixel.
J2K_BPP = [0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


class PillowCodec(Codec):
    """! @brief Codec qui passe par Image.save() de Pillow ; le débit est toujours réel."""

    ## Famille des codecs de référence.
    family = "Classique"
    ## Format Pillow (argument de Image.save()).
    fmt = ""
    ## Fonctionnalité Pillow requise (features.check()), ou None.
    feature = None

    def available(self):
        """! @copydoc aidge_in_space.codecs.base.Codec.available"""
        if self.feature and not features.check(self.feature):
            return False, f"Pillow compilé sans {self.feature}"
        return True, ""

    def save_kwargs(self, level, mode):
        """! @brief Options de Image.save() pour ce niveau.

        @param level  Level du codec.
        @param mode   @c "gris" ou @c "rgb".
        @return Dictionnaire d'options.
        """
        raise NotImplementedError

    def run(self, img, level, mode, backend=None, real_rate=True):
        """! @copydoc aidge_in_space.codecs.base.Codec.run

        @p backend et @p real_rate sont ignorés (pas de moteur, débit toujours réel).
        """
        pil = Image.fromarray(img)
        buf = io.BytesIO()
        t0 = time.perf_counter()
        pil.save(buf, self.fmt, **self.save_kwargs(level, mode))
        t_enc = time.perf_counter() - t0
        data = buf.getvalue()
        t0 = time.perf_counter()
        rec = Image.open(io.BytesIO(data))
        rec.load()
        t_dec = time.perf_counter() - t0
        return Result(convert(rec, mode), 8 * len(data), "réel", {"encodage": t_enc, "décodage": t_dec}, {})


class Jpeg(PillowCodec):
    """! @brief JPEG (libjpeg)."""

    ## Identifiant, nom affiché et format Pillow.
    key, label, fmt = "jpeg", "JPEG", "JPEG"
    ## Description affichée.
    description ="libjpeg, tables de Huffman optimisées, sous-échantillonnage 4:2:0 en couleur."

    def levels(self, mode):
        """! @copydoc aidge_in_space.codecs.base.Codec.levels"""
        return [Level(q, f"qualité {q}") for q in JPEG_QUALITIES]

    def save_kwargs(self, level, mode):
        """! @copydoc aidge_in_space.codecs.classical.PillowCodec.save_kwargs"""
        return {"quality": level.value, "optimize": True}


class Jpeg2000(PillowCodec):
    """! @brief JPEG 2000 (OpenJPEG), à débit cible."""

    ## Identifiant, nom affiché, format Pillow et fonctionnalité requise.
    key, label, fmt, feature = "jpeg2000", "JPEG 2000", "JPEG2000", "jpg_2000"
    ## Description affichée.
    description =("OpenJPEG, ondelette 9/7 irréversible, une couche à débit cible. "
                   "Proche de CCSDS 122, le standard spatial de compression d'image.")

    def levels(self, mode):
        """! @copydoc aidge_in_space.codecs.base.Codec.levels"""
        return [Level(b, f"≈ {b} bpp") for b in J2K_BPP]

    def save_kwargs(self, level, mode):
        """! @brief Le débit cible est converti en taux de compression (8 ou 24 bpp bruts / débit)."""
        raw_bpp = 8 if mode == "gris" else 24
        return {"quality_mode": "rates", "quality_layers": [raw_bpp / level.value], "irreversible": True}


class WebP(PillowCodec):
    """! @brief WebP avec pertes (libwebp)."""

    ## Identifiant, nom affiché, format Pillow et fonctionnalité requise.
    key, label, fmt, feature = "webp", "WebP", "WEBP", "webp"
    ## Description affichée.
    description ="libwebp avec pertes (prédiction intra de VP8), effort maximal (method=6)."

    def levels(self, mode):
        """! @copydoc aidge_in_space.codecs.base.Codec.levels"""
        return [Level(q, f"qualité {q}") for q in WEBP_QUALITIES]

    def save_kwargs(self, level, mode):
        """! @copydoc aidge_in_space.codecs.classical.PillowCodec.save_kwargs"""
        return {"quality": level.value, "method": 6, "lossless": False}


def codecs():
    """! @brief JPEG, JPEG 2000 et WebP."""
    return [Jpeg(), Jpeg2000(), WebP()]
