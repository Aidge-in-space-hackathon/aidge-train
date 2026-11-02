"""! @brief Variantes optimisées par Aidge : un codec par série, découvert dans outputs/optim.

Chaque job terminé apporte un niveau par recette utilisable : décodeur et modèle d'entropie du checkpoint source, encodeur
reconstruit depuis la recette et exécuté par Aidge, tuile par tuile, décodage par tuiles. Les onglets Tester et Valider les
voient comme n'importe quel codec. La découverte n'importe pas Aidge : seule l'exécution le charge.
Contrat : docs/conception_optimisation.md (§ 8).
"""
from pathlib import Path

import numpy as np
import torch

from .. import backends
from ..nn import Balle2017
from ..optim import results
from ..optim.tiling import decode_tiles, encode_tiles
from ..training.config import resolve_path
from .balle2017 import slug
from .base import LearnedCodec, Level


class AidgeVariantCodec(LearnedCodec):
    """! @brief Une série : même nom de job, même architecture, même recette, même tuile ; un niveau par job."""

    ## L'encodeur s'exécute toujours sur Aidge : c'est ce qui est mesuré.
    backends = ("aidge",)

    def __init__(self, reports, recipe):
        """! @brief Crée le codec d'une série.

        @param reports  Rapports de la série (voir aidge_in_space.optim.results.read_reports).
        @param recipe   Recette.
        """
        super().__init__()
        ## Recette de la série.
        self.recipe = recipe
        ## Rapports triés par λ croissant, donc par débit croissant.
        self.reports = sorted(reports, key=lambda r: r["source"]["lmbda"])
        first = self.reports[0]
        src = first["source"]
        ## Canaux de l'image, canaux des couches et GDN finale.
        self.in_ch, self.N, self.gdn_last = src["in_ch"], src["N"], src.get("gdn_last", True)
        ## Taille de tuile (C, H, W).
        self.tile = tuple(first["tile"])
        ## Identifiant : nom du job, architecture, recette et tuile.
        self.key = f"aidge-{slug(first['name'])}-c{self.in_ch}-n{self.N}-{recipe}-t{self.tile[1]}"
        ## Nom affiché.
        self.label = f"Aidge · {recipe} · {first['name']} · N={self.N} · tuile {self.tile[1]}"
        ## Description affichée.
        self.description = (f"Variante {recipe} de « {first['name']} », encodeur exécuté par Aidge par tuiles de "
                            f"{self.tile[1]} x {self.tile[2]} : {len(self.reports)} niveau(x).")
        ## Un modèle gris n'accepte que le mode gris ; un modèle RGB accepte les deux.
        self.modes = ("gris",) if self.in_ch == 1 else ("rgb", "gris")
        ## Rapport de chaque niveau, par valeur de Level.
        self._by_value = {r["_dir"]: r for r in self.reports}

    def _fresh(self):
        """! @brief Rapports dont le checkpoint source existe et n'a pas changé."""
        return [r for r in self.reports
                if results.file_sha256(resolve_path(r["source"]["checkpoint"])) == r["source"].get("file_sha256")]

    def available(self):
        """! @copydoc aidge_in_space.codecs.base.Codec.available"""
        ok, why = backends.available("aidge")
        if not ok:
            return False, why
        if not self._fresh():
            return False, "checkpoint source absent ou modifié depuis l'optimisation"
        return True, ""

    def levels(self, mode):
        """! @copydoc aidge_in_space.codecs.base.Codec.levels

        La valeur de chaque niveau est le dossier de résultats du job.
        """
        return [Level(r["_dir"], f"λ={r['source']['lmbda']:g} ({r['source']['step']:,} pas) · job {r['id']}".replace(",", " "))
                for r in self._fresh()]

    def make_model(self, level):
        """! @copydoc aidge_in_space.codecs.base.LearnedCodec.make_model"""
        return Balle2017(self.in_ch, self.N, self.gdn_last)

    def load_weights(self, model, level):
        """! @copydoc aidge_in_space.codecs.base.LearnedCodec.load_weights"""
        ck = torch.load(resolve_path(self._by_value[level.value]["source"]["checkpoint"]), map_location="cpu")
        model.load_state_dict(ck["state_dict"])

    def architecture(self, level):
        """! @copydoc aidge_in_space.codecs.base.LearnedCodec.architecture"""
        return f"N={self.N}" + ("" if self.gdn_last else ", sans GDN finale")

    def _pad_multiple(self, model):
        """! @copydoc aidge_in_space.codecs.base.LearnedCodec._pad_multiple

        L'image est complétée à un multiple de la tuile.
        """
        return self.tile[1]

    def _graph_runner(self, level):
        """! @brief Exécuteur Aidge d'une tuile, reconstruit depuis la recette puis gardé sur le modèle.

        @param level  Level du codec.
        @return aidge_in_space.optim.graph.Runner.
        @throws RuntimeError si la recette ne produit plus de graphe.
        """
        model = self.model(level)
        if not hasattr(model, "_ais_tile_runner"):
            from ..optim import graph, ptq
            report = self._by_value[level.value]
            encoder = self.encoder(level)
            ptq_cfg = report["config"]["ptq"]
            calibration = np.load(Path(report["_dir"]) / "calib.npz")["tiles"]
            if self.recipe == "fp32":
                g = graph.import_encoder(encoder, self.tile)
            elif self.recipe == "mixte":
                g = ptq.build_mixed(encoder, self.tile, calibration, ptq_cfg["clipping"], ptq_cfg["per_channel"])
            else:
                g = ptq.try_int8(encoder, self.tile, calibration, ptq_cfg["clipping"], ptq_cfg["per_channel"])["graph"]
                if g is None:
                    raise RuntimeError(f"la recette {self.recipe} ne produit plus de graphe")
            model._ais_tile_runner = graph.Runner(g)
        return model._ais_tile_runner

    def _analysis_runner(self, level, shape, backend):
        """! @copydoc aidge_in_space.codecs.base.LearnedCodec._analysis_runner

        Encode tuile par tuile avec Aidge, quel que soit @p backend, et assemble les latents.
        """
        tile_runner = self._graph_runner(level)
        return lambda x: encode_tiles(tile_runner, torch.from_numpy(x), self.tile[1], self.tile[2]).numpy()

    def _synthesize(self, model, y_hat):
        """! @copydoc aidge_in_space.codecs.base.LearnedCodec._synthesize

        Décode tuile par tuile.
        """
        d = model.downsampling
        return decode_tiles(model.g_s, y_hat, self.tile[1] // d, self.tile[2] // d)


def codecs():
    """! @brief Un codec par série de variantes utilisables trouvées dans outputs/optim.

    Les recettes sans graphe (sans effet, en échec) n'y figurent pas : l'onglet d'optimisation les signale. Une clé déjà
    prise reçoit un suffixe @c -2, @c -3, etc.
    """
    groups = {}
    for report in results.read_reports():
        for recipe, variant in report.get("variants", {}).items():
            if variant.get("status") == "ok":
                groups.setdefault(results.series_key(report, recipe), (recipe, []))[1].append(report)
    found, used = [], set()
    for recipe, reports in groups.values():
        codec = AidgeVariantCodec(reports, recipe)
        base, k = codec.key, 2
        while codec.key in used:
            codec.key, k = f"{base}-{k}", k + 1
        used.add(codec.key)
        found.append(codec)
    return found
