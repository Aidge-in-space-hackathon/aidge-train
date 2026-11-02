"""! @brief Encodeur Ballé 2017 vers GraphView Aidge : import, rapport de quantifiabilité, compte des nœuds, exécution.

Tout graphe passe par import_graph() : le format de données doit être fixé à NCHW **avant** la compilation, sinon
l'export C++ produit un binaire faux sans erreur à l'exécution (docs/conception_optimisation.md, § 1.5).
"""
import collections
import os
import re
import tempfile

import numpy as np
import torch

import aidge_core  # en premier : charge la DLL dont dépend le backend
import aidge_backend_cpu  # noqa: F401  (enregistre les noyaux CPU)
import aidge_onnx
import aidge_quantization

from .. import backends
from ..nn import Balle2017

## Lignes du rapport de quantifiabilité : « - [ok] Conv2D x3 ».
_REPORT_LINE = re.compile(r"- \[(\S+)\] (\S+) x(\d+)")
## États du rapport, traduits : « ? » signifie qu'Aidge ne sait pas quantifier l'opérateur.
_REPORT_STATES = {"ok": "ok", "?": "inconnu"}


def load_model(checkpoint):
    """! @brief Charge un modèle Ballé 2017 depuis un checkpoint.

    @param checkpoint  Chemin du checkpoint.
    @return Couple (modèle en évaluation, checkpoint lu).
    @throws Exception si le fichier est absent ou incompatible.
    """
    ck = torch.load(checkpoint, map_location="cpu")
    cfg = ck["cfg"]
    model = Balle2017(cfg["in_ch"], cfg["N"], cfg.get("gdn_last", True))
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck


def conv_names(encoder):
    """! @brief Noms des convolutions de l'encodeur, dans l'ordre : ce que la recette mixte quantifie.

    @param encoder  Encodeur gelé.
    @return Liste des noms (conv1, conv2, conv3).
    """
    return [name for name, _ in encoder.named_children() if name.startswith("conv")]


def import_graph(onnx_path, shape):
    """! @brief Charge un fichier ONNX et le compile sur le backend CPU, en flottant et au format NCHW.

    @param onnx_path  Fichier ONNX.
    @param shape      Forme de l'entrée, (1, C, H, W).
    @return GraphView compilé.
    """
    graph = aidge_onnx.load_onnx(str(onnx_path))
    graph.set_dataformat(aidge_core.dformat.nchw)
    graph.compile("cpu", aidge_core.dtype.float32, dims=[list(shape)])
    return graph


def import_encoder(encoder, tile):
    """! @brief Encodeur gelé vers GraphView, en passant par backends.export_onnx.

    @param encoder  Encodeur gelé.
    @param tile     Taille de tuile (C, H, W).
    @return GraphView compilé.
    """
    shape = (1, *tile)
    return import_graph(backends.export_onnx(encoder, shape), shape)


def tensor(array):
    """! @brief Tenseur Aidge au format NCHW depuis un tableau flottant.

    @param array  Tableau 1 x C x H x W.
    @return aidge_core.Tensor.
    """
    t = aidge_core.Tensor(np.ascontiguousarray(array, dtype=np.float32))
    t.set_data_format(aidge_core.dformat.nchw)
    return t


def count_nodes(graph):
    """! @brief Nombre de nœuds du graphe par type, triés par type.

    @param graph  GraphView.
    @return Dictionnaire type -> nombre.
    """
    return dict(sorted(collections.Counter(n.type() for n in graph.get_nodes()).items()))


class Runner:
    """! @brief Exécution répétée d'un GraphView : tableau numpy vers tableau numpy."""

    def __init__(self, graph):
        """! @brief Prépare l'ordonnanceur.

        @param graph  GraphView compilé.
        """
        ## Graphe exécuté.
        self.graph = graph
        ## Ordonnanceur séquentiel.
        self.scheduler = aidge_core.SequentialScheduler(graph)
        self.scheduler.generate_scheduling()
        ## Nœud dont la sortie est celle du graphe.
        self.output = graph.get_ordered_outputs()[0].node

    def __call__(self, array):
        """! @brief Exécute le graphe.

        @param array  Tableau 1 x C x H x W.
        @return Copie de la sortie.
        """
        self.scheduler.forward(data=[tensor(array)])
        return np.array(self.output.get_operator().get_output(0)).copy()


def capture_stdout(fn):
    """! @brief Exécute @p fn en capturant ce qu'il écrit sur le descripteur 1.

    Le code C++ d'Aidge écrit directement sur ce descripteur : ni sys.stdout ni contextlib ne le voient. Agit sur tout
    le processus : ne pas appeler depuis un serveur qui a d'autres fils (l'interface passe par un sous-processus).
    @param fn  Fonction sans argument.
    @return Texte écrit par @p fn.
    """
    with tempfile.TemporaryFile("w+b") as buf:
        saved = os.dup(1)
        try:
            os.dup2(buf.fileno(), 1)
            fn()
        finally:
            os.dup2(saved, 1)
            os.close(saved)
        buf.seek(0)
        return buf.read().decode("utf-8", errors="replace")


def parse_quantizability(text):
    """! @brief Rapport de quantifiabilité d'Aidge vers des données structurées.

    @param text  Texte écrit par aidge_quantization.quantizability_report.
    @return Dictionnaire @c quantizable (vrai si tous les opérateurs sont pris en charge) et @c operators (liste de
            dictionnaires @c type, @c count, @c state).
    """
    ops = [{"type": t, "count": int(n), "state": _REPORT_STATES.get(s, s)} for s, t, n in _REPORT_LINE.findall(text)]
    return {"quantizable": bool(ops) and all(o["state"] == "ok" for o in ops), "operators": ops}


def quantizability(graph):
    """! @brief Rapport de quantifiabilité d'un graphe, en données structurées.

    @param graph  GraphView.
    @return Voir parse_quantizability().
    """
    return parse_quantizability(capture_stdout(lambda: aidge_quantization.quantizability_report(graph)))
