"""! @brief Plan mémoire statique d'un graphe : pic d'activations, poids, comparaison à la BRAM du Zynq-7020.

Un graphe mixte exige une inférence avant le plan mémoire, sinon Aidge refuse de planifier son Dequantizer.
"""
import numpy as np

import aidge_core

from .. import config
from .graph import tensor


def weights(graph):
    """! @brief Taille des poids du graphe, sans les échelles de déquantification ajoutées par la recette mixte.

    @param graph  GraphView.
    @return Couple (octets de tous les poids, octets de ceux des convolutions, hors GDN).
    """
    total = conv = 0
    for node in graph.get_nodes():
        if node.type() == "Producer" and "_dq_" not in node.name():
            size = int(np.array(node.get_operator().get_output(0)).nbytes)
            total += size
            conv += size if node.name().startswith("conv") else 0
    return total, conv


def summarize(peak_bytes, weights_bytes, conv_weights_bytes, bram_bytes=None):
    """! @brief Compare un plan mémoire à la BRAM.

    @param peak_bytes          Pic d'activations.
    @param weights_bytes       Tous les poids.
    @param conv_weights_bytes  Poids des convolutions seules : ce qui reste en PL quand la GDN est sur le PS.
    @param bram_bytes          Capacité ; None : config.BRAM_BYTES.
    @return Dictionnaire de la clé @c memory de report.json.
    """
    bram = config.BRAM_BYTES if bram_bytes is None else bram_bytes
    return {"peak_bytes": int(peak_bytes), "weights_bytes": int(weights_bytes),
            "conv_weights_bytes": int(conv_weights_bytes), "bram_bytes": int(bram),
            "bram_ratio": round(peak_bytes / bram, 3), "total_bram_ratio": round((peak_bytes + weights_bytes) / bram, 3),
            "fits_bram": bool(peak_bytes <= bram), "weights_fit_bram": bool(weights_bytes <= bram),
            "total_fits_bram": bool(peak_bytes + weights_bytes <= bram),
            "conv_weights_fit_bram": bool(conv_weights_bytes <= bram)}


def memory_report(graph, sample):
    """! @brief Plan mémoire d'un graphe compilé.

    @param graph   GraphView compilé ; une inférence y est exécutée.
    @param sample  Entrée de l'inférence préalable, 1 x C x H x W.
    @return Voir summarize().
    @throws RuntimeError si Aidge ne sait pas planifier le graphe.
    """
    scheduler = aidge_core.SequentialScheduler(graph)
    scheduler.generate_scheduling()
    scheduler.forward(data=[tensor(sample)])
    peak = aidge_core.generate_optimized_memory_info(scheduler, wrapping=False, display_names=False)[0]
    total, conv = weights(graph)
    return summarize(peak[0], total, conv)
