"""! @brief Calibration et quantification : tentative INT8 sur le graphe entier, recette mixte.

En Aidge 0.10.1, quantize_network() rend la main sans exception sur un graphe qu'il refuse (Sqrt de la GDN) : try_int8()
détecte donc obligatoirement une quantification sans effet. build_mixed() quantifie chaque convolution seule, puis les
replace dans le graphe flottant, ce qui laisse la GDN en flottant. Contrat : docs/conception_optimisation.md (§ 1).
"""
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch

import aidge_core
import aidge_quantization as aq

from .. import backends
from ..images import list_images, load_image, to_tensor
from ..training import ConfigError
from .graph import Runner, conv_names, count_nodes, import_encoder, import_graph, tensor

## Types flottant et entiers de la recette.
_F32, _INT8, _INT32 = aidge_core.dtype.float32, aidge_core.dtype.int8, aidge_core.dtype.int32
## Statuts d'une variante dans report.json.
STATUS_OK, STATUS_NO_EFFECT, STATUS_FAILED = "ok", "sans_effet", "echec"
## Message affiché quand la tentative INT8 n'a rien changé.
NO_EFFECT_REASON = ("Aidge ne sait pas quantifier Sqrt, la racine de la GDN : quantize_network n'a rien changé. "
                    "La recette mixte garde la GDN en flottant et quantifie les convolutions.")


def calibration_tiles(folder, n, tile, seed):
    """! @brief Tire des tuiles de calibration dans un jeu d'images, de façon reproductible.

    @param folder  Dossier d'images.
    @param n       Nombre de tuiles.
    @param tile    Taille de tuile (C, H, W).
    @param seed    Graine du tirage.
    @return Tableau float32 n x C x H x W.
    @throws ConfigError si le dossier ne contient aucune image assez grande.
    """
    channels, height, width = tile
    paths = list_images(folder)
    if not paths:
        raise ConfigError(f"aucune image dans le jeu de calibration : {folder}")
    rng = np.random.default_rng(seed)
    tiles = []
    for _ in range(n * 20):
        if len(tiles) == n:
            break
        path = paths[int(rng.integers(len(paths)))]
        img = to_tensor(load_image(path, "gris" if channels == 1 else "rgb"), channels=channels)[0].numpy()
        if img.shape[1] < height or img.shape[2] < width:
            continue
        y, x = int(rng.integers(img.shape[1] - height + 1)), int(rng.integers(img.shape[2] - width + 1))
        tiles.append(img[:, y:y + height, x:x + width])
    if len(tiles) < n:
        raise ConfigError(f"images trop petites pour des tuiles {height} x {width} dans : {folder}")
    return np.stack(tiles).astype(np.float32)


def outputs_identical(a, b):
    """! @brief Vrai si deux sorties sont identiques au bit près."""
    return bool(np.array_equal(a, b))


def no_effect_signals(check_ok, nodes_before, nodes_after, same_outputs):
    """! @brief Signaux d'une quantification sans effet ; un seul suffit (fonction pure).

    @param check_ok      Résultat de check_architecture.
    @param nodes_before  Compte des nœuds avant.
    @param nodes_after   Compte des nœuds après.
    @param same_outputs  Sorties identiques au bit près.
    @return Dictionnaire des trois signaux, et @c no_effect.
    """
    signals = {"check_architecture": bool(check_ok), "nodes_unchanged": nodes_before == nodes_after,
               "outputs_identical": bool(same_outputs)}
    signals["no_effect"] = (not signals["check_architecture"]) or signals["nodes_unchanged"] or signals["outputs_identical"]
    return signals


def try_int8(encoder, tile, calibration, clipping="MAX", per_channel=False):
    """! @brief Tente la PTQ du graphe entier et détecte une quantification sans effet.

    @param encoder      Encodeur gelé.
    @param tile         Taille de tuile.
    @param calibration  Tuiles de calibration, n x C x H x W.
    @param clipping     Nom d'un mode de aidge_quantization.Clipping.
    @param per_channel  Quantification par canal.
    @return Dictionnaire @c status, @c reason, @c signals, @c nodes_before, @c nodes_after et @c graph (None sauf succès).
    """
    graph = import_encoder(encoder, tile)
    sample = calibration[:1]
    before = count_nodes(graph)
    reference = Runner(graph)(sample)
    check_ok = aq.check_architecture(graph)
    result = {"status": STATUS_FAILED, "reason": None, "signals": None, "nodes_before": before, "nodes_after": before,
              "graph": None}
    try:
        aq.auto_assign_node_precision(graph, _INT8, _INT8, _INT32)
        aq.quantize_network(graph, calibration_set=[tensor(t[None]) for t in calibration],
                            clipping_mode=getattr(aq.Clipping, clipping), channel_wise=per_channel)
        result["nodes_after"] = count_nodes(graph)
        output = Runner(graph)(sample)
    except Exception as e:      # Aidge lève des RuntimeError et des IndexError selon le point de rupture
        result["reason"] = f"quantification impossible : {e}"
        return result
    signals = no_effect_signals(check_ok, before, result["nodes_after"], outputs_identical(reference, output))
    result["signals"] = signals
    if signals["no_effect"]:
        result["status"], result["reason"] = STATUS_NO_EFFECT, NO_EFFECT_REASON
    else:
        result["status"], result["graph"] = STATUS_OK, graph
    return result


def _conv_inputs(encoder, names, calibration):
    """! @brief Activations à l'entrée de chaque convolution, sur les tuiles de calibration.

    @param encoder      Encodeur gelé.
    @param names        Noms des convolutions.
    @param calibration  Tuiles n x C x H x W.
    @return Dictionnaire nom -> liste de tableaux 1 x C x H x W.
    """
    inputs = {name: [] for name in names}
    hooks = [getattr(encoder, name).register_forward_hook(
        lambda _m, i, _o, name=name: inputs[name].append(i[0].numpy().copy())) for name in names]
    try:
        with torch.no_grad(), backends.limited_threads():
            for t in calibration:
                encoder(torch.from_numpy(t[None]))
    finally:
        for hook in hooks:
            hook.remove()
    return inputs


def _dequantizer_scale(node):
    """! @brief Échelle d'un Dequantizer : le Producer du Mul de son micro-graphe.

    @param node  Nœud Dequantizer.
    @return Tableau de l'échelle.
    @throws RuntimeError si le micro-graphe n'a pas la forme attendue.
    """
    for inner in node.get_operator().get_micro_graph().get_nodes():
        if inner.type() == "Mul":
            for parent in inner.get_parents():
                if parent is not None and parent.type() == "Producer":
                    return np.array(parent.get_operator().get_output(0)).copy()
    raise RuntimeError(f"Dequantizer sans échelle : {node.name()}")


def _explicit_dequantizer(segment, prefix):
    """! @brief Remplace le Dequantizer d'un segment par Cast(float32) puis Mul(échelle).

    Avec un Dequantizer, l'export plante dans constant_folding (« Select input out of range »).
    @param segment  GraphView du segment quantifié.
    @param prefix   Préfixe des nouveaux nœuds.
    @throws RuntimeError si le remplacement échoue.
    """
    node = next(n for n in segment.get_nodes() if n.type() == "Dequantizer")
    scale = np.asarray(_dequantizer_scale(node), dtype=np.float32)
    cast = aidge_core.Cast(_F32, name=f"{prefix}_dq_cast")
    mul = aidge_core.Mul(name=f"{prefix}_dq_mul")
    producer = aidge_core.Producer(aidge_core.Tensor(scale), f"{prefix}_dq_scale", True)
    cast.add_child(mul, 0, 0)
    producer.add_child(mul, 0, 1)
    new, old = aidge_core.GraphView(), aidge_core.GraphView()
    for n in (cast, mul, producer):
        new.add(n, False)
    old.add(node, False)
    if not aidge_core.GraphView.replace(old, new):
        raise RuntimeError(f"remplacement du Dequantizer impossible : {prefix}")
    for n in (cast, mul, producer):
        n.get_operator().set_backend("cpu")
    cast.get_operator().set_datatype(_F32)
    mul.get_operator().set_datatype(_F32)


def _quantize_segment(module, name, inputs, clipping, per_channel, folder):
    """! @brief Quantifie une convolution seule.

    @param module       Convolution PyTorch.
    @param name         Nom de la convolution, préfixe des nœuds du segment.
    @param inputs       Activations d'entrée, liste de tableaux 1 x C x H x W.
    @param clipping     Nom d'un mode de aidge_quantization.Clipping.
    @param per_channel  Quantification par canal.
    @param folder       Dossier temporaire pour le fichier ONNX.
    @return GraphView du segment, sans Dequantizer, aux nœuds préfixés.
    @throws RuntimeError si la quantification n'a rien changé.
    """
    path = Path(folder) / f"{name}.onnx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(module.eval(), (torch.zeros(inputs[0].shape),), str(path), input_names=["image"],
                          output_names=["latent"], opset_version=backends.ONNX_OPSET, do_constant_folding=True, dynamo=False)
    segment = import_graph(path, inputs[0].shape)
    aidge_core.SequentialScheduler(segment).generate_scheduling()
    before = count_nodes(segment)
    aq.auto_assign_node_precision(segment, _INT8, _INT8, _INT32)
    aq.quantize_network(segment, calibration_set=[aidge_core.Tensor(t) for t in inputs],
                        clipping_mode=getattr(aq.Clipping, clipping), channel_wise=per_channel, equalize=False,
                        rescale_inputs=True, dequantize=True)
    if count_nodes(segment) == before:
        raise RuntimeError(f"quantification sans effet sur la convolution {name}")
    _explicit_dequantizer(segment, name)
    for node in segment.get_nodes():     # les segments portent les mêmes noms : sans préfixe, l'export plante
        if not node.name().startswith(f"{name}_dq_"):
            node.set_name(f"{name}_{node.name()}")
    return segment


def build_mixed(encoder, tile, calibration, clipping="MAX", per_channel=False):
    """! @brief Graphe mixte : convolutions en INT8, GDN en flottant.

    Reconstruction déterministe depuis l'encodeur et les tuiles de calibration : c'est ainsi qu'une variante persiste,
    car load_model_folder perd le type des tenseurs d'un graphe quantifié.
    @param encoder      Encodeur gelé.
    @param tile         Taille de tuile.
    @param calibration  Tuiles de calibration, n x C x H x W.
    @param clipping     Nom d'un mode de aidge_quantization.Clipping.
    @param per_channel  Quantification par canal.
    @return GraphView mixte, prêt à exécuter. Un nouvel appel en est nécessaire pour chaque export, qui le modifie.
    @throws RuntimeError si une étape échoue.
    """
    names = conv_names(encoder)
    inputs = _conv_inputs(encoder, names, calibration)
    graph = import_encoder(encoder, tile)
    with tempfile.TemporaryDirectory() as folder:
        for name in names:
            node = next(n for n in graph.get_nodes() if n.name() == f"{name}_Conv")
            old = aidge_core.GraphView()
            old.add(node)
            for parent in node.get_parents():
                if parent is not None and parent.type() == "Producer":
                    old.add(parent)
            segment = _quantize_segment(getattr(encoder, name), name, inputs[name], clipping, per_channel, folder)
            if not aidge_core.GraphView.replace(old, segment):
                raise RuntimeError(f"remplacement de la convolution impossible : {name}")
    graph.forward_dims([[1, *tile]], True)
    return graph
