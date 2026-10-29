"""! @brief Exécution de l'encodeur embarqué (transformée d'analyse g_a) sur plusieurs moteurs.

- @c torch       : référence PyTorch ;
- @c onnxruntime : le modèle ONNX qui sert d'entrée à Aidge ;
- @c aidge       : backend CPU d'Aidge, première étape vers le code C/C++ généré.

Les mesures de temps se font sur THREADS thread(s) pour rester comparables entre moteurs (et plus
proches d'un CPU embarqué) ; elles restent indicatives tant qu'elles sont prises sur PC.

Les modèles ONNX sont exportés une fois par (poids, taille d'entrée) dans outputs/onnx, et les
moteurs préparés (session, graphe compilé) sont gardés dans un cache LRU de MAX_RUNNERS entrées.
"""
import contextlib
import hashlib
import os
import warnings
from collections import OrderedDict

import numpy as np
import torch

from .config import ONNX_DIR

## Moteurs connus : identifiant -> nom affiché.
BACKENDS = {"torch": "PyTorch", "onnxruntime": "ONNX Runtime", "aidge": "Aidge (CPU)"}
## Nombre de threads de calcul utilisés pour toutes les mesures.
THREADS = 1
## Version d'opset des exports ONNX (prise en charge par aidge_onnx 0.10).
ONNX_OPSET = 13
## Nombre de moteurs préparés gardés en mémoire (chacun retient un modèle, une session ou un graphe).
MAX_RUNNERS = 8
## Nombre de fichiers ONNX gardés dans outputs/onnx ; les plus anciens sont supprimés.
MAX_ONNX_FILES = 32

## Cache LRU des exécuteurs : (moteur, empreinte des poids, forme) -> fonction numpy -> numpy.
_runners = OrderedDict()


def available(key):
    """! @brief Indique si un moteur est utilisable (bibliothèques importables).

    @param key  Identifiant de BACKENDS.
    @return Couple (disponible, message d'erreur ou chaîne vide).
    """
    try:
        if key == "onnxruntime":
            import onnxruntime  # noqa: F401
        elif key == "aidge":
            import aidge_core  # noqa: F401  (en premier : charge la DLL dont dépend le backend)
            import aidge_backend_cpu  # noqa: F401
            import aidge_onnx  # noqa: F401
        elif key != "torch":
            return False, f"moteur inconnu : {key}"
    except ImportError as e:
        return False, str(e)
    return True, ""


def available_backends():
    """! @brief Liste des identifiants de moteurs utilisables, dans l'ordre de BACKENDS."""
    return [k for k in BACKENDS if available(k)[0]]


@contextlib.contextmanager
def limited_threads():
    """! @brief Contexte dans lequel PyTorch calcule sur THREADS thread(s).

    Utilisé pour tout ce qui est chronométré (encodeur, hyperprior, décodage), afin que les temps
    restent comparables d'un codec et d'un moteur à l'autre.
    """
    old = torch.get_num_threads()
    torch.set_num_threads(THREADS)
    try:
        yield
    finally:
        torch.set_num_threads(old)


def fingerprint(module):
    """! @brief Empreinte des poids d'un module (SHA-1 tronqué), mise en cache sur le module.

    Un modèle ré-entraîné sous le même nom donne une autre empreinte, donc un autre fichier ONNX.
    @param module  torch.nn.Module.
    @return Chaîne hexadécimale de 12 caractères.
    """
    fp = getattr(module, "_ais_fingerprint", None)
    if fp is None:
        h = hashlib.sha1()
        for k, v in module.state_dict().items():
            h.update(k.encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
        fp = h.hexdigest()[:12]
        module._ais_fingerprint = fp
    return fp


def export_onnx(module, shape):
    """! @brief Exporte un module en ONNX pour une taille d'entrée fixe, ou réutilise l'export existant.

    L'export est écrit dans un fichier temporaire puis renommé : un export interrompu ne laisse
    jamais de fichier tronqué à l'emplacement final.
    @param module  Module aux GDN déjà gelées (aidge_in_space.nn.freeze()).
    @param shape   Forme de l'entrée, (1, C, H, W).
    @return Chemin du fichier ONNX (entrée @c "image", sortie @c "latent").
    """
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    path = ONNX_DIR / f"enc_{fingerprint(module)}_{'x'.join(map(str, shape[1:]))}.onnx"
    if path.exists():
        path.touch()                                   # récemment utilisé : protégé du nettoyage
        return path
    tmp = path.with_suffix(".onnx.tmp")
    kw = dict(input_names=["image"], output_names=["latent"], opset_version=ONNX_OPSET,
              do_constant_folding=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(module.eval(), (torch.zeros(shape),), str(tmp), dynamo=False, **kw)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    _prune_onnx(keep=path)
    return path


def _prune_onnx(keep):
    """! @brief Supprime les exports ONNX les moins récemment utilisés au-delà de MAX_ONNX_FILES.

    @param keep  Fichier à ne jamais supprimer (celui qui vient d'être écrit).
    """
    files = sorted(ONNX_DIR.glob("enc_*.onnx"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in files[MAX_ONNX_FILES:]:
        if p != keep:
            p.unlink(missing_ok=True)


def _torch_runner(module):
    """! @brief Exécuteur PyTorch du module, sur THREADS thread(s)."""
    module = module.eval()

    def run(x):
        with torch.no_grad(), limited_threads():
            return module(torch.from_numpy(x)).numpy()
    return run


def _onnxruntime_runner(path):
    """! @brief Exécuteur ONNX Runtime (CPU, THREADS thread(s)) d'un fichier ONNX."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = THREADS
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    return lambda x: sess.run(None, {"image": x})[0]


def _aidge_runner(path, shape):
    """! @brief Exécuteur Aidge (backend CPU, ordonnanceur séquentiel) d'un fichier ONNX.

    @param path   Fichier ONNX.
    @param shape  Forme de l'entrée, nécessaire à la compilation du graphe.
    """
    import aidge_core
    import aidge_backend_cpu  # noqa: F401  (enregistre les noyaux CPU)
    import aidge_onnx
    graph = aidge_onnx.load_onnx(str(path))
    graph.compile("cpu", aidge_core.dtype.float32, dims=[list(shape)])
    scheduler = aidge_core.SequentialScheduler(graph)
    out_node = graph.get_ordered_outputs()[0][0]

    def run(x):
        scheduler.forward(data=[aidge_core.Tensor(np.ascontiguousarray(x, dtype=np.float32))])
        return np.array(out_node.get_operator().get_output(0)).copy()
    return run


def get_runner(backend, module, shape):
    """! @brief Fonction numpy -> numpy qui exécute @p module sur @p backend.

    L'exécuteur est préparé une fois par (moteur, poids, taille) puis gardé dans un cache LRU de
    MAX_RUNNERS entrées.
    @param backend  Identifiant de BACKENDS.
    @param module   Module aux GDN gelées.
    @param shape    Forme de l'entrée, (1, C, H, W).
    @return Fonction qui prend un tableau float32 de forme @p shape et renvoie la sortie du module.
    @throws RuntimeError si le moteur est indisponible.
    """
    ok, why = available(backend)
    if not ok:
        raise RuntimeError(f"moteur {backend} indisponible : {why}")
    shape = tuple(int(s) for s in shape)
    key = (backend, fingerprint(module), shape)
    if key not in _runners:
        if backend == "torch":
            _runners[key] = _torch_runner(module)
        elif backend == "onnxruntime":
            _runners[key] = _onnxruntime_runner(export_onnx(module, shape))
        else:
            _runners[key] = _aidge_runner(export_onnx(module, shape), shape)
        while len(_runners) > MAX_RUNNERS:
            _runners.popitem(last=False)
    _runners.move_to_end(key)
    return _runners[key]


def clear_runners():
    """! @brief Vide le cache des exécuteurs (libère modèles, sessions et graphes)."""
    _runners.clear()
