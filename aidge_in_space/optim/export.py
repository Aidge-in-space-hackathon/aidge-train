"""! @brief Export C++ d'Aidge : génération, compilation si g++ est là, vérification, taille, zip.

Un binaire peut compiler et tourner en étant faux : la vérification face au graphe Aidge de la même recette est donc
obligatoire et bloquante (EXPORT_MIN_IDENTICAL_PCT). Compilation directe par g++, sans CMake, seul g++ existant sur le
poste Windows. Contrat : docs/conception_optimisation.md (§ 1.5 et § 6).
"""
import re
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np

import aidge_core.export_utils as export_utils

from .evaluate import symbols_identical_pct
from .graph import tensor

## En dessous de ce taux de symboles identiques au graphe Aidge, la variante passe en échec.
EXPORT_MIN_IDENTICAL_PCT = 99.0
## Commande de compilation, sans les sources.
GXX_FLAGS = ["-O2", "-std=c++17", "-DSAVE_OUTPUTS", "-Idnn/include", "-I."]
## Dossier de vérification dans l'export : hors du zip.
VERIF_DIR = "verif"
## Délai maximal d'une compilation, en secondes.
COMPILE_TIMEOUT_S = 3600
## Délai maximal d'une exécution du binaire, en secondes.
RUN_TIMEOUT_S = 600
## Période de scrutation de la demande d'arrêt pendant la compilation, en secondes.
POLL_S = 0.5
## Nombres du format de SAVE_OUTPUTS.
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|nan|-?inf")

## Programme de vérification : une tuile en entrée, la sortie du modèle en sortie.
VERIF_MAIN = """#include <cstdio>
#include "forward.hpp"
int main(int argc, char **argv) {{
    static {in_type} input[{n_in}];
    {out_type} *output = nullptr;
    FILE *f = std::fopen(argv[1], "rb");
    if (!f || std::fread(input, sizeof(input[0]), {n_in}, f) != {n_in}) return 2;
    std::fclose(f);
    model_forward(input, &output);
    f = std::fopen(argv[2], "wb");
    if (!f || std::fwrite(output, sizeof(output[0]), {n_out}, f) != {n_out}) return 3;
    std::fclose(f);
    return 0;
}}
"""


class CompileStopped(Exception):
    """! @brief Compilation interrompue par une demande d'arrêt."""


def find_compiler():
    """! @brief Chemin de g++, ou None s'il est absent."""
    return shutil.which("g++")


def compiler_version(compiler):
    """! @brief Première ligne de @c g++ --version, ou chaîne vide."""
    try:
        out = subprocess.run([compiler, "--version"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.splitlines()[0] if out else ""


def generate(graph, folder, sample):
    """! @brief Génère l'export C++ d'un graphe.

    Le format NCHW en entrée et en sortie, et @c dtype="float32" même pour un graphe mixte, dont les types sont déjà
    fixés : les autres réglages produisent un binaire faux ou plantent (§ 1.5). L'export modifie le graphe.
    @param graph   GraphView, à ne plus utiliser ensuite.
    @param folder  Dossier de l'export, recréé.
    @param sample  Tuile d'exemple, 1 x C x H x W.
    @return Manifeste de l'export (@c inputs, @c outputs).
    """
    folder = Path(folder)
    shutil.rmtree(folder, ignore_errors=True)
    folder.parent.mkdir(parents=True, exist_ok=True)
    cfg = export_utils.config.ExportConfig(target="host", export_folder=str(folder), compile=False, cuda=False,
                                           dtype="float32", legacy=False, dformat="nchw")
    artifact = export_utils.export(graph=graph, config=cfg, input_tensors=[tensor(sample)], labels=None, nb_test=0)
    return artifact.manifest


def compile_export(folder, manifest, compiler, should_stop=None):
    """! @brief Compile l'export avec le programme de vérification, sans CMake.

    @param folder       Dossier de l'export.
    @param manifest     Manifeste de generate().
    @param compiler     Chemin de g++.
    @param should_stop  Fonction scrutée pendant la compilation ; une valeur vraie l'interrompt.
    @return Couple (chemin du binaire ou None, fin de stderr en cas d'échec).
    @throws CompileStopped si l'arrêt est demandé.
    """
    folder = Path(folder)
    verif = folder / VERIF_DIR
    verif.mkdir(parents=True, exist_ok=True)
    i, o = manifest.inputs[0], manifest.outputs[0]
    n_in, n_out = int(np.prod(i["dims"])), int(np.prod(o["dims"]))
    main = verif / "verif_main.cpp"
    main.write_text(VERIF_MAIN.format(in_type=i["dtype"], out_type=o["dtype"], n_in=n_in, n_out=n_out), encoding="utf-8")
    sources = [str(p.relative_to(folder)) for p in sorted(folder.rglob("*.cpp"))
               if p.name != "main.cpp" and p.parts[len(folder.parts)] != "data" and p != main]
    binary = verif / "verif"
    cmd = [compiler, *GXX_FLAGS, "-o", str(binary.relative_to(folder)), str(main.relative_to(folder)), *sources]
    proc = subprocess.Popen(cmd, cwd=folder, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    deadline = time.monotonic() + COMPILE_TIMEOUT_S
    while proc.poll() is None:
        if (should_stop is not None and should_stop()) or time.monotonic() > deadline:
            proc.kill()
            proc.wait()
            if time.monotonic() > deadline:
                return None, f"compilation interrompue après {COMPILE_TIMEOUT_S} s"
            raise CompileStopped()
        time.sleep(POLL_S)
    if proc.returncode != 0:
        return None, proc.stderr.read()[-2000:]
    return (binary if binary.exists() else binary.with_suffix(".exe")), ""


def run_binary(binary, tile_array, shape):
    """! @brief Exécute le binaire de vérification sur une tuile.

    @param binary      Binaire de compile_export().
    @param tile_array  Tuile 1 x C x H x W.
    @param shape       Forme de la sortie.
    @return Sortie, dans @p shape.
    @throws RuntimeError si l'exécution échoue.
    """
    run_dir = Path(binary).parent
    (run_dir / "data" / "export_outputs").mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(tile_array, dtype=np.float32).tofile(run_dir / "in.bin")
    r = subprocess.run([str(Path(binary).resolve()), "in.bin", "out.bin"], cwd=run_dir, capture_output=True,
                       timeout=RUN_TIMEOUT_S)
    if r.returncode != 0:
        raise RuntimeError(f"le binaire a échoué (code {r.returncode})")
    return np.fromfile(run_dir / "out.bin", dtype=np.float32).reshape(shape)


def verify(binary, tiles, aidge_fn, pytorch_fn):
    """! @brief Compare le binaire au graphe Aidge de la même recette et, pour information, à PyTorch.

    @param binary      Binaire de compile_export().
    @param tiles       Tuiles de contrôle, n x C x H x W.
    @param aidge_fn    Fonction tuile vers latents, par le graphe Aidge.
    @param pytorch_fn  Fonction tuile vers latents, par PyTorch.
    @return Dictionnaire @c symbols_identical_pct (face à Aidge), @c symbols_identical_vs_pytorch_pct et @c ok (seuil).
    """
    vs_aidge, vs_torch = [], []
    for t in tiles:
        ref = aidge_fn(t[None])
        out = run_binary(binary, t[None], ref.shape)
        vs_aidge.append(symbols_identical_pct(ref, out))
        vs_torch.append(symbols_identical_pct(pytorch_fn(t[None]), out))
    worst = min(vs_aidge)
    return {"symbols_identical_pct": round(worst, 3), "symbols_identical_vs_pytorch_pct": round(float(np.mean(vs_torch)), 3),
            "ok": worst >= EXPORT_MIN_IDENTICAL_PCT}


def binary_sizes(binary):
    """! @brief Taille des sections d'un binaire.

    @param binary  Binaire.
    @return Dictionnaire @c text_bytes et @c bss_bytes ; valeurs None si l'outil @c size est absent.
    """
    tool = shutil.which("size")
    if tool:
        try:
            last = subprocess.run([tool, str(binary)], capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[-1].split()
            return {"text_bytes": int(last[0]), "bss_bytes": int(last[2])}
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            pass
    return {"text_bytes": None, "bss_bytes": None}


def make_zip(folder, zip_path):
    """! @brief Archive l'export, sans le dossier de vérification.

    @param folder    Dossier de l'export.
    @param zip_path  Archive à écrire.
    @return Chemin de l'archive.
    """
    folder, zip_path = Path(folder), Path(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(folder.rglob("*")):
            if path.is_file() and VERIF_DIR not in path.relative_to(folder).parts:
                z.write(path, path.relative_to(folder).as_posix())
    return zip_path


def read_saved_output(path, dims, nhwc):
    """! @brief Relit la sortie d'une couche écrite par @c -DSAVE_OUTPUTS, en NCHW (fonction pure sur le fichier).

    @param path  Fichier texte.
    @param dims  Dimensions (C, H, W).
    @param nhwc  Vrai si le fichier suit l'ordre H, W, C.
    @return Tableau C x H x W.
    @throws ValueError si le nombre de valeurs ne correspond pas.
    """
    values = np.array([float(v) for v in _NUMBER.findall(Path(path).read_text(encoding="utf-8", errors="replace"))])
    c, h, w = dims
    if values.size != c * h * w:
        raise ValueError(f"{path} : {values.size} valeurs pour {c} x {h} x {w}")
    return values.reshape(h, w, c).transpose(2, 0, 1) if nhwc else values.reshape(c, h, w)


def _layer_dims(folder, layer):
    """! @brief Dimensions et format d'une couche, lus dans son en-tête généré.

    @param folder  Dossier de l'export.
    @param layer   Nom de la couche, par exemple @c _7_PadConv_0.
    @return Couple ((C, H, W), vrai si NHWC).
    """
    text = (Path(folder) / "dnn" / "include" / "layers" / f"{layer}.h").read_text(encoding="utf-8")

    def value(key):
        """! @brief Valeur entière d'une macro de l'en-tête."""
        return int(re.search(rf"_OUTPUT_0_{key}\s+(\d+)", text).group(1))
    fmt = re.search(r"_OUTPUT_0_DEV_FMT\s+\S*::(\w+)", text)
    return (value("NB_OUTPUTS"), value("OUT_HEIGHT"), value("OUT_WIDTH")), bool(fmt and fmt.group(1) == "NHWC")


def conv_layer_outputs(binary, folder):
    """! @brief Sorties int8 des convolutions et sorties des GDN du dernier passage du binaire, couche par couche.

    Dans l'ordre d'exécution, la sortie int8 d'une convolution est le premier Cast après le Clip qui la suit.
    @param binary  Binaire de compile_export(), déjà exécuté.
    @param folder  Dossier de l'export.
    @return Dictionnaire nom (@c conv1, @c gdn1...) vers tableau C x H x W.
    """
    saved = Path(binary).parent / "data" / "export_outputs"
    names = sorted((p.name[:-len("_output_0.txt")] for p in saved.glob("*_output_0.txt")),
                   key=lambda s: int(s.split("_")[1]))
    out, k, armed = {}, 0, False
    gdn = 0
    for name in names:
        kind = name.split("_")[2]
        if kind == "PadConv":
            k, armed = k + 1, False
        elif kind == "Clip" and k:
            armed = True
        elif kind == "Cast" and armed:
            dims, nhwc = _layer_dims(folder, name)
            out[f"conv{k}"] = read_saved_output(saved / f"{name}_output_0.txt", dims, nhwc)
            armed = False
        elif kind == "Div":
            gdn += 1
            dims, nhwc = _layer_dims(folder, name)
            out[f"gdn{gdn}"] = read_saved_output(saved / f"{name}_output_0.txt", dims, nhwc)
    return out
