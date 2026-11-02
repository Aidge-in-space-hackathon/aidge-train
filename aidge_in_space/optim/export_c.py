"""! @brief Générateur C pur pour LoCod : du GraphView Aidge quantifié vers un fichier @c main.c unique.

Export Aidge réutilisable, pensé pour pouvoir rejoindre le framework : le module n'importe que numpy, aidge_core et
aidge_quantization, rien de ce dépôt, et il suit la forme de aidge_export_cpp. Tout vient du graphe **après** quantification
(poids, échelles, attributs, ordre des couches), jamais de PyTorch. Un registre (EMITTERS) associe chaque type de nœud à
sa traduction ; un nœud inconnu lève ExportCError au lieu d'être ignoré.

Le fichier produit répond aux contraintes de LoCod : un seul fichier sans inclusion locale, en C, poids compris.
- Les convolutions INT8 sont des accélérateurs : deux pointeurs de structures de taille fixe, aucun pointeur interne,
  poids statiques (donc en BRAM avec la politique LSS), Quantizer de sortie en virgule fixe, sans aucun flottant.
- Tout le reste s'exécute sur le CPU, dont la GDN flottante : LOCOD_FPGA retire ce code de ce que voit Bambu.
Contrat : docs/conception_optimisation.md (§ 10).
"""
import math
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import aidge_quantization

## Macro définie par LoCod quand le fichier passe dans Bambu.
FPGA_MACRO = "LOCOD_FPGA"
## Macro de compilation native : les accélérateurs deviennent des appels directs, et la sortie de chaque couche peut être écrite.
NATIVE_MACRO = "AIDGE_C_NATIVE"
## Bits du multiplicateur de la virgule fixe.
FIXED_POINT_BITS = 31
## Options de la compilation native.
GCC_FLAGS = ["-O2", "-std=c11", f"-D{NATIVE_MACRO}"]
## Délai maximal de la compilation native, en secondes.
COMPILE_TIMEOUT_S = 1800
## Délai maximal d'une exécution native, en secondes.
RUN_TIMEOUT_S = 600


class ExportCError(ValueError):
    """! @brief Graphe que le générateur ne sait pas traduire : motif inconnu ou attribut non pris en charge."""


@dataclass
class Buffer:
    """! @brief Tableau C d'une sortie de nœud, aux dimensions NCHW sans le lot."""

    ## Nom de la variable C.
    name: str
    ## Dimensions (C, H, W).
    dims: tuple
    ## Type C de l'élément.
    ctype: str = "float"


@dataclass
class ConvLayer:
    """! @brief Convolution INT8 et son Quantizer de sortie : un accélérateur."""

    ## Nom de la fonction C.
    name: str
    ## Poids int8, O x I x KH x KW.
    weights: np.ndarray
    ## Biais int32, de taille O.
    biases: np.ndarray
    ## Tampon d'entrée (int8).
    input: Buffer
    ## Tampon de sortie (int8).
    output: Buffer
    ## Pas de la convolution.
    stride: int
    ## Padding constant nul, identique sur les quatre côtés.
    pad: int
    ## Multiplicateur de la virgule fixe.
    multiplier: int
    ## Décalage de la virgule fixe.
    shift: int
    ## Bornes de saturation.
    clip: tuple = (-128, 127)


@dataclass
class CpuOp:
    """! @brief Opération exécutée sur le CPU : du code C déjà écrit, et le tampon qu'elle produit."""

    ## Tampon de sortie.
    output: Buffer
    ## Corps en C, indenté par l'appelant.
    code: str
    ## Déclarations statiques nécessaires (données de poids).
    data: str = ""


@dataclass
class LocodExport:
    """! @brief Résultat de export_locod()."""

    ## Fichier C.
    path: Path
    ## Nombre d'accélérateurs.
    accelerators: int
    ## Octets de poids et de biais gardés en PL.
    pl_weights_bytes: int
    ## Octets échangés entre PS et PL par tuile.
    transfers_per_tile_bytes: int
    ## Noms des couches écrites par le banc natif, dans l'ordre : (nom, type C, dimensions).
    layers: list = field(default_factory=list)


def fixed_point(scale, bits=FIXED_POINT_BITS):
    """! @brief Échelle réelle vers virgule fixe : @c scale ≈ multiplier / 2^shift (fonction pure).

    @param scale  Échelle strictement positive.
    @param bits   Bits du multiplicateur.
    @return Couple (multiplier, shift), avec multiplier < 2^bits et shift >= 1.
    @throws ExportCError si l'échelle est invalide ou trop grande.
    """
    if not (math.isfinite(scale) and scale > 0):
        raise ExportCError(f"échelle invalide : {scale}")
    mantissa, exponent = math.frexp(scale)
    multiplier = round(mantissa * (1 << bits))
    shift = bits - exponent
    if multiplier == (1 << bits):
        multiplier //= 2
        shift -= 1
    if shift < 1:
        raise ExportCError(f"échelle trop grande pour la virgule fixe : {scale}")
    return multiplier, shift


def c_array(values, fmt="{}"):
    """! @brief Initialiseur C complet d'un tableau numpy : LoCod exige des tableaux entièrement initialisés.

    @param values  Tableau à au moins une dimension.
    @param fmt     Format de chaque valeur.
    @return Texte de l'initialiseur.
    """
    if values.ndim == 1:
        return "{" + ",".join(fmt.format(v) for v in values.tolist()) + "}"
    return "{" + ",\n".join(c_array(v, fmt) for v in values) + "}"


def _dims(node):
    """! @brief Dimensions (C, H, W) de la sortie d'un nœud, le lot retiré.

    @throws ExportCError si la sortie n'a pas la forme 1 x C x H x W.
    """
    dims = list(node.get_operator().get_output(0).dims)
    if len(dims) != 4 or dims[0] != 1:
        raise ExportCError(f"{node.name()} : sortie 1 x C x H x W attendue (reçu {dims})")
    return tuple(dims[1:])


def _symbol(name):
    """! @brief Nom Aidge vers identifiant C."""
    return "".join(c if c.isalnum() else "_" for c in name)


def _producer(node, input_index):
    """! @brief Valeur du Producer branché sur une entrée d'un nœud, ou None."""
    parents = node.get_parents()
    parent = parents[input_index] if input_index < len(parents) else None
    if parent is None or parent.type() != "Producer":
        return None
    return np.array(parent.get_operator().get_output(0)).copy()


def _quantizer_params(node):
    """! @brief Échelle, décalage et bornes d'un nœud Quantizer.

    @param node  Nœud Quantizer.
    @return Triplet (échelle, bornes (min, max)).
    @throws ExportCError si le décalage n'est pas nul.
    """
    scale = float(aidge_quantization.get_scaling_factor(node))
    low, high = -128.0, 127.0
    for inner in node.get_operator().get_micro_graph().get_nodes():
        if inner.type() == "Add":
            zero = _producer(inner, 1)
            zero = _producer(inner, 0) if zero is None else zero
            if zero is not None and float(zero) != 0:
                raise ExportCError(f"{node.name()} : décalage non nul non pris en charge ({float(zero)})")
        elif inner.type() == "Clip":
            low, high = float(_producer(inner, 1)), float(_producer(inner, 2))
    return scale, (int(low), int(high))


def _conv_attributes(node):
    """! @brief Pas et padding d'une convolution, atomique ou PaddedConv2D.

    @param node  Nœud de convolution.
    @return Couple (pas, padding).
    @throws ExportCError si les attributs ne sont pas pris en charge (dilatation, non-carré, padding non nul ou asymétrique).
    """
    op = node.get_operator()
    inner = list(op.get_micro_graph().get_nodes()) if not op.is_atomic() else [node]
    conv = next(n for n in inner if n.type() == "Conv2D")
    attrs = conv.get_operator().attr.dict()
    stride, dilation = attrs["stride_dims"], attrs["dilation_dims"]
    if stride[0] != stride[1] or list(dilation) != [1, 1]:
        raise ExportCError(f"{node.name()} : pas carré sans dilatation attendu (pas {stride}, dilatation {dilation})")
    pad = 0
    for n in inner:
        if n.type() == "Pad":
            pa = n.get_operator().attr.dict()
            pads = list(pa["pads"])
            if len(set(pads)) != 1 or float(np.array(pa["constant_value"])) != 0:
                raise ExportCError(f"{node.name()} : padding constant nul et symétrique attendu ({pads})")
            pad = int(pads[0])
    return int(stride[0]), pad


class _Translator:
    """! @brief Parcourt le graphe dans l'ordre d'exécution et construit la liste des couches."""

    def __init__(self, graph):
        """! @brief Prépare le parcours.

        @param graph  GraphView quantifié, dimensions propagées.
        """
        ## Graphe parcouru.
        self.graph = graph
        ## Tampon de sortie de chaque nœud déjà traduit, par nom de nœud.
        self.buffers = {}
        ## Couches dans l'ordre : ConvLayer ou CpuOp.
        self.layers = []
        ## Tampon d'entrée du modèle.
        self.model_input = None
        ## Quantizer d'entrée en attente de sa convolution.
        self.pending_conv = None

    def source(self, node, index=0):
        """! @brief Tampon alimentant une entrée d'un nœud (le tampon du modèle si l'entrée est libre)."""
        parents = node.get_parents()
        parent = parents[index] if index < len(parents) else None
        if parent is None:
            if self.model_input is None:
                op = node.get_operator()
                dims = list(op.get_input(index).dims)
                self.model_input = Buffer("model_input", tuple(dims[1:]))
            return self.model_input
        if parent.name() not in self.buffers:
            raise ExportCError(f"{node.name()} : entrée {parent.name()} non traduite")
        return self.buffers[parent.name()]

    def emit_cpu(self, node, code, ctype="float", data=""):
        """! @brief Ajoute une opération CPU produisant la sortie du nœud."""
        out = Buffer("buf_" + _symbol(node.name()), _dims(node), ctype)
        self.buffers[node.name()] = out
        self.layers.append(CpuOp(out, code.replace("OUT", out.name), data))
        return out

    def translate(self):
        """! @brief Traduit tout le graphe.

        @return Liste des couches.
        @throws ExportCError au premier nœud non pris en charge.
        """
        for node in self.graph.get_ordered_nodes():
            kind = node.type()
            if kind == "Producer":
                continue
            emit = EMITTERS.get(kind)
            if emit is None:
                raise ExportCError(f"opérateur non pris en charge par le générateur C : {kind} ({node.name()})")
            emit(self, node)
        if self.pending_conv is not None:
            raise ExportCError(f"{self.pending_conv[0].name()} : Quantizer d'entrée sans convolution")
        return self.layers


def _loop(dims):
    """! @brief Variables et bornes d'un parcours C x H x W : la boucle sur toutes les valeurs d'un tampon."""
    c, h, w = dims
    return f"for (c = 0; c < {c}; c++) for (y = 0; y < {h}; y++) for (x = 0; x < {w}; x++)"


def _emit_quantizer(tr, node):
    """! @brief Quantizer : celui d'une entrée de convolution attend la convolution ; celui de sa sortie la complète."""
    parents = node.get_parents()
    parent = parents[0] if parents else None
    if parent is not None and parent.type() in ("PaddedConv2D", "Conv2D") and parent.name() in tr.buffers:
        layer = tr.buffers[parent.name()]
        if not isinstance(layer, ConvLayer):
            raise ExportCError(f"{node.name()} : Quantizer de sortie après une convolution flottante")
        scale, clip = _quantizer_params(node)
        layer.multiplier, layer.shift = fixed_point(scale)
        layer.clip = clip
        layer.output = Buffer(f"result_{layer.name}.out", _dims(node), "signed char")
        tr.buffers[node.name()] = layer.output
        tr.layers.append(layer)
        return
    source = tr.source(node)
    scale, clip = _quantizer_params(node)
    dims = _dims(node)
    target = Buffer("quant_" + _symbol(node.name()), dims, "signed char")
    tr.buffers[node.name()] = target
    code = (f"{_loop(dims)} {{\n"
            f"    float v = roundf({source.name}[c][y][x] * {scale:.9e}f);\n"
            f"    {target.name}[c][y][x] = (signed char)(v > {clip[1]}.0f ? {clip[1]} : (v < {clip[0]}.0f ? {clip[0]} : (int)v));\n"
            f"}}")
    tr.layers.append(CpuOp(target, code))
    tr.pending_conv = (node, target)


def _emit_conv(tr, node):
    """! @brief Convolution : entière après un Quantizer d'entrée (accélérateur), flottante sinon (CPU)."""
    weights, biases = _producer(node, 1), _producer(node, 2)
    if weights is None:
        raise ExportCError(f"{node.name()} : poids absents")
    stride, pad = _conv_attributes(node)
    if tr.pending_conv is not None and weights.dtype == np.int8:
        quantizer, source = tr.pending_conv
        if node.get_parents()[0] is not quantizer and node.get_parents()[0].name() != quantizer.name():
            raise ExportCError(f"{node.name()} : le Quantizer en attente n'est pas son entrée")
        tr.pending_conv = None
        name = _symbol(node.name()).removesuffix("_Conv")
        bias = np.zeros(weights.shape[0], dtype=np.int32) if biases is None else biases.astype(np.int32)
        tr.buffers[node.name()] = ConvLayer(name, weights, bias, source, None, stride, pad, 0, 0)
        return
    if weights.dtype != np.float32:
        raise ExportCError(f"{node.name()} : poids {weights.dtype} sans Quantizer d'entrée")
    source = tr.source(node)
    o, i, kh, kw = weights.shape
    if pad != 0 or kh != 1 or kw != 1 or stride != 1:
        raise ExportCError(f"{node.name()} : seule une convolution flottante 1 x 1 est prise en charge sur le CPU")
    symbol = _symbol(node.name())
    bias = np.zeros(o, dtype=np.float32) if biases is None else biases
    data = (f"static const float W_{symbol}[{o}][{i}] = {c_array(weights[:, :, 0, 0], '{:.9e}f')};\n"
            f"static const float B_{symbol}[{o}] = {c_array(bias, '{:.9e}f')};\n")
    code = (f"{_loop(_dims(node))} {{\n"
            f"    float s = B_{symbol}[c];\n"
            f"    for (i = 0; i < {i}; i++) s += W_{symbol}[c][i] * {source.name}[i][y][x];\n"
            f"    OUT[c][y][x] = s;\n"
            f"}}")
    tr.emit_cpu(node, code, data=data)


def _emit_cast(tr, node):
    """! @brief Cast : la conversion int8 vers flottant d'une déquantification, sinon refusé."""
    source = tr.source(node)
    if source.ctype != "signed char":
        raise ExportCError(f"{node.name()} : seul le Cast int8 vers flottant est pris en charge")
    tr.emit_cpu(node, f"{_loop(_dims(node))} OUT[c][y][x] = (float){source.name}[c][y][x];")


def _emit_mul(tr, node):
    """! @brief Mul : par une constante (déquantification) ou élément par élément (x² de la GDN)."""
    scale = _producer(node, 1)
    source = tr.source(node, 0)
    if scale is not None:
        tr.emit_cpu(node, f"{_loop(_dims(node))} OUT[c][y][x] = {source.name}[c][y][x] * {float(scale):.9e}f;")
        return
    other = tr.source(node, 1)
    tr.emit_cpu(node, f"{_loop(_dims(node))} OUT[c][y][x] = {source.name}[c][y][x] * {other.name}[c][y][x];")


def _emit_sqrt(tr, node):
    """! @brief Sqrt : racine flottante, sur le CPU."""
    source = tr.source(node)
    tr.emit_cpu(node, f"{_loop(_dims(node))} OUT[c][y][x] = sqrtf({source.name}[c][y][x]);")


def _emit_div(tr, node):
    """! @brief Div : division flottante élément par élément, sur le CPU."""
    a, b = tr.source(node, 0), tr.source(node, 1)
    tr.emit_cpu(node, f"{_loop(_dims(node))} OUT[c][y][x] = {a.name}[c][y][x] / {b.name}[c][y][x];")


## Traduction de chaque type de nœud ; ajouter un opérateur revient à ajouter une entrée ici.
EMITTERS = {"Quantizer": _emit_quantizer, "PaddedConv2D": _emit_conv, "Conv2D": _emit_conv, "Cast": _emit_cast,
            "Mul": _emit_mul, "Sqrt": _emit_sqrt, "Div": _emit_div}


def _accelerator_source(layer):
    """! @brief Texte C d'un accélérateur : structures de taille fixe, poids statiques, sortie en virgule fixe, sans flottant.

    @param layer  ConvLayer.
    @return Texte de l'accélérateur.
    """
    o, i, kh, kw = layer.weights.shape
    c_in, h_in, w_in = layer.input.dims
    _, h_out, w_out = layer.output.dims
    name = layer.name
    low, high = layer.clip
    return f"""static const signed char WEIGHTS_{name}[{o}][{i}][{kh}][{kw}] = {c_array(layer.weights)};
static const int BIASES_{name}[{o}] = {c_array(layer.biases)};

struct param_{name} {{
    signed char in[{c_in}][{h_in}][{w_in}];
}};

struct result_{name} {{
    signed char out[{o}][{h_out}][{w_out}];
}};

void {name}(struct param_{name} *param, struct result_{name} *result) {{
    int c, oy, ox, i, ky, kx;
    for (c = 0; c < {o}; c++) {{
        for (oy = 0; oy < {h_out}; oy++) {{
            for (ox = 0; ox < {w_out}; ox++) {{
                int sum = BIASES_{name}[c];
                long long product;
                int rounded;
                for (i = 0; i < {i}; i++) {{
                    for (ky = 0; ky < {kh}; ky++) {{
                        int iy = oy * {layer.stride} + ky - {layer.pad};
                        if (iy < 0 || iy >= {h_in}) continue;
                        for (kx = 0; kx < {kw}; kx++) {{
                            int ix = ox * {layer.stride} + kx - {layer.pad};
                            if (ix < 0 || ix >= {w_in}) continue;
                            sum += (int)WEIGHTS_{name}[c][i][ky][kx] * (int)param->in[i][iy][ix];
                        }}
                    }}
                }}
                product = (long long)sum * {layer.multiplier}LL;
                rounded = (int)((product + (1LL << {layer.shift - 1})) >> {layer.shift});
                result->out[c][oy][ox] = (signed char)(rounded > {high} ? {high} : (rounded < {low} ? {low} : rounded));
            }}
        }}
    }}
}}
"""


def _declaration(buffer):
    """! @brief Déclaration C statique d'un tampon."""
    c, h, w = buffer.dims
    return f"static {buffer.ctype} {buffer.name}[{c}][{h}][{w}];"


def _indent(text, spaces=4):
    """! @brief Indente chaque ligne d'un bloc de code."""
    pad = " " * spaces
    return "\n".join(pad + line for line in text.splitlines())


def comment_text(text):
    """! @brief Texte libre sûr dans un commentaire C : ni fin de commentaire, ni faux appel d'accélérateur pour locod.sh.

    @param text  Texte, par exemple le nom donné à l'optimisation.
    @return Texte où tout sauf lettres, chiffres, espaces, points et tirets devient « _ ».
    """
    return re.sub(r"[^\w .-]", "_", text)


def generate_source(layers, model_input, name):
    """! @brief Texte complet de @c main.c.

    Le motif d'appel aux accélérateurs est écrit une fois par couche, et nulle part ailleurs : locod.sh trouve les
    accélérateurs par recherche de texte dans le fichier, commentaires compris.
    @param layers       Couches de _Translator.translate().
    @param model_input  Tampon d'entrée.
    @param name         Nom du modèle, pour le commentaire d'en-tête.
    @return Texte C.
    """
    convs = [l for l in layers if isinstance(l, ConvLayer)]
    last = layers[-1].output
    c_in, h_in, w_in = model_input.dims
    c_out, h_out, w_out = last.dims
    head = [f"/* Encodeur {comment_text(name)} : généré depuis le graphe Aidge quantifié. Ne pas modifier. */",
            f"#ifndef {FPGA_MACRO}", "#include <math.h>", "#include <stdio.h>", "#include <string.h>", f"#ifndef {NATIVE_MACRO}",
            '#include "locod.h"', "#endif", "#endif", ""]
    body = [_accelerator_source(l) for l in convs]
    cpu = [f"#ifndef {FPGA_MACRO}"]
    decls, steps = [], []
    index = 0
    for layer in layers:
        if isinstance(layer, ConvLayer):
            source, target = layer.input, layer.output
            decls.append(f"static struct param_{layer.name} param_{layer.name};")
            decls.append(f"static struct result_{layer.name} result_{layer.name};")
            steps.append(f"memcpy(param_{layer.name}.in, {source.name}, sizeof(param_{layer.name}.in));")
            call = (f"#ifdef {NATIVE_MACRO}\n{layer.name}(&param_{layer.name}, &result_{layer.name});\n#else\n"
                    f"FPGA({layer.name}, &param_{layer.name}, &result_{layer.name}, {index});\n"
                    f"wait_accelerator(&result_{layer.name}, {index});\n#endif")
            steps.append(call)
            steps.append(f"DUMP(\"{layer.name}\", {target.name});")
            index += 1
        else:
            if layer.data:
                decls.append(layer.data.rstrip())
            decls.append(_declaration(layer.output))
            steps.append(layer.code)
            steps.append(f"DUMP(\"{layer.output.name}\", {layer.output.name});")
    cpu += decls
    cpu += ["", "#ifdef " + NATIVE_MACRO, "static const char *dump_dir = 0;",
            "static void dump_layer(const char *layer, const void *data, unsigned long size) {",
            "    char path[1024];", "    FILE *f;", "    if (!dump_dir) return;",
            '    snprintf(path, sizeof(path), "%s/%s.bin", dump_dir, layer);', '    f = fopen(path, "wb");',
            "    if (f) { fwrite(data, 1, size, f); fclose(f); }", "}",
            "#define DUMP(layer, buffer) dump_layer(layer, buffer, sizeof(buffer))", "#else",
            "#define DUMP(layer, buffer)", "#endif", "",
            f"static void model_forward(const float {model_input.name}[{c_in}][{h_in}][{w_in}], float output[{c_out}][{h_out}][{w_out}]) {{",
            "    int c, y, x, i;", "    (void)i;"]
    cpu += [_indent(s) for s in steps]
    cpu += [f"    memcpy(output, {last.name}, sizeof(float) * {c_out} * {h_out} * {w_out});", "}", "",
            "int main(int argc, char **argv) {",
            f"    static float input[{c_in}][{h_in}][{w_in}];", f"    static float output[{c_out}][{h_out}][{w_out}];", "    FILE *f;",
            '    if (argc < 3) { fprintf(stderr, "usage : %s entree.bin sortie.bin\\n", argv[0]); return 1; }',
            '    f = fopen(argv[1], "rb");', "    if (!f || fread(input, sizeof(input), 1, f) != 1) return 2;", "    fclose(f);",
            f"#ifdef {NATIVE_MACRO}", "    if (argc > 3) dump_dir = argv[3];", "#else", f"    init_locod({len(convs)});", "#endif",
            "    model_forward(input, output);", f"#ifndef {NATIVE_MACRO}", "    deinit_locod();", "#endif",
            '    f = fopen(argv[2], "wb");', "    if (!f || fwrite(output, sizeof(output), 1, f) != 1) return 3;", "    fclose(f);",
            "    return 0;", "}", f"#endif /* {FPGA_MACRO} */", ""]
    return "\n".join(head) + "\n" + "\n".join(body) + "\n" + "\n".join(cpu)


def export_locod(graph, out_dir, *, name="model"):
    """! @brief Génère @c main.c pour LoCod depuis un graphe Aidge quantifié.

    @param graph    GraphView quantifié, dimensions propagées et inférence déjà exécutée (le graphe n'est pas modifié).
    @param out_dir  Dossier de sortie, créé si besoin.
    @param name     Nom du modèle, pour l'en-tête.
    @return LocodExport.
    @throws ExportCError si le graphe contient un motif non pris en charge, ou aucune convolution INT8.
    """
    translator = _Translator(graph)
    layers = translator.translate()
    convs = [l for l in layers if isinstance(l, ConvLayer)]
    if not convs:
        raise ExportCError("aucune convolution INT8 dans le graphe : rien à accélérer")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "main.c"
    path.write_text(generate_source(layers, translator.model_input, name), encoding="utf-8")
    pl_weights = sum(int(l.weights.nbytes + l.biases.nbytes) for l in convs)
    transfers = sum(int(np.prod(l.input.dims) + np.prod(l.output.dims)) for l in convs)
    described = [(l.name if isinstance(l, ConvLayer) else l.output.name,
                  "int8" if isinstance(l, ConvLayer) or l.output.ctype == "signed char" else "float32", l.output.dims)
                 for l in layers]
    return LocodExport(path, len(convs), pl_weights, transfers, described)


def compile_native(main_c, compiler):
    """! @brief Compile @c main.c en natif, avec écriture possible de la sortie de chaque couche.

    @param main_c    Fichier C.
    @param compiler  Compilateur C.
    @return Couple (chemin du binaire ou None, fin de stderr en cas d'échec).
    """
    main_c = Path(main_c)
    binary = main_c.with_name("verif")
    r = subprocess.run([compiler, *GCC_FLAGS, "-o", binary.name, main_c.name, "-lm"], cwd=main_c.parent,
                       capture_output=True, text=True, timeout=COMPILE_TIMEOUT_S)
    if r.returncode != 0:
        return None, r.stderr[-2000:]
    return (binary if binary.exists() else binary.with_suffix(".exe")), ""


def run_native(binary, sample, export):
    """! @brief Exécute le binaire natif sur une tuile et relit la sortie de chaque couche.

    @param binary  Binaire de compile_native().
    @param sample  Tuile 1 x C x H x W.
    @param export  LocodExport.
    @return Dictionnaire nom de couche -> tableau C x H x W.
    @throws RuntimeError si l'exécution échoue.
    """
    binary = Path(binary)
    work = binary.parent / "couches"
    work.mkdir(exist_ok=True)
    source = binary.parent / "in.bin"
    np.ascontiguousarray(sample[0], dtype=np.float32).tofile(source)
    r = subprocess.run([str(binary.resolve()), source.name, "out.bin", work.name], cwd=binary.parent, capture_output=True,
                       timeout=RUN_TIMEOUT_S)
    if r.returncode != 0:
        raise RuntimeError(f"le binaire natif a échoué (code {r.returncode})")
    out = {}
    for layer, kind, dims in export.layers:
        out[layer] = np.fromfile(work / f"{layer}.bin", dtype=np.int8 if kind == "int8" else np.float32).reshape(dims)
    return out
