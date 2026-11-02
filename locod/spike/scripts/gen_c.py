"""Spike LoCod : extrait deux convolutions INT8 d'Aidge vers C pur, avec vecteurs de test, et l'enveloppe C++."""
import pathlib
import re
import sys
import warnings

import numpy as np
import torch
import aidge_core
import aidge_backend_cpu  # noqa: F401
import aidge_onnx
import aidge_quantization as q

sys.path.insert(0, "/workspace")
from aidge_in_space.nn import Balle2017, freeze  # noqa: E402
from PIL import Image  # noqa: E402

aidge_core.Log.set_console_level(aidge_core.Level.Error)
OUT = pathlib.Path("/out")
int8, int32, f32 = aidge_core.dtype.int8, aidge_core.dtype.int32, aidge_core.dtype.float32
ck = torch.load("/workspace/outputs/checkpoints/bls2017_c3_N128_lambda1024_run9.pt", map_location="cpu")
m = Balle2017(3, 128, True)
m.load_state_dict(ck["state_dict"])
enc = freeze(m.g_a).eval()

rng = np.random.default_rng(0)
paths = sorted(pathlib.Path("/workspace/data/kodak").glob("*.png"))
tiles = []
for i in range(16):
    im = np.asarray(Image.open(paths[i % len(paths)]).convert("RGB"), dtype=np.float32) / 255
    y, x = rng.integers(0, im.shape[0] - 64), rng.integers(0, im.shape[1] - 64)
    tiles.append(np.ascontiguousarray(im[y:y + 64, x:x + 64].transpose(2, 0, 1)[None]))

caps = {}
hooks = [getattr(enc, n).register_forward_hook(lambda _m, i, o, n=n: caps.setdefault(n, []).append(i[0].numpy().copy()))
         for n in ("conv1", "conv2")]
with torch.no_grad():
    for t in tiles:
        enc(torch.from_numpy(t))
for h in hooks:
    h.remove()


def arr_c(a, depth=0):
    if a.ndim == 1:
        return "{" + ",".join(str(int(v)) for v in a) + "}"
    return "{" + ",\n".join(arr_c(s, depth + 1) for s in a) + "}"


for name in ("conv1", "conv2"):
    shape = caps[name][0].shape
    p = OUT / f"seg_{name}.onnx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(getattr(enc, name).eval(), (torch.zeros(shape),), str(p), input_names=["image"],
                          output_names=["latent"], opset_version=13, do_constant_folding=True, dynamo=False)
    g = aidge_onnx.load_onnx(str(p))
    g.set_dataformat(aidge_core.dformat.nchw)
    g.compile("cpu", f32, dims=[list(shape)])
    aidge_core.SequentialScheduler(g).generate_scheduling()
    q.auto_assign_node_precision(g, int8, int8, int32)
    q.quantize_network(g, calibration_set=[aidge_core.Tensor(t) for t in caps[name]], clipping_mode=q.Clipping.MAX,
                       equalize=False, rescale_inputs=True, dequantize=True)
    s = aidge_core.SequentialScheduler(g)
    s.generate_scheduling()
    x = caps[name][3]
    s.forward(data=[aidge_core.Tensor(np.ascontiguousarray(x, dtype=np.float32))])

    nodes = {n.type(): n for n in g.get_nodes() if n.type() != "Quantizer"}
    quant = {n.name(): n for n in g.get_nodes() if n.type() == "Quantizer"}
    conv = nodes["PaddedConv2D"]
    prod = {n.name(): np.array(n.get_operator().get_output(0)).copy() for n in g.get_nodes() if n.type() == "Producer"}
    W = next(v for k, v in prod.items() if k.startswith("weight"))
    B = next(v for k, v in prod.items() if k.startswith("bias"))
    q_in = np.array(quant["InputRescaling"].get_operator().get_output(0)).copy()
    conv_out = np.array(conv.get_operator().get_output(0)).copy()
    q_out_node = next(v for k, v in quant.items() if k != "InputRescaling")
    q_out = np.array(q_out_node.get_operator().get_output(0)).copy()
    in_scale, out_scale = q.get_scaling_factor(quant["InputRescaling"]), q.get_scaling_factor(q_out_node)
    dq = None
    for mn in nodes["Dequantizer"].get_operator().get_micro_graph().get_nodes():
        if mn.type() == "Mul":
            for pp in mn.get_parents():
                if pp is not None and pp.type() == "Producer":
                    dq = float(np.array(pp.get_operator().get_output(0)))
    attrs = conv.get_operator().get_micro_graph()
    print(f"{name}: W{W.shape} {W.dtype} [{W.min()},{W.max()}], B{B.shape} {B.dtype}, in {q_in.shape} {q_in.dtype}, "
          f"conv {conv_out.shape} {conv_out.dtype} [{conv_out.min()},{conv_out.max()}], q_out {q_out.shape} {q_out.dtype}, "
          f"échelles in={in_scale} out={out_scale} dq={dq}")
    O, I, KH, KW = W.shape
    _, _, H, Wd = q_in.shape
    _, _, OH, OW = conv_out.shape
    stride = H // OH
    pad = (KH - 1) // 2 if name == "conv1" else 2
    # Virgule fixe : M sur 31 bits, décalage n, tel que M / 2^n approche l'échelle
    n = 31 + int(np.ceil(-np.log2(out_scale))) if out_scale < 1 else 31
    M = int(round(out_scale * 2 ** n))
    while M >= 2 ** 31:
        n -= 1
        M = int(round(out_scale * 2 ** n))
    h = [f"#define C_IN {I}", f"#define H_IN {H}", f"#define W_IN {Wd}", f"#define C_OUT {O}", f"#define H_OUT {OH}",
         f"#define W_OUT {OW}", f"#define KH {KH}", f"#define KW {KW}", f"#define STRIDE {stride}", f"#define PAD {pad}",
         f"#define OUT_SCALE {float(out_scale):.9e}f", f"#define FIX_M {M}LL", f"#define FIX_N {n}"]
    (OUT / f"{name}_dims.h").write_text("\n".join(h) + "\n")
    (OUT / f"{name}_weights.h").write_text(
        f"static const signed char WEIGHTS[{O}][{I}][{KH}][{KW}] = {arr_c(W)};\n"
        f"static const int BIASES[{O}] = {arr_c(B)};\n")
    (OUT / f"{name}_weights_param.h").write_text(f"static const signed char WEIGHTS_INIT[{O}][{I}][{KH}][{KW}] = {arr_c(W)};\n"
                                                 f"static const int BIASES_INIT[{O}] = {arr_c(B)};\n")
    q_in.astype(np.int8).tofile(OUT / f"{name}_in.bin")
    conv_out.astype(np.int32).tofile(OUT / f"{name}_conv.bin")
    q_out.astype(np.int8).tofile(OUT / f"{name}_qout.bin")
    print(f"   stride={stride} pad={pad} M={M} n={n} approx={M / 2 ** n:.9e} poids={W.size} o")

# Enveloppe C++ : l'appel exact généré par l'export pour la première convolution
EXP = pathlib.Path("/workspace/outputs/spike/mixedc_nouveau_float_real")
fw = (EXP / "dnn/src/forward.cpp").read_text()
call = re.search(r"export_cpp::convolution_forward<_7_PADCONV_0.*?;", fw, re.S)
print("appel C++ :", call.group(0)[:1800] if call else "INTROUVABLE")
(OUT / "conv1_call.txt").write_text(call.group(0) if call else "")
print("param hpp:", (EXP / "dnn/include/parameters/_7_PadConv_0_weights.hpp").read_text()[:300] if (EXP / "dnn/include/parameters/_7_PadConv_0_weights.hpp").exists() else "?")
print("rescaling:", [l for l in (EXP / "dnn/include/layers/_7_PadConv_0.h").read_text().splitlines() if "RESCALING" in l or "Rescaling" in l])
