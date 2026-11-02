"""Couche réduite issue des vrais poids de conv1 : 4 filtres, tuile 16 x 16. Vecteurs pour la simulation Verilog."""
import pathlib
import re
import subprocess

import numpy as np

OUT = pathlib.Path("/out/small")
OUT.mkdir(exist_ok=True)
txt = (pathlib.Path("/out") / "conv1_weights.h").read_text()
w_txt, b_txt = txt.split("static const int BIASES")
nums = lambda s: np.array([int(v) for v in re.findall(r"-?\d+", s.split("=", 1)[1])], dtype=np.int64)
W = nums(w_txt).reshape(128, 3, 9, 9).astype(np.int8)[:4]
B = nums(b_txt).astype(np.int32)[:4]
x = np.fromfile("/out/conv1_in.bin", dtype=np.int8).reshape(3, 64, 64)[:, 20:36, 20:36]
dims = pathlib.Path("/out/conv1_dims.h").read_text()
scale = re.search(r"OUT_SCALE (\S+)", dims).group(1)
m = re.search(r"FIX_M (\S+)", dims).group(1)
n = re.search(r"FIX_N (\S+)", dims).group(1)
(OUT / "small_dims.h").write_text(f"""#define C_IN 3
#define H_IN 16
#define W_IN 16
#define C_OUT 4
#define H_OUT 4
#define W_OUT 4
#define KH 9
#define KW 9
#define STRIDE 4
#define PAD 4
#define OUT_SCALE {scale}
#define FIX_M {m}
#define FIX_N {n}
""")


def c(a):
    return "{" + ",".join(str(int(v)) for v in a) + "}" if a.ndim == 1 else "{" + ",".join(c(s) for s in a) + "}"


(OUT / "small_weights.h").write_text(f"static const signed char WEIGHTS[4][3][9][9] = {c(W)};\nstatic const int BIASES[4] = {c(B)};\n")
x.tofile(OUT / "in.bin")

# Valeurs attendues : calculées par la couche C compilée en natif, prouvée identique à Aidge
for mode, dtype, ctype in (("MODE_INT32", np.int32, "int"), ("MODE_FIXQ", np.int8, "signed char"), ("MODE_FLOATQ", np.int8, "signed char")):
    subprocess.check_call(["gcc", "-O2", "-std=c11", "-DNATIVE_TEST", f"-D{mode}", "-I/out/small", "-DLAYER_DIMS=\"small_dims.h\"",
                           "-DLAYER_WEIGHTS=\"small_weights.h\"", "/out/conv_c.c", "-o", "/tmp/ns"])
    subprocess.check_call(["/tmp/ns", str(OUT / "in.bin"), f"/tmp/o_{mode}.bin"])
    e = np.fromfile(f"/tmp/o_{mode}.bin", dtype=dtype)
    (OUT / f"expected_{mode}.h").write_text(f"static const {ctype} EXPECTED[{e.size}] = {c(e)};\n")
    print(mode, e.size, "valeurs ; extrait", e[:6])
(OUT / "input.h").write_text(f"static const signed char INPUT[3][16][16] = {c(x)};\n")

# Banc auto-vérifiant en C : le circuit renvoie le nombre de valeurs fausses
for mode in ("MODE_INT32", "MODE_FIXQ", "MODE_FLOATQ"):
    (OUT / f"check_{mode}.c").write_text(f"""#define {mode}
#define ACC_QUAL static __attribute__((always_inline)) inline
#define LAYER_DIMS "small_dims.h"
#define LAYER_WEIGHTS "small_weights.h"
#include "conv_c.c"
#include "input.h"
#include "expected_{mode}.h"
int check(int dummy) {{
    static struct param_conv p;
    static struct result_conv r;
    int c, y, x, k = 0, bad = 0;
    for (c = 0; c < C_IN; c++) for (y = 0; y < H_IN; y++) for (x = 0; x < W_IN; x++) p.in[c][y][x] = INPUT[c][y][x];
    conv_acc(&p, &r);
    for (c = 0; c < C_OUT; c++) for (y = 0; y < H_OUT; y++) for (x = 0; x < W_OUT; x++) {{
        if (r.out[c][y][x] != EXPECTED[k]) bad++;
        k++;
    }}
    return bad + dummy;
}}
""")

# La même chose avec le gabarit C++ généré par l'export, paramètres réduits, entrée et sortie en NHWC et poids en OHWI
Wohwi = W.transpose(0, 2, 3, 1)
xhwc = x.transpose(1, 2, 0)
exp_chw = np.fromfile("/tmp/o_MODE_INT32.bin", dtype=np.int32).reshape(4, 4, 4)
exp_hwc = exp_chw.transpose(1, 2, 0).ravel()
(OUT / "check_cpp.cpp").write_text(f"""#include <cstdint>
#include "kernels/cpp/convolution.hpp"
static const int8_t W_T[4*9*9*3] = {c(Wohwi.ravel())};
static const int32_t B_T[4] = {c(B)};
static const int8_t X_T[16*16*3] = {c(xhwc.ravel())};
static const int32_t E_T[4*4*4] = {c(exp_hwc)};
static const export_cpp::NoScaling RS = {{}};
extern "C" int check(int dummy) {{
    static int8_t in[16*16*3];
    static int32_t out[4*4*4];
    int i, bad = 0;
    for (i = 0; i < 16*16*3; i++) in[i] = X_T[i];
    export_cpp::convolution_forward<1, 3, 16, 16, 4, 4, 4, 4, 4, 4, 4, 1, 1, 9, 9, export_cpp::ActivationFunction_T::Linear,
        0, 0, 0, 0, 3, 0, 0, 0, 0, 16>(in, out, W_T, B_T, RS);
    for (i = 0; i < 4*4*4; i++) if (out[i] != E_T[i]) bad++;
    return bad + dummy;
}}
#ifdef NATIVE_TEST
#include <cstdio>
int main() {{ printf("erreurs natives du gabarit réduit : %d\\n", check(0)); return 0; }}
#endif
""")
(OUT / "tb.xml").write_text('<?xml version="1.0"?>\n<function>\n  <testbench dummy="0"/>\n</function>\n')
print("ok")
