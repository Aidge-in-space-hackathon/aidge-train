"""Enveloppe FPGA() autour de l'appel C++ à gabarits généré par l'export, sans le modifier."""
import pathlib
import re
import shutil

EXP = pathlib.Path("/workspace/outputs/spike/mixedc_nouveau_float_real")
OUT = pathlib.Path("/out/cpp")
shutil.rmtree(OUT, ignore_errors=True)
OUT.mkdir(parents=True)
shutil.copytree(EXP / "dnn/include", OUT / "include")
for n in ("_7_PadConv_0_weights.cpp", "_7_PadConv_0_biases.cpp"):
    shutil.copy(EXP / "dnn/src/parameters" / n, OUT / n)
fw = (EXP / "dnn/src/forward.cpp").read_text()
m = re.search(r"export_cpp::convolution_forward<\s*_6_TRANSPOSE_0_OUTPUT_0_IN_BATCH.*?>\s*\((.*?)\);", fw, re.S)
print("arguments d'origine :", " ".join(m.group(1).split()))
tmpl = m.group(0)[: m.group(0).rindex("(")]
print("derniers paramètres :", " ".join(tmpl.split())[-260:])
incs = [l for l in fw.splitlines() if l.startswith("#include") and re.search(r"_6_Transpose_0\.h|_7_PadConv_0|kernels/cpp/convolution", l)]
print("inclusions :", incs)
body = "\n".join(l for l in incs if "parameters/" not in l)
(OUT / "main.cpp").write_text(f"""#include <cstdint>
#include "parameters/_7_PadConv_0_weights.hpp"
#include "parameters/_7_PadConv_0_biases.hpp"
{body}
#include "_7_PadConv_0_weights.cpp"
#include "_7_PadConv_0_biases.cpp"

struct param_conv {{ int8_t in[12288]; }};
struct result_conv {{ int32_t out[32768]; }};

extern "C" void conv_acc(struct param_conv *param, struct result_conv *result) {{
    {tmpl}(param->in, result->out, _7_PadConv_0_weights_output_0, _7_PadConv_0_biases_output_0, _7_PADCONV_0_RESCALING);
}}

#ifdef NATIVE_TEST
#include <cstdio>
int main() {{ static param_conv p; static result_conv r; conv_acc(&p, &r); printf("ok %d\\n", r.out[0]); return 0; }}
#endif
""")
print((OUT / "main.cpp").read_text()[:700])
