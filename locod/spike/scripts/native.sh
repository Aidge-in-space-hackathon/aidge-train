#!/bin/sh
# Compare la réécriture C à Aidge (mêmes entrées, sorties octet par octet ou taux d'égalité).
cd /out
for L in conv1 conv2; do
  for M in MODE_INT32 MODE_FLOATQ MODE_FIXQ; do
    gcc -O2 -std=c11 -DNATIVE_TEST -D$M "-DLAYER_DIMS=\"${L}_dims.h\"" "-DLAYER_WEIGHTS=\"${L}_weights.h\"" conv_c.c -o /tmp/t_${L}_$M || { echo "$L $M compilation KO"; continue; }
    /tmp/t_${L}_$M ${L}_in.bin /tmp/o_${L}_$M.bin
    python3 - "$L" "$M" <<'E'
import sys, numpy as np
L, M = sys.argv[1:]
o = np.fromfile(f"/tmp/o_{L}_{M}.bin", dtype=np.int32 if M == "MODE_INT32" else np.int8)
r = np.fromfile(f"/out/{L}_conv.bin" if M == "MODE_INT32" else f"/out/{L}_qout.bin", dtype=np.int32 if M == "MODE_INT32" else np.int8)
print(f"{L} {M}: {o.size} valeurs, identiques={np.mean(o == r) * 100:.3f}%, écart max={np.abs(o.astype(int) - r.astype(int)).max()}")
E
  done
  gcc -O2 -std=c11 -DNATIVE_TEST -DMODE_FIXQ -DWEIGHTS_IN_PARAM "-DLAYER_DIMS=\"${L}_dims.h\"" "-DLAYER_WEIGHTS=\"${L}_weights_param.h\"" conv_c.c -o /tmp/t_p && /tmp/t_p ${L}_in.bin /tmp/o_p.bin && cmp /tmp/o_p.bin /tmp/o_${L}_MODE_FIXQ.bin && echo "$L poids en param : identique"
done
python3 - <<'E'
import re, pathlib
fw = pathlib.Path("/workspace/outputs/spike/mixedc_nouveau_float_real/dnn/src/forward.cpp").read_text()
i = fw.find("convolution_forward<")
print(fw[i-20:i+1500])
print("--- entêtes des gabarits :", [str(p.name) for p in pathlib.Path("/workspace/outputs/spike/mixedc_nouveau_float_real/dnn/include/utils/cpp").glob("*")])
E
