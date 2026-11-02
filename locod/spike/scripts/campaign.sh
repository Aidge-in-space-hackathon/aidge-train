#!/bin/bash
# Campagne Bambu 0.9.8 avec les options exactes de locod.sh (ligne 47). Résultats : /out/res/<test>/
A=/w/bambu/squashfs-root
export APPDIR=$A PATH=$A/usr/bin:$PATH
export LD_LIBRARY_PATH=$A/usr/lib:$A/usr/lib/x86_64-linux-gnu:$A/usr/lib64:$A/lib:$A/lib/x86_64-linux-gnu
BAMBU=$A/usr/bin/bambu
OPT="--writer=V --generate-interface=MINIMAL --memory-allocation-policy=LSS --channels-type=MEM_ACC_11 --memory-ctrl-type=D21 --addr-bus-bitsize=32 --data-bus-bitsize=32 -DLOCOD_FPGA -O0"
RES=/out/res
mkdir -p $RES

run() { # nom, limite (s), fonction du sommet, fichier source, options supplémentaires...
  local name=$1 lim=$2 top=$3 src=$4; shift 4
  local w=/tmp/run/$name d=$RES/$name
  rm -rf "$w" "$d"; mkdir -p "$w" "$d"; cd "$w"
  local t0=$(date +%s)
  timeout "$lim" $BAMBU $OPT -v3 --top-fname=$top "$@" "$src" > log.txt 2>&1
  local rc=$?
  local dt=$(( $(date +%s) - t0 ))
  local ok="ÉCHEC"; [ -f "$top.v" ] && ok="VERILOG"
  [ $rc = 124 ] && ok="LIMITE DE TEMPS"
  cp log.txt "$d/"
  [ -f "$top.v" ] && { wc -l "$top.v" > "$d/verilog_lignes.txt"; du -b "$top.v" >> "$d/verilog_lignes.txt"; ls *.mem 2>/dev/null | wc -l >> "$d/verilog_lignes.txt"; du -cb *.mem 2>/dev/null | tail -1 >> "$d/verilog_lignes.txt"; }
  local area=$(grep -a 'Total estimated area' log.txt | tail -1 | sed 's/.*: //')
  local dsp=$(grep -a 'Estimated number of DSPs' log.txt | tail -1 | sed 's/.*: //')
  local mhz=$(grep -a 'Estimated max frequency' log.txt | tail -1 | sed 's/.*: //' | cut -c1-6)
  local ffs=$(grep -a 'Total number of flip-flops' log.txt | tail -1 | sed 's/.*: //')
  echo "[$name] $ok rc=$rc durée=${dt}s surface=${area:-?} DSP=${dsp:-?} FF=${ffs:-?} MHz=${mhz:-?} v_lignes=$( [ -f "$top.v" ] && wc -l < "$top.v" || echo 0) mem=$(ls *.mem 2>/dev/null | wc -l)" | tee -a $RES/summary.txt
  rm -rf "$w"
}

variant() { # nom, couche, mode, poids en param ?
  local f=/out/v_$1.c
  { echo "#define $3"; echo "#define LAYER_DIMS \"$2_dims.h\""
    if [ "$4" = param ]; then echo "#define WEIGHTS_IN_PARAM"; echo "#define LAYER_WEIGHTS \"$2_weights_param.h\""; else echo "#define LAYER_WEIGHTS \"$2_weights.h\""; fi
    echo '#include "conv_c.c"'; } > "$f"
  echo "$f"
}

case "$1" in
base)
  EX=/w/LoCod/demo/example_readme/main.c
  run A0_exemple_acc0 600 acc0 "$EX"
  run A1_exemple_acc1_float 900 acc1 "$EX"
  ;;
cpp)
  run B0_cpp_defaut 900 conv_acc /out/cpp/main.cpp -I/out/cpp/include
  for c in I386_GCC7 I386_CLANG12 I386_CLANG6; do
    run "B_cpp_$c" 900 conv_acc /out/cpp/main.cpp -I/out/cpp/include --compiler=$c --std=c++14
  done
  ;;
c)
  run C1_conv1_int32_bram 3000 conv_acc "$(variant C1 conv1 MODE_INT32 bram)" -I/out
  run C2_conv1_fixq_bram 3000 conv_acc "$(variant C2 conv1 MODE_FIXQ bram)" -I/out
  run C3_conv1_floatq_bram 3000 conv_acc "$(variant C3 conv1 MODE_FLOATQ bram)" -I/out
  run C4_conv1_fixq_param 3000 conv_acc "$(variant C4 conv1 MODE_FIXQ param)" -I/out
  run C5_gdn_float 1800 gdn_acc /out/gdn.c -I/out
  ;;
big)
  run D1_conv2_fixq_bram 7200 conv_acc "$(variant D1 conv2 MODE_FIXQ bram)" -I/out
  run D2_conv2_fixq_param 3600 conv_acc "$(variant D2 conv2 MODE_FIXQ param)" -I/out
  ;;
esac
