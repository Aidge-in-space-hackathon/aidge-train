#!/bin/bash
# Preuve manuelle : le main.c du générateur passe-t-il dans Bambu, avec les options exactes de locod.sh ?
# Usage : bash locod/verifier_bambu.sh DOSSIER_BAMBU main.c [DOSSIER_RESULTATS]
#   DOSSIER_BAMBU : AppImage de Bambu 0.9.8 extrait (squashfs-root), voir locod/spike/README.md.
# Les accélérateurs sont cherchés comme le fait locod.sh, puis chacun passe seul dans Bambu : un fichier Verilog par accélérateur.
# Il faut iverilog ou verilator, et gcc-multilib. Une convolution N=128 prend de quelques secondes à une minute.
set -u
A=$1
SRC=$(realpath "$2")
RES=$(realpath -m "${3:-resultats_bambu}")
export APPDIR=$A PATH=$A/usr/bin:$PATH
export LD_LIBRARY_PATH=$A/usr/lib:$A/usr/lib/x86_64-linux-gnu:$A/usr/lib64:$A/lib:$A/lib/x86_64-linux-gnu
OPT="--writer=V --generate-interface=MINIMAL --memory-allocation-policy=LSS --channels-type=MEM_ACC_11 --memory-ctrl-type=D21 --addr-bus-bitsize=32 --data-bus-bitsize=32 -DLOCOD_FPGA -O0"
mkdir -p "$RES"
FUNCS=$(grep -oP '(?<=FPGA\().*?(?=,)' "$SRC")
echo "Accélérateurs : $FUNCS"
STATUS=0
for ACC in $FUNCS; do
  W=$(mktemp -d)
  cp "$SRC" "$W/main.c"
  mkdir "$W/bambu"
  (cd "$W/bambu" && timeout 3600 "$A/usr/bin/bambu" $OPT -v3 --top-fname="$ACC" ../main.c > "$RES/$ACC.log" 2>&1)
  RC=$?
  if [ -f "$W/bambu/$ACC.v" ]; then
    cp "$W/bambu/$ACC.v" "$RES/"
    L=$RES/$ACC.log
    echo "[$ACC] VERILOG rc=$RC lignes=$(wc -l < "$RES/$ACC.v")" \
      "surface=$(grep -a 'Total estimated area' "$L" | tail -1 | sed 's/.*: //')" \
      "DSP=$(grep -a 'Estimated number of DSPs' "$L" | tail -1 | sed 's/.*: //')" \
      "MHz=$(grep -a 'Estimated max frequency' "$L" | tail -1 | sed 's/.*: //' | cut -c1-6)" \
      "BRAM=$(grep -a 'Internally allocated memory:' "$L" | tail -1 | sed 's/.*: //')o" \
      "flottant=$(grep -a -c -E '__float|FloatingPoint|fp_' "$RES/$ACC.v")"
  else
    echo "[$ACC] ÉCHEC rc=$RC : $(grep -a -E 'error' "$RES/$ACC.log" | head -1 | cut -c1-160)"
    STATUS=1
  fi
  rm -rf "$W"
done
exit $STATUS
