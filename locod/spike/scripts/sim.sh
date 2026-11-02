#!/bin/bash
# Simulation Verilog auto-vérifiante, options exactes de locod.sh. check() renvoie le nombre de valeurs fausses.
A=/w/bambu/squashfs-root
export APPDIR=$A PATH=$A/usr/bin:$PATH
export LD_LIBRARY_PATH=$A/usr/lib:$A/usr/lib/x86_64-linux-gnu:$A/usr/lib64:$A/lib:$A/lib/x86_64-linux-gnu
OPT="--writer=V --generate-interface=MINIMAL --memory-allocation-policy=LSS --channels-type=MEM_ACC_11 --memory-ctrl-type=D21 --addr-bus-bitsize=32 --data-bus-bitsize=32 -DLOCOD_FPGA -O0"
SIMOPT="--simulate --simulator=VERILATOR --generate-tb=/out/small/tb.xml"
R=/out/res_sim; mkdir -p $R
sim() { # nom, limite, source, options
  local n=$1 lim=$2 src=$3; shift 3
  local w=/tmp/sim/$n; rm -rf $w; mkdir -p $w; cd $w
  local t0=$(date +%s)
  timeout $lim $A/usr/bin/bambu $OPT $SIMOPT -v3 --top-fname=check "$@" "$src" > log.txt 2>&1
  local rc=$?
  cp log.txt $R/$n.log
  { echo "[$n] rc=$rc durée=$(( $(date +%s)-t0 ))s"
    grep -a -E 'Total cycles|Number of executions|Average execution|Simulation|error ->|Estimated max frequency|Total estimated area|Estimated number of DSPs' log.txt | tail -8 | cut -c1-120
    ls results* 2>/dev/null; [ -f results.txt ] && { echo "results.txt :"; head -3 results.txt; }
    find . -maxdepth 3 -name 'results*' | head -3; } | tee -a $R/summary.txt
  cp -f results.txt $R/$n.results.txt 2>/dev/null
}
: > $R/summary.txt
sim sim_c_int32 1500 /out/small/check_MODE_INT32.c -I/out -I/out/small
sim sim_c_fixq 1500 /out/small/check_MODE_FIXQ.c -I/out -I/out/small
sim sim_c_floatq 2400 /out/small/check_MODE_FLOATQ.c -I/out -I/out/small
sim sim_cpp_int32 1500 /out/small/check_cpp.cpp -I/out/cpp/include
