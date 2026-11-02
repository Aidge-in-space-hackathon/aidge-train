#!/bin/bash
# Contre-exemple : une valeur attendue faussée doit se voir. Le code de sortie Verilog est recherché dans le banc.
A=/w/bambu/squashfs-root
export APPDIR=$A PATH=$A/usr/bin:$PATH
export LD_LIBRARY_PATH=$A/usr/lib:$A/usr/lib/x86_64-linux-gnu:$A/usr/lib64:$A/lib:$A/lib/x86_64-linux-gnu
OPT="--writer=V --generate-interface=MINIMAL --memory-allocation-policy=LSS --channels-type=MEM_ACC_11 --memory-ctrl-type=D21 --addr-bus-bitsize=32 --data-bus-bitsize=32 -DLOCOD_FPGA -O0"
SIMOPT="--simulate --simulator=VERILATOR --generate-tb=/out/small/tb.xml"
mkdir -p /out/small/mut; cd /out/small/mut
cp ../small_dims.h ../small_weights.h ../input.h .
sed -E '0,/\{\{|\{/s/\{(-?[0-9]+)/{7777/' ../expected_MODE_INT32.h > expected_MODE_INT32.h
head -c 90 expected_MODE_INT32.h; echo
sed -e 's/#define MODE_INT32/#define MODE_INT32/' ../check_MODE_INT32.c > check_mut.c
go() { # nom, source
  local w=/tmp/s2/$1; rm -rf $w; mkdir -p $w; cd $w
  timeout 900 $A/usr/bin/bambu $OPT $SIMOPT -v3 --no-clean --top-fname=check -I/out/small/mut -I/out -I/out/small "$2" > log.txt 2>&1
  echo "[$1] rc=$?"; grep -a -E 'Simulation|error ->|ex_return|Value found|Return|return' log.txt | tail -6 | cut -c1-130
  ls; cat results.txt 2>/dev/null | head -3
  find . -name 'values*' -o -name 'test.xml' -o -name 'simulation*' | head;
  [ -f HLS_output/simulation/values.txt ] && tail -5 HLS_output/simulation/values.txt | cut -c1-120
  grep -rl 'ex_return_port' HLS_output 2>/dev/null | head -3
  grep -n -E 'ex_return_port' HLS_output/simulation/testbench_check_tb.v 2>/dev/null | head -8 | cut -c1-140
}
go juste /out/small/check_MODE_INT32.c
go faux /out/small/mut/check_mut.c
