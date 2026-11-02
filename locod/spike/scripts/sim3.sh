#!/bin/bash
# Le gabarit C++ réduit, simulé avec chaque compilateur embarqué.
A=/w/bambu/squashfs-root
export APPDIR=$A PATH=$A/usr/bin:$PATH
export LD_LIBRARY_PATH=$A/usr/lib:$A/usr/lib/x86_64-linux-gnu:$A/usr/lib64:$A/lib:$A/lib/x86_64-linux-gnu
OPT="--writer=V --generate-interface=MINIMAL --memory-allocation-policy=LSS --channels-type=MEM_ACC_11 --memory-ctrl-type=D21 --addr-bus-bitsize=32 --data-bus-bitsize=32 -DLOCOD_FPGA -O0"
SIMOPT="--simulate --simulator=VERILATOR --generate-tb=/out/small/tb.xml"
for cfg in "defaut:" "GCC7:--compiler=I386_GCC7 --std=c++14" "CLANG12:--compiler=I386_CLANG12 --std=c++14" "CLANG6:--compiler=I386_CLANG6 --std=c++14"; do
  n=${cfg%%:*}; x=${cfg#*:}
  w=/tmp/s3/$n; rm -rf $w; mkdir -p $w; cd $w
  timeout 500 $A/usr/bin/bambu $OPT $SIMOPT -v3 $x --top-fname=check -I/out/cpp/include /out/small/check_cpp.cpp > log.txt 2>&1
  echo "=== [$n] rc=$?"
  grep -a -E 'return_port =|Total cycles|Total estimated area|error ->' log.txt | tail -4 | cut -c1-120
  grep -a -B4 -m1 'Parse error' log.txt | cut -c1-200
done
