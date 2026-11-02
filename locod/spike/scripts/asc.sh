#!/bin/bash
# Le chemin réel de LoCod : le fichier devient main.c, sans -I.
A=/w/bambu/squashfs-root
export APPDIR=$A PATH=$A/usr/bin:$PATH
export LD_LIBRARY_PATH=$A/usr/lib:$A/usr/lib/x86_64-linux-gnu:$A/usr/lib64:$A/lib:$A/lib/x86_64-linux-gnu
OPT="--writer=V --generate-interface=MINIMAL --memory-allocation-policy=LSS --channels-type=MEM_ACC_11 --memory-ctrl-type=D21 --addr-bus-bitsize=32 --data-bus-bitsize=32 -DLOCOD_FPGA -O0"
mkdir -p /tmp/as /tmp/as2; cd /tmp/as; cp /out/cpp/main.cpp main.c
timeout 300 $A/usr/bin/bambu $OPT --top-fname=conv_acc -I/out/cpp/include main.c > log.txt 2>&1; echo "C++ nommé main.c, avec -I : rc=$?"
grep -a -E 'error' log.txt | head -5 | cut -c1-170
cd /tmp/as2; cp /out/cpp/main.cpp main.c
timeout 300 $A/usr/bin/bambu $OPT --top-fname=conv_acc main.c > log.txt 2>&1; echo "C++ nommé main.c, sans -I (exactement locod.sh) : rc=$?"
grep -a -E 'error' log.txt | head -3 | cut -c1-170
echo "--- compilateur retenu par défaut :"; grep -a -iE 'compiler|clang|gcc' /out/res/B0_cpp_defaut/log.txt | head -4 | cut -c1-170
echo "--- ressources du C++ (GCC7) :"; grep -a -E 'mult|DSP|BRAM|memor' /out/res/B_cpp_I386_GCC7/log.txt | head -12 | cut -c1-140
