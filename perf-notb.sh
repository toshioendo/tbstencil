#!/usr/bin/bash
for i in 8 16 32 ; do
    gcc -O2 -fopenmp -DNX=$((1000*i)) -DNY=$((1000*i))  comp-notb.c
    export OMP_NUM_THREADS=56
    echo num56
    perf stat -e LLC-loads,LLC-load-misses,L1-dcache-loads,L1-dcache-load-misses ./a.out
done
