#!/usr/bin/bash
export OMP_NUM_THREADS=56

for n in 4 8 16 32; do
  for bt in 1 2 5 ; do
    gcc -O2 -g -fopenmp -DNX=$((1000*n)) -DNY=$((1000*n)) -DBT=$((bt)) comp-v2.c
    perf stat -e LLC-loads,LLC-load-misses,L1-dcache-loads,L1-dcache-load-misses ./a.out
  done
done
