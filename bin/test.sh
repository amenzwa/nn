#!/usr/bin/env sh
# Usage: ~/nn/test.sh

make clean all
./lir lir-xor2
./lir lir-enc8
./som som-rgb
./som som-mst
