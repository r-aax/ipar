#!/bin/bash

COMP="$1"

if [[ -z "$COMP" ]]
then
    COMP="mpiicc"
fi

FLAGS="-DINTEL -O2 -unroll=0 -xmic-avx512"
INFO_FLAGS="-qopt-report=5"
EXE="VECMatrices.out"

rm -f $EXE

$COMP \
    -DDEBUG \
    *.cpp ../../Utils/*.cpp \
    $FLAGS $INFO_FLAGS \
    -lm -fopenmp \
    -S
$COMP \
    -DDEBUG \
    *.cpp ../../Utils/*.cpp \
    $FLAGS \
    -o $EXE \
    -lm -fopenmp
