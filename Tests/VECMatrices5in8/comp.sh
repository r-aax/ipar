#!/bin/bash

#set -x

COMP="$1"

if [[ -z "$COMP" ]]
then
    COMP="mpiicc"
fi

if [[ "$COMP" = "mpic++" ]]
then
    FLAGS="-O2"
    INFO_FLAGS=""
fi

if [[ "$COMP" = "mpiicc" ]]
then
    FLAGS="-DINTEL -O2 -xmic-avx512 -inline-level=0"
    INFO_FLAGS="-qopt-report=5"
fi

EXE="VECMatrices5in8.out"

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
