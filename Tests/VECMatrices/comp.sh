#!/bin/bash

COMP="$1"

if [[ -z "$COMP" ]]
then
    COMP="mpiicc"
fi

#FLAGS="-xmic-avx512"
INFO_FLAGS="-qopt-report=5"
EXE="VECMatrices.out"

rm -f $EXE

#$COMP \
#    -DDEBUG \
#    *.cpp ../../Utils/*.cpp \
#    -O2 -unroll=0 \
#    $FLAGS \
#    -lm -fopenmp \
#    -S
$COMP \
    -DDEBUG \
    *.cpp ../../Utils/*.cpp \
    -O2 -unroll=0 \
    $FLAGS \
    -o $EXE \
    -lm -fopenmp
