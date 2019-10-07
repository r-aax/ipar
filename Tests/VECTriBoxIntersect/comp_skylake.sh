#!/bin/bash

set -x

COMP="$1"

if [[ -z "$COMP" ]]
then
    COMP="mpiicc"
fi

#FLAGS="-DINTEL -O3 -xmic-avx512 -qopt-prefetch"
FLAGS="-O3"
#INFO_FLAGS="-qopt-report=5"
INFO_FLAGS=""
EXE="VECTriBoxIntersect_skylake.out"

rm -f $EXE

$COMP \
    -DDEBUG \
    *.cpp ../../Utils/*.cpp \
    $FLAGS $INFO_FLAGS \
    -lm -fopenmp \
    -o $EXE
