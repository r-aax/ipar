#!/bin/bash

COMP="$1"

if [[ -z "$COMP" ]]
then
    COMP="mpicxx"
fi

EXE="VECMatrices.out"

rm -f $EXE

$COMP \
    -DDEBUG \
    *.cpp ../../Utils/*.cpp \
    -o $EXE \
    -lm -fopenmp
