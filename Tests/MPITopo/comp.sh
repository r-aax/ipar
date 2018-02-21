#!/bin/bash

COMP="$1"

if [[ -z "$COMP" ]]
then
    COMP="mpicxx"
fi

EXE="MPITopo.out"

rm -f $EXE

$COMP \
    -DDEBUG -DUSEMPI \
    *.cpp ../../Utils/*.cpp \
    -o $EXE \
    -lm -fopenmp
