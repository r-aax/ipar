#!/bin/bash

COMP="$1"

if [[ -z "$COMP" ]]
then
    COMP="mpicxx"
fi

EXE="MPIInit.out"

rm -f $EXE

$COMP \
    -DDEBUG -DUSEMPI \
    *.cpp ../../Utils/*.cpp \
    -o $EXE \
    -lm -fopenmp
