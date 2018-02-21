#!/bin/bash

COMP="$1"

if [[ -z "$COMP" ]]
then
    COMP="mpicxx"
fi

EXE="MPIReduce.out"

rm -f $EXE

$COMP -g \
    -DDEBUG -DUSEMPI \
    *.cpp ../../Utils/*.cpp \
    -o $EXE \
    -lm -fopenmp
