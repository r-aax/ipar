#!/bin/sh

COMP="mpic++"
EXE="MPIExchangeNtoN.out"

rm -f $EXE

$COMP \
    -DDEBUG -DUSEMPI \
    *.cpp ../../Utils/*.cpp \
    -o $EXE \
    -lm -fopenmp
