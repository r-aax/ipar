#!/bin/sh

COMP="mpic++"
EXE="MPITopo.out"

rm -f $EXE

$COMP \
    -DDEBUG -DUSEMPI \
    *.cpp ../../Utils/*.cpp \
    -o $EXE \
    -lm -fopenmp
