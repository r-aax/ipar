#!/bin/sh

COMP="mpic++"
EXE="MPIInit.out"

rm -f $EXE

$COMP \
    -DDEBUG -DUSEMPI \
    *.cpp ../../Utils/*.cpp \
    -o $EXE \
    -lm -fopenmp
