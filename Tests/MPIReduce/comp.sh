#!/bin/sh

COMP="mpic++"
EXE="MPIReduce.out"

rm -f $EXE

$COMP -g \
    -DDEBUG -DUSEMPI \
    *.cpp ../../Utils/*.cpp \
    -o $EXE \
    -lm -fopenmp
