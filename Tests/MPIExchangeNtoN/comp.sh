#!/bin/sh

COMP="g++"

$COMP *.cpp ../../Utils/*.cpp -o MPIExchangeNtoN.out -lm -fopenmp
