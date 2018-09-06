#!/bin/bash

EXE="VECMatrices8.out"

if [[ -f "$EXE" ]]
then
    srun -p knl -n 1 ./${EXE} $1 > _result.txt
    cat _result.txt
    cat _result.txt >> result.txt
    rm -f _result.txt
fi
