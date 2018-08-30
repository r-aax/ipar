#!/bin/bash

EXE="VECShellSort.out"

if [[ -f "$EXE" ]]
then
    srun -p knl -n 1 --constraint=cache,a2a ./${EXE} $1 > _result.txt
    cat _result.txt
    cat _result.txt >> result.txt
    rm -f _result.txt
fi
