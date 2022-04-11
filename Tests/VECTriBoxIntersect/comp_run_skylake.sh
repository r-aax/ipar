#!/bin/bash

rm VECTriBoxIntersect_skylake.out
./comp_skylake.sh

if [ -f "VECTriBoxIntersect_skylake.out" ]
then
    ./run_skylake.sh
fi

