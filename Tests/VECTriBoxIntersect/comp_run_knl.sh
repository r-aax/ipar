#!/bin/bash

rm VECTriBoxIntersect_knl.out
./comp_knl.sh

if [ -f "VECTriBoxIntersect_knl.out" ]
then
    ./run_knl.sh
fi

