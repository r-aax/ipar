#!/bin/bash

set -x

rm -f VECRiemannSample.exe
./comp.sh
if [ -f VECRiemannSample.out ]
then
    ./run_knl.sh $1
fi
