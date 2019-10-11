#!/bin/bash

set -x

mpirun -s knl -np 1 -maxtime 4 ./VECTriBoxIntersect_knl.out 100
