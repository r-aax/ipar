#!/bin/bash

set -x

mpirun -s knl -np 1 ./VECTriBoxIntersect_knl.out
