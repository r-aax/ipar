#!/bin/bash

set -x

mpirun -s skylake -np 1 ./VECTriBoxIntersect_skylake.out
