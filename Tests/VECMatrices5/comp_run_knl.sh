#!/bin/bash

set -x

./comp.sh
./run_knl.sh $1
