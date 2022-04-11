#!/bin/sh

set -x

rm -f VECTriBoxIntersect.out
rm -f gmon.out
rm -f *.gcda
rm -f *.gcno
rm -f *.gcov

g++ \
    *.cpp ../../Utils/*.cpp \
    -g -pg \
    -fprofile-arcs -ftest-coverage \
    -lm -fopenmp \
    -o VECTriBoxIntersect.out

./VECTriBoxIntersect.out

gprof VECTriBoxIntersect.out gmon.out -p > prof_p.txt
gprof VECTriBoxIntersect.out gmon.out -q > prof_q.txt
gprof VECTriBoxIntersect.out gmon.out -A > prof_A.txt

gcov ./tri_box_intersect.cpp