/// @file
/// @brief Functions realization.

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "tri_box_intersect.h"
using namespace std;

#ifdef INTEL
#include <immintrin.h>
#endif

/// @brief Original function.
///
/// @param [in] ax-zh - Datas.
/// @param [in] c - Count.
/// @param [out] r - Results.
void
tri_box_intersects_orig(float *ax, float *ay, float *az,
                        float *bx, float *by, float *bz,
                        float *cx, float *cy, float *cz,
                        float *xl, float *xh, float *yl, float *yh, float *zl, float *zh,
                        int c,
                        bool *r)
{
    for (int i = 0; i < c; i++)
    {
        r[i] = true;
    }
}

/// @brief Optimized fucntion.
///
/// @param [in] ax-zh - Datas.
/// @param [in] c - Count.
/// @param [out] r - Results.
void
tri_box_intersects_opt(float *ax, float *ay, float *az,
                       float *bx, float *by, float *bz,
                       float *cx, float *cy, float *cz,
                       float *xl, float *xh, float *yl, float *yh, float *zl, float *zh,
                       int c,
                       bool *r)
{
    for (int i = 0; i < c; i++)
    {
        r[i] = true;
    }
}
