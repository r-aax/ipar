/// @file
/// @brief Functions realization.

#include "tri_box_intersect.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "../../Utils/Maths.h"
#include "avx512.h"

using namespace std;

//#ifdef INTEL
#include <immintrin.h>
//#endif

/// @brief Upgrade interval.
///
/// @return
/// <c>true</c>, if current solve interval is not zero,
/// <c>false</c>, otherwise.
bool
upgrade_solution_orig(float * __restrict__ lo,
                      float * __restrict__ hi,
                      float f0,
                      float f1)
{
    if (f0 > 0.0)
    {
        *hi = Utils::Min(*hi, -f1 / f0);
    }
    else if (f0 < 0.0)
    {
        *lo = Utils::Max(*lo, -f1 / f0);
    }
    else
    {
        // Interval didn't change.
        return f1 <= 0.0;
    }

    return *lo <= *hi;
}

/// @brief Origin single case.
bool
tri_box_intersect_orig(float xa,
                       float ya,
                       float za,
                       float xb,
                       float yb,
                       float zb,
                       float xc,
                       float yc,
                       float zc,
                       float xl,
                       float xh,
                       float yl,
                       float yh,
                       float zl,
                       float zh)
{
    float lo = 0.0;
    float hi = 1.0;

    const int basic_eqns_count = 8;
    float b[basic_eqns_count][3];
    b[0][0] = xb - xa;
    b[0][1] = xc - xa;
    b[0][2] = -(xh - xa);
    b[1][0] = -(xb - xa);
    b[1][1] = -(xc - xa);
    b[1][2] = xl - xa;
    b[2][0] = yb - ya;
    b[2][1] = yc - ya;
    b[2][2] = -(yh - ya);
    b[3][0] = -(yb - ya);
    b[3][1] = -(yc - ya);
    b[3][2] = yl - ya;
    b[4][0] = zb - za;
    b[4][1] = zc - za;
    b[4][2] = -(zh - za);
    b[5][0] = -(zb - za);
    b[5][1] = -(zc - za);
    b[5][2] = zl - za;
    b[6][0] = 1.0;
    b[6][1] = 0.0;
    b[6][2] = -1.0;
    b[7][0] = -1.0;
    b[7][1] = 0.0;
    b[7][2] = 0.0;

    // Выполнение свертки.
    for (int i = 0; i < basic_eqns_count; i++)
    {
        float bi0 = b[i][0];

        if (bi0 == 0.0)
        {
            if (!upgrade_solution_orig(&lo, &hi, b[i][1], b[i][2]))
            {
                return false;
            }
        }
        else
        {
            for (int j = i + 1; j < basic_eqns_count; j++)
            {
                if (bi0 * b[j][0] < 0.0)
                {
                    float f0 = bi0 * b[j][1] - b[j][0] * b[i][1];
                    float f1 = bi0 * b[j][2] - b[j][0] * b[i][2];

                    if (bi0 < 0.0)
                    {
                        f0 = -f0;
                        f1 = -f1;
                    }

                    if (!upgrade_solution_orig(&lo, &hi, f0, f1))
                    {
                        return false;
                    }
                }
            }
        }
    }

    return true;
}

/// @brief Upgrade interval.
///
/// @return
/// <c>true</c>, if current solve interval is not zero,
/// <c>false</c>, otherwise.
bool
upgrade_solution_opt(float * __restrict__ lo,
                     float * __restrict__ hi,
                     float f0,
                     float f1)
{
    if (f0 == 0.0)
    {
        return f1 <= 0.0;
    }
    else
    {
        float k = -f1 / f0;

        if (f0 > 0.0)
        {
            *hi = Utils::Min(*hi, -f1 / f0);
        }
        else
        {
            *lo = Utils::Max(*lo, -f1 / f0);
        }

        return *lo <= *hi;
    }
}

/// @brief Origin single case.
bool
tri_box_intersect_opt(float xa,
                      float ya,
                      float za,
                      float xb,
                      float yb,
                      float zb,
                      float xc,
                      float yc,
                      float zc,
                      float xl,
                      float xh,
                      float yl,
                      float yh,
                      float zl,
                      float zh)
{
    float lo = 0.0;
    float hi = 1.0;

    const int basic_eqns_count = 8;
    float b[basic_eqns_count][3];
    b[0][0] = xb - xa;
    b[0][1] = xc - xa;
    b[0][2] = -(xh - xa);
    b[1][0] = -(xb - xa);
    b[1][1] = -(xc - xa);
    b[1][2] = xl - xa;
    b[2][0] = yb - ya;
    b[2][1] = yc - ya;
    b[2][2] = -(yh - ya);
    b[3][0] = -(yb - ya);
    b[3][1] = -(yc - ya);
    b[3][2] = yl - ya;
    b[4][0] = zb - za;
    b[4][1] = zc - za;
    b[4][2] = -(zh - za);
    b[5][0] = -(zb - za);
    b[5][1] = -(zc - za);
    b[5][2] = zl - za;
    b[6][0] = 1.0;
    b[6][1] = 0.0;
    b[6][2] = -1.0;
    b[7][0] = -1.0;
    b[7][1] = 0.0;
    b[7][2] = 0.0;

    // Выполнение свертки.
    for (int i = 0; i < basic_eqns_count; i++)
    {
        float bi0 = b[i][0];

        if (bi0 == 0.0)
        {
            if (!upgrade_solution_opt(&lo, &hi, b[i][1], b[i][2]))
            {
                return false;
            }
        }
        else
        {
            for (int j = i + 1; j < basic_eqns_count; j++)
            {
                if (bi0 * b[j][0] < 0.0)
                {
                    float f0 = bi0 * b[j][1] - b[j][0] * b[i][1];
                    float f1 = bi0 * b[j][2] - b[j][0] * b[i][2];

                    if (bi0 < 0.0)
                    {
                        f0 = -f0;
                        f1 = -f1;
                    }

                    if (!upgrade_solution_opt(&lo, &hi, f0, f1))
                    {
                        return false;
                    }
                }
            }
        }
    }

    return true;
}

#define COND_EXE(command, pred) \
if (pred) \
{ \
    command; \
}

void
update_lo_hi_orig(bool m,
                  float f0,
                  float f1,
                  float *lo,
                  float *hi)
{
    float k;

    COND_EXE(*lo = *hi + 1.0, m && (f0 == 0.0) && (f1 > 0.0));
    COND_EXE(k = -f1 / f0, !(m && (f0 == 0.0)));
    COND_EXE(*hi = Utils::Min(*hi, k), m && (f0 > 0.0));
    COND_EXE(*lo = Utils::Max(*lo, k), m && (f0 < 0.0));
}

void
tri_box_intersects_orig_16(float * __restrict__ xa,
                           float * __restrict__ ya,
                           float * __restrict__ za,
                           float * __restrict__ xb,
                           float * __restrict__ yb,
                           float * __restrict__ zb,
                           float * __restrict__ xc,
                           float * __restrict__ yc,
                           float * __restrict__ zc,
                           float * __restrict__ xl,
                           float * __restrict__ xh,
                           float * __restrict__ yl,
                           float * __restrict__ yh,
                           float * __restrict__ zl,
                           float * __restrict__ zh,
                           int * __restrict__ r)
{
    const int basic_eqns_count = 8;
    float lo, hi;
    float b[basic_eqns_count][3][VEC_WIDTH];

    // Init.
    for (int w = 0; w < VEC_WIDTH; w++)
    {
        b[0][0][w] = xb[w] - xa[w];
        b[0][1][w] = xc[w] - xa[w];
        b[0][2][w] = -(xh[w] - xa[w]);
        b[1][0][w] = -(xb[w] - xa[w]);
        b[1][1][w] = -(xc[w] - xa[w]);
        b[1][2][w] = xl[w] - xa[w];
        b[2][0][w] = yb[w] - ya[w];
        b[2][1][w] = yc[w] - ya[w];
        b[2][2][w] = -(yh[w] - ya[w]);
        b[3][0][w] = -(yb[w] - ya[w]);
        b[3][1][w] = -(yc[w] - ya[w]);
        b[3][2][w] = yl[w] - ya[w];
        b[4][0][w] = zb[w] - za[w];
        b[4][1][w] = zc[w] - za[w];
        b[4][2][w] = -(zh[w] - za[w]);
        b[5][0][w] = -(zb[w] - za[w]);
        b[5][1][w] = -(zc[w] - za[w]);
        b[5][2][w] = zl[w] - za[w];
        b[6][0][w] = 1.0;
        b[6][1][w] = 0.0;
        b[6][2][w] = -1.0;
        b[7][0][w] = -1.0;
        b[7][1][w] = 0.0;
        b[7][2][w] = 0.0;
    }

    // Result init.
    for (int w = 0; w < VEC_WIDTH; w++)
    {
        r[w] = 1;
    }

    // Main loop.
    for (int w = 0; w < VEC_WIDTH; w++)
    {
        lo = 0.0;
        hi = 1.0;

        for (int i = 0; i < basic_eqns_count; i++)
        {
            update_lo_hi_orig(b[i][0][w] == 0.0,
                              b[i][1][w],
                              b[i][2][w],
                              &lo, &hi);

            if (lo > hi)
            {
                break;
            }
        }

        for (int i = 0; i < basic_eqns_count; i++)
        {
            float bi0 = b[i][0][w];
            float abi0 = fabs(bi0);

            for (int j = i + 1; j < basic_eqns_count; j++)
            {
                float bj0 = b[j][0][w];
                float abj0 = fabs(bj0);

                update_lo_hi_orig(bi0 * bj0 < 0.0,
                                  abi0 * b[j][1][w] + abj0 * b[i][1][w],
                                  abi0 * b[j][2][w] + abj0 * b[i][2][w],
                                  &lo, &hi);

                if (lo > hi)
                {
                    break;
                }
            }

            if (lo > hi)
            {
                break;
            }
        }

        COND_EXE(r[w] = 0, lo > hi);
    }
}

//
//
//
// I).
// bi0 * bj0 < 0
// if (bi0 > 0) // bj0 < 0
//     bi0 * bj1 - bj0 * bi1
// if (bi0 < 0) // bj0 > 0
//     -bi0 * bj1 + bj0 * bi1
// RESUME : r = abs(bi0) * bj1 - abs(bj0) * bi1
//
// II).
// do-while (first iteration is always to be done).
//
// III).
// preprocessor predicate code.
//
//
//

#ifdef INTEL
__m512 z0 = SETZERO();
__m512 z1 = SETONE();
#endif

void
update_lo_hi_opt(bool m,
                 float f0,
                 float f1,
                 float *lo,
                 float *hi)
{
    float k;

    COND_EXE(*lo = *hi + 1.0, m && (f0 == 0.0) && (f1 > 0.0));
    COND_EXE(k = -f1 / f0, !(m && (f0 == 0.0)));
    COND_EXE(*hi = Utils::Min(*hi, k), m && (f0 > 0.0));
    COND_EXE(*lo = Utils::Max(*lo, k), m && (f0 < 0.0));
}

void
tri_box_intersects_opt_16(float * __restrict__ xa,
                          float * __restrict__ ya,
                          float * __restrict__ za,
                          float * __restrict__ xb,
                          float * __restrict__ yb,
                          float * __restrict__ zb,
                          float * __restrict__ xc,
                          float * __restrict__ yc,
                          float * __restrict__ zc,
                          float * __restrict__ xl,
                          float * __restrict__ xh,
                          float * __restrict__ yl,
                          float * __restrict__ yh,
                          float * __restrict__ zl,
                          float * __restrict__ zh,
                          int * __restrict__ r)
{
    const int basic_eqns_count = 8;
    float lo, hi;
    float b[basic_eqns_count][3][VEC_WIDTH];

    // Init.
    ST(&b[0][0][0], SUB(LD(xb), LD(xa)));
    ST(&b[0][1][0], SUB(LD(xc), LD(xa)));
    ST(&b[0][2][0], SUB(LD(xa), LD(xh)));
    ST(&b[1][0][0], SUB(LD(xa), LD(xb)));
    ST(&b[1][1][0], SUB(LD(xa), LD(xc)));
    ST(&b[1][2][0], SUB(LD(xl), LD(xa)));
    ST(&b[2][0][0], SUB(LD(yb), LD(ya)));
    ST(&b[2][1][0], SUB(LD(yc), LD(ya)));
    ST(&b[2][2][0], SUB(LD(ya), LD(yh)));
    ST(&b[3][0][0], SUB(LD(ya), LD(yb)));
    ST(&b[3][1][0], SUB(LD(ya), LD(yc)));
    ST(&b[3][2][0], SUB(LD(yl), LD(ya)));
    ST(&b[4][0][0], SUB(LD(zb), LD(za)));
    ST(&b[4][1][0], SUB(LD(zc), LD(za)));
    ST(&b[4][2][0], SUB(LD(za), LD(zh)));
    ST(&b[5][0][0], SUB(LD(za), LD(zb)));
    ST(&b[5][1][0], SUB(LD(za), LD(zc)));
    ST(&b[5][2][0], SUB(LD(zl), LD(za)));
    ST(&b[6][0][0], z1);
    ST(&b[6][1][0], z0);
    ST(&b[6][2][0], SET1(-1.0));
    ST(&b[7][0][0], SET1(-1.0));
    ST(&b[7][1][0], z0);
    ST(&b[7][2][0], z0);

    // Result init.
    for (int w = 0; w < VEC_WIDTH; w++)
    {
        r[w] = 1;
    }

    // Main loop.
    for (int w = 0; w < VEC_WIDTH; w++)
    {
        lo = 0.0;
        hi = 1.0;

        for (int i = 0; i < basic_eqns_count; i++)
        {
            update_lo_hi_opt(b[i][0][w] == 0.0,
                             b[i][1][w],
                             b[i][2][w],
                             &lo, &hi);

            if (lo > hi)
            {
                break;
            }
        }

        for (int i = 0; i < basic_eqns_count; i++)
        {
            float bi0 = b[i][0][w];
            float abi0 = fabs(bi0);

            for (int j = i + 1; j < basic_eqns_count; j++)
            {
                float bj0 = b[j][0][w];
                float abj0 = fabs(bj0);

                update_lo_hi_opt(bi0 * bj0 < 0.0,
                                 abi0 * b[j][1][w] + abj0 * b[i][1][w],
                                 abi0 * b[j][2][w] + abj0 * b[i][2][w],
                                 &lo, &hi);

                if (lo > hi)
                {
                    break;
                }
            }

            if (lo > hi)
            {
                break;
            }
        }

        COND_EXE(r[w] = 0, lo > hi);
    }
}

/// @brief Original function.
///
/// @param [in] ax-zh - Datas.
/// @param [in] c - Count.
/// @param [out] r - Results.
void
tri_box_intersects_orig(float * __restrict__ ax,
                        float * __restrict__ ay,
                        float * __restrict__ az,
                        float * __restrict__ bx,
                        float * __restrict__ by,
                        float * __restrict__ bz,
                        float * __restrict__ cx,
                        float * __restrict__ cy,
                        float * __restrict__ cz,
                        float * __restrict__ xl,
                        float * __restrict__ xh,
                        float * __restrict__ yl,
                        float * __restrict__ yh,
                        float * __restrict__ zl,
                        float * __restrict__ zh,
                        int c,
                        int * __restrict__ r)
{
    for (int i = 0; i < c; i += VEC_WIDTH)
    {
        tri_box_intersects_orig_16(&ax[i], &ay[i], &az[i],
                                   &bx[i], &by[i], &bz[i],
                                   &cx[i], &cy[i], &cz[i],
                                   &xl[i], &xh[i],
                                   &yl[i], &yh[i],
                                   &zl[i], &zh[i],
                                   &r[i]);
    }
}

/// @brief Optimized fucntion.
///
/// @param [in] ax-zh - Datas.
/// @param [in] c - Count.
/// @param [out] r - Results.
void
tri_box_intersects_opt(float * __restrict__ ax,
                       float * __restrict__ ay,
                       float * __restrict__ az,
                       float * __restrict__ bx,
                       float * __restrict__ by,
                       float * __restrict__ bz,
                       float * __restrict__ cx,
                       float * __restrict__ cy,
                       float * __restrict__ cz,
                       float * __restrict__ xl,
                       float * __restrict__ xh,
                       float * __restrict__ yl,
                       float * __restrict__ yh,
                       float * __restrict__ zl,
                       float * __restrict__ zh,
                       int c,
                       int * __restrict__ r)
{
    for (int i = 0; i < c; i += VEC_WIDTH)
    {
        tri_box_intersects_opt_16(&ax[i], &ay[i], &az[i],
                                   &bx[i], &by[i], &bz[i],
                                   &cx[i], &cy[i], &cz[i],
                                   &xl[i], &xh[i],
                                   &yl[i], &yh[i],
                                   &zl[i], &zh[i],
                                   &r[i]);
    }
}
