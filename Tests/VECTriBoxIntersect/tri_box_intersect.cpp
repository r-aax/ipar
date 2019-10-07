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

using namespace std;

#ifdef INTEL
#include <immintrin.h>
#endif

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
    const int basic_eqns_count = 10;
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
    b[8][0] = 0.0;
    b[8][1] = 1.0;
    b[8][2] = -1.0;
    b[9][0] = 0.0;
    b[9][1] = -1.0;
    b[9][2] = 0.0;

    int n = 0;
    float f[18][2];

    // Выполнение свертки.
    for (int i = 0; i < basic_eqns_count; i++)
    {
        if (b[i][0] == 0.0)
        {
            // Нулевой коэффициент, неравенство переходит as is.
            f[n][0] = b[i][1];
            f[n][1] = b[i][2];
            n++;
        }
        else
        {
            // Коэффициент ненулевой.
            // Работаем с двумя неравенствами.
            for (int j = i + 1; j < basic_eqns_count; j++)
            {
                if (b[i][0] * b[j][0] < 0.0)
                {
                    // Нашли разнознаковые неравенства.
                    // Определяем, кто какого знака.

                    int p, q;

                    if (b[i][0] > 0.0)
                    {
                        p = i;
                        q = j;
                    }
                    else
                    {
                        p = j;
                        q = i;
                    }

                    // Сворачиваем два неравенства в одно.
                    f[n][0] = b[p][0] * b[q][1] - b[q][0] * b[p][1];
                    f[n][1] = b[p][0] * b[q][2] - b[q][0] * b[p][2];
                    n++;
                }
            }
        }
    }

    float lo = 0.0;
    float hi = 0.0;
    bool is_lo_init = false;
    bool is_hi_init = false;

    for (int i = 0; i < n; i++)
    {
        if (f[i][0] > 0.0)
        {
            float k = -f[i][1] / f[i][0];

            // Неравенство kx + v <= 0 (k > 0).
            // Верхняя граница.
            if (!is_hi_init || (k < hi))
            {
                hi = k;
                is_hi_init = true;
            }
        }
        else if (f[i][0] < 0.0)
        {
            float k = -f[i][1] / f[i][0];

            // Неравенство kx + v <= 0 (k < 0).
            // Нижняя граница.
            // kx <= -v => x >= -v / k
            if (!is_lo_init || k > lo)
            {
                lo = k;
                is_lo_init = true;
            }
        }
        else
        {
            // Нулевой коэффициент, проверяем тождество.
            if (f[i][1] > 0.0)
            {
                // Встретили неравенство positive_value <= 0.0.
                // Система неразрешима.
                return false;
            }
        }

        if (is_lo_init
            && is_hi_init
            && (hi < lo))
        {
            // Пересечение уже нулевое.
            return false;
        }
    }

    // Если не отвалились до сих пор, то решение есть.
    return true;
}

/// @brief Upgrade interval.
///
/// @return
/// <c>true</c>, if current solve interval is not zero,
/// <c>false</c>, otherwise.
bool
upgrade_solution(float *lo,
                 float *hi,
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
        if (b[i][0] == 0.0)
        {
            if (!upgrade_solution(&lo, &hi, b[i][1], b[i][2]))
            {
                return false;
            }
        }
        else
        {
            for (int j = i + 1; j < basic_eqns_count; j++)
            {
                if (b[i][0] * b[j][0] < 0.0)
                {
                    int p, q;

                    if (b[i][0] > 0.0)
                    {
                        p = i;
                        q = j;
                    }
                    else
                    {
                        p = j;
                        q = i;
                    }

                    if (!upgrade_solution(&lo, &hi,
                                          b[p][0] * b[q][1] - b[q][0] * b[p][1],
                                          b[p][0] * b[q][2] - b[q][0] * b[p][2]))
                    {
                        return false;
                    }
                }
            }
        }
    }

    return true;
}

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
        r[i] = tri_box_intersect_orig(ax[i], ay[i], az[i],
                                      bx[i], by[i], bz[i],
                                      cx[i], cy[i], cz[i],
                                      xl[i], xh[i],
                                      yl[i], yh[i],
                                      zl[i], zh[i]);
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
        r[i] = tri_box_intersect_opt(ax[i], ay[i], az[i],
                                      bx[i], by[i], bz[i],
                                      cx[i], cy[i], cz[i],
                                      xl[i], xh[i],
                                      yl[i], yh[i],
                                      zl[i], zh[i]);
    }
}
