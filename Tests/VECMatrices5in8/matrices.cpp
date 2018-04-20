/// \file
/// \brief Matrix operations implementations.

#include "matrices.h"
#include "../../Utils/Maths.h"
#include <stdlib.h>
#include <math.h>
#include "avx512debug.h"

#ifdef INTEL
#include <immintrin.h>
#endif

/// \brief Macro for 2 swiz + 2 add + 1 blend.
#define SWIZ_2_ADD_2_BLEND_1(X, Y, SWIZ_TYPE, BLEND_MASK) \
    _mm512_mask_blend_ps(BLEND_MASK, \
                         _mm512_add_ps(X, _mm512_swizzle_ps(X, SWIZ_TYPE)), \
                         _mm512_add_ps(Y, _mm512_swizzle_ps(Y, SWIZ_TYPE)))

/// \brief Macro for 2 perm + 2 add + 1 blend.
#define PERM_2_ADD_2_BLEND_1(X, Y, PERM_TYPE, BLEND_MASK) \
    _mm512_mask_blend_ps(BLEND_MASK, \
                         _mm512_add_ps(X, _mm512_permute4f128_ps(X, PERM_TYPE)), \
                         _mm512_add_ps(Y, _mm512_permute4f128_ps(Y, PERM_TYPE)))

/// \brief Macro for 2 swiz + 2 blend + 1 add.
#define SWIZ_2_BLEND_2_ADD_1(X, Y, SWIZ_TYPE, BLEND_MASK) \
    _mm512_add_ps(_mm512_mask_blend_ps(BLEND_MASK, X, Y), \
                  _mm512_mask_blend_ps(BLEND_MASK, \
                                       _mm512_swizzle_ps(X, SWIZ_TYPE), \
                                       _mm512_swizzle_ps(Y, SWIZ_TYPE)))

/// \brief Macro for 2 perm + 2 blend + 1 add.
#define PERM_2_BLEND_2_ADD_1(X, Y, PERM_TYPE, BLEND_MASK) \
    _mm512_add_ps(_mm512_mask_blend_ps(BLEND_MASK, X, Y), \
                  _mm512_mask_blend_ps(BLEND_MASK, \
                                       _mm512_permute4f128_ps(X, PERM_TYPE), \
                                       _mm512_permute4f128_ps(Y, PERM_TYPE)))

/// \brief Multiplication 5*5-matrix (in 8*8 memory) on 5-vector (in 8 memory).
///
/// Original version.
///
/// \param m - matrix
/// \param v - vector
/// \param r - result
void matvec5in8_orig(float * __restrict m, float * __restrict v, float * __restrict r)
{
    for (int i = 0; i < V5; i++)
    {
        float sum = 0.0;
        int ii = i * V8;

        for (int j = 0; j < V5; j++)
        {
            sum = sum + m[ii + j] * v[j];
        }

        r[i] = sum;
    }
}

/// \brief Multiplication 5*5-matrix (in 8*8 memory) on 8-vector (in 5 memory).
///
/// Optimized version.
///
/// \param m - matrix
/// \param v - vector
/// \param r - result 
void matvec5in8_opt(float * __restrict m, float * __restrict v, float * __restrict r)
{

#ifndef INTEL

    matvec5in8_orig(m, v, r);

#else

    __assume_aligned(m, 64);
    __assume_aligned(v, 64);
    __assume_aligned(r, 64);

    __m512 z = _mm512_setzero_ps();
    __m512 vec = _mm512_mask_i32gather_ps(z, 0x1F1F,
                                          _mm512_set_epi32(0, 0, 0, 4, 3, 2, 1, 0,
                                                           0, 0, 0, 4, 3, 2, 1, 0),
                                          v, _MM_SCALE_4);
    __m512 m0, m2, m4, x0, x2, x4;

    m0 = _mm512_mul_ps(_mm512_load_ps(&m[0]), vec);
    m2 = _mm512_mul_ps(_mm512_load_ps(&m[2 * V8]), vec);
    m4 = _mm512_mul_ps(_mm512_load_ps(&m[4 * V8]), vec);
    x0 = SWIZ_2_ADD_2_BLEND_1(m0, m2, _MM_SWIZ_REG_CDAB, 0xAAAA);
    x2 = SWIZ_2_ADD_2_BLEND_1(m4, z, _MM_SWIZ_REG_CDAB, 0xAAAA);
    m0 = SWIZ_2_ADD_2_BLEND_1(x0, x2, _MM_SWIZ_REG_BADC, 0xCCCC);
    x0 = _mm512_add_ps(m0, _mm512_permute4f128_ps(m0, _MM_PERM_CDAB));
    _mm512_mask_i32scatter_ps(r, 0x307,
                              _mm512_set_epi32(0, 0, 0, 0, 0, 0, 3, 1,
                                               0, 0, 0, 0, 0, 4, 2, 0),
                              x0, _MM_SCALE_4);

#endif

}

/// \brief Multiplication of two 5*5 matrices (in 8*8 memory).
///
/// Original version.
///
/// \param a - first matrix
/// \param b - second matrix
/// \param r - result matrix
void matmat5in8_orig(float * __restrict a, float * __restrict b, float * __restrict r)
{
    for (int i = 0; i < V5; i++)
    {
        int ii = i * V8;

        for (int j = 0; j < V5; j++)
        {
            float sum = 0.0;

            for (int k = 0; k < V5; k++)
            {
                int kk = k * V8;

                sum = sum + a[ii + k] * b[kk + j];
            }

            r[ii + j] = sum;
        }
    }
}

/// \brief Multiplication of two 5*5 matrices (in 8*8 memory).
///
/// Optimized version.
///
/// \param a - first matrix
/// \param b - second matrix
/// \param r - result matrix
void matmat5in8_opt(float * __restrict a, float * __restrict b, float * __restrict r)
{

#ifndef INTEL

    matmat5in8_orig(a, b, r);

#else

    __assume_aligned(a, 64);
    __assume_aligned(b, 64);
    __assume_aligned(r, 64);

    __m512 bj, bj2, a0, a1, a2, m0, m1, m2, m3, m4, m5, x0, x1, x2;

    __m512 z = _mm512_setzero_ps();
    __m512i ind = _mm512_set_epi32(0, 0, 0, 4 * V8 + 1, 3 * V8 + 1, 2 * V8 + 1, V8 + 1, 1,
                                   0, 0, 0,     4 * V8,     3 * V8,     2 * V8,     V8, 0);

    // Loop for b matrix.
    for (int j = 0; j < V8 - 2; j += 2)
    {
        bj = _mm512_mask_i32gather_ps(z, 0x1F1F, ind, &b[j], _MM_SCALE_4);
        bj2 = _mm512_permute4f128_ps(bj, _MM_PERM_BADC);

        int ii0 = 0;
        int ii1 = 2 * V8;
        int ii2 = 4 * V8;
        a0 = _mm512_mask_load_ps(z, 0x1F1F, &a[ii0]);
        a1 = _mm512_mask_load_ps(z, 0x1F1F, &a[ii1]);
        a2 = _mm512_mask_load_ps(z, 0x1F1F, &a[ii2]);
        m0 = _mm512_mul_ps(a0, bj);
        m1 = _mm512_mul_ps(a0, bj2);
        m2 = _mm512_mul_ps(a1, bj);
        m3 = _mm512_mul_ps(a1, bj2);
        m4 = _mm512_mul_ps(a2, bj);
        m5 = _mm512_mul_ps(a2, bj2);
        //
        x0 = SWIZ_2_ADD_2_BLEND_1(m0, m1, _MM_SWIZ_REG_CDAB, 0xAAAA);
        x1 = SWIZ_2_ADD_2_BLEND_1(m2, m3, _MM_SWIZ_REG_CDAB, 0xAAAA);
        x2 = SWIZ_2_ADD_2_BLEND_1(m4, m5, _MM_SWIZ_REG_CDAB, 0xAAAA);
        m0 = SWIZ_2_ADD_2_BLEND_1(x0, x1, _MM_SWIZ_REG_BADC, 0xCCCC);
        m1 = SWIZ_2_ADD_2_BLEND_1(x2, z, _MM_SWIZ_REG_BADC, 0xCCCC);
        x0 = PERM_2_ADD_2_BLEND_1(m0, m1, _MM_PERM_CDAB, 0xF0F0);
        //
        if (j < 4)
        {
            _mm512_mask_i32scatter_ps(r, 0x1F1F,
                                      _mm512_set_epi32(0, 0, 0,
                                                       ii2 + V8 + j + 1,
                                                       ii1 + V8 + j,
                                                       ii1 + V8 + j + 1,
                                                       ii0 + V8 + j,
                                                       ii0 + V8 + j + 1,
                                                       0, 0, 0,
                                                       ii2 + j,
                                                       ii1 + j + 1,
                                                       ii1 + j,
                                                       ii0 + j + 1,
                                                       ii0 + j),
                                      x0, _MM_SCALE_4);
        }
        else
        {
            _mm512_mask_i32scatter_ps(r, 0x1F,
                                      _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0,
                                                       0, 0, 0,
                                                       ii2 + j,
                                                       ii1 + j + 1,
                                                       ii1 + j,
                                                       ii0 + j + 1,
                                                       ii0 + j),
                                      x0, _MM_SCALE_4);
        }
    }

#endif

}

/// \brief Invert 5*5 matrix (in 8*8 memory).
///
/// Original version.
///
/// \param m - matrix
/// \param r - result matrix
///
/// \return
/// 0 - if the matrix is successfully inverted,
/// 1 - if error has occured.
int invmat5in8_orig(float * __restrict m, float * __restrict r)
{
    // Set E-matrix to r.
    for (int i = 0; i < V5; i++)
    {
        for (int j = 0; j < V5; j++)
        {
            r[i * V8 + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    for (int i = 0; i < V5; i++)
    {
        // For q < i, w < i we have
        // r[q, w] = 0, if q != w,
        // r[q, w] = 1, if q == w.

        // Find lead line from i to V5 - 1.
        int lead_i = i;
        for (int j = i + 1; j < V5; j++)
        {
            if (fabs(m[j * V8 + i]) > fabs(m[lead_i * V8 + i]))
            {
                lead_i = j;
            }
        }
        if (fabs(m[lead_i * V8 + i]) < MATHS_EPS)
        {
            return 1;
        }

        // Interchange i-th and lead_i-th lines.
        if (lead_i != i)
        {
            for (int j = 0; j < V5; j++)
            {
                float tmp_m = m[lead_i * V8 + j];
                m[lead_i * V8 + j] = m[i * V8 + j];
                m[i * V8 + j] = tmp_m;

                float tmp_r = r[lead_i * V8 + j];
                r[lead_i * V8 + j] = r[i * V8 + j];
                r[i * V8 + j] = tmp_r;
            }
        }

        // Scale i-th line.
        float d = 1.0 / m[i * V8 + i];
        for (int j = 0; j < V5; j++)
        {
            m[i * V8 + j] *= d;
            r[i * V8 + j] *= d;
        }

        // Zero all other lines.
        for (int j = 0; j < V5; j++)
        {
            if (j != i)
            {
                float t = m[j * V8 + i];

                for (int k = 0; k < V5; k++)
                {
                    m[j * V8 + k] -= m[i * V8 + k] * t;
                    r[j * V8 + k] -= r[i * V8 + k] * t;
                }
            }
        }
    }

    return 0;
}

/// \brief Invert 5*5 matrix (int 8*8 memory).
///
/// Optimized version.
///
/// \param m - matrix
/// \param r - result matrix
///
/// \return
/// 0 - if the matrix is successfully inverted,
/// 1 - if error has occured.
int invmat5in8_opt(float * __restrict m, float * __restrict r)
{

#ifndef INTEL

    return invmat5in8_orig(m, r);

#else

    __declspec(align(64)) float t[2 * V64];

    __assume_aligned(m, 64);
    __assume_aligned(r, 64);
    __assume_aligned(&t[0], 64);

    __m512 vd, vi, vj, vl;

    __m512i ind = _mm512_set_epi32(V16 + 7, V16 + 6, V16 + 5, V16 + 4,
                                   V16 + 3, V16 + 2, V16 + 1,     V16,
                                         7,       6,       5,       4,
                                         3,       2,       1,       0);

    // Copy m to tmp.

    vd = _mm512_setzero_ps();

    for (int i = 0; i < V8 - 2; i += 2)
    {
        vi = _mm512_load_ps(&m[i * V8]);
        _mm512_i32scatter_ps(&t[i * V16], ind, vi, _MM_SCALE_4);
        _mm512_i32scatter_ps(&t[i * V16 + V8], ind, vd, _MM_SCALE_4);
    }

    __m512i ind1 = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0,
                                    7 * V16 + 7,
                                    6 * V16 + 6,
                                    5 * V16 + 5,
                                    4 * V16 + 4,
                                    3 * V16 + 3,
                                    2 * V16 + 2,
                                        V16 + 1,
                                              0);
    _mm512_mask_i32scatter_ps(&t[V8], 0xFF, ind1, _mm512_set1_ps(1.0), _MM_SCALE_4);

    for (int i = 0; i < V5; i++)
    {
        int ii = i * V16;

        // For q < i, w < i we have
        // r[q, w] = 0, if q != w,
        // r[q, w] = 1, if q == w.

        // Find lead line from i to V8 - 1.
        int lead_i = i;
        for (int j = i + 1; j < V5; j++)
        {
            if (fabs(t[j * V16 + i]) > fabs(t[lead_i * V16 + i]))
            {
                lead_i = j;
            }
        }
        if (fabs(t[lead_i * V16 + i]) < MATHS_EPS)
        {
            return 1;
        }

        // Interchange i-th and lead_i-th lines.
        if (lead_i != i)
        {
            int ll = lead_i * V16;

            vi = _mm512_load_ps(&t[ii]);
            vl = _mm512_load_ps(&t[ll]);
            _mm512_store_ps(&t[ll], vi);
            _mm512_store_ps(&t[ii], vl);
        }

        // Scale i-th line.
        vd = _mm512_set1_ps(1.0 / t[ii + i]);
        vi = _mm512_load_ps(&t[ii]);
        vi = _mm512_mul_ps(vi, vd);
        _mm512_store_ps(&t[ii], vi);

        // Zero all other lines.
        for (int j = 0; j < V5; j++)
        {
            int jj = j * V16;

            if (j != i)
            {
                vd = _mm512_set1_ps(-t[jj + i]);
                vj = _mm512_load_ps(&t[jj]);
                vi = _mm512_load_ps(&t[ii]);
                vj = _mm512_fmadd_ps(vi, vd, vj);
                _mm512_store_ps(&t[jj], vj);
            }
        }
    }

    // Copy to r.
    for (int i = 0; i < V8 - 2; i += 2)
    {
        vi = _mm512_i32gather_ps(ind, &t[i * V16 + V8], _MM_SCALE_4);
        _mm512_store_ps(&r[i * V8], vi);
    }

    return 0;

#endif

}
