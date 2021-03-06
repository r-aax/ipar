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

/// \brief Multiplication 8*8-matrix on 8-vector.
///
/// Original version.
///
/// \param m - matrix
/// \param v - vector
/// \param r - result
void matvec8_orig(float * __restrict m, float * __restrict v, float * __restrict r)
{
    for (int i = 0; i < V8; i++)
    {
        float sum = 0.0;
        int ii = i * V8;

        for (int j = 0; j < V8; j++)
        {
            sum = sum + m[ii + j] * v[j];
        }

        r[i] = sum;
    }
}

/// \brief Multiplication 8*8-matrix on 8-vector.
///
/// Optimized version.
///
/// \param m - matrix
/// \param v - vector
/// \param r - result 
void matvec8_opt(float * __restrict m, float * __restrict v, float * __restrict r)
{

#ifndef INTEL

    matvec8_orig(m, v, r);

#else

    __assume_aligned(m, 64);
    __assume_aligned(v, 64);
    __assume_aligned(r, 64);

    // Bad performance (2x slower than common case).
    //__m512 vec = _mm512_i32gather_ps(_mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0,
    //                                                  7, 6, 5, 4, 3, 2, 1, 0),
    //                                 v, _MM_SCALE_4);
    //__m512i ind = _mm512_set_epi32(0, 0, 0, 0, 7, 5, 3, 1,
    //                               0, 0, 0, 0, 6, 4, 2, 0);
    //__m512 ml, sh, ad;
    //
    //for (int i = 0; i < V8 / 2; i++)
    //{
    //    ml = _mm512_mul_ps(_mm512_load_ps(&m[i * 2 * V8]), vec);
    //    sh = _mm512_swizzle_ps(ml, _MM_SWIZ_REG_CDAB);
    //    ad = _mm512_add_ps(ml, sh);
    //    sh = _mm512_swizzle_ps(ad, _MM_SWIZ_REG_BADC);
    //    ml = _mm512_add_ps(ad, sh);
    //    sh = _mm512_permute4f128_ps(ml, _MM_PERM_CDAB);
    //    ad = _mm512_add_ps(ml, sh);
    //    _mm512_mask_i32scatter_ps(r, 0x101 << i, ind, ad, _MM_SCALE_4);
    //}

    // Bad performance (20% faster than common case).
    __m512 vec = _mm512_i32gather_ps(_mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0,
                                                      7, 6, 5, 4, 3, 2, 1, 0),
                                     v, _MM_SCALE_4);
    __m512 m0, m2, m4, m6, x0, x2, x4, x6;

    m0 = _mm512_mul_ps(_mm512_load_ps(&m[0]), vec);
    m2 = _mm512_mul_ps(_mm512_load_ps(&m[2 * V8]), vec);
    m4 = _mm512_mul_ps(_mm512_load_ps(&m[4 * V8]), vec);
    m6 = _mm512_mul_ps(_mm512_load_ps(&m[6 * V8]), vec);
    x0 = SWIZ_2_ADD_2_BLEND_1(m0, m2, _MM_SWIZ_REG_CDAB, 0xAAAA);
    x2 = SWIZ_2_ADD_2_BLEND_1(m4, m6, _MM_SWIZ_REG_CDAB, 0xAAAA);
    m0 = SWIZ_2_ADD_2_BLEND_1(x0, x2, _MM_SWIZ_REG_BADC, 0xCCCC);
    x0 = _mm512_add_ps(m0, _mm512_permute4f128_ps(m0, _MM_PERM_CDAB));
    _mm512_mask_i32scatter_ps(r, 0xF0F,
                              _mm512_set_epi32(0, 0, 0, 0, 7, 5, 3, 1,
                                               0, 0, 0, 0, 6, 4, 2, 0),
                              x0, _MM_SCALE_4);

    //__m512i mi, vi;
    //__m512 m0, m2, m4, m6, v0, v2, v4, v6;
    //
    //mi = _mm512_set_epi32(7 * V8 + 1, 6 * V8 + 1, 5 * V8 + 1, 4 * V8 + 1,
    //                      3 * V8 + 1, 2 * V8 + 1,     V8 + 1,          1,
    //                          7 * V8,     6 * V8,     5 * V8,     4 * V8,
    //                          3 * V8,     2 * V8,         V8,          0);
    //vi = _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0);
    //m0 = _mm512_i32gather_ps(mi, &m[0], _MM_SCALE_4);
    //m2 = _mm512_i32gather_ps(mi, &m[2], _MM_SCALE_4);
    //m4 = _mm512_i32gather_ps(mi, &m[4], _MM_SCALE_4);
    //m6 = _mm512_i32gather_ps(mi, &m[6], _MM_SCALE_4);
    //v0 = _mm512_i32gather_ps(vi, &v[0], _MM_SCALE_4);
    //v2 = _mm512_i32gather_ps(vi, &v[2], _MM_SCALE_4);
    //v4 = _mm512_i32gather_ps(vi, &v[4], _MM_SCALE_4);
    //v6 = _mm512_i32gather_ps(vi, &v[6], _MM_SCALE_4);
    //m0 = _mm512_mul_ps(m0, v0);
    //m2 = _mm512_mul_ps(m2, v2);
    //m4 = _mm512_mul_ps(m4, v4);
    //m6 = _mm512_mul_ps(m6, v6);
    //m0 = _mm512_add_ps(m0, m2);
    //m4 = _mm512_add_ps(m4, m6);
    //m0 = _mm512_add_ps(m0, m4);
    //m0 = _mm512_add_ps(m0, _mm512_permute4f128_ps(m0, _MM_PERM_BADC));
    //_mm512_store_ps(r, m0);

#endif

}

/// \brief Multiplication 8*8-matrix on 8-vector.
///
/// Optimized version.
///
/// \param m - matrix
/// \param v - vector
/// \param r - result 
void matvec8_opt2(float * __restrict m, float * __restrict v, float * __restrict r)
{

#ifndef INTEL

    matvec8_orig(m, v, r);

#else

    __declspec(align(64)) float t[V64];

    __assume_aligned(m, 64);
    __assume_aligned(v, 64);
    __assume_aligned(r, 64);
    __assume_aligned(&t[0], 64);

    __m512i ind = _mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0,
                                   7, 6, 5, 4, 3, 2, 1, 0);
    __m512 vec = _mm512_i32gather_ps(ind, v, _MM_SCALE_4);

    __m512 mi, mul;

    for (int i = 0; i < V8; i += 2)
    {
        int ii = i * V8;

        mi = _mm512_load_ps(&m[ii]);
        mul = _mm512_mul_ps(mi, vec);

        _mm512_store_ps(&t[ii], mul);
    }

    ind = _mm512_set_epi32(     0,      0,      0,      0,
                                0,      0,      0,      0,
                           7 * V8, 6 * V8, 5 * V8, 4 * V8,
                           3 * V8, 2 * V8,     V8,      0);
    __m512 res =_mm512_setzero_ps();
    __m512 ti;

    for (int i = 0; i < V8; i++)
    {
        ti = _mm512_mask_i32gather_ps(ti, 0xFF, ind, &t[i], _MM_SCALE_4);
        res = _mm512_mask_add_ps(res, 0xFF, res, ti);
    }

    _mm512_mask_store_ps(r, 0xFF, res);

#endif

}

/// \brief Multiplication 16*16-matrix on 16-vector.
///
/// Original version.
///
/// \param m - matrix
/// \param v - vector
/// \param r - result
void matvec16_orig(float * __restrict m, float * __restrict v, float * __restrict r)
{
    for (int i = 0; i < V16; i++)
    {
        float sum = 0.0;
        int ii = i * V16;

        for (int j = 0; j < V16; j++)
        {
            sum = sum + m[ii + j] * v[j];
        }

        r[i] = sum;
    }
}

/// \brief Multiplication 16*16-matrix on 16-vector.
///
/// Optimized version.
///
/// \param m - matrix
/// \param v - vector
/// \param r - result
void matvec16_opt(float * __restrict m, float * __restrict v, float * __restrict r)
{

#ifndef INTEL

    matvec16_orig(m, v, r);

#else

    __assume_aligned(m, 64);
    __assume_aligned(v, 64);
    __assume_aligned(r, 64);

    __m512 vec = _mm512_load_ps(v);
    __m512 m00, m01, m02, m03, m04, m05, m06, m07, m08, m09, m10, m11, m12, m13, m14, m15,
           x00, x01, x02, x03, x04, x05, x06, x07, x08, x09, x10, x11, x12, x13, x14, x15;

    // Fetch out 16x16 matrix.
    m00 = _mm512_mul_ps(_mm512_load_ps(&m[0]), vec);
    m01 = _mm512_mul_ps(_mm512_load_ps(&m[V16]), vec);
    m02 = _mm512_mul_ps(_mm512_load_ps(&m[2 * V16]), vec);
    m03 = _mm512_mul_ps(_mm512_load_ps(&m[3 * V16]), vec);
    m04 = _mm512_mul_ps(_mm512_load_ps(&m[4 * V16]), vec);
    m05 = _mm512_mul_ps(_mm512_load_ps(&m[5 * V16]), vec);
    m06 = _mm512_mul_ps(_mm512_load_ps(&m[6 * V16]), vec);
    m07 = _mm512_mul_ps(_mm512_load_ps(&m[7 * V16]), vec);
    m08 = _mm512_mul_ps(_mm512_load_ps(&m[8 * V16]), vec);
    m09 = _mm512_mul_ps(_mm512_load_ps(&m[9 * V16]), vec);
    m10 = _mm512_mul_ps(_mm512_load_ps(&m[10 * V16]), vec);
    m11 = _mm512_mul_ps(_mm512_load_ps(&m[11 * V16]), vec);
    m12 = _mm512_mul_ps(_mm512_load_ps(&m[12 * V16]), vec);
    m13 = _mm512_mul_ps(_mm512_load_ps(&m[13 * V16]), vec);
    m14 = _mm512_mul_ps(_mm512_load_ps(&m[14 * V16]), vec);
    m15 = _mm512_mul_ps(_mm512_load_ps(&m[15 * V16]), vec);

#if 1

    // This variant is faster (2 swiz + 2 add + 1 blend is better than
    //                         2 swiz + 2 blend + 1 add).

    // Stage 1.
    x00 = SWIZ_2_ADD_2_BLEND_1(m00, m01, _MM_SWIZ_REG_CDAB, 0xAAAA);
    x01 = SWIZ_2_ADD_2_BLEND_1(m02, m03, _MM_SWIZ_REG_CDAB, 0xAAAA);
    x02 = SWIZ_2_ADD_2_BLEND_1(m04, m05, _MM_SWIZ_REG_CDAB, 0xAAAA);
    x03 = SWIZ_2_ADD_2_BLEND_1(m06, m07, _MM_SWIZ_REG_CDAB, 0xAAAA);
    x04 = SWIZ_2_ADD_2_BLEND_1(m08, m09, _MM_SWIZ_REG_CDAB, 0xAAAA);
    x05 = SWIZ_2_ADD_2_BLEND_1(m10, m11, _MM_SWIZ_REG_CDAB, 0xAAAA);
    x06 = SWIZ_2_ADD_2_BLEND_1(m12, m13, _MM_SWIZ_REG_CDAB, 0xAAAA);
    x07 = SWIZ_2_ADD_2_BLEND_1(m14, m15, _MM_SWIZ_REG_CDAB, 0xAAAA);

    // Stage 2.
    m00 = SWIZ_2_ADD_2_BLEND_1(x00, x01, _MM_SWIZ_REG_BADC, 0xCCCC);
    m01 = SWIZ_2_ADD_2_BLEND_1(x02, x03, _MM_SWIZ_REG_BADC, 0xCCCC);
    m02 = SWIZ_2_ADD_2_BLEND_1(x04, x05, _MM_SWIZ_REG_BADC, 0xCCCC);
    m03 = SWIZ_2_ADD_2_BLEND_1(x06, x07, _MM_SWIZ_REG_BADC, 0xCCCC);

    // Stage 3.
    x00 = PERM_2_ADD_2_BLEND_1(m00, m01, _MM_PERM_CDAB, 0xF0F0);
    x01 = PERM_2_ADD_2_BLEND_1(m02, m03, _MM_PERM_CDAB, 0xF0F0);

    // Stage 4.
    m00 = PERM_2_ADD_2_BLEND_1(x00, x01, _MM_PERM_BADC, 0xFF00);

#else

    // Stage 1.
    x00 = SWIZ_2_BLEND_2_ADD_1(m00, m01, _MM_SWIZ_REG_CDAB, 0xAAAA);
    x01 = SWIZ_2_BLEND_2_ADD_1(m02, m03, _MM_SWIZ_REG_CDAB, 0xAAAA);
    x02 = SWIZ_2_BLEND_2_ADD_1(m04, m05, _MM_SWIZ_REG_CDAB, 0xAAAA);
    x03 = SWIZ_2_BLEND_2_ADD_1(m06, m07, _MM_SWIZ_REG_CDAB, 0xAAAA);
    x04 = SWIZ_2_BLEND_2_ADD_1(m08, m09, _MM_SWIZ_REG_CDAB, 0xAAAA);
    x05 = SWIZ_2_BLEND_2_ADD_1(m10, m11, _MM_SWIZ_REG_CDAB, 0xAAAA);
    x06 = SWIZ_2_BLEND_2_ADD_1(m12, m13, _MM_SWIZ_REG_CDAB, 0xAAAA);
    x07 = SWIZ_2_BLEND_2_ADD_1(m14, m15, _MM_SWIZ_REG_CDAB, 0xAAAA);

    // Stage 2.
    m00 = SWIZ_2_BLEND_2_ADD_1(x00, x01, _MM_SWIZ_REG_BADC, 0xCCCC);
    m01 = SWIZ_2_BLEND_2_ADD_1(x02, x03, _MM_SWIZ_REG_BADC, 0xCCCC);
    m02 = SWIZ_2_BLEND_2_ADD_1(x04, x05, _MM_SWIZ_REG_BADC, 0xCCCC);
    m03 = SWIZ_2_BLEND_2_ADD_1(x06, x07, _MM_SWIZ_REG_BADC, 0xCCCC);

    // Stage 3.
    x00 = PERM_2_BLEND_2_ADD_1(m00, m01, _MM_PERM_CDAB, 0xF0F0);
    x01 = PERM_2_BLEND_2_ADD_1(m02, m03, _MM_PERM_CDAB, 0xF0F0);

    // Stage 4.
    m00 = PERM_2_BLEND_2_ADD_1(x00, x01, _MM_PERM_BADC, 0xFF00);

#endif

    _mm512_store_ps(r, m00);

#endif

}

/// \brief Multiplication 16*16-matrix on 16-vector.
///
/// Optimized version.
///
/// \param m - matrix
/// \param v - vector
/// \param r - result
void matvec16_opt2(float * __restrict m, float * __restrict v, float * __restrict r)
{

#ifndef INTEL

    matvec16_orig(m, v, r);

#else

    __declspec(align(64)) float t[V256];

    __assume_aligned(m, 64);
    __assume_aligned(v, 64);
    __assume_aligned(r, 64);
    __assume_aligned(&t[0], 64);

    __m512 vec = _mm512_load_ps(v);
    __m512 mi, mul;

    for (int i = 0; i < V16; i++)
    {
        int ii = i * V16;

        mi = _mm512_load_ps(&m[ii]);
        mul = _mm512_mul_ps(mi, vec);
        _mm512_store_ps(&t[ii], mul);
    }

    __m512i ind = _mm512_set_epi32(15 * V16, 14 * V16, 13 * V16, 12 * V16,
                                   11 * V16, 10 * V16,  9 * V16,  8 * V16,
                                    7 * V16,  6 * V16,  5 * V16,  4 * V16,
                                    3 * V16,  2 * V16,      V16,        0);
    __m512 res = _mm512_setzero_ps();
    __m512 ti;

    for (int i = 0; i < V16; i++)
    {
        ti = _mm512_i32gather_ps(ind, &t[i], _MM_SCALE_4);
        res = _mm512_add_ps(res, ti);
    }

    _mm512_store_ps(r, res);

#endif

}

/// \brief Multiplication of two 8*8 matrices.
///
/// Original version.
///
/// \param a - first matrix
/// \param b - second matrix
/// \param r - result matrix
void matmat8_orig(float * __restrict a, float * __restrict b, float * __restrict r)
{
    for (int i = 0; i < V8; i++)
    {
        int ii = i * V8;

        for (int j = 0; j < V8; j++)
        {
            float sum = 0.0;

            for (int k = 0; k < V8; k++)
            {
                int kk = k * V8;

                sum = sum + a[ii + k] * b[kk + j];
            }

            r[ii + j] = sum;
        }
    }
}

/// \brief Multiplication of two 8*8 matrices.
///
/// Optimized version.
///
/// \param a - first matrix
/// \param b - second matrix
/// \param r - result matrix
void matmat8_opt(float * __restrict a, float * __restrict b, float * __restrict r)
{

#ifndef INTEL

    matmat8_orig(a, b, r);

#else

    __assume_aligned(a, 64);
    __assume_aligned(b, 64);
    __assume_aligned(r, 64);

    __m512 bj, bj2, a0, a1, a2, a3, m0, m1, m2, m3, m4, m5, m6, m7, x0, x1, x2, x3, x4, x5, x6, x7;

    __m512i ind = _mm512_set_epi32(7 * V8 + 1, 6 * V8 + 1, 5 * V8 + 1, 4 * V8 + 1,
                                   3 * V8 + 1, 2 * V8 + 1,     V8 + 1,          1,
                                       7 * V8,     6 * V8,     5 * V8,     4 * V8,
                                       3 * V8,     2 * V8,         V8,          0);

    // Loop for b matrix.
    for (int j = 0; j < V8; j += 2)
    {
        bj = _mm512_i32gather_ps(ind, &b[j], _MM_SCALE_4);
        bj2 = _mm512_permute4f128_ps(bj, _MM_PERM_BADC);

        // Bad performance.
        //for (int i = 0; i < V8; i += 2)
        //{
        //    int ii = i * V8;
        //
        //    a0 = _mm512_load_ps(&a[ii]);
        //    m0 = _mm512_mul_ps(a0, bj);
        //    m1 = _mm512_mul_ps(a0, bj2);
        //
        //    r[ii + j] = _mm512_mask_reduce_add_ps(0xFF, m0);
        //    r[ii + j + 1] = _mm512_mask_reduce_add_ps(0xFF, m1);
        //    r[ii + V8 + j] = _mm512_mask_reduce_add_ps(0xFF00, m1);
        //    r[ii + V8 + j + 1] = _mm512_mask_reduce_add_ps(0xFF00, m0);
        //}

        int ii0 = 0;
        int ii1 = 2 * V8;
        int ii2 = 4 * V8;
        int ii3 = 6 * V8;
        a0 = _mm512_load_ps(&a[ii0]);
        a1 = _mm512_load_ps(&a[ii1]);
        a2 = _mm512_load_ps(&a[ii2]);
        a3 = _mm512_load_ps(&a[ii3]);
        m0 = _mm512_mul_ps(a0, bj);
        m1 = _mm512_mul_ps(a0, bj2);
        m2 = _mm512_mul_ps(a1, bj);
        m3 = _mm512_mul_ps(a1, bj2);
        m4 = _mm512_mul_ps(a2, bj);
        m5 = _mm512_mul_ps(a2, bj2);
        m6 = _mm512_mul_ps(a3, bj);
        m7 = _mm512_mul_ps(a3, bj2);
        //
        x0 = SWIZ_2_ADD_2_BLEND_1(m0, m1, _MM_SWIZ_REG_CDAB, 0xAAAA);
        x1 = SWIZ_2_ADD_2_BLEND_1(m2, m3, _MM_SWIZ_REG_CDAB, 0xAAAA);
        x2 = SWIZ_2_ADD_2_BLEND_1(m4, m5, _MM_SWIZ_REG_CDAB, 0xAAAA);
        x3 = SWIZ_2_ADD_2_BLEND_1(m6, m7, _MM_SWIZ_REG_CDAB, 0xAAAA);
        m0 = SWIZ_2_ADD_2_BLEND_1(x0, x1, _MM_SWIZ_REG_BADC, 0xCCCC);
        m1 = SWIZ_2_ADD_2_BLEND_1(x2, x3, _MM_SWIZ_REG_BADC, 0xCCCC);
        x0 = PERM_2_ADD_2_BLEND_1(m0, m1, _MM_PERM_CDAB, 0xF0F0);
        //
        _mm512_i32scatter_ps(r,
                             _mm512_set_epi32(ii3 + V8 + j,
                                              ii3 + V8 + j + 1,
                                              ii2 + V8 + j,
                                              ii2 + V8 + j + 1,
                                              ii1 + V8 + j,
                                              ii1 + V8 + j + 1,
                                              ii0 + V8 + j,
                                              ii0 + V8 + j + 1,
                                              ii3 + j + 1,
                                              ii3 + j,
                                              ii2 + j + 1,
                                              ii2 + j,
                                              ii1 + j + 1,
                                              ii1 + j,
                                              ii0 + j + 1,
                                              ii0 + j),
                             x0, _MM_SCALE_4);
    }

#endif

}

/// \brief Multiplication of two 16*16 matrices.
///
/// Original version.
///
/// \param a - first matrix
/// \param b - second matrix
/// \param r - result matrix
void matmat16_orig(float * __restrict a, float * __restrict b, float * __restrict r)
{
    for (int i = 0; i < V16; i++)
    {
        int ii = i * V16;

        for (int j = 0; j < V16; j++)
        {
            float sum = 0.0;

            for (int k = 0; k < V16; k++)
            {
                int kk = k * V16;

                sum = sum + a[ii + k] * b[kk + j];
            }

            r[ii + j] = sum;
        }
    }
}

/// \brief Multiplication of two 16*16 matrices.
///
/// Optimized version.
///
/// \param a - first matrix
/// \param b - second matrix
/// \param r - result matrix
void matmat16_opt(float * __restrict a, float * __restrict b, float * __restrict r)
{

#ifndef INTEL

    matmat16_orig(a, b, r);

#else

    __assume_aligned(a, 64);
    __assume_aligned(b, 64);
    __assume_aligned(r, 64);

    __m512 bj, mul;
    __m512i ind = _mm512_set_epi32(15 * V16, 14 * V16, 13 * V16, 12 * V16,
                                   11 * V16, 10 * V16,  9 * V16,  8 * V16,
                                    7 * V16,  6 * V16,  5 * V16,  4 * V16,
                                    3 * V16,  2 * V16,      V16,        0);
    __m512 a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15,
           x00, x01, x02, x03, x04, x05, x06, x07, x08, x09, x10, x11, x12, x13, x14, x15;

    // Loop for b matrix.
    for (int j = 0; j < V16; j++)
    {
        bj = _mm512_i32gather_ps(ind, &b[j], _MM_SCALE_4);

        // Load and mul.
        a00 = _mm512_mul_ps(_mm512_load_ps(&a[0]), bj);
        a01 = _mm512_mul_ps(_mm512_load_ps(&a[V16]), bj);
        a02 = _mm512_mul_ps(_mm512_load_ps(&a[2 * V16]), bj);
        a03 = _mm512_mul_ps(_mm512_load_ps(&a[3 * V16]), bj);
        a04 = _mm512_mul_ps(_mm512_load_ps(&a[4 * V16]), bj);
        a05 = _mm512_mul_ps(_mm512_load_ps(&a[5 * V16]), bj);
        a06 = _mm512_mul_ps(_mm512_load_ps(&a[6 * V16]), bj);
        a07 = _mm512_mul_ps(_mm512_load_ps(&a[7 * V16]), bj);
        a08 = _mm512_mul_ps(_mm512_load_ps(&a[8 * V16]), bj);
        a09 = _mm512_mul_ps(_mm512_load_ps(&a[9 * V16]), bj);
        a10 = _mm512_mul_ps(_mm512_load_ps(&a[10 * V16]), bj);
        a11 = _mm512_mul_ps(_mm512_load_ps(&a[11 * V16]), bj);
        a12 = _mm512_mul_ps(_mm512_load_ps(&a[12 * V16]), bj);
        a13 = _mm512_mul_ps(_mm512_load_ps(&a[13 * V16]), bj);
        a14 = _mm512_mul_ps(_mm512_load_ps(&a[14 * V16]), bj);
        a15 = _mm512_mul_ps(_mm512_load_ps(&a[15 * V16]), bj);

        // Stage 1.
        x00 = SWIZ_2_ADD_2_BLEND_1(a00, a01, _MM_SWIZ_REG_CDAB, 0xAAAA);
        x01 = SWIZ_2_ADD_2_BLEND_1(a02, a03, _MM_SWIZ_REG_CDAB, 0xAAAA);
        x02 = SWIZ_2_ADD_2_BLEND_1(a04, a05, _MM_SWIZ_REG_CDAB, 0xAAAA);
        x03 = SWIZ_2_ADD_2_BLEND_1(a06, a07, _MM_SWIZ_REG_CDAB, 0xAAAA);
        x04 = SWIZ_2_ADD_2_BLEND_1(a08, a09, _MM_SWIZ_REG_CDAB, 0xAAAA);
        x05 = SWIZ_2_ADD_2_BLEND_1(a10, a11, _MM_SWIZ_REG_CDAB, 0xAAAA);
        x06 = SWIZ_2_ADD_2_BLEND_1(a12, a13, _MM_SWIZ_REG_CDAB, 0xAAAA);
        x07 = SWIZ_2_ADD_2_BLEND_1(a14, a15, _MM_SWIZ_REG_CDAB, 0xAAAA);

        // Stage 2.
        a00 = SWIZ_2_ADD_2_BLEND_1(x00, x01, _MM_SWIZ_REG_BADC, 0xCCCC);
        a01 = SWIZ_2_ADD_2_BLEND_1(x02, x03, _MM_SWIZ_REG_BADC, 0xCCCC);
        a02 = SWIZ_2_ADD_2_BLEND_1(x04, x05, _MM_SWIZ_REG_BADC, 0xCCCC);
        a03 = SWIZ_2_ADD_2_BLEND_1(x06, x07, _MM_SWIZ_REG_BADC, 0xCCCC);

        // Stage 3.
        x00 = PERM_2_ADD_2_BLEND_1(a00, a01, _MM_PERM_CDAB, 0xF0F0);
        x01 = PERM_2_ADD_2_BLEND_1(a02, a03, _MM_PERM_CDAB, 0xF0F0);

        // Stage 4.
        a00 = PERM_2_ADD_2_BLEND_1(x00, x01, _MM_PERM_BADC, 0xFF00);

        _mm512_i32scatter_ps(r,
                             _mm512_set_epi32(15 * V16 + j,
                                              14 * V16 + j,
                                              13 * V16 + j,
                                              12 * V16 + j,
                                              11 * V16 + j,
                                              10 * V16 + j,
                                               9 * V16 + j,
                                               8 * V16 + j,
                                               7 * V16 + j,
                                               6 * V16 + j,
                                               5 * V16 + j,
                                               4 * V16 + j,
                                               3 * V16 + j,
                                               2 * V16 + j,
                                                   V16 + j,
                                                         j),
                             a00, _MM_SCALE_4);
    }

#endif

}

/// \brief Invert 8*8 matrix.
///
/// Original version.
///
/// \param m - matrix
/// \param r - result matrix
///
/// \return
/// 0 - if the matrix is successfully inverted,
/// 1 - if error has occured.
int invmat8_orig(float * __restrict m, float * __restrict r)
{
    // Set E-matrix to r.
    for (int i = 0; i < V8; i++)
    {
        for (int j = 0; j < V8; j++)
        {
            r[i * V8 + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    for (int i = 0; i < V8; i++)
    {
        // For q < i, w < i we have
        // r[q, w] = 0, if q != w,
        // r[q, w] = 1, if q == w.

        // Find lead line from i to V8 - 1.
        int lead_i = i;
        for (int j = i + 1; j < V8; j++)
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
            for (int j = 0; j < V8; j++)
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
        for (int j = 0; j < V8; j++)
        {
            m[i * V8 + j] *= d;
            r[i * V8 + j] *= d;
        }

        // Zero all other lines.
        for (int j = 0; j < V8; j++)
        {
            if (j != i)
            {
                float t = m[j * V8 + i];

                for (int k = 0; k < V8; k++)
                {
                    m[j * V8 + k] -= m[i * V8 + k] * t;
                    r[j * V8 + k] -= r[i * V8 + k] * t;
                }
            }
        }
    }

    return 0;
}

/// \brief Invert 8*8 matrix.
///
/// Optimized version.
///
/// \param m - matrix
/// \param r - result matrix
///
/// \return
/// 0 - if the matrix is successfully inverted,
/// 1 - if error has occured.
int invmat8_opt(float * __restrict m, float * __restrict r)
{

#ifndef INTEL

    return invmat8_orig(m, r);

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

    for (int i = 0; i < V8; i += 2)
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

    for (int i = 0; i < V8; i++)
    {
        int ii = i * V16;

        // For q < i, w < i we have
        // r[q, w] = 0, if q != w,
        // r[q, w] = 1, if q == w.

        // Find lead line from i to V8 - 1.
        int lead_i = i;
        for (int j = i + 1; j < V8; j++)
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
        for (int j = 0; j < V8; j++)
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
    for (int i = 0; i < V8; i += 2)
    {
        vi = _mm512_i32gather_ps(ind, &t[i * V16 + V8], _MM_SCALE_4);
        _mm512_store_ps(&r[i * V8], vi);
    }

    return 0;

#endif

}

/// \brief Invert 16*16 matrix.
///
/// Original version.
///
/// \param m - matrix
/// \param r - result matrix
///
/// \return
/// 0 - if the matrix is successfully inverted,
/// 1 - if error has occured.
int invmat16_orig(float * __restrict m, float * __restrict r)
{
    // Set E-matrix to r.
    for (int i = 0; i < V16; i++)
    {
        for (int j = 0; j < V16; j++)
        {
            r[i * V16 + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    for (int i = 0; i < V16; i++)
    {
        // For q < i, w < i we have
        // r[q, w] = 0, if q != w,
        // r[q, w] = 1, if q == w.

        // Find lead line from i to V8 - 1.
        int lead_i = i;
        for (int j = i + 1; j < V16; j++)
        {
            if (fabs(m[j * V16 + i]) > fabs(m[lead_i * V16 + i]))
            {
                lead_i = j;
            }
        }
        if (fabs(m[lead_i * V16 + i]) < MATHS_EPS)
        {
            return 1;
        }

        // Interchange i-th and lead_i-th lines.
        if (lead_i != i)
        {
            for (int j = 0; j < V16; j++)
            {
                float tmp_m = m[lead_i * V16 + j];
                m[lead_i * V16 + j] = m[i * V16 + j];
                m[i * V16 + j] = tmp_m;

                float tmp_r = r[lead_i * V16 + j];
                r[lead_i * V16 + j] = r[i * V16 + j];
                r[i * V16 + j] = tmp_r;
            }
        }

        // Scale i-th line.
        float d = 1.0 / m[i * V16 + i];
        for (int j = 0; j < V16; j++)
        {
            m[i * V16 + j] *= d;
            r[i * V16 + j] *= d;
        }

        // Zero all other lines.
        for (int j = 0; j < V16; j++)
        {
            if (j != i)
            {
                float t = m[j * V16 + i];

                for (int k = 0; k < V16; k++)
                {
                    m[j * V16 + k] -= m[i * V16 + k] * t;
                    r[j * V16 + k] -= r[i * V16 + k] * t;
                }
            }
        }
    }

    return 0;
}

/// \brief Invert 16*16 matrix.
///
/// Optimized version.
///
/// \param m - matrix
/// \param r - result matrix
///
/// \return
/// 0 - if the matrix is successfully inverted,
/// 1 - if error has occured.
int invmat16_opt(float * __restrict m, float * __restrict r)
{

#ifndef INTEL

    return invmat16_orig(m, r);

#else

    __assume_aligned(m, 64);
    __assume_aligned(r, 64);

    // Set E-matrix to r.
    // First we zero all elements of matrix,
    // then set diagonal elements to 1.0.

    __m512 vd, vi, vj, vl;

    vd = _mm512_setzero_ps();

    for (int i = 0; i < V16; i++)
    {
        int ii = i * V16;

        _mm512_store_ps(&r[ii], vd);
    }

    int d = V16 + 1;
    __m512i ind = _mm512_set_epi32(15 * d, 14 * d, 13 * d, 12 * d,
                                   11 * d, 10 * d,  9 * d,  8 * d,
                                    7 * d,  6 * d,  5 * d,  4 * d,
                                    3 * d,  2 * d,      d,      0);

    _mm512_i32scatter_ps(r, ind, _mm512_set1_ps(1.0), _MM_SCALE_4);

    for (int i = 0; i < V16; i++)
    {
        int ii = i * V16;

        // For q < i, w < i we have
        // m[q, w] = 0, if q != w,
        // m[q, w] = 1, if q == w.

        // Find lead line from i to V8 - 1.
        int lead_i = i;
        for (int j = i + 1; j < V16; j++)
        {
            if (fabs(m[j * V16 + i]) > fabs(m[lead_i * V16 + i]))
            {
                lead_i = j;
            }
        }
        if (fabs(m[lead_i * V16 + i]) < MATHS_EPS)
        {
            return 1;
        }

        // Interchange i-th and lead_i-th lines.
        if (lead_i != i)
        {
            int ll = lead_i * V16;

            vi = _mm512_load_ps(&m[ii]);
            vl = _mm512_load_ps(&m[ll]);
            _mm512_store_ps(&m[ll], vi);
            _mm512_store_ps(&m[ii], vl);

            vi = _mm512_load_ps(&r[ii]);
            vl = _mm512_load_ps(&r[ll]);
            _mm512_store_ps(&r[ll], vi);
            _mm512_store_ps(&r[ii], vl);
        }

        // Scale i-th line.
        vd = _mm512_set1_ps(1.0 / m[ii + i]);
        vi = _mm512_load_ps(&m[ii]);
        vi = _mm512_mul_ps(vi, vd);
        _mm512_store_ps(&m[ii], vi);
        vi = _mm512_load_ps(&r[ii]);
        vi = _mm512_mul_ps(vi, vd);
        _mm512_store_ps(&r[ii], vi);

        // Zero all other lines.
        for (int j = 0; j < V16; j++)
        {
            if (j != i)
            {
                vd = _mm512_set1_ps(-m[j * V16 + i]);
                vj = _mm512_load_ps(&m[j * V16]);
                vi = _mm512_load_ps(&m[ii]);
                vj = _mm512_fmadd_ps(vi, vd, vj);
                _mm512_store_ps(&m[j * V16], vj);
                vj = _mm512_load_ps(&r[j * V16]);
                vi = _mm512_load_ps(&r[ii]);
                vj = _mm512_fmadd_ps(vi, vd, vj);
                _mm512_store_ps(&r[j * V16], vj);
            }
        }
    }

    return 0;

#endif

}
