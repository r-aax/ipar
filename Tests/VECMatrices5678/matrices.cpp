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

//--------------------------------------------------------------------------------------------------

// Short names.
#define LD(ADDR) _mm512_load_ps(ADDR)
#define ST(ADDR, VAL) _mm512_store_ps(ADDR, VAL)
#define ADD(va, vb) _mm512_add_ps(va, vb)
#define MUL(va, vb) _mm512_mul_ps(va, vb)
#define SUB(va, vb) _mm512_sub_ps(va, vb)
#define DIV(va, vb) _mm512_div_ps(va, vb)
#define POW(va, vb) _mm512_pow_ps(va, vb)
#define SQRT(va) _mm512_sqrt_ps(va)
#define ABS(v) _mm512_abs_ps(v)
#define MAX(va, vb) _mm512_max_ps(va, vb)
#define MIN(va, vb) _mm512_min_ps(va, vb)
#define CMP(va, vb, cmp) _mm512_cmp_ps_mask(va, vb, cmp)
#define SETZERO() _mm512_setzero_ps()
#define SET1(v) _mm512_set1_ps(v)

//--------------------------------------------------------------------------------------------------

/// \brief Macro for 2 swiz + 2 add + 1 blend.
#define SWIZ_2_ADD_2_BLEND_1(X, Y, SWIZ_TYPE, BLEND_MASK) \
    _mm512_mask_blend_ps(BLEND_MASK, \
                         ADD(X, _mm512_swizzle_ps(X, SWIZ_TYPE)), \
                         ADD(Y, _mm512_swizzle_ps(Y, SWIZ_TYPE)))

/// \brief Macro for 2 perm + 2 add + 1 blend.
#define PERM_2_ADD_2_BLEND_1(X, Y, PERM_TYPE, BLEND_MASK) \
    _mm512_mask_blend_ps(BLEND_MASK, \
                         ADD(X, _mm512_permute4f128_ps(X, PERM_TYPE)), \
                         ADD(Y, _mm512_permute4f128_ps(Y, PERM_TYPE)))

/// \brief Macro for 2 swiz + 2 blend + 1 add.
#define SWIZ_2_BLEND_2_ADD_1(X, Y, SWIZ_TYPE, BLEND_MASK) \
    ADD(_mm512_mask_blend_ps(BLEND_MASK, X, Y), \
        _mm512_mask_blend_ps(BLEND_MASK, \
                             _mm512_swizzle_ps(X, SWIZ_TYPE), \
                             _mm512_swizzle_ps(Y, SWIZ_TYPE)))

/// \brief Macro for 2 perm + 2 blend + 1 add.
#define PERM_2_BLEND_2_ADD_1(X, Y, PERM_TYPE, BLEND_MASK) \
    ADD(_mm512_mask_blend_ps(BLEND_MASK, X, Y), \
        _mm512_mask_blend_ps(BLEND_MASK, \
                             _mm512_permute4f128_ps(X, PERM_TYPE), \
                             _mm512_permute4f128_ps(Y, PERM_TYPE)))

//--------------------------------------------------------------------------------------------------

/// \brief Multiplication of two 8*8 matrices.
///
/// Original version.
///
/// \param a - first matrix
/// \param b - second matrix
/// \param r - result matrix
void om_mult_mm_8x8_orig(float * __restrict a,
                         float * __restrict b,
                         float * __restrict r)
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
void om_mult_mm_8x8_opt(float * __restrict a,
                        float * __restrict b,
                        float * __restrict r)
{

#ifndef INTEL

    om_mult_mm_8x8_orig(a, b, r);

#else

    __assume_aligned(a, 64);
    __assume_aligned(b, 64);
    __assume_aligned(r, 64);

    int j;
    int ii0 = 0;
    int ii1 = 2 * V8;
    int ii2 = 4 * V8;
    int ii3 = 6 * V8;
    __m512 bj, bj2,
           m0, m1, m2, m3, m4, m5, m6, m7;

    // Index vector for reading the vector V8 twice.
    __m512i ind_vv = _mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0,
                                      7, 6, 5, 4, 3, 2, 1, 0);

    // Index vector for reading two adjacent matrix colemns.
    __m512i ind_cc = _mm512_set_epi32(7 * V8 + 1, 6 * V8 + 1, 5 * V8 + 1, 4 * V8 + 1,
                                      3 * V8 + 1, 2 * V8 + 1,     V8 + 1,          1,
                                      7 * V8    , 6 * V8    , 5 * V8    , 4 * V8    ,
                                      3 * V8    , 2 * V8    ,     V8    ,          0);
    // Index vector for result storing.
    __m512i ind_st = _mm512_set_epi32(7 * V8    , 7 * V8 + 1, 5 * V8    , 5 * V8 + 1,
                                      3 * V8    , 3 * V8 + 1,     V8    ,     V8 + 1,
                                      6 * V8 + 1, 6 * V8    , 4 * V8 + 1, 4 * V8    ,
                                      2 * V8 + 1, 2 * V8    ,     1     ,          0);

    // Load left matrix.
    __m512 a0 = LD(&a[ii0]);
    __m512 a1 = LD(&a[ii1]);
    __m512 a2 = LD(&a[ii2]);
    __m512 a3 = LD(&a[ii3]);

    // Loop for b matrix.
    for (j = 0; j < V8; j += 2)
    {
        bj = _mm512_i32gather_ps(ind_cc, &b[j], _MM_SCALE_4);
        bj2 = _mm512_permute4f128_ps(bj, _MM_PERM_BADC);

        m0 = MUL(a0, bj);
        m1 = MUL(a0, bj2);
        m2 = MUL(a1, bj);
        m3 = MUL(a1, bj2);
        m4 = MUL(a2, bj);
        m5 = MUL(a2, bj2);
        m6 = MUL(a3, bj);
        m7 = MUL(a3, bj2);

        m0 = SWIZ_2_ADD_2_BLEND_1(m0, m1, _MM_SWIZ_REG_CDAB, 0xAAAA);
        m1 = SWIZ_2_ADD_2_BLEND_1(m2, m3, _MM_SWIZ_REG_CDAB, 0xAAAA);
        m2 = SWIZ_2_ADD_2_BLEND_1(m4, m5, _MM_SWIZ_REG_CDAB, 0xAAAA);
        m3 = SWIZ_2_ADD_2_BLEND_1(m6, m7, _MM_SWIZ_REG_CDAB, 0xAAAA);
        m4 = SWIZ_2_ADD_2_BLEND_1(m0, m1, _MM_SWIZ_REG_BADC, 0xCCCC);
        m5 = SWIZ_2_ADD_2_BLEND_1(m2, m3, _MM_SWIZ_REG_BADC, 0xCCCC);
        m6 = PERM_2_ADD_2_BLEND_1(m4, m5, _MM_PERM_CDAB, 0xF0F0);

        _mm512_i32scatter_ps(&r[j], ind_st, m6, _MM_SCALE_4);
    }

#endif

}

//--------------------------------------------------------------------------------------------------
