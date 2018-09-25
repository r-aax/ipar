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

// Modes.
#define OLD_VEC     0
#define FULL_UNROLL 0
#define NEW_VEC     1

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
#define FMADD(v1, v2, v3) _mm512_fmadd_ps(v1, v2, v3)
#define PERMXV(ind, v) _mm512_permutexvar_ps(ind, v)

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

#define ADR(I, J) ((I) * V8 + (J))

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

#if OLD_VEC == 1

    // Old version of vectorization with gather/scatter.
    // sp 2.59

    __assume_aligned(a, 64);
    __assume_aligned(b, 64);
    __assume_aligned(r, 64);

    __m512 bj, bj2,
           m0, m1, m2, m3, m4, m5, m6, m7;

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
    __m512 a0 = LD(&a[0]);
    __m512 a1 = LD(&a[2 * V8]);
    __m512 a2 = LD(&a[4 * V8]);
    __m512 a3 = LD(&a[6 * V8]);

    // Loop for b matrix.
    for (int j = 0; j < V8; j += 2)
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
        m0 = SWIZ_2_ADD_2_BLEND_1(m0, m1, _MM_SWIZ_REG_BADC, 0xCCCC);
        m1 = SWIZ_2_ADD_2_BLEND_1(m2, m3, _MM_SWIZ_REG_BADC, 0xCCCC);
        m2 = PERM_2_ADD_2_BLEND_1(m0, m1, _MM_PERM_CDAB, 0xF0F0);

        _mm512_i32scatter_ps(&r[j], ind_st, m2, _MM_SCALE_4);
    }

#endif

#if FULL_UNROLL

    // Full unroll.
    // sp 3.71

    __assume_aligned(a, 64);
    __assume_aligned(b, 64);
    __assume_aligned(r, 64);


    r[ADR(0, 0)] = a[ADR(0,0)]*b[ADR(0,0)]+a[ADR(0,1)]*b[ADR(1,0)]+a[ADR(0,2)]*b[ADR(2,0)]+a[ADR(0,3)]*b[ADR(3,0)]
                 + a[ADR(0,4)]*b[ADR(4,0)]+a[ADR(0,5)]*b[ADR(5,0)]+a[ADR(0,6)]*b[ADR(6,0)]+a[ADR(0,7)]*b[ADR(7,0)];
    r[ADR(0, 1)] = a[ADR(0,0)]*b[ADR(0,1)]+a[ADR(0,1)]*b[ADR(1,1)]+a[ADR(0,2)]*b[ADR(2,1)]+a[ADR(0,3)]*b[ADR(3,1)]
                 + a[ADR(0,4)]*b[ADR(4,1)]+a[ADR(0,5)]*b[ADR(5,1)]+a[ADR(0,6)]*b[ADR(6,1)]+a[ADR(0,7)]*b[ADR(7,1)];
    r[ADR(0, 2)] = a[ADR(0,0)]*b[ADR(0,2)]+a[ADR(0,1)]*b[ADR(1,2)]+a[ADR(0,2)]*b[ADR(2,2)]+a[ADR(0,3)]*b[ADR(3,2)]
                 + a[ADR(0,4)]*b[ADR(4,2)]+a[ADR(0,5)]*b[ADR(5,2)]+a[ADR(0,6)]*b[ADR(6,2)]+a[ADR(0,7)]*b[ADR(7,2)];
    r[ADR(0, 3)] = a[ADR(0,0)]*b[ADR(0,3)]+a[ADR(0,1)]*b[ADR(1,3)]+a[ADR(0,2)]*b[ADR(2,3)]+a[ADR(0,3)]*b[ADR(3,3)]
                 + a[ADR(0,4)]*b[ADR(4,3)]+a[ADR(0,5)]*b[ADR(5,3)]+a[ADR(0,6)]*b[ADR(6,3)]+a[ADR(0,7)]*b[ADR(7,3)];
    r[ADR(0, 4)] = a[ADR(0,0)]*b[ADR(0,4)]+a[ADR(0,1)]*b[ADR(1,4)]+a[ADR(0,2)]*b[ADR(2,4)]+a[ADR(0,3)]*b[ADR(3,4)]
                 + a[ADR(0,4)]*b[ADR(4,4)]+a[ADR(0,5)]*b[ADR(5,4)]+a[ADR(0,6)]*b[ADR(6,4)]+a[ADR(0,7)]*b[ADR(7,4)];
    r[ADR(0, 5)] = a[ADR(0,0)]*b[ADR(0,5)]+a[ADR(0,1)]*b[ADR(1,5)]+a[ADR(0,2)]*b[ADR(2,5)]+a[ADR(0,3)]*b[ADR(3,5)]
                 + a[ADR(0,4)]*b[ADR(4,5)]+a[ADR(0,5)]*b[ADR(5,5)]+a[ADR(0,6)]*b[ADR(6,5)]+a[ADR(0,7)]*b[ADR(7,5)];
    r[ADR(0, 6)] = a[ADR(0,0)]*b[ADR(0,6)]+a[ADR(0,1)]*b[ADR(1,6)]+a[ADR(0,2)]*b[ADR(2,6)]+a[ADR(0,3)]*b[ADR(3,6)]
                 + a[ADR(0,4)]*b[ADR(4,6)]+a[ADR(0,5)]*b[ADR(5,6)]+a[ADR(0,6)]*b[ADR(6,6)]+a[ADR(0,7)]*b[ADR(7,6)];
    r[ADR(0, 7)] = a[ADR(0,0)]*b[ADR(0,7)]+a[ADR(0,1)]*b[ADR(1,7)]+a[ADR(0,2)]*b[ADR(2,7)]+a[ADR(0,3)]*b[ADR(3,7)]
                 + a[ADR(0,4)]*b[ADR(4,7)]+a[ADR(0,5)]*b[ADR(5,7)]+a[ADR(0,6)]*b[ADR(6,7)]+a[ADR(0,7)]*b[ADR(7,7)];

    r[ADR(1, 0)] = a[ADR(1,0)]*b[ADR(0,0)]+a[ADR(1,1)]*b[ADR(1,0)]+a[ADR(1,2)]*b[ADR(2,0)]+a[ADR(1,3)]*b[ADR(3,0)]
                 + a[ADR(1,4)]*b[ADR(4,0)]+a[ADR(1,5)]*b[ADR(5,0)]+a[ADR(1,6)]*b[ADR(6,0)]+a[ADR(1,7)]*b[ADR(7,0)];
    r[ADR(1, 1)] = a[ADR(1,0)]*b[ADR(0,1)]+a[ADR(1,1)]*b[ADR(1,1)]+a[ADR(1,2)]*b[ADR(2,1)]+a[ADR(1,3)]*b[ADR(3,1)]
                 + a[ADR(1,4)]*b[ADR(4,1)]+a[ADR(1,5)]*b[ADR(5,1)]+a[ADR(1,6)]*b[ADR(6,1)]+a[ADR(1,7)]*b[ADR(7,1)];
    r[ADR(1, 2)] = a[ADR(1,0)]*b[ADR(0,2)]+a[ADR(1,1)]*b[ADR(1,2)]+a[ADR(1,2)]*b[ADR(2,2)]+a[ADR(1,3)]*b[ADR(3,2)]
                 + a[ADR(1,4)]*b[ADR(4,2)]+a[ADR(1,5)]*b[ADR(5,2)]+a[ADR(1,6)]*b[ADR(6,2)]+a[ADR(1,7)]*b[ADR(7,2)];
    r[ADR(1, 3)] = a[ADR(1,0)]*b[ADR(0,3)]+a[ADR(1,1)]*b[ADR(1,3)]+a[ADR(1,2)]*b[ADR(2,3)]+a[ADR(1,3)]*b[ADR(3,3)]
                 + a[ADR(1,4)]*b[ADR(4,3)]+a[ADR(1,5)]*b[ADR(5,3)]+a[ADR(1,6)]*b[ADR(6,3)]+a[ADR(1,7)]*b[ADR(7,3)];
    r[ADR(1, 4)] = a[ADR(1,0)]*b[ADR(0,4)]+a[ADR(1,1)]*b[ADR(1,4)]+a[ADR(1,2)]*b[ADR(2,4)]+a[ADR(1,3)]*b[ADR(3,4)]
                 + a[ADR(1,4)]*b[ADR(4,4)]+a[ADR(1,5)]*b[ADR(5,4)]+a[ADR(1,6)]*b[ADR(6,4)]+a[ADR(1,7)]*b[ADR(7,4)];
    r[ADR(1, 5)] = a[ADR(1,0)]*b[ADR(0,5)]+a[ADR(1,1)]*b[ADR(1,5)]+a[ADR(1,2)]*b[ADR(2,5)]+a[ADR(1,3)]*b[ADR(3,5)]
                 + a[ADR(1,4)]*b[ADR(4,5)]+a[ADR(1,5)]*b[ADR(5,5)]+a[ADR(1,6)]*b[ADR(6,5)]+a[ADR(1,7)]*b[ADR(7,5)];
    r[ADR(1, 6)] = a[ADR(1,0)]*b[ADR(0,6)]+a[ADR(1,1)]*b[ADR(1,6)]+a[ADR(1,2)]*b[ADR(2,6)]+a[ADR(1,3)]*b[ADR(3,6)]
                 + a[ADR(1,4)]*b[ADR(4,6)]+a[ADR(1,5)]*b[ADR(5,6)]+a[ADR(1,6)]*b[ADR(6,6)]+a[ADR(1,7)]*b[ADR(7,6)];
    r[ADR(1, 7)] = a[ADR(1,0)]*b[ADR(0,7)]+a[ADR(1,1)]*b[ADR(1,7)]+a[ADR(1,2)]*b[ADR(2,7)]+a[ADR(1,3)]*b[ADR(3,7)]
                 + a[ADR(1,4)]*b[ADR(4,7)]+a[ADR(1,5)]*b[ADR(5,7)]+a[ADR(1,6)]*b[ADR(6,7)]+a[ADR(1,7)]*b[ADR(7,7)];

    r[ADR(2, 0)] = a[ADR(2,0)]*b[ADR(0,0)]+a[ADR(2,1)]*b[ADR(1,0)]+a[ADR(2,2)]*b[ADR(2,0)]+a[ADR(2,3)]*b[ADR(3,0)]
                 + a[ADR(2,4)]*b[ADR(4,0)]+a[ADR(2,5)]*b[ADR(5,0)]+a[ADR(2,6)]*b[ADR(6,0)]+a[ADR(2,7)]*b[ADR(7,0)];
    r[ADR(2, 1)] = a[ADR(2,0)]*b[ADR(0,1)]+a[ADR(2,1)]*b[ADR(1,1)]+a[ADR(2,2)]*b[ADR(2,1)]+a[ADR(2,3)]*b[ADR(3,1)]
                 + a[ADR(2,4)]*b[ADR(4,1)]+a[ADR(2,5)]*b[ADR(5,1)]+a[ADR(2,6)]*b[ADR(6,1)]+a[ADR(2,7)]*b[ADR(7,1)];
    r[ADR(2, 2)] = a[ADR(2,0)]*b[ADR(0,2)]+a[ADR(2,1)]*b[ADR(1,2)]+a[ADR(2,2)]*b[ADR(2,2)]+a[ADR(2,3)]*b[ADR(3,2)]
                 + a[ADR(2,4)]*b[ADR(4,2)]+a[ADR(2,5)]*b[ADR(5,2)]+a[ADR(2,6)]*b[ADR(6,2)]+a[ADR(2,7)]*b[ADR(7,2)];
    r[ADR(2, 3)] = a[ADR(2,0)]*b[ADR(0,3)]+a[ADR(2,1)]*b[ADR(1,3)]+a[ADR(2,2)]*b[ADR(2,3)]+a[ADR(2,3)]*b[ADR(3,3)]
                 + a[ADR(2,4)]*b[ADR(4,3)]+a[ADR(2,5)]*b[ADR(5,3)]+a[ADR(2,6)]*b[ADR(6,3)]+a[ADR(2,7)]*b[ADR(7,3)];
    r[ADR(2, 4)] = a[ADR(2,0)]*b[ADR(0,4)]+a[ADR(2,1)]*b[ADR(1,4)]+a[ADR(2,2)]*b[ADR(2,4)]+a[ADR(2,3)]*b[ADR(3,4)]
                 + a[ADR(2,4)]*b[ADR(4,4)]+a[ADR(2,5)]*b[ADR(5,4)]+a[ADR(2,6)]*b[ADR(6,4)]+a[ADR(2,7)]*b[ADR(7,4)];
    r[ADR(2, 5)] = a[ADR(2,0)]*b[ADR(0,5)]+a[ADR(2,1)]*b[ADR(1,5)]+a[ADR(2,2)]*b[ADR(2,5)]+a[ADR(2,3)]*b[ADR(3,5)]
                 + a[ADR(2,4)]*b[ADR(4,5)]+a[ADR(2,5)]*b[ADR(5,5)]+a[ADR(2,6)]*b[ADR(6,5)]+a[ADR(2,7)]*b[ADR(7,5)];
    r[ADR(2, 6)] = a[ADR(2,0)]*b[ADR(0,6)]+a[ADR(2,1)]*b[ADR(1,6)]+a[ADR(2,2)]*b[ADR(2,6)]+a[ADR(2,3)]*b[ADR(3,6)]
                 + a[ADR(2,4)]*b[ADR(4,6)]+a[ADR(2,5)]*b[ADR(5,6)]+a[ADR(2,6)]*b[ADR(6,6)]+a[ADR(2,7)]*b[ADR(7,6)];
    r[ADR(2, 7)] = a[ADR(2,0)]*b[ADR(0,7)]+a[ADR(2,1)]*b[ADR(1,7)]+a[ADR(2,2)]*b[ADR(2,7)]+a[ADR(2,3)]*b[ADR(3,7)]
                 + a[ADR(2,4)]*b[ADR(4,7)]+a[ADR(2,5)]*b[ADR(5,7)]+a[ADR(2,6)]*b[ADR(6,7)]+a[ADR(2,7)]*b[ADR(7,7)];

    r[ADR(3, 0)] = a[ADR(3,0)]*b[ADR(0,0)]+a[ADR(3,1)]*b[ADR(1,0)]+a[ADR(3,2)]*b[ADR(2,0)]+a[ADR(3,3)]*b[ADR(3,0)]
                 + a[ADR(3,4)]*b[ADR(4,0)]+a[ADR(3,5)]*b[ADR(5,0)]+a[ADR(3,6)]*b[ADR(6,0)]+a[ADR(3,7)]*b[ADR(7,0)];
    r[ADR(3, 1)] = a[ADR(3,0)]*b[ADR(0,1)]+a[ADR(3,1)]*b[ADR(1,1)]+a[ADR(3,2)]*b[ADR(2,1)]+a[ADR(3,3)]*b[ADR(3,1)]
                 + a[ADR(3,4)]*b[ADR(4,1)]+a[ADR(3,5)]*b[ADR(5,1)]+a[ADR(3,6)]*b[ADR(6,1)]+a[ADR(3,7)]*b[ADR(7,1)];
    r[ADR(3, 2)] = a[ADR(3,0)]*b[ADR(0,2)]+a[ADR(3,1)]*b[ADR(1,2)]+a[ADR(3,2)]*b[ADR(2,2)]+a[ADR(3,3)]*b[ADR(3,2)]
                 + a[ADR(3,4)]*b[ADR(4,2)]+a[ADR(3,5)]*b[ADR(5,2)]+a[ADR(3,6)]*b[ADR(6,2)]+a[ADR(3,7)]*b[ADR(7,2)];
    r[ADR(3, 3)] = a[ADR(3,0)]*b[ADR(0,3)]+a[ADR(3,1)]*b[ADR(1,3)]+a[ADR(3,2)]*b[ADR(2,3)]+a[ADR(3,3)]*b[ADR(3,3)]
                 + a[ADR(3,4)]*b[ADR(4,3)]+a[ADR(3,5)]*b[ADR(5,3)]+a[ADR(3,6)]*b[ADR(6,3)]+a[ADR(3,7)]*b[ADR(7,3)];
    r[ADR(3, 4)] = a[ADR(3,0)]*b[ADR(0,4)]+a[ADR(3,1)]*b[ADR(1,4)]+a[ADR(3,2)]*b[ADR(2,4)]+a[ADR(3,3)]*b[ADR(3,4)]
                 + a[ADR(3,4)]*b[ADR(4,4)]+a[ADR(3,5)]*b[ADR(5,4)]+a[ADR(3,6)]*b[ADR(6,4)]+a[ADR(3,7)]*b[ADR(7,4)];
    r[ADR(3, 5)] = a[ADR(3,0)]*b[ADR(0,5)]+a[ADR(3,1)]*b[ADR(1,5)]+a[ADR(3,2)]*b[ADR(2,5)]+a[ADR(3,3)]*b[ADR(3,5)]
                 + a[ADR(3,4)]*b[ADR(4,5)]+a[ADR(3,5)]*b[ADR(5,5)]+a[ADR(3,6)]*b[ADR(6,5)]+a[ADR(3,7)]*b[ADR(7,5)];
    r[ADR(3, 6)] = a[ADR(3,0)]*b[ADR(0,6)]+a[ADR(3,1)]*b[ADR(1,6)]+a[ADR(3,2)]*b[ADR(2,6)]+a[ADR(3,3)]*b[ADR(3,6)]
                 + a[ADR(3,4)]*b[ADR(4,6)]+a[ADR(3,5)]*b[ADR(5,6)]+a[ADR(3,6)]*b[ADR(6,6)]+a[ADR(3,7)]*b[ADR(7,6)];
    r[ADR(3, 7)] = a[ADR(3,0)]*b[ADR(0,7)]+a[ADR(3,1)]*b[ADR(1,7)]+a[ADR(3,2)]*b[ADR(2,7)]+a[ADR(3,3)]*b[ADR(3,7)]
                 + a[ADR(3,4)]*b[ADR(4,7)]+a[ADR(3,5)]*b[ADR(5,7)]+a[ADR(3,6)]*b[ADR(6,7)]+a[ADR(3,7)]*b[ADR(7,7)];

    r[ADR(4, 0)] = a[ADR(4,0)]*b[ADR(0,0)]+a[ADR(4,1)]*b[ADR(1,0)]+a[ADR(4,2)]*b[ADR(2,0)]+a[ADR(4,3)]*b[ADR(3,0)]
                 + a[ADR(4,4)]*b[ADR(4,0)]+a[ADR(4,5)]*b[ADR(5,0)]+a[ADR(4,6)]*b[ADR(6,0)]+a[ADR(4,7)]*b[ADR(7,0)];
    r[ADR(4, 1)] = a[ADR(4,0)]*b[ADR(0,1)]+a[ADR(4,1)]*b[ADR(1,1)]+a[ADR(4,2)]*b[ADR(2,1)]+a[ADR(4,3)]*b[ADR(3,1)]
                 + a[ADR(4,4)]*b[ADR(4,1)]+a[ADR(4,5)]*b[ADR(5,1)]+a[ADR(4,6)]*b[ADR(6,1)]+a[ADR(4,7)]*b[ADR(7,1)];
    r[ADR(4, 2)] = a[ADR(4,0)]*b[ADR(0,2)]+a[ADR(4,1)]*b[ADR(1,2)]+a[ADR(4,2)]*b[ADR(2,2)]+a[ADR(4,3)]*b[ADR(3,2)]
                 + a[ADR(4,4)]*b[ADR(4,2)]+a[ADR(4,5)]*b[ADR(5,2)]+a[ADR(4,6)]*b[ADR(6,2)]+a[ADR(4,7)]*b[ADR(7,2)];
    r[ADR(4, 3)] = a[ADR(4,0)]*b[ADR(0,3)]+a[ADR(4,1)]*b[ADR(1,3)]+a[ADR(4,2)]*b[ADR(2,3)]+a[ADR(4,3)]*b[ADR(3,3)]
                 + a[ADR(4,4)]*b[ADR(4,3)]+a[ADR(4,5)]*b[ADR(5,3)]+a[ADR(4,6)]*b[ADR(6,3)]+a[ADR(4,7)]*b[ADR(7,3)];
    r[ADR(4, 4)] = a[ADR(4,0)]*b[ADR(0,4)]+a[ADR(4,1)]*b[ADR(1,4)]+a[ADR(4,2)]*b[ADR(2,4)]+a[ADR(4,3)]*b[ADR(3,4)]
                 + a[ADR(4,4)]*b[ADR(4,4)]+a[ADR(4,5)]*b[ADR(5,4)]+a[ADR(4,6)]*b[ADR(6,4)]+a[ADR(4,7)]*b[ADR(7,4)];
    r[ADR(4, 5)] = a[ADR(4,0)]*b[ADR(0,5)]+a[ADR(4,1)]*b[ADR(1,5)]+a[ADR(4,2)]*b[ADR(2,5)]+a[ADR(4,3)]*b[ADR(3,5)]
                 + a[ADR(4,4)]*b[ADR(4,5)]+a[ADR(4,5)]*b[ADR(5,5)]+a[ADR(4,6)]*b[ADR(6,5)]+a[ADR(4,7)]*b[ADR(7,5)];
    r[ADR(4, 6)] = a[ADR(4,0)]*b[ADR(0,6)]+a[ADR(4,1)]*b[ADR(1,6)]+a[ADR(4,2)]*b[ADR(2,6)]+a[ADR(4,3)]*b[ADR(3,6)]
                 + a[ADR(4,4)]*b[ADR(4,6)]+a[ADR(4,5)]*b[ADR(5,6)]+a[ADR(4,6)]*b[ADR(6,6)]+a[ADR(4,7)]*b[ADR(7,6)];
    r[ADR(4, 7)] = a[ADR(4,0)]*b[ADR(0,7)]+a[ADR(4,1)]*b[ADR(1,7)]+a[ADR(4,2)]*b[ADR(2,7)]+a[ADR(4,3)]*b[ADR(3,7)]
                 + a[ADR(4,4)]*b[ADR(4,7)]+a[ADR(4,5)]*b[ADR(5,7)]+a[ADR(4,6)]*b[ADR(6,7)]+a[ADR(4,7)]*b[ADR(7,7)];

    r[ADR(5, 0)] = a[ADR(5,0)]*b[ADR(0,0)]+a[ADR(5,1)]*b[ADR(1,0)]+a[ADR(5,2)]*b[ADR(2,0)]+a[ADR(5,3)]*b[ADR(3,0)]
                 + a[ADR(5,4)]*b[ADR(4,0)]+a[ADR(5,5)]*b[ADR(5,0)]+a[ADR(5,6)]*b[ADR(6,0)]+a[ADR(5,7)]*b[ADR(7,0)];
    r[ADR(5, 1)] = a[ADR(5,0)]*b[ADR(0,1)]+a[ADR(5,1)]*b[ADR(1,1)]+a[ADR(5,2)]*b[ADR(2,1)]+a[ADR(5,3)]*b[ADR(3,1)]
                 + a[ADR(5,4)]*b[ADR(4,1)]+a[ADR(5,5)]*b[ADR(5,1)]+a[ADR(5,6)]*b[ADR(6,1)]+a[ADR(5,7)]*b[ADR(7,1)];
    r[ADR(5, 2)] = a[ADR(5,0)]*b[ADR(0,2)]+a[ADR(5,1)]*b[ADR(1,2)]+a[ADR(5,2)]*b[ADR(2,2)]+a[ADR(5,3)]*b[ADR(3,2)]
                 + a[ADR(5,4)]*b[ADR(4,2)]+a[ADR(5,5)]*b[ADR(5,2)]+a[ADR(5,6)]*b[ADR(6,2)]+a[ADR(5,7)]*b[ADR(7,2)];
    r[ADR(5, 3)] = a[ADR(5,0)]*b[ADR(0,3)]+a[ADR(5,1)]*b[ADR(1,3)]+a[ADR(5,2)]*b[ADR(2,3)]+a[ADR(5,3)]*b[ADR(3,3)]
                 + a[ADR(5,4)]*b[ADR(4,3)]+a[ADR(5,5)]*b[ADR(5,3)]+a[ADR(5,6)]*b[ADR(6,3)]+a[ADR(5,7)]*b[ADR(7,3)];
    r[ADR(5, 4)] = a[ADR(5,0)]*b[ADR(0,4)]+a[ADR(5,1)]*b[ADR(1,4)]+a[ADR(5,2)]*b[ADR(2,4)]+a[ADR(5,3)]*b[ADR(3,4)]
                 + a[ADR(5,4)]*b[ADR(4,4)]+a[ADR(5,5)]*b[ADR(5,4)]+a[ADR(5,6)]*b[ADR(6,4)]+a[ADR(5,7)]*b[ADR(7,4)];
    r[ADR(5, 5)] = a[ADR(5,0)]*b[ADR(0,5)]+a[ADR(5,1)]*b[ADR(1,5)]+a[ADR(5,2)]*b[ADR(2,5)]+a[ADR(5,3)]*b[ADR(3,5)]
                 + a[ADR(5,4)]*b[ADR(4,5)]+a[ADR(5,5)]*b[ADR(5,5)]+a[ADR(5,6)]*b[ADR(6,5)]+a[ADR(5,7)]*b[ADR(7,5)];
    r[ADR(5, 6)] = a[ADR(5,0)]*b[ADR(0,6)]+a[ADR(5,1)]*b[ADR(1,6)]+a[ADR(5,2)]*b[ADR(2,6)]+a[ADR(5,3)]*b[ADR(3,6)]
                 + a[ADR(5,4)]*b[ADR(4,6)]+a[ADR(5,5)]*b[ADR(5,6)]+a[ADR(5,6)]*b[ADR(6,6)]+a[ADR(5,7)]*b[ADR(7,6)];
    r[ADR(5, 7)] = a[ADR(5,0)]*b[ADR(0,7)]+a[ADR(5,1)]*b[ADR(1,7)]+a[ADR(5,2)]*b[ADR(2,7)]+a[ADR(5,3)]*b[ADR(3,7)]
                 + a[ADR(5,4)]*b[ADR(4,7)]+a[ADR(5,5)]*b[ADR(5,7)]+a[ADR(5,6)]*b[ADR(6,7)]+a[ADR(5,7)]*b[ADR(7,7)];

    r[ADR(6, 0)] = a[ADR(6,0)]*b[ADR(0,0)]+a[ADR(6,1)]*b[ADR(1,0)]+a[ADR(6,2)]*b[ADR(2,0)]+a[ADR(6,3)]*b[ADR(3,0)]
                 + a[ADR(6,4)]*b[ADR(4,0)]+a[ADR(6,5)]*b[ADR(5,0)]+a[ADR(6,6)]*b[ADR(6,0)]+a[ADR(6,7)]*b[ADR(7,0)];
    r[ADR(6, 1)] = a[ADR(6,0)]*b[ADR(0,1)]+a[ADR(6,1)]*b[ADR(1,1)]+a[ADR(6,2)]*b[ADR(2,1)]+a[ADR(6,3)]*b[ADR(3,1)]
                 + a[ADR(6,4)]*b[ADR(4,1)]+a[ADR(6,5)]*b[ADR(5,1)]+a[ADR(6,6)]*b[ADR(6,1)]+a[ADR(6,7)]*b[ADR(7,1)];
    r[ADR(6, 2)] = a[ADR(6,0)]*b[ADR(0,2)]+a[ADR(6,1)]*b[ADR(1,2)]+a[ADR(6,2)]*b[ADR(2,2)]+a[ADR(6,3)]*b[ADR(3,2)]
                 + a[ADR(6,4)]*b[ADR(4,2)]+a[ADR(6,5)]*b[ADR(5,2)]+a[ADR(6,6)]*b[ADR(6,2)]+a[ADR(6,7)]*b[ADR(7,2)];
    r[ADR(6, 3)] = a[ADR(6,0)]*b[ADR(0,3)]+a[ADR(6,1)]*b[ADR(1,3)]+a[ADR(6,2)]*b[ADR(2,3)]+a[ADR(6,3)]*b[ADR(3,3)]
                 + a[ADR(6,4)]*b[ADR(4,3)]+a[ADR(6,5)]*b[ADR(5,3)]+a[ADR(6,6)]*b[ADR(6,3)]+a[ADR(6,7)]*b[ADR(7,3)];
    r[ADR(6, 4)] = a[ADR(6,0)]*b[ADR(0,4)]+a[ADR(6,1)]*b[ADR(1,4)]+a[ADR(6,2)]*b[ADR(2,4)]+a[ADR(6,3)]*b[ADR(3,4)]
                 + a[ADR(6,4)]*b[ADR(4,4)]+a[ADR(6,5)]*b[ADR(5,4)]+a[ADR(6,6)]*b[ADR(6,4)]+a[ADR(6,7)]*b[ADR(7,4)];
    r[ADR(6, 5)] = a[ADR(6,0)]*b[ADR(0,5)]+a[ADR(6,1)]*b[ADR(1,5)]+a[ADR(6,2)]*b[ADR(2,5)]+a[ADR(6,3)]*b[ADR(3,5)]
                 + a[ADR(6,4)]*b[ADR(4,5)]+a[ADR(6,5)]*b[ADR(5,5)]+a[ADR(6,6)]*b[ADR(6,5)]+a[ADR(6,7)]*b[ADR(7,5)];
    r[ADR(6, 6)] = a[ADR(6,0)]*b[ADR(0,6)]+a[ADR(6,1)]*b[ADR(1,6)]+a[ADR(6,2)]*b[ADR(2,6)]+a[ADR(6,3)]*b[ADR(3,6)]
                 + a[ADR(6,4)]*b[ADR(4,6)]+a[ADR(6,5)]*b[ADR(5,6)]+a[ADR(6,6)]*b[ADR(6,6)]+a[ADR(6,7)]*b[ADR(7,6)];
    r[ADR(6, 7)] = a[ADR(6,0)]*b[ADR(0,7)]+a[ADR(6,1)]*b[ADR(1,7)]+a[ADR(6,2)]*b[ADR(2,7)]+a[ADR(6,3)]*b[ADR(3,7)]
                 + a[ADR(6,4)]*b[ADR(4,7)]+a[ADR(6,5)]*b[ADR(5,7)]+a[ADR(6,6)]*b[ADR(6,7)]+a[ADR(6,7)]*b[ADR(7,7)];

    r[ADR(7, 0)] = a[ADR(7,0)]*b[ADR(0,0)]+a[ADR(7,1)]*b[ADR(1,0)]+a[ADR(7,2)]*b[ADR(2,0)]+a[ADR(7,3)]*b[ADR(3,0)]
                 + a[ADR(7,4)]*b[ADR(4,0)]+a[ADR(7,5)]*b[ADR(5,0)]+a[ADR(7,6)]*b[ADR(6,0)]+a[ADR(7,7)]*b[ADR(7,0)];
    r[ADR(7, 1)] = a[ADR(7,0)]*b[ADR(0,1)]+a[ADR(7,1)]*b[ADR(1,1)]+a[ADR(7,2)]*b[ADR(2,1)]+a[ADR(7,3)]*b[ADR(3,1)]
                 + a[ADR(7,4)]*b[ADR(4,1)]+a[ADR(7,5)]*b[ADR(5,1)]+a[ADR(7,6)]*b[ADR(6,1)]+a[ADR(7,7)]*b[ADR(7,1)];
    r[ADR(7, 2)] = a[ADR(7,0)]*b[ADR(0,2)]+a[ADR(7,1)]*b[ADR(1,2)]+a[ADR(7,2)]*b[ADR(2,2)]+a[ADR(7,3)]*b[ADR(3,2)]
                 + a[ADR(7,4)]*b[ADR(4,2)]+a[ADR(7,5)]*b[ADR(5,2)]+a[ADR(7,6)]*b[ADR(6,2)]+a[ADR(7,7)]*b[ADR(7,2)];
    r[ADR(7, 3)] = a[ADR(7,0)]*b[ADR(0,3)]+a[ADR(7,1)]*b[ADR(1,3)]+a[ADR(7,2)]*b[ADR(2,3)]+a[ADR(7,3)]*b[ADR(3,3)]
                 + a[ADR(7,4)]*b[ADR(4,3)]+a[ADR(7,5)]*b[ADR(5,3)]+a[ADR(7,6)]*b[ADR(6,3)]+a[ADR(7,7)]*b[ADR(7,3)];
    r[ADR(7, 4)] = a[ADR(7,0)]*b[ADR(0,4)]+a[ADR(7,1)]*b[ADR(1,4)]+a[ADR(7,2)]*b[ADR(2,4)]+a[ADR(7,3)]*b[ADR(3,4)]
                 + a[ADR(7,4)]*b[ADR(4,4)]+a[ADR(7,5)]*b[ADR(5,4)]+a[ADR(7,6)]*b[ADR(6,4)]+a[ADR(7,7)]*b[ADR(7,4)];
    r[ADR(7, 5)] = a[ADR(7,0)]*b[ADR(0,5)]+a[ADR(7,1)]*b[ADR(1,5)]+a[ADR(7,2)]*b[ADR(2,5)]+a[ADR(7,3)]*b[ADR(3,5)]
                 + a[ADR(7,4)]*b[ADR(4,5)]+a[ADR(7,5)]*b[ADR(5,5)]+a[ADR(7,6)]*b[ADR(6,5)]+a[ADR(7,7)]*b[ADR(7,5)];
    r[ADR(7, 6)] = a[ADR(7,0)]*b[ADR(0,6)]+a[ADR(7,1)]*b[ADR(1,6)]+a[ADR(7,2)]*b[ADR(2,6)]+a[ADR(7,3)]*b[ADR(3,6)]
                 + a[ADR(7,4)]*b[ADR(4,6)]+a[ADR(7,5)]*b[ADR(5,6)]+a[ADR(7,6)]*b[ADR(6,6)]+a[ADR(7,7)]*b[ADR(7,6)];
    r[ADR(7, 7)] = a[ADR(7,0)]*b[ADR(0,7)]+a[ADR(7,1)]*b[ADR(1,7)]+a[ADR(7,2)]*b[ADR(2,7)]+a[ADR(7,3)]*b[ADR(3,7)]
                 + a[ADR(7,4)]*b[ADR(4,7)]+a[ADR(7,5)]*b[ADR(5,7)]+a[ADR(7,6)]*b[ADR(6,7)]+a[ADR(7,7)]*b[ADR(7,7)];

#endif

#if NEW_VEC == 1

    // New version of vectorization with load/store operations.
    // sp 5.73

    __assume_aligned(a, 64);
    __assume_aligned(b, 64);
    __assume_aligned(r, 64);

    // Indices.
    __m512i ind_df = _mm512_set_epi32( 7,  6,  5,  4,  3,  2, 1, 0,
                                       7,  6,  5,  4,  3,  2, 1, 0);
    __m512i ind_ds = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8,
                                      15, 14, 13, 12, 11, 10, 9, 8);

    // Load full b matrix lines.
    __m512 b0 = LD(&b[0]);
    __m512 b1 = _mm512_permutexvar_ps(ind_ds, b0);
    __m512 b2 = LD(&b[2 * V8]);
    __m512 b3 = _mm512_permutexvar_ps(ind_ds, b2);
    __m512 b4 = LD(&b[4 * V8]);
    __m512 b5 = _mm512_permutexvar_ps(ind_ds, b4);
    __m512 b6 = LD(&b[6 * V8]);
    __m512 b7 = _mm512_permutexvar_ps(ind_ds, b6);

    // Correct even matrices lines.
    b0 = _mm512_permutexvar_ps(ind_df, b0);
    b2 = _mm512_permutexvar_ps(ind_df, b2);
    b4 = _mm512_permutexvar_ps(ind_df, b4);
    b6 = _mm512_permutexvar_ps(ind_df, b6);

    // Load full a matrix lines.
    __m512 a0 = LD(&a[0]);
    __m512 a2 = LD(&a[2 * V8]);
    __m512 a4 = LD(&a[4 * V8]);
    __m512 a6 = LD(&a[6 * V8]);

    // Indices.
    __m512i ind_0 = _mm512_set_epi32( 8,  8,  8,  8,  8,  8,  8,  8,
                                      0,  0,  0,  0,  0,  0,  0,  0);
    __m512i ind_1 = _mm512_set_epi32( 9,  9,  9,  9,  9,  9,  9,  9,
                                      1,  1,  1,  1,  1,  1,  1,  1);
    __m512i ind_2 = _mm512_set_epi32(10, 10, 10, 10, 10, 10, 10, 10,
                                      2,  2,  2,  2,  2,  2,  2,  2);
    __m512i ind_3 = _mm512_set_epi32(11, 11, 11, 11, 11, 11, 11, 11,
                                      3,  3,  3,  3,  3,  3,  3,  3);
    __m512i ind_4 = _mm512_set_epi32(12, 12, 12, 12, 12, 12, 12, 12,
                                      4,  4,  4,  4,  4,  4,  4,  4);
    __m512i ind_5 = _mm512_set_epi32(13, 13, 13, 13, 13, 13, 13, 13,
                                      5,  5,  5,  5,  5,  5,  5,  5);
    __m512i ind_6 = _mm512_set_epi32(14, 14, 14, 14, 14, 14, 14, 14,
                                      6,  6,  6,  6,  6,  6,  6,  6);
    __m512i ind_7 = _mm512_set_epi32(15, 15, 15, 15, 15, 15, 15, 15,
                                      7,  7,  7,  7,  7,  7,  7,  7);

#define BLOCK(N, A)                                                    \
    ST(&r[N * V8],                                                     \
       FMADD(PERMXV(ind_0, A), b0,                                     \
             FMADD(PERMXV(ind_1, A), b1,                               \
                   FMADD(PERMXV(ind_2, A), b2,                         \
                         FMADD(PERMXV(ind_3, A), b3,                   \
                               FMADD(PERMXV(ind_4, A), b4,             \
                                     FMADD(PERMXV(ind_5, A), b5,       \
                                           FMADD(PERMXV(ind_6, A), b6, \
                                                 MUL(PERMXV(ind_7, A), b7)))))))));

    // Four blocks for storing result.
    BLOCK(0, a0);
    BLOCK(2, a2);
    BLOCK(4, a4);
    BLOCK(6, a6);

#undef BLOCK

#endif

#endif

}

//--------------------------------------------------------------------------------------------------
