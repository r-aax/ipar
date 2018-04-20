/// \file
/// \brief Matrix operations implementations.

#include "matrices.h"
#include "../../Utils/Maths.h"
#include "../../Utils/Intel.h"
#include "../../Utils/Bits.h"
#include <stdlib.h>
#include <math.h>
#include "avx512debug.h"

#ifdef INTEL
#include <immintrin.h>
#endif

#ifdef INTEL

// Global variables.

/// \brief Index for loading the matrix.
__m512i matvec5_m_load_ind;

/// \brief Index for loading the vector.
__m512i matvec5_v_load_ind;

/// \brief Index N 1 for scatter in matvec5.
__m512i matvec5_sc1_ind;

/// \brief Index N 2 for scatter in matvec5.
__m512i matvec5_sc2_ind;

/// \brief Index N 3 for scatter in matvec5.
__m512i matvec5_sc3_ind;

/// \brief Index N 4 for scatter in matvec5.
__m512i matvec5_sc4_ind;

/// \brief Index N 5 for scatter in matvec5.
__m512i matvec5_sc5_ind;

/// \brief Index N 6 for scatter in matvec5.
__m512i matvec5_sc6_ind;

/// \brief Index for saving the result.
__m512i matvec5_save_ind;

#endif

/// \brief Multiplication 5*8-matrix on 5-vector.
///
/// Original version.
///
/// \param m - matrix
/// \param v - vector
/// \param r - result
void matvec5_orig(float * __restrict m, float * __restrict v, float * __restrict r)
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

/// \brief Multiplication 5*8-matrix on 5-vector. Initilization.
void matvec5_opt_init()
{

#ifdef INTEL

    matvec5_m_load_ind = _mm512_set_epi32(                                                     0,
                                          2 * V8 + 4, 2 * V8 + 3, 2 * V8 + 2, 2 * V8 + 1, 2 * V8,
                                              V8 + 4,     V8 + 3,     V8 + 2,     V8 + 1,     V8,
                                                   4,          3,          2,          1,      0);
    matvec5_v_load_ind = _mm512_set_epi32(0, 4, 3, 2, 1, 0, 4, 3, 2, 1, 0, 4, 3, 2, 1, 0);
    matvec5_sc1_ind = _mm512_set_epi32(0, 10, 0, 9, 0, 8, 0, 6, 0, 5, 4, 2, 0, 1, 0, 0);
    matvec5_sc2_ind = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 18, 0, 17, 16, 14, 0, 13, 0, 12);
    matvec5_sc3_ind = _mm512_set_epi32(0, 30, 0, 29, 0, 28, 0, 26, 0, 25, 24, 22, 0, 21, 0, 20);
    matvec5_sc4_ind = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 38, 0, 37, 36, 34, 0, 33, 0, 32);
    matvec5_sc5_ind = _mm512_set_epi32(0, 50, 0, 49, 0, 48, 0, 46, 0, 45, 44, 42, 0, 41, 0, 40);
    matvec5_sc6_ind = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 58, 0, 57, 56, 54, 0, 53, 0, 52);
    matvec5_save_ind = _mm512_set_epi32(         0, 2 * V8 + 1, V8 + 2, 3,
                                        2 * V8 + 4,     2 * V8, V8 + 1, 2,
                                        2 * V8 + 3,     V8 + 4,     V8, 1,
                                        2 * V8 + 2,     V8 + 3,      4, 0);

#endif

}

/// \brief Multiplication 5*8-matrix on 5-vector.
///
/// Optimized version.
///
/// \param m - matrix
/// \param v - vector
/// \param r - result 
void matvec5_3x_opt(float * __restrict m, float * __restrict v, float * __restrict r)
{

#ifndef INTEL

    matvec5_orig(m, v, r);
    matvec5_orig(m + V48, v + V8, r + V8);
    matvec5_orig(m + 2 * V48, v + 2 * V8, r + 2 * V8);

#else

    __assume_aligned(m, 64);
    __assume_aligned(v, 64);
    __assume_aligned(r, 64);

    // Zero vector.
    __m512 z = _mm512_setzero_ps();

    // Load matrix.
    __m512 abc = _mm512_mask_i32gather_ps(z, 0x7FFF, matvec5_m_load_ind, m,                    _MM_SCALE_4);
    __m512 de  = _mm512_mask_i32gather_ps(z, 0x3FF,  matvec5_m_load_ind, m + 3 * V8,           _MM_SCALE_4);
    __m512 fgh = _mm512_mask_i32gather_ps(z, 0x7FFF, matvec5_m_load_ind, m + V48,              _MM_SCALE_4);
    __m512 ij  = _mm512_mask_i32gather_ps(z, 0x3FF,  matvec5_m_load_ind, m + V48 + 3 * V8,     _MM_SCALE_4);
    __m512 klm = _mm512_mask_i32gather_ps(z, 0x7FFF, matvec5_m_load_ind, m + 2 * V48,          _MM_SCALE_4);
    __m512 no  = _mm512_mask_i32gather_ps(z, 0x3FF,  matvec5_m_load_ind, m + 2 * V48 + 3 * V8, _MM_SCALE_4);

    // Load vectors.
    __m512 v1 = _mm512_mask_i32gather_ps(z, 0x7FFF, matvec5_v_load_ind, v, _MM_SCALE_4);
    __m512 v2 = _mm512_mask_i32gather_ps(z, 0x7FFF, matvec5_v_load_ind, v + V8, _MM_SCALE_4);
    __m512 v3 = _mm512_mask_i32gather_ps(z, 0x7FFF, matvec5_v_load_ind, v + 2 * V8, _MM_SCALE_4);

    // Multiplication.
    abc = _mm512_mask_mul_ps(z, 0x7FFF, abc, v1);
    de  = _mm512_mask_mul_ps(z, 0x3FF,  de,  v1);
    fgh = _mm512_mask_mul_ps(z, 0x7FFF, fgh, v2);
    ij  = _mm512_mask_mul_ps(z, 0x3FF,  ij,  v2);
    klm = _mm512_mask_mul_ps(z, 0x7FFF, klm, v3);
    no  = _mm512_mask_mul_ps(z, 0x3FF,  no,  v3);

    // First swizzle phase.
    abc = _mm512_mask_add_ps(abc, 0x3FCF, abc, _mm512_swizzle_ps(abc, _MM_SWIZ_REG_CDAB));
    de  = _mm512_mask_add_ps( de,  0x3CF,  de, _mm512_swizzle_ps( de, _MM_SWIZ_REG_CDAB));
    fgh = _mm512_mask_add_ps(fgh, 0x3FCF, fgh, _mm512_swizzle_ps(fgh, _MM_SWIZ_REG_CDAB));
    ij  = _mm512_mask_add_ps( ij,  0x3CF,  ij, _mm512_swizzle_ps( ij, _MM_SWIZ_REG_CDAB));
    klm = _mm512_mask_add_ps(klm, 0x3FCF, klm, _mm512_swizzle_ps(klm, _MM_SWIZ_REG_CDAB));
    no  = _mm512_mask_add_ps( no,  0x3CF,  no, _mm512_swizzle_ps( no, _MM_SWIZ_REG_CDAB));

    // Save to memory.
    float _q[64];
    _mm512_mask_i32scatter_ps(&_q[0], 0x5575, matvec5_sc1_ind, abc, _MM_SCALE_4);
    _mm512_mask_i32scatter_ps(&_q[0], 0x175,  matvec5_sc2_ind, de,  _MM_SCALE_4);
    _mm512_mask_i32scatter_ps(&_q[0], 0x5575, matvec5_sc3_ind, fgh, _MM_SCALE_4);
    _mm512_mask_i32scatter_ps(&_q[0], 0x175,  matvec5_sc4_ind, ij,  _MM_SCALE_4);
    _mm512_mask_i32scatter_ps(&_q[0], 0x5575, matvec5_sc5_ind, klm, _MM_SCALE_4);
    _mm512_mask_i32scatter_ps(&_q[0], 0x175,  matvec5_sc6_ind, no,  _MM_SCALE_4);

    // Load and find sums of 4 neighbours elements.
    __m512 w0 = _mm512_load_ps(&_q[0]);
    __m512 w1 = _mm512_load_ps(&_q[16]);
    __m512 w2 = _mm512_load_ps(&_q[32]);
    __m512 w3 = _mm512_load_ps(&_q[48]);
    w0 = INTEL_SWIZ_2_ADD_2_BLEND_1(w0, w1, _MM_SWIZ_REG_CDAB, 0xAAAA);
    w1 = INTEL_SWIZ_2_ADD_2_BLEND_1(w2, w3, _MM_SWIZ_REG_CDAB, 0xAAAA);
    w0 = INTEL_SWIZ_2_ADD_2_BLEND_1(w0, w1, _MM_SWIZ_REG_BADC, 0xCCCC);

    // Save the result.
    _mm512_mask_i32scatter_ps(r, 0x7FFF, matvec5_save_ind, w0, _MM_SCALE_4);

#endif

}

