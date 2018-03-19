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

    // Indices.
    __m512i ind_m = _mm512_set_epi32(                                                     0,
                                     2 * V8 + 4, 2 * V8 + 3, 2 * V8 + 2, 2 * V8 + 1, 2 * V8,
                                         V8 + 4,     V8 + 3,     V8 + 2,     V8 + 1,     V8,
                                              4,          3,          2,          1,      0);
    __m512i ind_v = _mm512_set_epi32(0, 4, 3, 2, 1, 0, 4, 3, 2, 1, 0, 4, 3, 2, 1, 0);
    //
    __m512 abc = _mm512_mask_i32gather_ps(z, 0x7FFF, ind_m, m,                    _MM_SCALE_4);
    __m512 de  = _mm512_mask_i32gather_ps(z, 0x3FF,  ind_m, m + 3 * V8,           _MM_SCALE_4);
    __m512 fgh = _mm512_mask_i32gather_ps(z, 0x7FFF, ind_m, m + V48,              _MM_SCALE_4);
    __m512 ij  = _mm512_mask_i32gather_ps(z, 0x3FF,  ind_m, m + V48 + 3 * V8,     _MM_SCALE_4);
    __m512 klm = _mm512_mask_i32gather_ps(z, 0x7FFF, ind_m, m + 2 * V48,          _MM_SCALE_4);
    __m512 no  = _mm512_mask_i32gather_ps(z, 0x3FF,  ind_m, m + 2 * V48 + 3 * V8, _MM_SCALE_4);
    //
    __m512 v1 = _mm512_mask_i32gather_ps(z, 0x7FFF, ind_v, v, _MM_SCALE_4);
    __m512 v2 = _mm512_mask_i32gather_ps(z, 0x7FFF, ind_v, v + V8, _MM_SCALE_4);
    __m512 v3 = _mm512_mask_i32gather_ps(z, 0x7FFF, ind_v, v + 2 * V8, _MM_SCALE_4);
    //
    abc = _mm512_mask_mul_ps(z, 0x7FFF, abc, v1);
    de  = _mm512_mask_mul_ps(z, 0x3FF,  de,  v1);
    fgh = _mm512_mask_mul_ps(z, 0x7FFF, fgh, v2);
    ij  = _mm512_mask_mul_ps(z, 0x3FF,  ij,  v2);
    klm = _mm512_mask_mul_ps(z, 0x7FFF, klm, v3);
    no  = _mm512_mask_mul_ps(z, 0x3FF,  no,  v3);
    //
    __m512 abc1 = _mm512_mask_add_ps(abc, 0x3FCF, abc, _mm512_swizzle_ps(abc, _MM_SWIZ_REG_CDAB));
    __m512 de1  = _mm512_mask_add_ps( de,  0x3CF,  de, _mm512_swizzle_ps( de, _MM_SWIZ_REG_CDAB));
    __m512 fgh1 = _mm512_mask_add_ps(fgh, 0x3FCF, fgh, _mm512_swizzle_ps(fgh, _MM_SWIZ_REG_CDAB));
    __m512 ij1  = _mm512_mask_add_ps( ij,  0x3CF,  ij, _mm512_swizzle_ps( ij, _MM_SWIZ_REG_CDAB));
    __m512 klm1 = _mm512_mask_add_ps(klm, 0x3FCF, klm, _mm512_swizzle_ps(klm, _MM_SWIZ_REG_CDAB));
    __m512 no1  = _mm512_mask_add_ps( no,  0x3CF,  no, _mm512_swizzle_ps( no, _MM_SWIZ_REG_CDAB));
    //
    float _t[16];
    _mm512_mask_i32scatter_ps(&_t[0], 0x5575,
                              _mm512_set_epi32(0, 8, 0, 7, 0, 6, 0, 5, 0, 4, 3, 2, 0, 1, 0, 0),
                              abc1, _MM_SCALE_4);
    _mm512_mask_i32scatter_ps(&_t[0], 0x175,
                              _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 14, 0, 13, 12, 11, 0, 10, 0, 9),
                              de1, _MM_SCALE_4);
    abc1 = _mm512_load_ps(&_t[0]);
    _mm512_mask_i32scatter_ps(&_t[0], 0x5575,
                              _mm512_set_epi32(0, 8, 0, 7, 0, 6, 0, 5, 0, 4, 3, 2, 0, 1, 0, 0),
                              fgh1, _MM_SCALE_4);
    _mm512_mask_i32scatter_ps(&_t[0], 0x175,
                              _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 14, 0, 13, 12, 11, 0, 10, 0, 9),
                              ij1, _MM_SCALE_4);
    fgh1 = _mm512_load_ps(&_t[0]);
    _mm512_mask_i32scatter_ps(&_t[0], 0x5575,
                              _mm512_set_epi32(0, 8, 0, 7, 0, 6, 0, 5, 0, 4, 3, 2, 0, 1, 0, 0),
                              klm1, _MM_SCALE_4);
    _mm512_mask_i32scatter_ps(&_t[0], 0x175,
                              _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 14, 0, 13, 12, 11, 0, 10, 0, 9),
                              no1, _MM_SCALE_4);
    klm1 = _mm512_load_ps(&_t[0]);
    //
    __m512 abc2 = _mm512_mask_add_ps(abc1, 0x3CF3, abc1, _mm512_swizzle_ps(abc1, _MM_SWIZ_REG_CDAB));
    __m512 fgh2 = _mm512_mask_add_ps(fgh1, 0x3CF3, fgh1, _mm512_swizzle_ps(fgh1, _MM_SWIZ_REG_CDAB));
    __m512 klm2 = _mm512_mask_add_ps(klm1, 0x3CF3, klm1, _mm512_swizzle_ps(klm1, _MM_SWIZ_REG_CDAB));
    //
    _mm512_mask_i32scatter_ps(&_t[0], 0x5b6d,
                              _mm512_set_epi32(0, 9, 0, 8, 7, 0, 6, 5, 0, 4, 3, 0, 2, 1, 0, 0),
                              abc2, _MM_SCALE_4);
    __m512 abc3 = _mm512_load_ps(&_t[0]);
    _mm512_mask_i32scatter_ps(&_t[0], 0x5b6d,
                              _mm512_set_epi32(0, 9, 0, 8, 7, 0, 6, 5, 0, 4, 3, 0, 2, 1, 0, 0),
                              fgh2, _MM_SCALE_4);
    __m512 fgh3 = _mm512_load_ps(&_t[0]);
    _mm512_mask_i32scatter_ps(&_t[0], 0x5b6d,
                              _mm512_set_epi32(0, 9, 0, 8, 7, 0, 6, 5, 0, 4, 3, 0, 2, 1, 0, 0),
                              klm2, _MM_SCALE_4);
    __m512 klm3 = _mm512_load_ps(&_t[0]);
    //
    __m512 abc4 = _mm512_mask_add_ps(abc3, 0x3FF, abc3, _mm512_swizzle_ps(abc3, _MM_SWIZ_REG_CDAB));
    __m512 fgh4 = _mm512_mask_add_ps(fgh3, 0x3FF, fgh3, _mm512_swizzle_ps(fgh3, _MM_SWIZ_REG_CDAB));
    __m512 klm4 = _mm512_mask_add_ps(klm3, 0x3FF, klm3, _mm512_swizzle_ps(klm3, _MM_SWIZ_REG_CDAB));
    //
    _mm512_mask_i32scatter_ps(r, 0x155,            _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 4, 0, 3, 0, 2, 0, 1, 0, 0),
                              abc4, _MM_SCALE_4);
    _mm512_mask_i32scatter_ps(r + V8, 0x155,       _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 4, 0, 3, 0, 2, 0, 1, 0, 0),
                              fgh4, _MM_SCALE_4);
    _mm512_mask_i32scatter_ps(r + 2 * V8, 0x155,   _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 4, 0, 3, 0, 2, 0, 1, 0, 0),
                              klm4, _MM_SCALE_4);

#endif

}
