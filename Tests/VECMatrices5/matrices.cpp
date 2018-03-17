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

#else

    __assume_aligned(m, 64);
    __assume_aligned(v, 64);
    __assume_aligned(r, 64);

    __m512i ind_m = _mm512_set_epi32(                                                     0,
                                     2 * V8 + 4, 2 * V8 + 3, 2 * V8 + 2, 2 * V8 + 1, 2 * V8,
                                         V8 + 4,     V8 + 3,     V8 + 2,     V8 + 1,     V8,
                                              4,          3,          2,          1,      0);
    __m512i ind_v = _mm512_set_epi32(0, 4, 3, 2, 1, 0, 4, 3, 2, 1, 0, 4, 3, 2, 1, 0);
    __m512 z = _mm512_setzero_ps();
    //
    __m512 abc = _mm512_mask_i32gather_ps(z, 0x7FFF, ind_m, m,                    _MM_SCALE_4);
    __m512 de  = _mm512_mask_i32gather_ps(z, 0x3FF,  ind_m, m + 3 * V8,           _MM_SCALE_4);
    __m512 fgh = _mm512_mask_i32gather_ps(z, 0x7FFF, ind_m, m + V48,              _MM_SCALE_4);
    __m512 ij  = _mm512_mask_i32gather_ps(z, 0x3FF,  ind_m, m + V48 + 3 * V8,     _MM_SCALE_4);
    __m512 klm = _mm512_mask_i32gather_ps(z, 0x7FFF, ind_m, m + 2 * V48,          _MM_SCALE_4);
    __m512 no  = _mm512_mask_i32gather_ps(z, 0x3FF,  ind_m, m + 2 * V48 + 3 * V8, _MM_SCALE_4);
    //
    __m512 v1 = _mm512_mask_i32gather_ps(z, 0x7FFF, ind_v, v, _MM_SCALE_4);
    __m512 v2 = _mm512_mask_i32gather_ps(z, 0x7FFF, ind_v, v, _MM_SCALE_4);
    __m512 v3 = _mm512_mask_i32gather_ps(z, 0x7FFF, ind_v, v, _MM_SCALE_4);
    //
    abc = _mm512_mask_mul_ps(z, 0x7FFF, abc, v1);
    de  = _mm512_mask_mul_ps(z, 0x3FF,  de,  v1);
    fgh = _mm512_mask_mul_ps(z, 0x7FFF, fgh, v2);
    ij  = _mm512_mask_mul_ps(z, 0x3FF,  ij,  v2);
    klm = _mm512_mask_mul_ps(z, 0x7FFF, klm, v3);
    no  = _mm512_mask_mul_ps(z, 0x3FF,  no,  v3);

    r[0] = _mm512_mask_reduce_add_ps(0x1F,   abc);
    r[1] = _mm512_mask_reduce_add_ps(0x3E0,  abc);
    r[2] = _mm512_mask_reduce_add_ps(0x7C00, abc);
    r[3] = _mm512_mask_reduce_add_ps(0x1F,   de);
    r[4] = _mm512_mask_reduce_add_ps(0x3E0,  de);
    r[V8]     = _mm512_mask_reduce_add_ps(0x1F,   fgh);
    r[V8 + 1] = _mm512_mask_reduce_add_ps(0x3E0,  fgh);
    r[V8 + 2] = _mm512_mask_reduce_add_ps(0x7C00, fgh);
    r[V8 + 3] = _mm512_mask_reduce_add_ps(0x1F,   ij);
    r[V8 + 4] = _mm512_mask_reduce_add_ps(0x3E0,  ij);
    r[2 * V8]     = _mm512_mask_reduce_add_ps(0x1F,   klm);
    r[2 * V8 + 1] = _mm512_mask_reduce_add_ps(0x3E0,  klm);
    r[2 * V8 + 2] = _mm512_mask_reduce_add_ps(0x7C00, klm);
    r[2 * V8 + 3] = _mm512_mask_reduce_add_ps(0x1F,   no);
    r[2 * V8 + 4] = _mm512_mask_reduce_add_ps(0x3E0,  no);

    matvec5_orig(m, v, r);
    matvec5_orig(m + V48, v + V8, r + V8);
    matvec5_orig(m + 2 * V48, v + 2 * V8, r + 2 * V8);

#endif

}
