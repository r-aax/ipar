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

/// \brief Multiplication 8*8-matrix on 8-vector.
///
/// Original version.
///
/// \param matr - matrix
/// \param vect - vector
/// \param matv - result
void matvec8_orig(float * __restrict matr, float * __restrict vect, float * __restrict matv)
{
    for (int i = 0; i < V8; i++)
    {
        float sum = 0.0;
        int ii = i * V8;

        for (int j = 0; j < V8; j++)
        {
            sum = sum + matr[ii + j] * vect[j];
        }

        matv[i] = sum;
    }
}

/// \brief Multiplication 8*8-matrix on 8-vector.
///
/// Optimized version.
///
/// \param matr - matrix
/// \param vect - vector
/// \param matv - result 
void matvec8_opt(float * __restrict matr, float * __restrict vect, float * __restrict matv)
{

#ifndef INTEL

    matvec8_orig(matr, vect, matv);

#else

    __assume_aligned(&matr[0], 64);
    __assume_aligned(&vect[0], 64);
    __assume_aligned(&matv[0], 64);

    // Hint for loading two copies of vector vect.
    __m512 v1 = _mm512_setzero_ps();
    __m512 v2 = _mm512_setzero_ps();
    v1 = _mm512_mask_extload_ps(v1, 0xF0F, &vect[0],
                                _MM_UPCONV_PS_NONE,
                                _MM_BROADCAST_4X16,
                                _MM_HINT_NONE);
    v2 = _mm512_mask_extload_ps(v2, 0xF0F0, &vect[4],
                                _MM_UPCONV_PS_NONE,
                                _MM_BROADCAST_4X16,
                                _MM_HINT_NONE);
    __m512 v = _mm512_add_ps(v1, v2);

    for (int i = 0; i < V8; i += 2)
    {
        int ii = i * V8;

        __m512 m = _mm512_load_ps(&matr[ii]);
        __m512 mv = _mm512_mul_ps(m, v);

        matv[i] = _mm512_mask_reduce_add_ps(0xFF, mv);
        matv[i + 1] = _mm512_mask_reduce_add_ps(0xFF00, mv);
    }

#endif

}

/// \brief Multiplication 8*8-matrix on 8-vector.
///
/// Optimized version.
///
/// \param matr - matrix
/// \param vect - vector
/// \param matv - result 
void matvec8_opt2(float * __restrict matr, float * __restrict vect, float * __restrict matv)
{

#ifndef INTEL

    matvec8_orig(matr, vect, matv);

#else

    __declspec(align(64)) float tmp[V64];

    __assume_aligned(&matr[0], 64);
    __assume_aligned(&vect[0], 64);
    __assume_aligned(&matv[0], 64);
    __assume_aligned(&tmp[0], 64);

    // Hint for loading two copies of vector vect.
    __m512 v1 = _mm512_setzero_ps();
    __m512 v2 = _mm512_setzero_ps();
    v1 = _mm512_mask_extload_ps(v1, 0xF0F, &vect[0],
                                _MM_UPCONV_PS_NONE,
                                _MM_BROADCAST_4X16,
                                _MM_HINT_NONE);
    v2 = _mm512_mask_extload_ps(v2, 0xF0F0, &vect[4],
                                _MM_UPCONV_PS_NONE,
                                _MM_BROADCAST_4X16,
                                _MM_HINT_NONE);
    __m512 v = _mm512_add_ps(v1, v2);

    for (int i = 0; i < V8; i += 2)
    {
        int ii = i * V8;

        __m512 m = _mm512_load_ps(&matr[ii]);
        __m512 mv = _mm512_mul_ps(m, v);

        _mm512_store_ps(&tmp[ii], mv);
    }

    __m512i ind = _mm512_set_epi32(     0,      0,      0,      0,
                                        0,      0,      0,      0,
                                   7 * V8, 6 * V8, 5 * V8, 4 * V8,
                                   3 * V8, 2 * V8,     V8,      0);

    __m512 r, ri;

    r = _mm512_mask_i32gather_ps(r, 0xFF, ind, &tmp[0], _MM_SCALE_4);

    for (int i = 1; i < V8; i++)
    {
        ri = _mm512_mask_i32gather_ps(ri, 0xFF, ind, &tmp[i], _MM_SCALE_4);
        r = _mm512_mask_add_ps(r, 0xFF, r, ri);
    }

    _mm512_mask_store_ps(&matv[0], 0xFF, r);

#endif

}

/// \brief Multiplication 16*16-matrix on 16-vector.
///
/// Original version.
///
/// \param matr - matrix
/// \param vect - vector
/// \param matv - result
void matvec16_orig(float * __restrict matr, float * __restrict vect, float * __restrict matv)
{
    for (int i = 0; i < V16; i++)
    {
        float sum = 0.0;
        int ii = i * V16;

        for (int j = 0; j < V16; j++)
        {
            sum = sum + matr[ii + j] * vect[j];
        }

        matv[i] = sum;
    }
}

/// \brief Multiplication 16*16-matrix on 16-vector.
///
/// Optimized version.
///
/// \param matr - matrix
/// \param vect - vector
/// \param matv - result
void matvec16_opt(float * __restrict matr, float * __restrict vect, float * __restrict matv)
{

#ifndef INTEL

    matvec16_orig(matr, vect, matv);

#else

    __assume_aligned(&matr[0], 64);
    __assume_aligned(&vect[0], 64);
    __assume_aligned(&matv[0], 64);

    __m512 v = _mm512_load_ps(vect);

    for (int i = 0; i < V16; i++)
    {
        int ii = i * V16;

        __m512 m = _mm512_load_ps(&matr[ii]);
        __m512 mv = _mm512_mul_ps(m, v);

        matv[i] = _mm512_reduce_add_ps(mv);
    }

#endif

}

/// \brief Multiplication 16*16-matrix on 16-vector.
///
/// Optimized version.
///
/// \param matr - matrix
/// \param vect - vector
/// \param matv - result
void matvec16_opt2(float * __restrict matr, float * __restrict vect, float * __restrict matv)
{

#ifndef INTEL

    matvec16_orig(matr, vect, matv);

#else

    __declspec(align(64)) float tmp[V256];

    __assume_aligned(&matr[0], 64);
    __assume_aligned(&vect[0], 64);
    __assume_aligned(&matv[0], 64);
    __assume_aligned(&tmp[0], 64);

    __m512 v = _mm512_load_ps(vect);

    for (int i = 0; i < V16; i++)
    {
        int ii = i * V16;

        __m512 m = _mm512_load_ps(&matr[ii]);
        __m512 mv = _mm512_mul_ps(m, v);

        _mm512_store_ps(&tmp[ii], mv);
    }

    __m512i ind = _mm512_set_epi32(15 * V16, 14 * V16, 13 * V16, 12 * V16,
                                   11 * V16, 10 * V16,  9 * V16,  8 * V16,
                                    7 * V16,  6 * V16,  5 * V16,  4 * V16,
                                    3 * V16,  2 * V16,      V16,        0);

    __m512 r, ri;

    r = _mm512_i32gather_ps(ind, &tmp[0], _MM_SCALE_4);

    for (int i = 1; i < V16; i++)
    {
        ri = _mm512_i32gather_ps(ind, &tmp[i], _MM_SCALE_4);
        r = _mm512_add_ps(r, ri);
    }

    _mm512_store_ps(matv, r);

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

    __m512 bs, bs2, as, mul1, mul2;

    __m512i ind = _mm512_set_epi32(7 * V8 + 1, 6 * V8 + 1,
                                   5 * V8 + 1, 4 * V8 + 1,
                                   3 * V8 + 1, 2 * V8 + 1,
                                       V8 + 1,          1,
                                       7 * V8,     6 * V8,
                                       5 * V8,     4 * V8,
                                       3 * V8,     2 * V8,
                                           V8,          0);

    // Loop for b matrix.
    for (int j = 0; j < V8; j += 2)
    {
        bs = _mm512_i32gather_ps(ind, &b[j], _MM_SCALE_4);
        bs2 = _mm512_permute4f128_ps(bs, _MM_PERM_BADC);

        // Loop for a matrix.
        for (int i = 0; i < V8; i += 2)
        {
            int ii = i * V8;

            as = _mm512_load_ps(&a[ii]);
            mul1 = _mm512_mul_ps(as, bs);
            mul2 = _mm512_mul_ps(as, bs2);

            r[ii + j] = _mm512_mask_reduce_add_ps(0xFF, mul1);
            r[ii + j + 1] = _mm512_mask_reduce_add_ps(0xFF, mul2);
            r[ii + V8 + j] = _mm512_mask_reduce_add_ps(0xFF00, mul2);
            r[ii + V8 + j + 1] = _mm512_mask_reduce_add_ps(0xFF00, mul1);
        }
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

    __m512 bs, as, mul;
    __m512i ind = _mm512_set_epi32(15 * V16, 14 * V16, 13 * V16, 12 * V16,
                                   11 * V16, 10 * V16,  9 * V16,  8 * V16,
                                    7 * V16,  6 * V16,  5 * V16,  4 * V16,
                                    3 * V16,  2 * V16,      V16,        0);

    // Loop for b matrix.
    for (int j = 0; j < V16; j++)
    {
        bs = _mm512_i32gather_ps(ind, &b[j], _MM_SCALE_4);

        // Loop for a matrix.
        for (int i = 0; i < V16; i++)
        {
            int ii = i * V16;

            as = _mm512_load_ps(&a[ii]);
            mul = _mm512_mul_ps(as, bs);
            r[ii + j] = _mm512_reduce_add_ps(mul);
        }
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
            if (fabs(m[j * V8 + j]) > fabs(m[lead_i * V8 + lead_i]))
            {
                lead_i = j;
            }
        }
        if (fabs(m[lead_i * V8 + lead_i]) < MATHS_EPS)
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
        float d = m[i * V8 + i];
        for (int j = i; j < V8; j++)
        {
            m[i * V8 + j] /= d;
            r[i * V8 + j] /= d;
        }

        // Zero all other lines.
        for (int j = 0; j < V8; j++)
        {
            if (j != i)
            {
                float t = m[j * V8 + i];

                for (int k = i; k < V8; k++)
                {
                    m[j * V8 + k] -= m[i * V8 + k] * t;
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
            if (fabs(m[j * V8 + j]) > fabs(m[lead_i * V8 + lead_i]))
            {
                lead_i = j;
            }
        }
        if (fabs(m[lead_i * V8 + lead_i]) < MATHS_EPS)
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
        float d = m[i * V8 + i];
        for (int j = i; j < V8; j++)
        {
            m[i * V8 + j] /= d;
            r[i * V8 + j] /= d;
        }

        // Zero all other lines.
        for (int j = 0; j < V8; j++)
        {
            if (j != i)
            {
                float t = m[j * V8 + i];

                for (int k = i; k < V8; k++)
                {
                    m[j * V8 + k] -= m[i * V8 + k] * t;
                }
            }
        }
    }

    return 0;
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
            if (fabs(m[j * V16 + j]) > fabs(m[lead_i * V16 + lead_i]))
            {
                lead_i = j;
            }
        }
        if (fabs(m[lead_i * V16 + lead_i]) < MATHS_EPS)
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
        float d = m[i * V16 + i];
        for (int j = i; j < V16; j++)
        {
            m[i * V16 + j] /= d;
            r[i * V16 + j] /= d;
        }

        // Zero all other lines.
        for (int j = 0; j < V16; j++)
        {
            if (j != i)
            {
                float t = m[j * V16 + i];

                for (int k = i; k < V16; k++)
                {
                    m[j * V16 + k] -= m[i * V16 + k] * t;
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
            if (fabs(m[j * V16 + j]) > fabs(m[lead_i * V16 + lead_i]))
            {
                lead_i = j;
            }
        }
        if (fabs(m[lead_i * V16 + lead_i]) < MATHS_EPS)
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
        float d = m[i * V16 + i];
        for (int j = i; j < V16; j++)
        {
            m[i * V16 + j] /= d;
            r[i * V16 + j] /= d;
        }

        // Zero all other lines.
        for (int j = 0; j < V16; j++)
        {
            if (j != i)
            {
                float t = m[j * V16 + i];

                for (int k = i; k < V16; k++)
                {
                    m[j * V16 + k] -= m[i * V16 + k] * t;
                }
            }
        }
    }

    return 0;
}
