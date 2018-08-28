/// \file
/// \brief Shell sort realization.

#include "shell_sort.h"
#include "../../Utils/Maths.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "avx512debug.h"

#ifdef INTEL
#include <immintrin.h>
#endif

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

#ifdef INTEL
__m512i ind_straight;
__m512i ind_k;
#endif

/// \brief Shell sort.
///
/// \param m - array
/// \param n - array element count
void shell_sort_orig(float *m, int n)
{
    int i, j, k;

    for (k = n / 2; k > 0; k /= 2)
    {
        for (i = k; i < n; i++)
        {
            float t = m[i];

            for (j = i; j >= k; j -= k)
            {
                if (t < m[j - k])
                {
                    m[j] = m[j - k];
                }
                else
                {
                    break;
                }
            }

            m[j] = t;
        }
    }
}

#ifdef INTEL

/// \brief Insert from i position for step k in shell sorting.
///
/// \param m - array
/// \param n - array size
/// \param k - step size
/// \param i - insert position
void shell_sort_k_i(float *m, int n, int k, int i)
{
    int j;
    float t = m[i];

    for (j = i; j >= k; j -= k)
    {
        if (t < m[j - k])
        {
             m[j] = m[j - k];
        }
        else
        {
            break;
        }
    }

    m[j] = t;
}

/// \brief Insert from i position for step k in shell sorting.
///
/// Version, modified for vectorization.
///
/// \param m - array
/// \param n - array size
/// \param k - step size
/// \param i - insert position
void shell_sort_k_i_mod(float *m, int n, int k, int i)
{
    int j = i;
    float t = m[j];

    do
    {
        bool p1 = (j >= k);

        if (!p1)
        {
            break;
        }

        float q = m[j - k];

        bool p2 = (t < q);

        if (!p2)
        {
            break;
        }

        m[j] = q;
        j -= k;
    }
    while (true);

    m[j] = t;
}

/// \brief Insert from i position for step k in shell sorting (start with given mask).
///
/// \param m - array
/// \param n - array size
/// \param k - step size
/// \param i - insert position
/// \param w - wide
void shell_sort_k_i_wide(float *m, int n, int k, int i, int w)
{
    int j = i;
    __mmask16 mask = ((unsigned int)0xFFFF) >> (16 - w);
    __m512i ind_j = _mm512_add_epi32(_mm512_set1_epi32(i), ind_straight);
    __m512 t, q;

    t = _mm512_mask_load_ps(t, mask, &m[j]);

    do
    {
        mask = mask & _mm512_mask_cmp_epi32_mask(mask, ind_j, ind_k, _MM_CMPINT_GE);
        q = _mm512_mask_load_ps(q, mask, &m[j - k]);
        mask = mask & _mm512_mask_cmp_ps_mask(mask, t, q, _MM_CMPINT_LT);
        _mm512_mask_store_ps(&m[j], mask, q);
        ind_j = _mm512_mask_sub_epi32(ind_j, mask, ind_j, ind_k);
        j -= k;
    }
    while (mask != 0x0);

    _mm512_i32scatter_ps(m, ind_j, t, _MM_SCALE_4);
}

/// \brief Shell sort step.
///
/// \param m - array
/// \param n - array size
/// \param k - step size
/// \param w - wide
void shell_sort_k(float *m, int n, int k, int w)
{
    int i = k;

    ind_k = _mm512_set1_epi32(k);

    while (i + 15 < n)
    {
        shell_sort_k_i_wide(m, n, k, i, w);
        i += w;
    }

    if (i + 1 < n)
    {
        shell_sort_k_i_wide(m, n, k, i, n - i);
    }
    else if (i < n)
    {
        shell_sort_k_i(m, n, k, i);
    }
}

/// \brief Shell sort small step.
///
/// \param m - array
/// \param n - array size
void shell_sort_1(float *m, int n)
{
    for (int i = 1; i < n; i++)
    {
        shell_sort_k_i(m, n, 1, i);
    }
}

#endif

/// \brief Prepare for optimized version of shell sort.
void shell_sort_opt_prepare()
{

#ifdef INTEL

    ind_straight = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8,
                                     7,  6,  5,  4,  3,  2, 1, 0);

#endif

}

/// \brief Shell sort.
///
/// \param m - array
/// \param n - array element count
void shell_sort_opt(float *m, int n)
{

#ifndef INTEL

    shell_sort_orig(m, n);

#else

    int k;

    for (k = n / 2; k > 1; k /= 2)
    {
        shell_sort_k(m, n, k, MIN(16, k));
    }

    for (; k > 0; k /= 2)
    {
        shell_sort_1(m, n);
    }

#endif

}
