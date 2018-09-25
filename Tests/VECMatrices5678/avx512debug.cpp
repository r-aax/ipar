/// \file
/// \brief AVX-512 debug functions implementations.

#include "avx512debug.h"
#include "../../Utils/IO.h"

#ifdef INTEL

#include <immintrin.h>

using namespace std;

/// \brief Get <c>i</c>-th element from vector.
///
/// \param[in] v - vector
/// \param[in] i - index
///
/// \return
/// Element.
float get(__m512 v, int i)
{
    float arr[16];

    _mm512_store_ps(&arr[0], v);

    return arr[i];
}

/// \brief Set <c>i</c>-th element in vector.
///
/// \param[in,out] v - vector
/// \param[in] i - index
/// \param[in] f - value
void set(__m512 *v, int i, float f)
{
    float arr[16];

    _mm512_store_ps(&arr[0], *v);
    arr[i] = f;
    *v = _mm512_load_ps(&arr[0]);
}

/// \brief Print __m512 vector.
///
/// \param v - vector
void print_m512(__m512 v)
{
    __declspec(align(64)) float tmp[16];

    __assume_aligned(&tmp[0], 64);

    _mm512_store_ps(&tmp[0], v);

    cout << "__m512 [";
    for (int i = 0; i < 16; i++)
    {
        cout << tmp[i] << ", ";
    }
    cout << "]" << endl;
}

#endif

/// \brief Print matrix 8x8.
///
/// \param Matrix pointer.
void print_matrix8x8(float *m)
{
    cout << "matrix 8x8" << endl;

    for (int i = 0; i < 8; i++)
    {
        int ii = i * 8;

        cout << " ";

        for (int j = 0; j < 8; j++)
        {
            cout << m[ii + j] << " ";
        }

        cout << endl;
    }
}
