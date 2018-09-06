/// \file
/// \brief AVX-512 debug functions implementations.

#include "avx512debug.h"
#include "../../Utils/IO.h"

#ifdef INTEL

#include <immintrin.h>

using namespace std;

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
