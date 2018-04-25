/// \file
/// \brief AVX-512 debug functions implementations.

#include "avx512debug.h"
#include "../../Utils/IO.h"

#ifdef INTEL

#include <immintrin.h>

using namespace std;

/// \brief Print __m512d vector.
///
/// \param v - vector
void print_m512d(__m512d v)
{
    __declspec(align(64)) double tmp[8];

    __assume_aligned(&tmp[0], 64);

    _mm512_store_pd(&tmp[0], v);

    cout << "__m512dd [";
    for (int i = 0; i < 8; i++)
    {
        cout << tmp[i] << ", ";
    }
    cout << "]" << endl;
}

/// \brief Print __m512i vector.
///
/// \param vi - vector
void print_m512i(__m512i vi)
{
    __declspec(align(64)) int tmp[16];

    __assume_aligned(&tmp[0], 64);

    _mm512_store_epi32(&tmp[0], vi);

    cout << "__m512i [";
    for (int i = 0; i < 16; i++)
    {
	cout << tmp[i] << ", ";
    }
    cout << "]" << endl;
}

#endif
