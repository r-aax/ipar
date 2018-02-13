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
