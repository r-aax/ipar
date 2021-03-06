/// \file
/// \brief AVX-512 debug functions declarations.

#ifndef AVX_512_DEBUG_H
#define AVX_512_DEBUG_H

#ifdef INTEL

#include <immintrin.h>

void print_m512(__m512 v);
void print_m512i(__m512i vi);

#endif

void print_matrix8x8(float *m);

#endif
