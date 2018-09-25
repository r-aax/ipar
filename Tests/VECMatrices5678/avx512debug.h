/// \file
/// \brief AVX-512 debug functions declarations.

#ifndef AVX_512_DEBUG_H
#define AVX_512_DEBUG_H

#ifdef INTEL

#include <immintrin.h>

float get(__m512 v, int i);
void set(__m512 *v, int i, float f);
void print_m512(__m512 v);

#endif

void print_matrix8x8(float *m);

#endif
