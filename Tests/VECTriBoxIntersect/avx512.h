#ifndef AVX512_H
#define AVX512_H

#define LD(addr) _mm512_load_ps(addr)
#define ST(addr, v) _mm512_store_ps(addr, v)
#define ADD(a, b) _mm512_add_ps(a, b)
#define SUB(a, b) _mm512_sub_ps(a, b)
#define SETZERO() _mm512_setzero_ps()
#define SET1(v) _mm512_set1_ps(v)
#define SETONE(v) SET1(1.0)

#endif

