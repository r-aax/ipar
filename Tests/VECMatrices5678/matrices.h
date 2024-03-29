/// \file
/// \brief Matrix operations declarations.

#ifndef MATRICES_H
#define MATRICES_H

/// \brief Constant 8.
#define V8 8

/// \brief Constant 64.
#define V64 (V8 * V8)

//--------------------------------------------------------------------------------------------------
void om_mult_mm_8x8_orig(float * __restrict a, float * __restrict b, float * __restrict r);
void om_mult_mm_8x8_opt(float * __restrict a, float * __restrict b, float * __restrict r);
//--------------------------------------------------------------------------------------------------
void om_mult_mm_7x7_orig(float * __restrict a, float * __restrict b, float * __restrict r);
void om_mult_mm_7x7_opt(float * __restrict a, float * __restrict b, float * __restrict r);
//--------------------------------------------------------------------------------------------------
void om_mult_mm_6x6_orig(float * __restrict a, float * __restrict b, float * __restrict r);
void om_mult_mm_6x6_opt(float * __restrict a, float * __restrict b, float * __restrict r);
//--------------------------------------------------------------------------------------------------
void om_mult_mm_5x5_orig(float * __restrict a, float * __restrict b, float * __restrict r);
void om_mult_mm_5x5_opt(float * __restrict a, float * __restrict b, float * __restrict r);
//--------------------------------------------------------------------------------------------------

#endif
