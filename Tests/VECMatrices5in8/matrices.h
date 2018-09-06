/// \file
/// \brief Matrix operations declarations.

#ifndef MATRICES_H
#define MATRICES_H

/// \brief Constant 5.
#define V5 5

/// \brief Constant 8.
#define V8 8

/// \brief Constant 16.
#define V16 16

/// \brief Constant 64.
#define V64 (V8 * V8)

//--------------------------------------------------------------------------------------------------
void matvec5in8_orig(float * __restrict m, float * __restrict v, float * __restrict r);
void matvec5in8_opt(float * __restrict m, float * __restrict v, float * __restrict r);
//--------------------------------------------------------------------------------------------------
void matmat5in8_orig(float * __restrict a, float * __restrict b, float * __restrict r);
void matmat5in8_opt(float * __restrict a, float * __restrict b, float * __restrict r);
//--------------------------------------------------------------------------------------------------
int invmat5in8_orig(float * __restrict m, float * __restrict r);
int invmat5in8_opt(float * __restrict m, float * __restrict r);
//--------------------------------------------------------------------------------------------------

#endif
