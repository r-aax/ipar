/// \file
/// \brief Matrix operations declarations.

#ifndef MATRICES_H
#define MATRICES_H

/// \brief Constant 8.
#define V8 8

// brief Constant 16.
#define V16 16

/// \brief Constant 64.
#define V64 (V8 * V8)

/// \brief Constant 256.
#define V256 (V16 * V16)

//--------------------------------------------------------------------------------------------------
void matvec8_orig(float * __restrict matr, float * __restrict vect, float * __restrict matv);
void matvec8_opt(float * __restrict matr, float * __restrict vect, float * __restrict matv);
void matvec8_opt2(float * __restrict matr, float * __restrict vect, float * __restrict matv);
//--------------------------------------------------------------------------------------------------
void matvec16_orig(float * __restrict matr, float * __restrict vect, float * __restrict matv);
void matvec16_opt(float * __restrict matr, float * __restrict vect, float * __restrict matv);
void matvec16_opt2(float * __restrict matr, float * __restrict vect, float * __restrict matv);
//--------------------------------------------------------------------------------------------------
void matmat8_orig(float * __restrict a, float * __restrict b, float * __restrict r);
void matmat8_opt(float * __restrict a, float * __restrict b, float * __restrict r);
//--------------------------------------------------------------------------------------------------
void matmat16_orig(float * __restrict a, float * __restrict b, float * __restrict r);
void matmat16_opt(float * __restrict a, float * __restrict b, float * __restrict r);
//--------------------------------------------------------------------------------------------------
int invmat8_orig(float * __restrict m, float * __restrict r);
int invmat8_opt(float * __restrict m, float * __restrict r);
//--------------------------------------------------------------------------------------------------
int invmat16_orig(float * __restrict m, float * __restrict r);
int invmat16_opt(float * __restrict m, float * __restrict r);
//--------------------------------------------------------------------------------------------------

#endif
