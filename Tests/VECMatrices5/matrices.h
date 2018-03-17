/// \file
/// \brief Matrix operations declarations.

#ifndef MATRICES_H
#define MATRICES_H

/// \brief Constant 5.
#define V5 5

/// \brief Constant 8.
#define V8 8

/// \brief Constant 64.
#define V40 (V5 * V8)

//--------------------------------------------------------------------------------------------------
void matvec5_orig(float * __restrict m, float * __restrict v, float * __restrict r);
void matvec5_opt(float * __restrict m, float * __restrict v, float * __restrict r);
//--------------------------------------------------------------------------------------------------

#endif
