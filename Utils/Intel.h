/// \file
/// \brief Intel specific functions.

#ifndef UTILS_INTEL_H
#define UTILS_INTEL_H

#ifdef INTEL

#include <immintrin.h>

/// \brief Macro for 2 swiz + 2 add + 1 blend.
///
/// \param X - first vector
/// \param Y - second vector
/// \param SWIZ_TYPE - swizzle parameter
/// \param BLEND_MASK - blend mask
#define INTEL_SWIZ_2_ADD_2_BLEND_1(X, Y, SWIZ_TYPE, BLEND_MASK) \
    _mm512_mask_blend_ps(BLEND_MASK, \
                         _mm512_add_ps(X, _mm512_swizzle_ps(X, SWIZ_TYPE)), \
                         _mm512_add_ps(Y, _mm512_swizzle_ps(Y, SWIZ_TYPE)))

/// \brief Macro for 2 perm + 2 add + 1 blend.
///
/// \param X - first vector
/// \param Y - second vector
/// \param PERM_TYPE - permute parameter
/// \param BLEND_MASK - blend mask
#define INTEL_PERM_2_ADD_2_BLEND_1(X, Y, PERM_TYPE, BLEND_MASK) \
    _mm512_mask_blend_ps(BLEND_MASK, \
                         _mm512_add_ps(X, _mm512_permute4f128_ps(X, PERM_TYPE)), \
                         _mm512_add_ps(Y, _mm512_permute4f128_ps(Y, PERM_TYPE)))

/// \brief Macro for 2 swiz + 2 blend + 1 add.
///
/// \param X - first vector
/// \param Y - second vector
/// \param SWIZ_TYPE - swizzle parameter
/// \param BLEND_MASK - blend mask
#define INTEL_SWIZ_2_BLEND_2_ADD_1(X, Y, SWIZ_TYPE, BLEND_MASK) \
    _mm512_add_ps(_mm512_mask_blend_ps(BLEND_MASK, X, Y), \
                  _mm512_mask_blend_ps(BLEND_MASK, \
                                       _mm512_swizzle_ps(X, SWIZ_TYPE), \
                                       _mm512_swizzle_ps(Y, SWIZ_TYPE)))

/// \brief Macro for 2 perm + 2 blend + 1 add.
///
/// \param X - first vector
/// \param Y - second vector
/// \param SWIZ_TYPE - permute parameter
/// \param BLEND_MASK - blend mask
#define INTEL_PERM_2_BLEND_2_ADD_1(X, Y, PERM_TYPE, BLEND_MASK) \
    _mm512_add_ps(_mm512_mask_blend_ps(BLEND_MASK, X, Y), \
                  _mm512_mask_blend_ps(BLEND_MASK, \
                                       _mm512_permute4f128_ps(X, PERM_TYPE), \
                                       _mm512_permute4f128_ps(Y, PERM_TYPE)))

#endif

#endif