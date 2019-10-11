/// @file
/// @brief Functions declarations.

#ifndef TRI_BOX_INTERSECT_H
#define TRI_BOX_INTERSECT_H

/// @brief Elements count.
#define ELS_COUNT 15

/// @brief Vectorization length.
#define VEC_WIDTH 16

// Original function.
void
tri_box_intersects_orig(float * __restrict__ ax,
                        float * __restrict__ ay,
                        float * __restrict__ az,
                        float * __restrict__ bx,
                        float * __restrict__ by,
                        float * __restrict__ bz,
                        float * __restrict__ cx,
                        float * __restrict__ cy,
                        float * __restrict__ cz,
                        float * __restrict__ xl,
                        float * __restrict__ xh,
                        float * __restrict__ yl,
                        float * __restrict__ yh,
                        float * __restrict__ zl,
                        float * __restrict__ zh,
                        int c,
                        int * __restrict__ r);

// Optimized fucntion.
void
tri_box_intersects_opt(float * __restrict__ ax,
                       float * __restrict__ ay,
                       float * __restrict__ az,
                       float * __restrict__ bx,
                       float * __restrict__ by,
                       float * __restrict__ bz,
                       float * __restrict__ cx,
                       float * __restrict__ cy,
                       float * __restrict__ cz,
                       float * __restrict__ xl,
                       float * __restrict__ xh,
                       float * __restrict__ yl,
                       float * __restrict__ yh,
                       float * __restrict__ zl,
                       float * __restrict__ zh,
                       int c,
                       int * __restrict__ r);

#endif // !TRI_BOX_INTERSECTS
