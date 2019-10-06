/// @file
/// @brief Functions declarations.

#ifndef TRI_BOX_INTERSECT_H
#define TRI_BOX_INTERSECT_H

/// @brief Elements count.
#define ELS_COUNT 15

// Original function.
void
tri_box_intersects_orig(float *ax, float *ay, float *az,
                        float *bx, float *by, float *bz,
                        float *cx, float *cy, float *cz,
                        float *xl, float *xh, float *yl, float *yh, float *zl, float *zh,
                        int c,
                        bool *r);

// Optimized fucntion.
void
tri_box_intersects_opt(float *ax, float *ay, float *az,
                       float *bx, float *by, float *bz,
                       float *cx, float *cy, float *cz,
                       float *xl, float *xh, float *yl, float *yh, float *zl, float *zh,
                       int c,
                       bool *r);

#endif // !TRI_BOX_INTERSECTS
