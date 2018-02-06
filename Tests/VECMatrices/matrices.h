#ifndef MATRICES_H
#define MATRICES_H

/// \brief Constant 8.
#define V8 8

/// \brief Constant 64.
#define V64 (V8 * V8)

void matvec8_orig(float *matr, float *vect, float *matv);
void matvec8_opt(float *matr, float *vect, float *matv);

#endif
