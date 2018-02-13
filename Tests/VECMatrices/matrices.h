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
void matvec8_orig(float *matr, float *vect, float *matv);
void matvec8_opt(float *matr, float *vect, float *matv);
void matvec8_opt2(float *matr, float *vect, float *matv);
//--------------------------------------------------------------------------------------------------
void matvec16_orig(float *matr, float *vect, float *matv);
void matvec16_opt(float *matr, float *vect, float *matv);
//--------------------------------------------------------------------------------------------------
void matmat8_orig(float *a, float *b, float *r);
void matmat8_opt(float *a, float *b, float *r);
//--------------------------------------------------------------------------------------------------
void matmat16_orig(float *a, float *b, float *r);
void matmat16_opt(float *a, float *b, float *r);
//--------------------------------------------------------------------------------------------------
int invmat8_orig(float *m, float *r);
int invmat8_opt(float *m, float *r);
//--------------------------------------------------------------------------------------------------
int invmat16_orig(float *m, float *r);
int invmat16_opt(float *m, float *r);
//--------------------------------------------------------------------------------------------------

#endif
