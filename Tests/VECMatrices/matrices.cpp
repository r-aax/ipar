#include "matrices.h"

/// \brief Multiplication 8*8-matrix on 8-vector.
///
/// Original version.
///
/// \param matr - matrix
/// \param vect - vector
/// \param matv - result
void matvec8_orig(float *matr, float *vect, float *matv)
{
    for (int i = 0; i < V8; i++)
    {
        float sum = 0.0;
        int ii = i * V8;

        for (int j = 0; j < V8; j++)
        {
            sum = sum + matr[ii + j] * vect[j];
        }

        matv[i] = sum;
    }
}

/// \brief Multiplication 8*8-matrix on 8-vector.
///
/// Optimized version.
///
/// \param matr - matrix
/// \param vect - vector
/// \param matv - result 
void matvec8_opt(float *matr, float *vect, float *matv)
{
    for (int i = 0; i < V8; i++)
    {
        float sum = 0.0;
        int ii = i * V8;

        for (int j = 0; j < V8; j++)
        {
            sum = sum + matr[ii + j] * vect[j];
        }

        matv[i] = sum;
    }
}

