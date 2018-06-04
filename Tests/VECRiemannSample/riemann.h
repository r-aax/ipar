/// \file
/// \brief Functions declarations for Riemann solver.

#ifndef RIEMANN_H
#define RIEMANN_H

// \brief Tests count.
#define TESTS_COUNT 222640

/// \brief Gama value.
#define GAMA 1.4

/// \brief Gama 1.
#define G1 (((GAMA) - 1.0) / (2.0 * (GAMA)))

/// \brief Gama 2.
#define G2 (((GAMA) + 1.0) / (2.0 * (GAMA)))

/// \brief Gama 3.
#define G3 (2.0 * (GAMA) / ((GAMA) - 1.0))

/// \brief Gama 4.
#define G4 (2.0 / ((GAMA) - 1.0))

/// \brief Gama 5.
#define G5 (2.0 / ((GAMA) + 1.0))

/// \brief Gama 6.
#define G6 (((GAMA) - 1.0) / ((GAMA) + 1.0))

/// \brief Gama 7.
#define G7 (((GAMA) - 1.0) / 2.0)

/// \brief Gama 8.
#define G8 ((GAMA) - 1.0)

// Sample functions prototypes.
void sample_orig(float dl, float ul, float pl, float cl,
                 float dr, float ur, float pr, float cr,
                 const float pm, const float um,
                 float &d, float &u, float &p);
void sample_opt(float dl, float ul, float pl, float cl,
                float dr, float ur, float pr, float cr,
                const float pm, const float um,
                float &d, float &u, float &p);
void samples_orig(float *dls, float *uls, float *pls, float *cls,
                  float *drs, float *urs, float *prs, float *crs,
                  float *pms, float *ums,
                  float *ds, float *us, float *ps);
void samples_opt(float *dls, float *uls, float *pls, float *cls,
                 float *drs, float *urs, float *prs, float *crs,
                 float *pms, float *ums,
                 float *ds, float *us, float *ps);

#endif
