/// \file
/// \brief Functions declarations for Riemann solver.

#ifndef RIEMANN_H
#define RIEMANN_H

// \brief Test mode.
//
// 0 - small test,
// 1 - big test.
#define TEST_MODE 0

// \brief Tests count.
#if TEST_MODE == 0
#define TESTS_COUNT 32
#else
#define TESTS_COUNT 419984
#endif

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
void samples_orig(float *dls, float *uls, float *pls, float *cls,
                  float *drs, float *urs, float *prs, float *crs,
                  float *pms, float *ums,
                  float *ds, float *us, float *ps);
void samples_opt(float *dls, float *uls, float *pls, float *cls,
                 float *drs, float *urs, float *prs, float *crs,
                 float *pms, float *ums,
                 float *ds, float *us, float *ps);

#endif
