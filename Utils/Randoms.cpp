/// \file
/// \brief Random generation realization.

#include "Randoms.h"
#include <stdlib.h>

namespace Utils {

/// \brief Generate random number from 0.0 to 1.0.
///
/// \return
/// Random value.
double Randoms::Rand01()
{
    return ((double) rand()) / ((double) RAND_MAX);
}

/// \brief Generate random number from lo to hi.
///
/// \return
/// Random number.
double Randoms::Rand(double lo, double hi)
{
    return lo + Rand01() * (hi - lo);
}

}