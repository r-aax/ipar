/// \file
/// \brief Random functions realization.

#ifndef UTILS_RANDOMS_H
#define UTILS_RANDOMS_H

namespace Utils {

/// \brief Class for random numbers generation.
class Randoms
{

public:

    // Random double values.
    static double Rand01();
    static double Rand(double lo, double hi);

};

}

#endif
