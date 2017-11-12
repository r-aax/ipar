/// \file
/// \brief OpenMP library description.

#ifndef UTILS_OMP_H
#define UTILS_OMP_H

namespace Utils {

/// \brief OMP class.
class OMP
{

public:

    // Set and check dynamic and nested OpenMP.
    static void SetDynamic(bool v);
    static void SetNested(bool v);
    static bool IsDynamic();
    static bool IsNested();

    // Threads count.
    static int MaxThreads();
    static int ThreadNum();

    // Global time.
    static double GTime();
};

}

#endif
