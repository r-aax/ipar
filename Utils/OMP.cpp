/// \file
/// \brief OpenMP library implementation.

#include "OMP.h"
#include <omp.h>

namespace Utils {

/// \brief Set dynamic OpenMP.
///
/// \param v - value
void OMP::SetDynamic(bool v)
{
    omp_set_dynamic(v ? 1 : 0);
}

/// \brief Set nested OpenMP.
///
/// \param v - value
void OMP::SetNested(bool v)
{
    omp_set_nested(v ? 1 : 0);
}

/// \brief Check dynamic OpenMP.
///
/// \return
/// Value.
bool OMP::IsDynamic()
{
    return (omp_get_dynamic() == 1);
}

/// \brief Check nested OpenMP.
///
/// \return
/// Value.
bool OMP::IsNested()
{
    return (omp_get_nested() == 1);
}

/// \brief Get threads max count.
///
/// \return
/// Threads max count.
int OMP::MaxThreads()
{
    return omp_get_max_threads();
}

/// \brief Get thread number.
///
/// \return
/// Thread number.
int OMP::ThreadNum()
{
    return omp_get_thread_num();
}

/// \brief Get global time.
///
/// \return
/// Global time.
double OMP::GTime()
{
    return omp_get_wtime();
}

}
