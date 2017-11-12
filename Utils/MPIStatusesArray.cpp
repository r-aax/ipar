/// \file
/// \brief MPI statuses array implementation.

#include "MPIStatusesArray.h"
#include "MPI.h"
#include "Debug.h"

namespace Utils {

/// \brief Constructor.
///
/// \param n - count
MPIStatusesArray::MPIStatusesArray(int n)
{
    int ss = MPI::StatusSize();

    DEBUG_CHECK((ss % 4) == 0, "request status is not divisible on 4");

    Arr = new char[n * ss];
}

/// \brief Destructor.
MPIStatusesArray::~MPIStatusesArray()
{
    delete [] Arr;
}

}
