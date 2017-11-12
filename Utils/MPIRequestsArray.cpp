/// \file
/// \brief MPI statuses array.

#include "MPIRequestsArray.h"
#include "MPI.h"
#include "Debug.h"

namespace Utils {

/// \brief Constructor.
///
/// \param n - count
MPIRequestsArray::MPIRequestsArray(int n)
{
    int rs = MPI::RequestSize();

    DEBUG_CHECK((rs % 4) == 0, "request size is not divisible on 4");

    Arr = new char[n * rs];
}

/// \brief Destructor.
MPIRequestsArray::~MPIRequestsArray()
{
    delete [] Arr;
}

/// \brief Request pointer.
///
/// \param i - number
///
/// \return
/// Pointer.
char *MPIRequestsArray::ReqP(int i)
{
    return &Arr[i * MPI::RequestSize()];
}

}
