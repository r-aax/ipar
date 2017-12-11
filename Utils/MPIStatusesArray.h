/// \file
/// \brief MPI statuses array description.

#ifndef UTILS_MPI_STATUSES_ARRAY_H
#define UTILS_MPI_STATUSES_ARRAY_H

namespace Utils {

/// \brief Statuses array.
class MPIStatusesArray
{

public:

    // Constructor/destructor.
    MPIStatusesArray(int n);
    ~MPIStatusesArray();

    char *Arr; ///< array
};

}

#endif
