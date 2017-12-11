/// \file
/// \brief MPI requests array description.

#ifndef UTILS_MPI_REQUESTS_ARRAY_H
#define UTILS_MPI_REQUESTS_ARRAY_H

namespace Utils {

/// \brief Requests array.
class MPIRequestsArray
{

public:

    // Constructor/destructor.
    MPIRequestsArray(int n);
    ~MPIRequestsArray();

    // Request pointer.
    char *ReqP(int i);

    char *Arr; ///< array
};

}

#endif
