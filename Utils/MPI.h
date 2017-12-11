/// \file
/// \brief MPI library description.

#ifndef UTILS_MPI_H
#define UTILS_MPI_H

#include "MPIRequestsArray.h"
#include "MPIStatusesArray.h"

namespace Utils {

/// \brief MPI class.
class MPI
{

public:

    // Init and Finalize.
    static void Init(int *argc, char ***argv);
    static void Finalize();

    // Get process number and count of processes.
    static int Rank();
    static int Ranks();

    /// \brief Check if zero rank.
    ///
    /// \return
    /// true - if it is zero process,
    /// false - otherwise.
    static bool IsRank0()
    {
        return Rank() == 0;
    }

    // Get size of request and status.
    static int RequestSize();
    static int StatusSize();

    // Barrier.
    static void Barrier();

    // Global time.
    static double GTime();

    // Send data with MPI interface.
    static void ISendDoubles(double *buf, int count, int dst, int tag, char *req_p);
    static void IRecvDoubles(double *buf, int count, int src, int tag, char *req_p);

    // Waiting for processes.
    static void WaitAll(int count, MPIRequestsArray *reqs, MPIStatusesArray *stats);

    static const int MinRequestSize = 4; ///< minimum request size
    static const int MinStatusSize = 4;  ///< minimum status size
};

}

#endif
