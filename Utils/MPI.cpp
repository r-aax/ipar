/// \file
/// \brief MPI library implementation.

#ifdef USEMPI
#include <mpi.h>
#endif

#include "MPI.h"
#include "IO.h"
#include "Timer.h"

namespace Utils {

/// \brief Initialization.
///
/// \param argc - arguments count
/// \param argv - arguments
void MPI::Init(int *argc, char ***argv)
{

#ifdef USEMPI

    MPI_Init(argc, argv);

#endif

}

/// \brief Finalization.
void MPI::Finalize()
{

#ifdef USEMPI

    MPI_Finalize();

#endif

}

/// \brief Get process number.
///
/// \return
/// MPI process number.
int MPI::Rank()
{
    static int r = 0;
    static int is_r = false;

    if (!is_r)
    {

#ifdef USEMPI

        MPI_Comm_rank(MPI_COMM_WORLD, &r);

#else

        r = 0;

#endif

        is_r = true;
    }

    return r;
}

/// \brief MPI processes count.
///
/// \return
/// MPI processes count.
int MPI::Ranks()
{
    static int r = 0;
    static int is_r = false;

    if (!is_r)
    {

#ifdef USEMPI

        MPI_Comm_size(MPI_COMM_WORLD, &r);

#else

        r = 1;

#endif

        is_r = true;
    }

    return r;
}

/// \brief Request size.
///
/// \return
/// Request size.
int MPI::RequestSize()
{

#ifdef USEMPI

    return sizeof(MPI_Request);

#else

    // Return not zero value (for the reason not zero requests array length).
    return MinRequestSize;

#endif

}

/// \brief Status size.
///
/// \return
/// Status size.
int MPI::StatusSize()
{

#ifdef USEMPI

    return sizeof(MPI_Status);

#else

    // Return not zero value (for the reason not zero statuses array length).
    return MinStatusSize;

#endif

}

/// \brief Barrier.
void MPI::Barrier()
{

#ifdef USEMPI

    MPI_Barrier(MPI_COMM_WORLD);

#endif

}

/// \brief Global time.
///
/// \return
/// Time.
double MPI::GTime()
{

#ifdef USEMPI

    return MPI_Wtime();

#else

    Timer timer(Timer::Default);

    return timer.GTime();

#endif

}

/// \brief Send meessage.
///
/// \param buf - buffer
/// \param count - words count
/// \param dst - destination identifier
/// \param tag - send tag
/// \param req_p - request pointer
void MPI::ISendDoubles(double *buf, int count, int dst, int tag, char *req_p)
{

#ifdef USEMPI

    MPI_Isend(static_cast<void *>(buf),
              count, MPI_DOUBLE, dst, tag, MPI_COMM_WORLD,
              static_cast<MPI_Request *>(static_cast<void *>(req_p)));

#endif

}

/// \brief Receive message.
///
/// \param buf - buffer
/// \param count - words count
/// \param src - source identifier
/// \param tag - send tag
/// \param req - request pointer
void MPI::IRecvDoubles(double *buf, int count, int src, int tag, char *req_p)
{

#ifdef USEMPI

    MPI_Irecv(static_cast<void *>(buf),
              count, MPI_DOUBLE, src, tag, MPI_COMM_WORLD,
              static_cast<MPI_Request *>(static_cast<void *>(req_p)));

#endif

}

/// \brief Wait all requests.
///
/// \param count - requests count
/// \param reqs - requests array
/// \param stats - statuses array
void MPI::WaitAll(int count, MPIRequestsArray *reqs, MPIStatusesArray *stats)
{

#ifdef USEMPI

    MPI_Waitall(count,
                static_cast<MPI_Request *>(static_cast<void *>(reqs->Arr)),
                static_cast<MPI_Status *>(static_cast<void *>(stats->Arr)));

#endif

}

}
