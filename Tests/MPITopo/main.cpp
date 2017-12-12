/// \file
/// \brief MPI Topology detection.

#include <stdlib.h>
#include <mpi.h>
#include "../../Utils/FMPI.h"
#include "../../Utils/IO.h"
#include "../../Utils/Timer.h"
#include "../../Utils/ExchangeBuffer.h"
#include "../../Utils/Debug.h"

using namespace Utils;

/// \brief Size of data chunk.
#define CHUNK_SIZE 1024

/// \brief Times of exchanges TO this process FROM other.
Timer **Timers;

/// \brief Iterations count.
int Iterations = 1;

/// \brief Exchange buffer.
ExchangeBuffer *SendBuffer, *ReceiveBuffer;

/// \brief Request.
MPI_Request rq;

/// \brief Status.
MPI_Status st;

/// \brief Process exchange.
///
/// \param i - from which process we calculate time of exchange
/// \param j - to which process we calculate time of exchange
void ExchangeFromItoJ(int i, int j)
{
    if (FMPI::Rank() == i)
    {
        SendBuffer->Set1();
    }

    if (FMPI::Rank() == j)
    {
        ReceiveBuffer->Set0();
    }

    FMPI::Barrier();

    if (FMPI::Rank() == i)
    {
        // send
        MPI_Isend((void *)SendBuffer->BufferDoubles(), CHUNK_SIZE, MPI_DOUBLE, j, 99, MPI_COMM_WORLD, &rq);
    }

    if (FMPI::Rank() == j)
    {
        Timers[i]->Start();

        MPI_Recv((void *)ReceiveBuffer->BufferDoubles(), CHUNK_SIZE, MPI_DOUBLE, i, 99, MPI_COMM_WORLD, &st);

        Timers[i]->Stop();

        DEBUG_CHECK(ReceiveBuffer->Is1(), "wrong value of target buffer");
    }

    FMPI::Barrier();
}

/// \brief Process exchanges.
void Exchanges()
{
    int ranks = FMPI::Ranks();

    for (int i = 0; i < ranks; i++)
    {
        for (int j = 0; j < ranks; j++)
        {
            ExchangeFromItoJ(i, j);
        }
    }
}

/// \brief Collect times.
void CollectTimes()
{
    int ranks = FMPI::Ranks();

    for (int r = 0; r < ranks; r++)
    {
        if (FMPI::Rank() == r)
        {
            for (int i = 0; i < ranks; i++)
            {
                cout << "[ " << setw(3) << setfill('0') << i
                     << "_" << setw(3) << setfill('0') << FMPI::Rank()
                     << " ] " << setw(12) << setfill(' ') << Timers[i]->Time() << endl;
            }
        }

        FMPI::Barrier();
    }    
}

/// \brief Main function.
///
/// \param argc - arguments count
/// \param argv - arguments
int main(int argc, char **argv)
{
    FMPI::Init(&argc, &argv);

    int ranks = FMPI::Ranks();

    // Init.
    Timers = new Timer*[ranks];
    for (int i = 0; i < ranks; i++)
    {
        Timers[i] = new Timer(Timer::MPI);
    }
    SendBuffer = new ExchangeBuffer(CHUNK_SIZE);
    ReceiveBuffer = new ExchangeBuffer(CHUNK_SIZE);

    // Try to parse iterations count.
    if (argc == 2)        
    {
        Iterations = atoi(argv[1]);
    }

    // Hi.
    if (FMPI::IsRank0())
    {
        cout << "MPITopo : test begin, "
             << FMPI::Ranks() << " ranks are in use" << endl;
    }
    FMPI::Barrier();

    for (int i = 0; i < Iterations; i++)
    {
        Exchanges();
    }

    // Bye.
    FMPI::Barrier();
    if (FMPI::IsRank0())
    {
        cout << "MPITopo : test end" << endl;
    }

    // Collect times.
    CollectTimes();

    // Free memory.
    delete SendBuffer;
    delete ReceiveBuffer;
    for (int i = 0; i < ranks; i++)
    {
        delete Timers[i];
    }
    delete Timers;

    FMPI::Finalize();

    return 0;
}
