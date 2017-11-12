/// \file
/// \brief N to N exchange test realization.

#include "../../Utils/MPI.h"
#include "../../Utils/IO.h"
#include "../../Utils/ExchangeBuffer.h"
#include "../../Utils/Timer.h"
#include "../../Utils/Debug.h"
#include "../../Utils/MPIRequestsArray.h"
#include "../../Utils/MPIStatusesArray.h"

using namespace Utils;

/// \brief Maximum power of 2 for doubles array size for MPI exchanges.
#define MAX_2_POWER_FOR_DOUBLES_ARRAY_SIZE 25

/// \brief Perform exchanges for given power of array size.
///
/// \param power - power of 2 for doubles arrays size
void PerformExchanges(int power)
{
    int array_size = 1 << power;
    int rank = MPI::Rank();
    int ranks = MPI::Ranks();
    int ranksm1 = ranks - 1;
    ExchangeBuffer *buffers_snd[ranksm1];
    ExchangeBuffer *buffers_rcv[ranksm1];

    // Init buffers.
    for (int i = 0; i < ranksm1; i++)
    {
        buffers_snd[i] = new ExchangeBuffer(array_size);
        buffers_snd[i]->Set1();
        buffers_rcv[i] = new ExchangeBuffer(array_size);
        buffers_rcv[i]->Set0();
    }

    // Init requests and statuses.
    MPIRequestsArray *reqs = new MPIRequestsArray(2 * ranksm1);
    MPIStatusesArray *stats = new MPIStatusesArray(2 * ranksm1);

    // Start timer.
    Timer *timer = new Timer(Timer::MPI);
    MPI::Barrier();
    timer->Start();

    // Initialize asynchronous sends and receives.
    // After initialization - wait for all exchanges.
    for (int i = 0; i < ranksm1; i++)
    {
        int from_rank = (ranks + rank - i - 1) % ranks;
        MPI::IRecvDoubles(buffers_rcv[i]->BufferDoubles(), buffers_rcv[i]->BufferDoublesCount(),
                          from_rank, from_rank, reqs->ReqP(i));
    }
    for (int i = 0; i < ranksm1; i++)
    {
        int to_rank = (rank + i + 1) % ranks;
        MPI::ISendDoubles(buffers_snd[i]->BufferDoubles(), buffers_snd[i]->BufferDoublesCount(),
                          to_rank, rank, reqs->ReqP(ranksm1 + i));
    }
    MPI::WaitAll(2 * ranksm1, reqs, stats);
    MPI::Barrier();

    // Stop timer.
    double t = timer->Stop();
    if (MPI::IsRank0())
    {
        cout << "MPIExchangeNtoN : power " << power
             << ", time " << t << endl;
    }
    delete timer;

    // Delete requests and statuses.
    delete reqs;
    delete stats;

    // Destroy buffers.
    for (int i = 0; i < ranksm1; i++)
    {
        delete buffers_snd[i];
        DEBUG_CHECK(buffers_rcv[i]->Is1(), "wrong value of target buffer");
        delete buffers_rcv[i];
    }
}

/// \brief Perform exchanges.
void PerformExchanges()
{
    for (int i = 0; i <= MAX_2_POWER_FOR_DOUBLES_ARRAY_SIZE; i++)
    {
        PerformExchanges(i);
    }
}

/// \brief Main function.
///
/// \param argc - arguments count
/// \param argv - arguments
int main(int argc, char **argv)
{
    MPI::Init(&argc, &argv);

    // Hi.
    if (MPI::IsRank0())
    {
        cout << "MPIExchangeNtoN : test begin, "
             << MPI::Ranks() << " ranks are in use" << endl;
    }
    MPI::Barrier();
    cout << "MPIExchangeNtoN : proc " << MPI::Rank() << endl; 
    MPI::Barrier();

    // Body.
    PerformExchanges();

    // Bye.
    MPI::Barrier();
    if (MPI::IsRank0())
    {
        cout << "MPIExchangeNtoN : test end" << endl;
    }

    MPI::Finalize();

    return 0;
}
