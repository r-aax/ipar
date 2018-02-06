/// \file
/// \brief MPI Reduce test.

#include <mpi.h>
#include <stdlib.h>
#include "../../Utils/FMPI.h"
#include "../../Utils/IO.h"
#include "../../Utils/Timer.h"
#include "../../Utils/Debug.h"
#include "../../Utils/Maths.h"

using namespace Utils;

/// \brief Maximum power of 2 for array size for MPI reduces.
#define MAX_2_POWER_FOR_ARRAY_SIZE 25

/// \brief Pair of values - double value with its index.
typedef struct
{
    double D; //< double value
    int I;    //< index
}
DoubleWithIndex;

/// \brief Perform reduces for given power of array size.
///
/// \param power - power of 2 for doubles arrays size
void PerformReduces(int power)
{
    int array_size = 1 << power;    
    int rank = FMPI::Rank();
    int ranks = FMPI::Ranks();    

    // Create buffers.
    double *local = new double[array_size];
    double *max = new double[array_size];
    DoubleWithIndex *local2 = new DoubleWithIndex[array_size];
    DoubleWithIndex *maxloc = new DoubleWithIndex[array_size];

    for (int i = 0; i < array_size; i++)
    {
        local[i] = (double)(i + rank);
        max[i] = 0.0;
        local2[i].D = (double)(i + rank);
        local2[i].I = rank;
        maxloc[i].D = 0.0;
        maxloc[i].I = 0;
    }

    // Start timer.
    Timer *timer = new Timer(Timer::MPI);
    FMPI::Barrier();
    timer->Start();

    MPI_Allreduce(local, max, array_size, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(local2, maxloc, array_size, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

    FMPI::Barrier();

    // Stop timer.
    double t = timer->Stop();
    if (FMPI::IsRank0())
    {
        cout << "MPIReduce : power " << power
             << ", time " << t << endl;
    }
    delete timer;

    // Free arrays with checkers.
    for (int i = 0; i < array_size; i++)
    {
        DEBUG_CHECK(MATHS_IS_EQ(max[i], (double)(i + ranks - 1))
                    && MATHS_IS_EQ(max[i], maxloc[i].D)
                    && (maxloc[i].I == ranks - 1),
                    "wrong value after allreduce");
    }
    delete local;
    delete max;
    delete local2;
    delete maxloc;
}

/// \brief Perform exchanges.
void PerformReduces()
{
    for (int i = 0; i <= MAX_2_POWER_FOR_ARRAY_SIZE; i++)
    {
        PerformReduces(i);
    }
}

/// \brief Main function.
///
/// \param argc - arguments count
/// \param argv - arguments
int main(int argc, char **argv)
{
    FMPI::Init(&argc, &argv);

    // Exchanges count.
    int reduces_count = 1;

    // Parse only correct parameters string.
    if (argc == 2)
    {
        reduces_count = atoi(argv[1]);
    }

    // Hi.
    if (FMPI::IsRank0())
    {
        cout << "MPIReduce : test begin, "
             << FMPI::Ranks() << " ranks are in use" << endl;
    }
    FMPI::Barrier();

    // Body.
    for (int rd = 0; rd < reduces_count; rd++)
    {
        PerformReduces();
    }

    // Bye.
    FMPI::Barrier();
    if (FMPI::IsRank0())
    {
        cout << "MPIReduce : test end" << endl;
    }

    FMPI::Finalize();

    return 0;
}
