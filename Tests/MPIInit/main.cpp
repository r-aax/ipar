/// \file
/// \brief MPI Init and Finalization test.

#include "../../Utils/MPI.h"
#include "../../Utils/IO.h"

using namespace Utils;

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
        cout << "MPIInit : test begin, "
             << MPI::Ranks() << " ranks are in use" << endl;
    }
    MPI::Barrier();
    cout << "MPIInit : proc " << MPI::Rank() << endl; 

    // Bye.
    MPI::Barrier();
    if (MPI::IsRank0())
    {
        cout << "MPIInit : test end" << endl;
    }

    MPI::Finalize();

    return 0;
}
