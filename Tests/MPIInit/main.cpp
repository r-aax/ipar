/// \file
/// \brief MPI Init and Finalization test.

#include "../../Utils/FMPI.h"
#include "../../Utils/IO.h"

using namespace Utils;

/// \brief Main function.
///
/// \param argc - arguments count
/// \param argv - arguments
int main(int argc, char **argv)
{
    FMPI::Init(&argc, &argv);

    // Hi.
    if (FMPI::IsRank0())
    {
        cout << "MPIInit : test begin, "
             << FMPI::Ranks() << " ranks are in use" << endl;
    }
    FMPI::Barrier();
    cout << "MPIInit : proc " << FMPI::Rank() << endl; 

    // Bye.
    FMPI::Barrier();
    if (FMPI::IsRank0())
    {
        cout << "MPIInit : test end" << endl;
    }

    FMPI::Finalize();

    return 0;
}
