/// \file
/// \brief IO implementation.

#include "IO.h"

namespace Utils {

/// \brief File zeroing.
///
/// \param filename - file name
///
/// \throw runtime_error - if it is impossible to open the file
void IO::ClearFile(string filename)
{
    ofstream f;

    f.open(filename.c_str());

    if (!f.is_open())
    {
        throw runtime_error("can not open the file");
    }

    f.close();
}

}
