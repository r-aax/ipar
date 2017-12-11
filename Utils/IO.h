/// \file
/// \brief IO declaration.

#ifndef UTILS_IO_H
#define UTILS_IO_H

#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <sstream>

using namespace std;

namespace Utils {

/// \brief IO class.
class IO
{

public:

    /// \brief To string cast.
    ///
    /// \param T - type
    /// \param val - value
    ///
    /// \return
    /// String.
    template <typename T>
    static string ToString(T val)
    {
        ostringstream oss;

        oss << val;

        return oss.str();
    }

    // Files.
    static void ClearFile(string filename);
};

}

#endif
