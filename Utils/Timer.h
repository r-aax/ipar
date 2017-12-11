/// \file
/// \brief Basic timer.

#ifndef UTILS_TIMER_H
#define UTILS_TIMER_H

namespace Utils {

/// \brief Timer.
class Timer
{

public:

    /// \brief Timer type.
    typedef enum
    {
        MPI,              ///< MPI timer
        OMP,              ///< OpenMP timer
        SysTime,          ///< timer from  sys/time.h
        Default = SysTime ///< default timer type
    }
    Type;

    // Constructor/destructor.
    Timer(Type type);
    ~Timer();

    // Main commands.
    double Init();
    double Start();
    double Stop();
    double Time();
    double GTime();
    void Sleep(double d);

protected:

    // Type.
    Type Type_;

    // Total time.
    double Total_;

    // Active flag.
    bool IsActive_;

    // Last start point.
    double LastStart_;

    // Get timer.
    double LastTime();
};

}

#endif
