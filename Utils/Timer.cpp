/// \file
/// \brief Timer implementation.

#include "Timer.h"
#include "FMPI.h"
#include "OMP.h"
#include <stdlib.h>
#include <sys/time.h>

namespace Utils {

/// \brief Constructor.
///
/// \param type - timer type
Timer::Timer(Type type)
    : Type_(type)
{
    Init();
}

/// \brief Destructor.
Timer::~Timer()
{
}

/// \brief Initialization.
///
/// \return
/// Initialization time value (0.0).
double Timer::Init()
{
    Total_ = 0.0;
    IsActive_ = false;
    LastStart_ = 0.0;

    return 0.0;
}

/// \brief Start timer.
///
/// \return
/// Timer value.
double Timer::Start()
{
    if (!IsActive_)
    {
        LastStart_ = GTime();
        IsActive_ = true;
    }

    return Time();
}

/// \brief Stop timer.
///
/// \return
/// Timer value.
double Timer::Stop()
{
    if (IsActive_)
    {
        Total_ += LastTime();
        IsActive_ = false;
    }

    return Time();
}

/// \brief Get timer current value.
///
/// \return
/// Timer value.
double Timer::Time()
{
    double r = Total_;

    if (IsActive_)
    {
        r += LastTime();
    }

    return r;
}

/// \brief Global time.
///
/// \return
/// Global time.
double Timer::GTime()
{
    double gtime = 0.0;

    switch (Type_)
    {
        case MPI:

            gtime = FMPI::GTime();

            break;

        case OMP:

            gtime = OMP::GTime();

            break;

        case SysTime:
        default:

            struct timeval mytime;

            gettimeofday(&mytime, NULL);
            gtime = (double)mytime.tv_sec
                    + (double)mytime.tv_usec * 1.0e-6;

            break;
    }

    return gtime;
}

/// \brief Sleep.
///
/// \param d - sleep time
void Timer::Sleep(double d)
{
    double s = GTime();

    while (GTime() - s < d)
    {
        // Waiting ...
        ;
    }
}

/// \brief Last time period, if timer is acrive.
///
/// \return
/// Last time period.
double Timer::LastTime()
{
    return GTime() - LastStart_;
}

}
