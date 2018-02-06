/// \file
/// \brief Matrices operations vectorization.

#include "../../Utils/IO.h"
#include "../../Utils/Randoms.h"
#include "../../Utils/Timer.h"
#include "../../Utils/Maths.h"
#include "../../Utils/Debug.h"
#include "matrices.h"

/// \brief Repeats count.
#define REPEAT 100

/// \brief 
#define MATVEC8_COUNT 100000

/// \brief Matrices for matvec8 test.
float matvec8_matr[MATVEC8_COUNT * V64];

/// \brief Vectors for matvec8 test.
float matvec8_vect[MATVEC8_COUNT * V8];

/// \brief Matrices for matvec results.
float matvec8_matv[MATVEC8_COUNT * V64];

using namespace Utils;

/// \brief Sum of matvec8_matv array.
///
/// \return
/// Sum of matvec8_matv array.
static double matvec8_matv_sum()
{
    double res;

    for (int i = 0; i < MATVEC8_COUNT * V64; i++)
    {
        res += (double)matvec8_matv[i];
    }

    return res;
}

/// \brief Main function.
///
/// \param argc - arguments count
/// \param argv - arguments
int main(int argc, char **argv)
{
    cout << "VECMatrices : test begin" << endl;

    Timer *timer = new Timer(Timer::OMP);
    double time_orig, time_opt;
    double check_orig, check_opt;

    // Init.
    for (int i = 0; i < MATVEC8_COUNT * V64; i++)
    {
        matvec8_matr[i] = Randoms::Rand01();
        matvec8_matv[i] = Randoms::Rand01();
    }
    for (int i = 0; i < MATVEC8_COUNT * V8; i++)
    {
        matvec8_vect[i] = Randoms::Rand01();
    }

    // *---------*
    // | matvec8 |
    // *---------*

    // Original.
    timer->Init();
    for (int r = 0; r < REPEAT; r++)
    {
        for (int i = 0; i < MATVEC8_COUNT; i++)
        {
            matvec8_orig(&matvec8_matr[i * V64],
                         &matvec8_vect[i * V8],
                         &matvec8_matv[i * V64]);
        }
    }
    timer->Start();
    for (int r = 0; r < REPEAT; r++)
    {
        for (int i = 0; i < MATVEC8_COUNT; i++)
        {
            matvec8_orig(&matvec8_matr[i * V64],
                         &matvec8_vect[i * V8],
                         &matvec8_matv[i * V64]);
        }
    }
    timer->Stop();
    time_orig = timer->Time();
    check_orig = matvec8_matv_sum();

    // Optimized.
    timer->Init();
    for (int r = 0; r < REPEAT; r++)
    {
        for (int i = 0; i < MATVEC8_COUNT; i++)
        {
            matvec8_opt(&matvec8_matr[i * V64],
                        &matvec8_vect[i * V8],
                        &matvec8_matv[i * V64]);
        }
    }
    timer->Start();
    for (int r = 0; r < REPEAT; r++)
    {
        for (int i = 0; i < MATVEC8_COUNT; i++)
        {
            matvec8_opt(&matvec8_matr[i * V64],
                        &matvec8_vect[i * V8],
                        &matvec8_matv[i * V64]);
        }
    }
    timer->Stop();
    time_opt = timer->Time();
    check_opt = matvec8_matv_sum();

    cout << "VECMatrices : matvec8 : orig = " << time_orig
         << ", opt = " << time_opt << endl;
    cout << "VECMatrices : matvec8 check : orig = " << check_orig
         << ", opt = " << check_opt << endl;
    DEBUG_CHECK(MATHS_IS_EQ(check_orig, check_opt), "matvec8 check failed");

    delete timer;

    cout << "VECMatrices : test end" << endl;
}