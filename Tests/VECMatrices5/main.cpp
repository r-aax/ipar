/// \file
/// \brief Matrices operations vectorization.

#include "../../Utils/IO.h"
#include "../../Utils/Randoms.h"
#include "../../Utils/Timer.h"
#include "../../Utils/Maths.h"
#include "../../Utils/Debug.h"
#include "matrices.h"
#include "avx512debug.h"
#include <stdlib.h>

/// \brief Align on 64 macro.
#ifdef INTEL
#define ALIGN_64 __declspec(align(64))
#else
#define ALIGN_64
#endif

/// \brief matvec5 test cases count.
#define MATVEC5_COUNT 3000

/// \brief Matrices for matvec5 test.
ALIGN_64 float matvec5_m[MATVEC5_COUNT * V48];

/// \brief Vectors for matvec8 test.
ALIGN_64 float matvec5_v[MATVEC5_COUNT * V8];

/// \brief Matrices for matvec8 results.
ALIGN_64 float matvec5_r[MATVEC5_COUNT * V8];

using namespace Utils;

/// \brief Sum of float array elements.
///
/// \param a - array
/// \param c - elements count
///
/// \return
/// Sum of elements.
static double array_sum(float *a, int c)
{
    double s = 0.0;

    for (int i = 0; i < c; i++)
    {
        s += a[i];
    }

    return s;
}

/// \brief Clean array.
///
/// \param a - array
/// \param c - count of elements
static void clean_array(float *a, int c)
{
    for (int i = 0; i < c; i++)
    {
        a[i] = 0.0;
    }
}

/// \brief Set random values to array.
///
/// \param a - array
/// \param c - count of elements
static void random_array(float *a, int c)
{
    for (int i = 0; i < c; i++)
    {
        a[i] = Randoms::Rand01();
    }
}

/// \brief Copy one array to another.
///
/// \param from - first array (src)
/// \param to - second array (dst)
/// \param c - elements count
static void arrays_copy(float *from, float *to, int c)
{
    for (int i = 0; i < c; i++)
    {
        to[i] = from[i];
    }
}

/// \brief Run 2-arguments function in cycle.
///
/// \param c - iterations count
/// \param f - function
/// \param arr1 - first data array
/// \param arr2 - second data array
/// \param sc1 - first scale factor
/// \param sc2 - second scale factor
static void run2(int c,
                 int (*f)(float *, float *),
                 float *arr1, float *arr2,
                 int sc1, int sc2)
{
    for (int i = 0; i < c; i++)
    {
        f(&arr1[i * sc1], &arr2[i * sc2]);
    }
}

/// \brief Run 3-arguments function in cycle.
///
/// \param r - repeats count
/// \param c - iterations count
/// \param f - function
/// \param arr1 - first data array
/// \param arr2 - second data array
/// \param arr3 - third data array
/// \param sc1 - first scale factor
/// \param sc2 - second scale factor
/// \param sc3 - third scale factor
static void run3(int r, int c,
                 void (*f)(float *, float *, float *),
                 float *arr1, float *arr2, float *arr3,
                 int sc1, int sc2, int sc3)
{
    for (int j = 0; j < r; j++)
    {
        for (int i = 0; i < c; i++)
        {
            f(&arr1[i * sc1], &arr2[i * sc2], &arr3[i * sc3]);
        }
    }
}

/// \brief Clean results.
static void clean_res()
{
    clean_array(matvec5_r, MATVEC5_COUNT * V8);
}

/// \brief Main function.
///
/// \param argc - arguments count
/// \param argv - arguments
int main(int argc, char **argv)
{
    int repeats_count = 1;

    // Parse repeats count if given.
    if (argc == 2)
    {
	   repeats_count = atoi(argv[1]);
    }

    cout << "VECMatrices5 : test begin" << endl;
    cout << "------------------------------" << endl;

    Timer *timer = new Timer(Timer::OMP);
    double time_orig, time_opt, time_opt2;
    double check_orig, check_opt, check_opt2;

    // Init.
    random_array(matvec5_m, MATVEC5_COUNT * V48);
    random_array(matvec5_v, MATVEC5_COUNT * V8);

    // *---------*
    // | matvec5 |
    // *---------*

    if (MATVEC5_COUNT > 0)
    {
        // Original.
        clean_res();
        timer->Init();
        run3(repeats_count, MATVEC5_COUNT, matvec5_orig, matvec5_m, matvec5_v, matvec5_r, V48, V8, V8);
        timer->Start();
        run3(repeats_count, MATVEC5_COUNT, matvec5_orig, matvec5_m, matvec5_v, matvec5_r, V48, V8, V8);
        timer->Stop();
        time_orig = timer->Time();
        check_orig = array_sum(matvec5_r, MATVEC5_COUNT * V8);

        // Optimized.
        clean_res();
        timer->Init();
        run3(repeats_count, MATVEC5_COUNT / 3, matvec5_3x_opt, matvec5_m, matvec5_v, matvec5_r, 3 * V48, 3 * V8, 3 * V8);
        timer->Start();
        run3(repeats_count, MATVEC5_COUNT / 3, matvec5_3x_opt, matvec5_m, matvec5_v, matvec5_r, 3 * V48, 3 * V8, 3 * V8);
        timer->Stop();
        time_opt = timer->Time();
        check_opt = array_sum(matvec5_r, MATVEC5_COUNT * V8);

        cout << "VECMatrices5 : matvec5 : orig = " << time_orig
             << ", opt = " << time_opt << endl;
        DEBUG_CHECK(MATHS_IS_NEAR(check_orig, check_opt, 0.01), "matvec5 opt check failed");
        cout << "VECMatrices5 : matvec5 check : " << check_orig << endl;
        cout << "------------------------------" << endl;
    }

    delete timer;

    cout << "VECMatrices5 : test end" << endl;
}