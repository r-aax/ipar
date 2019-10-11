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

/// \brief Define align.
#ifdef INTEL
#define ALIGN __declspec(align(64))
#else
#define ALIGN
#endif

/// \brief Test cases count.
#define COUNT 100000

// Data.
ALIGN float a[COUNT * V64];
ALIGN float b[COUNT * V64];
ALIGN float r[COUNT * V64];

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
    clean_array(r, COUNT * V64);
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

    cout << "VECMatrices5678 : test begin" << endl;
    cout << "------------------------------" << endl;

    Timer *timer = new Timer(Timer::OMP);
    double time_orig, time_opt;
    double check_orig, check_opt;

    // Init.
    random_array(a, COUNT * V64);
    random_array(b, COUNT * V64);

    // *----------------*
    // | om_mult_mm_8x8 |
    // *----------------*

    if (COUNT > 0)
    {
        // Original.
        clean_res();
        timer->Init();
        run3(repeats_count, COUNT, om_mult_mm_8x8_orig, a, b, r, V64, V64, V64);
        timer->Start();
        run3(repeats_count, COUNT, om_mult_mm_8x8_orig, a, b, r, V64, V64, V64);
        timer->Stop();
        time_orig = timer->Time();
        check_orig = array_sum(r, COUNT * V64);

        // Optimized.
        clean_res();
        timer->Init();
        run3(repeats_count, COUNT, om_mult_mm_8x8_opt, a, b, r, V64, V64, V64);
        timer->Start();
        run3(repeats_count, COUNT, om_mult_mm_8x8_opt, a, b, r, V64, V64, V64);
        timer->Stop();
        time_opt = timer->Time();
        check_opt = array_sum(r, COUNT * V64);

        cout << "VECMatrices5678 : om_mult_mm_8x8 : orig = " << time_orig
             << ", opt = " << time_opt 
             << ", speedup = " << time_orig / time_opt  << endl;
        DEBUG_CHECK(MATHS_IS_NEAR(check_orig, check_opt, 0.01), "om_mult_mm_8x8 check failed");
        cout << "VECMatrices5678 : om_mult_mm_8x8 check : " << check_orig << endl;
        cout << "------------------------------" << endl;
    }

    // *----------------*
    // | om_mult_mm_7x7 |
    // *----------------*

    if (COUNT > 0)
    {
        // Original.
        clean_res();
        timer->Init();
        run3(repeats_count, COUNT, om_mult_mm_7x7_orig, a, b, r, V64, V64, V64);
        timer->Start();
        run3(repeats_count, COUNT, om_mult_mm_7x7_orig, a, b, r, V64, V64, V64);
        timer->Stop();
        time_orig = timer->Time();
        check_orig = array_sum(r, COUNT * V64);

        // Optimized.
        clean_res();
        timer->Init();
        run3(repeats_count, COUNT, om_mult_mm_7x7_opt, a, b, r, V64, V64, V64);
        timer->Start();
        run3(repeats_count, COUNT, om_mult_mm_7x7_opt, a, b, r, V64, V64, V64);
        timer->Stop();
        time_opt = timer->Time();
        check_opt = array_sum(r, COUNT * V64);

        cout << "VECMatrices5678 : om_mult_mm_7x7 : orig = " << time_orig
             << ", opt = " << time_opt 
             << ", speedup = " << time_orig / time_opt  << endl;
//        DEBUG_CHECK(MATHS_IS_NEAR(check_orig, check_opt, 0.01), "om_mult_mm_7x7 check failed");
        cout << "VECMatrices5678 : om_mult_mm_7x7 check : " << check_orig << endl;
        cout << "------------------------------" << endl;
    }

    // *----------------*
    // | om_mult_mm_6x6 |
    // *----------------*

    if (COUNT > 0)
    {
        // Original.
        clean_res();
        timer->Init();
        run3(repeats_count, COUNT, om_mult_mm_6x6_orig, a, b, r, V64, V64, V64);
        timer->Start();
        run3(repeats_count, COUNT, om_mult_mm_6x6_orig, a, b, r, V64, V64, V64);
        timer->Stop();
        time_orig = timer->Time();
        check_orig = array_sum(r, COUNT * V64);

        // Optimized.
        clean_res();
        timer->Init();
        run3(repeats_count, COUNT, om_mult_mm_6x6_opt, a, b, r, V64, V64, V64);
        timer->Start();
        run3(repeats_count, COUNT, om_mult_mm_6x6_opt, a, b, r, V64, V64, V64);
        timer->Stop();
        time_opt = timer->Time();
        check_opt = array_sum(r, COUNT * V64);

        cout << "VECMatrices5678 : om_mult_mm_6x6 : orig = " << time_orig
             << ", opt = " << time_opt 
             << ", speedup = " << time_orig / time_opt  << endl;
//        DEBUG_CHECK(MATHS_IS_NEAR(check_orig, check_opt, 0.01), "om_mult_mm_6x6 check failed");
        cout << "VECMatrices5678 : om_mult_mm_6x6 check : " << check_orig << endl;
        cout << "------------------------------" << endl;
    }

    // *----------------*
    // | om_mult_mm_5x5 |
    // *----------------*

    if (COUNT > 0)
    {
        // Original.
        clean_res();
        timer->Init();
        run3(repeats_count, COUNT, om_mult_mm_5x5_orig, a, b, r, V64, V64, V64);
        timer->Start();
        run3(repeats_count, COUNT, om_mult_mm_5x5_orig, a, b, r, V64, V64, V64);
        timer->Stop();
        time_orig = timer->Time();
        check_orig = array_sum(r, COUNT * V64);

        // Optimized.
        clean_res();
        timer->Init();
        run3(repeats_count, COUNT, om_mult_mm_5x5_opt, a, b, r, V64, V64, V64);
        timer->Start();
        run3(repeats_count, COUNT, om_mult_mm_5x5_opt, a, b, r, V64, V64, V64);
        timer->Stop();
        time_opt = timer->Time();
        check_opt = array_sum(r, COUNT * V64);

        cout << "VECMatrices5678 : om_mult_mm_5x5 : orig = " << time_orig
             << ", opt = " << time_opt 
             << ", speedup = " << time_orig / time_opt  << endl;
//        DEBUG_CHECK(MATHS_IS_NEAR(check_orig, check_opt, 0.01), "om_mult_mm_5x5 check failed");
        cout << "VECMatrices5678 : om_mult_mm_5x5 check : " << check_orig << endl;
        cout << "------------------------------" << endl;
    }

    delete timer;

    cout << "VECMatrices5678 : test end" << endl;
}