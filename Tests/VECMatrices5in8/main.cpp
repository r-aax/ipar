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

/// \brief matvec8 test cases count.
#define MATVEC5IN8_COUNT 10000

/// \brief matmat8 test cases count.
#define MATMAT5IN8_COUNT 10000

/// \brief invmat8 test cases count.
#define INVMAT5IN8_COUNT 5000

/// \brief Matrices for matvec5in8 test.
ALIGN_64 float matvec5in8_m[MATVEC5IN8_COUNT * V64];

/// \brief Vectors for matvec8 test.
ALIGN_64 float matvec5in8_v[MATVEC5IN8_COUNT * V8];

/// \brief Matrices for matvec8 results.
ALIGN_64 float matvec5in8_r[MATVEC5IN8_COUNT * V8];

/// \brief Matrices a for matmat8 test.
ALIGN_64 float matmat5in8_a[MATMAT5IN8_COUNT * V64];

/// \brief Matrices b for matmat8 test.
ALIGN_64 float matmat5in8_b[MATMAT5IN8_COUNT * V64];

/// \brief Matrices r for matmat8 test.
ALIGN_64 float matmat5in8_r[MATMAT5IN8_COUNT * V64];

/// \brief Matrices m for invmat8 test.
ALIGN_64 float invmat5in8_m[INVMAT5IN8_COUNT * V64];

/// \brief Tmp matrices m for invmat8 test.
ALIGN_64 float invmat5in8_t[INVMAT5IN8_COUNT * V64];

/// \brief Matrices r for invmat8 test.
ALIGN_64 float invmat5in8_r[INVMAT5IN8_COUNT * V64];

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
    clean_array(matvec5in8_r, MATVEC5IN8_COUNT * V8);
    clean_array(matmat5in8_r, MATMAT5IN8_COUNT * V64);
    clean_array(invmat5in8_r, INVMAT5IN8_COUNT * V64);
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

    cout << "VECMatrices5in8 : test begin" << endl;
    cout << "------------------------------" << endl;

    Timer *timer = new Timer(Timer::OMP);
    double time_orig, time_opt;
    double check_orig, check_opt;

    // Init.
    random_array(matvec5in8_m, MATVEC5IN8_COUNT * V64);
    random_array(matvec5in8_v, MATVEC5IN8_COUNT * V8);
    random_array(matmat5in8_a, MATMAT5IN8_COUNT * V64);
    random_array(matmat5in8_b, MATMAT5IN8_COUNT * V64);
    random_array(invmat5in8_m, INVMAT5IN8_COUNT * V64);
    for (int i = 0; i < INVMAT5IN8_COUNT; i++)
    {
        for (int j = 0; j < V8; j++)
        {
            invmat5in8_m[i * V64 + j * V8 + j] *= 2.0;
        }
    }

    // *---------*
    // | matvec5in8 |
    // *---------*

    if (MATVEC5IN8_COUNT > 0)
    {
        // Original.
        clean_res();
        timer->Init();
        run3(repeats_count, MATVEC5IN8_COUNT, matvec5in8_orig, matvec5in8_m, matvec5in8_v, matvec5in8_r, V64, V8, V8);
        timer->Start();
        run3(repeats_count, MATVEC5IN8_COUNT, matvec5in8_orig, matvec5in8_m, matvec5in8_v, matvec5in8_r, V64, V8, V8);
        timer->Stop();
        time_orig = timer->Time();
        check_orig = array_sum(matvec5in8_r, MATVEC5IN8_COUNT * V8);

        // Optimized.
        clean_res();
        timer->Init();
        run3(repeats_count, MATVEC5IN8_COUNT, matvec5in8_opt, matvec5in8_m, matvec5in8_v, matvec5in8_r, V64, V8, V8);
        timer->Start();
        run3(repeats_count, MATVEC5IN8_COUNT, matvec5in8_opt, matvec5in8_m, matvec5in8_v, matvec5in8_r, V64, V8, V8);
        timer->Stop();
        time_opt = timer->Time();
        check_opt = array_sum(matvec5in8_r, MATVEC5IN8_COUNT * V8);

        cout << "VECMatrices5in8 : matvec5in8 : orig = " << time_orig
             << ", opt = " << time_opt << endl;
//        DEBUG_CHECK(MATHS_IS_NEAR(check_orig, check_opt, 0.01), "matvec5in8 opt check failed");
        cout << "VECMatrices5in8 : matvec5in8 check : " << check_orig << endl;
        cout << "------------------------------" << endl;
    }

    // *---------*
    // | matmat5in8 |
    // *---------*

    if (MATMAT5IN8_COUNT > 0)
    {
        // Original.
        clean_res();
        timer->Init();
        run3(repeats_count, MATMAT5IN8_COUNT, matmat5in8_orig, matmat5in8_a, matmat5in8_b, matmat5in8_r, V64, V64, V64);
        timer->Start();
        run3(repeats_count, MATMAT5IN8_COUNT, matmat5in8_orig, matmat5in8_a, matmat5in8_b, matmat5in8_r, V64, V64, V64);
        timer->Stop();
        time_orig = timer->Time();
        check_orig = array_sum(matmat5in8_r, MATMAT5IN8_COUNT * V64);

        // Optimized.
        clean_res();
        timer->Init();
        run3(repeats_count, MATMAT5IN8_COUNT, matmat5in8_opt, matmat5in8_a, matmat5in8_b, matmat5in8_r, V64, V64, V64);
        timer->Start();
        run3(repeats_count, MATMAT5IN8_COUNT, matmat5in8_opt, matmat5in8_a, matmat5in8_b, matmat5in8_r, V64, V64, V64);
        timer->Stop();
        time_opt = timer->Time();
        check_opt = array_sum(matmat5in8_r, MATMAT5IN8_COUNT * V64);

        cout << "VECMatrices5in8 : matmat5in8 : orig = " << time_orig
             << ", opt = " << time_opt << endl;
//        DEBUG_CHECK(MATHS_IS_NEAR(check_orig, check_opt, 0.01), "matmat5in8 check failed");
        cout << "VECMatrices5in8 : matmat5in8 check : " << check_orig << endl;
        cout << "------------------------------" << endl;
    }

    // *---------*
    // | invmat5in8 |
    // *---------*

    if (INVMAT5IN8_COUNT > 0)
    {
        // Original.
        clean_res();
        timer->Init();
        for (int r = 0; r < repeats_count; r++)
        {
            arrays_copy(invmat5in8_m, invmat5in8_t, INVMAT5IN8_COUNT * V64);
            run2(INVMAT5IN8_COUNT, invmat5in8_orig, invmat5in8_t, invmat5in8_r, V64, V64);
        }
        for (int r = 0; r < repeats_count; r++)
        {
            arrays_copy(invmat5in8_m, invmat5in8_t, INVMAT5IN8_COUNT * V64);
            timer->Start();
            run2(INVMAT5IN8_COUNT, invmat5in8_orig, invmat5in8_t, invmat5in8_r, V64, V64);
            timer->Stop();
        }
        time_orig = timer->Time();
        check_orig = array_sum(invmat5in8_r, INVMAT5IN8_COUNT * V64);

        // Optimized.
        clean_res();
        timer->Init();
        for (int r = 0; r < repeats_count; r++)
        {
            arrays_copy(invmat5in8_m, invmat5in8_t, INVMAT5IN8_COUNT * V64);
            run2(INVMAT5IN8_COUNT, invmat5in8_opt, invmat5in8_t, invmat5in8_r, V64, V64);
        }  
        for (int r = 0; r < repeats_count; r++)
        {
            arrays_copy(invmat5in8_m, invmat5in8_t, INVMAT5IN8_COUNT * V64);
            timer->Start();
            run2(INVMAT5IN8_COUNT, invmat5in8_opt, invmat5in8_t, invmat5in8_r, V64, V64);
            timer->Stop();
        }
        time_opt = timer->Time();
        check_opt = array_sum(invmat5in8_r, INVMAT5IN8_COUNT * V64);

        cout << "VECMatrices5in8 : invmat5in8 : orig = " << time_orig
             << ", opt = " << time_opt << endl;
//        DEBUG_CHECK(MATHS_IS_NEAR(check_orig, check_opt, 0.01), "invmat5in8 check failed");
        cout << "VECMatrices5in8 : invmat5in8 check : " << check_orig << endl;
        cout << "------------------------------" << endl;
    }

    delete timer;

    cout << "VECMatrices8 : test end" << endl;
}