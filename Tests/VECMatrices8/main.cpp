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

/// \brief matvec8 test cases count.
#define MATVEC8_COUNT 10000

/// \brief matvec16 test cases count.
#define MATVEC16_COUNT 5000

/// \brief matmat8 test cases count.
#define MATMAT8_COUNT 10000

/// \brief matmat16 test cases cout.
#define MATMAT16_COUNT 5000

/// \brief invmat8 test cases count.
#define INVMAT8_COUNT 5000

/// \brief invmat16 test cases count.
#define INVMAT16_COUNT 1000

/// \brief Matrices for matvec8 test.
#ifdef INTEL
__declspec(align(64)) float matvec8_m[MATVEC8_COUNT * V64];
#else
float matvec8_m[MATVEC8_COUNT * V64];
#endif

/// \brief Vectors for matvec8 test.
#ifdef INTEL
__declspec(align(64)) float matvec8_v[MATVEC8_COUNT * V8];
#else
float matvec8_v[MATVEC8_COUNT * V8];
#endif

/// \brief Matrices for matvec8 results.
#ifdef INTEL
__declspec(align(64)) float matvec8_r[MATVEC8_COUNT * V8];
#else
float matvec8_r[MATVEC8_COUNT * V8];
#endif

/// \brief Matrices for matvec16 test.
#ifdef INTEL
__declspec(align(64)) float matvec16_m[MATVEC16_COUNT * V256];
#else
float matvec16_m[MATVEC16_COUNT * V256];
#endif

/// \brief Vectors for matvec16 test.
#ifdef INTEL
__declspec(align(64)) float matvec16_v[MATVEC16_COUNT * V16];
#else
float matvec16_v[MATVEC16_COUNT * V16];
#endif

/// \brief Matrices for matvec16 results.
#ifdef INTEL
__declspec(align(64)) float matvec16_r[MATVEC16_COUNT * V16];
#else
float matvec16_r[MATVEC16_COUNT * V16];
#endif

/// \brief Matrices a for matmat8 test.
#ifdef INTEL
__declspec(align(64)) float matmat8_a[MATMAT8_COUNT * V64];
#else
float matmat8_a[MATMAT8_COUNT * V64];
#endif

/// \brief Matrices b for matmat8 test.
#ifdef INTEL
__declspec(align(64)) float matmat8_b[MATMAT8_COUNT * V64];
#else
float matmat8_b[MATMAT8_COUNT * V64];
#endif

/// \brief Matrices r for matmat8 test.
#ifdef INTEL
__declspec(align(64)) float matmat8_r[MATMAT8_COUNT * V64];
#else
float matmat8_r[MATMAT8_COUNT * V64];
#endif

/// \brief Matrices a for matmat16 test.
#ifdef INTEL
__declspec(align(64)) float matmat16_a[MATMAT16_COUNT * V256];
#else
float matmat16_a[MATMAT16_COUNT * V256];
#endif

/// \brief Matrices b for matmat16 test.
#ifdef INTEL
__declspec(align(64)) float matmat16_b[MATMAT16_COUNT * V256];
#else
float matmat16_b[MATMAT16_COUNT * V256];
#endif

/// \brief Matrices r for matmat16 test.
#ifdef INTEL
__declspec(align(64)) float matmat16_r[MATMAT16_COUNT * V256];
#else
float matmat16_r[MATMAT16_COUNT * V256];
#endif

/// \brief Matrices m for invmat8 test.
#ifdef INTEL
__declspec(align(64)) float invmat8_m[INVMAT8_COUNT * V64];
#else
float invmat8_m[INVMAT8_COUNT * V64];
#endif

/// \brief Tmp matrices m for invmat8 test.
#ifdef INTEL
__declspec(align(64)) float invmat8_t[INVMAT8_COUNT * V64];
#else
float invmat8_t[INVMAT8_COUNT * V64];
#endif

/// \brief Matrices r for invmat8 test.
#ifdef INTEL
__declspec(align(64)) float invmat8_r[INVMAT8_COUNT * V64];
#else
float invmat8_r[INVMAT8_COUNT * V64];
#endif

/// \brief Matrices m for invmat16 test.
#ifdef INTEL
__declspec(align(64)) float invmat16_m[INVMAT16_COUNT * V256];
#else
float invmat16_m[INVMAT16_COUNT * V256];
#endif

/// \brief Tmp matrices m for invmat16 test.
#ifdef INTEL
__declspec(align(64)) float invmat16_t[INVMAT16_COUNT * V256];
#else
float invmat16_t[INVMAT16_COUNT * V256];
#endif

/// \brief Matrices r for invmat16 test.
#ifdef INTEL
__declspec(align(64)) float invmat16_r[INVMAT16_COUNT * V256];
#else
float invmat16_r[INVMAT16_COUNT * V256];
#endif

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
    clean_array(matvec8_r, MATVEC8_COUNT * V8);
    clean_array(matvec16_r, MATVEC16_COUNT * V16);
    clean_array(matmat8_r, MATMAT8_COUNT * V64);
    clean_array(matmat16_r, MATMAT16_COUNT * V256);
    clean_array(invmat8_r, INVMAT8_COUNT * V64);
    clean_array(invmat16_r, INVMAT16_COUNT * V256);
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

    cout << "VECMatrices8 : test begin" << endl;
    cout << "------------------------------" << endl;

    Timer *timer = new Timer(Timer::OMP);
    double time_orig, time_opt, time_opt2;
    double check_orig, check_opt, check_opt2;

    // Init.
    random_array(matvec8_m, MATVEC8_COUNT * V64);
    random_array(matvec8_v, MATVEC8_COUNT * V8);
    random_array(matvec16_m, MATVEC16_COUNT * V256);
    random_array(matvec16_v, MATVEC16_COUNT * V16);
    random_array(matmat8_a, MATMAT8_COUNT * V64);
    random_array(matmat8_b, MATMAT8_COUNT * V64);
    random_array(matmat16_a, MATMAT16_COUNT * V256);
    random_array(matmat16_b, MATMAT16_COUNT * V256);
    random_array(invmat8_m, INVMAT8_COUNT * V64);
    for (int i = 0; i < INVMAT8_COUNT; i++)
    {
        for (int j = 0; j < V8; j++)
        {
            invmat8_m[i * V64 + j * V8 + j] *= 2.0;
        }
    }
    random_array(invmat16_m, INVMAT16_COUNT * V256);
    for (int i = 0; i < INVMAT16_COUNT; i++)
    {
        for (int j = 0; j < V16; j++)
        {
            invmat16_m[i * V256 + j * V16 + j] *= 2.0;
        }
    }

    // *---------*
    // | matvec8 |
    // *---------*

    if (MATVEC8_COUNT > 0)
    {
        // Original.
        clean_res();
        timer->Init();
        run3(repeats_count, MATVEC8_COUNT, matvec8_orig, matvec8_m, matvec8_v, matvec8_r, V64, V8, V8);
        timer->Start();
        run3(repeats_count, MATVEC8_COUNT, matvec8_orig, matvec8_m, matvec8_v, matvec8_r, V64, V8, V8);
        timer->Stop();
        time_orig = timer->Time();
        check_orig = array_sum(matvec8_r, MATVEC8_COUNT * V8);

        // Optimized.
        clean_res();
        timer->Init();
        run3(repeats_count, MATVEC8_COUNT, matvec8_opt, matvec8_m, matvec8_v, matvec8_r, V64, V8, V8);
        timer->Start();
        run3(repeats_count, MATVEC8_COUNT, matvec8_opt, matvec8_m, matvec8_v, matvec8_r, V64, V8, V8);
        timer->Stop();
        time_opt = timer->Time();
        check_opt = array_sum(matvec8_r, MATVEC8_COUNT * V8);

        // Optimized 2.
        clean_res();
        timer->Init();
        run3(repeats_count, MATVEC8_COUNT, matvec8_opt2, matvec8_m, matvec8_v, matvec8_r, V64, V8, V8);
        timer->Start();
        run3(repeats_count, MATVEC8_COUNT, matvec8_opt2, matvec8_m, matvec8_v, matvec8_r, V64, V8, V8);
        timer->Stop();
        time_opt2 = timer->Time();
        check_opt2 = array_sum(matvec8_r, MATVEC8_COUNT * V8);

        cout << "VECMatrices8 : matvec8 : orig = " << time_orig
             << ", opt = " << time_opt 
             << ", opt2 = " << time_opt2 << endl;
        DEBUG_CHECK(MATHS_IS_NEAR(check_orig, check_opt, 0.01), "matvec8 opt check failed");
        DEBUG_CHECK(MATHS_IS_NEAR(check_orig, check_opt2, 0.01), "matvec8 opt2 check failed");
        cout << "VECMatrices8 : matvec8 check : " << check_orig << endl;
        cout << "------------------------------" << endl;
    }

    // *----------*
    // | matvec16 |
    // *----------*

    if (MATVEC16_COUNT > 0)
    {
        // Original.
        clean_res();
        timer->Init();
        run3(repeats_count, MATVEC16_COUNT, matvec16_orig, matvec16_m, matvec16_v, matvec16_r, V256, V16, V16);
        timer->Start();
        run3(repeats_count, MATVEC16_COUNT, matvec16_orig, matvec16_m, matvec16_v, matvec16_r, V256, V16, V16);
        timer->Stop();
        time_orig = timer->Time();
        check_orig = array_sum(matvec16_r, MATVEC16_COUNT * V16);

        // Optimized.
        clean_res();
        timer->Init();
        run3(repeats_count, MATVEC16_COUNT, matvec16_opt, matvec16_m, matvec16_v, matvec16_r, V256, V16, V16);
        timer->Start();
        run3(repeats_count, MATVEC16_COUNT, matvec16_opt, matvec16_m, matvec16_v, matvec16_r, V256, V16, V16);
        timer->Stop();
        time_opt = timer->Time();
        check_opt = array_sum(matvec16_r, MATVEC16_COUNT * V16);

        // Optimized 2.
        clean_res();
        timer->Init();
        run3(repeats_count, MATVEC16_COUNT, matvec16_opt2, matvec16_m, matvec16_v, matvec16_r, V256, V16, V16);
        timer->Start();
        run3(repeats_count, MATVEC16_COUNT, matvec16_opt2, matvec16_m, matvec16_v, matvec16_r, V256, V16, V16);
        timer->Stop();
        time_opt2 = timer->Time();
        check_opt2 = array_sum(matvec16_r, MATVEC16_COUNT * V16);

        cout << "VECMatrices8 : matvec16 : orig = " << time_orig
             << ", opt = " << time_opt
             << ", opt2 = " << time_opt2 << endl;
        DEBUG_CHECK(MATHS_IS_NEAR(check_orig, check_opt, 0.01), "matvec16 opt check failed");
        DEBUG_CHECK(MATHS_IS_NEAR(check_orig, check_opt2, 0.01), "matvec16 opt2 check failed");
        cout << "VECMatrices8 : matvec16 check : " << check_orig << endl;
        cout << "------------------------------" << endl;
    }

    // *---------*
    // | matmat8 |
    // *---------*

    if (MATMAT8_COUNT > 0)
    {
        // Original.
        clean_res();
        timer->Init();
        run3(repeats_count, MATMAT8_COUNT, matmat8_orig, matmat8_a, matmat8_b, matmat8_r, V64, V64, V64);
        timer->Start();
        run3(repeats_count, MATMAT8_COUNT, matmat8_orig, matmat8_a, matmat8_b, matmat8_r, V64, V64, V64);
        timer->Stop();
        time_orig = timer->Time();
        check_orig = array_sum(matmat8_r, MATMAT8_COUNT * V64);

        // Optimized.
        clean_res();
        timer->Init();
        run3(repeats_count, MATMAT8_COUNT, matmat8_opt, matmat8_a, matmat8_b, matmat8_r, V64, V64, V64);
        timer->Start();
        run3(repeats_count, MATMAT8_COUNT, matmat8_opt, matmat8_a, matmat8_b, matmat8_r, V64, V64, V64);
        timer->Stop();
        time_opt = timer->Time();
        check_opt = array_sum(matmat8_r, MATMAT8_COUNT * V64);

        cout << "VECMatrices8 : matmat8 : orig = " << time_orig
             << ", opt = " << time_opt << endl;
        DEBUG_CHECK(MATHS_IS_NEAR(check_orig, check_opt, 0.01), "matmat8 check failed");
        cout << "VECMatrices8 : matmat8 check : " << check_orig << endl;
        cout << "------------------------------" << endl;
    }

    // *----------*
    // | matmat16 |
    // *----------*

    if (MATMAT16_COUNT > 0)
    {
        // Original.
        clean_res();
        timer->Init();
        run3(repeats_count, MATMAT16_COUNT, matmat16_orig, matmat16_a, matmat16_b, matmat16_r, V256, V256, V256);
        timer->Start();
        run3(repeats_count, MATMAT16_COUNT, matmat16_orig, matmat16_a, matmat16_b, matmat16_r, V256, V256, V256);
        timer->Stop();
        time_orig = timer->Time();
        check_orig = array_sum(matmat16_r, MATMAT16_COUNT * V256);

        // Optimized.
        clean_res();
        timer->Init();
        run3(repeats_count, MATMAT16_COUNT, matmat16_opt, matmat16_a, matmat16_b, matmat16_r, V256, V256, V256);
        timer->Start();
        run3(repeats_count, MATMAT16_COUNT, matmat16_opt, matmat16_a, matmat16_b, matmat16_r, V256, V256, V256);
        timer->Stop();
        time_opt = timer->Time();
        check_opt = array_sum(matmat16_r, MATMAT16_COUNT * V256);

        cout << "VECMatrices8 : matmat16 : orig = " << time_orig
             << ", opt = " << time_opt << endl;
        DEBUG_CHECK(MATHS_IS_NEAR(check_orig, check_opt, 0.01), "matmat16 check failed");
        cout << "VECMatrices8 : matmat16 check : " << check_orig << endl;
        cout << "------------------------------" << endl;
    }

    // *---------*
    // | invmat8 |
    // *---------*

    if (INVMAT8_COUNT > 0)
    {
        // Original.
        clean_res();
        timer->Init();
        for (int r = 0; r < repeats_count; r++)
        {
            arrays_copy(invmat8_m, invmat8_t, INVMAT8_COUNT * V64);
            run2(INVMAT8_COUNT, invmat8_orig, invmat8_t, invmat8_r, V64, V64);
        }
        for (int r = 0; r < repeats_count; r++)
        {
            arrays_copy(invmat8_m, invmat8_t, INVMAT8_COUNT * V64);
            timer->Start();
            run2(INVMAT8_COUNT, invmat8_orig, invmat8_t, invmat8_r, V64, V64);
            timer->Stop();
        }
        time_orig = timer->Time();
        check_orig = array_sum(invmat8_r, INVMAT8_COUNT * V64);

        // Optimized.
        clean_res();
        timer->Init();
        for (int r = 0; r < repeats_count; r++)
        {
            arrays_copy(invmat8_m, invmat8_t, INVMAT8_COUNT * V64);
            run2(INVMAT8_COUNT, invmat8_opt, invmat8_t, invmat8_r, V64, V64);
        }  
        for (int r = 0; r < repeats_count; r++)
        {
            arrays_copy(invmat8_m, invmat8_t, INVMAT8_COUNT * V64);
            timer->Start();
            run2(INVMAT8_COUNT, invmat8_opt, invmat8_t, invmat8_r, V64, V64);
            timer->Stop();
        }
        time_opt = timer->Time();
        check_opt = array_sum(invmat8_r, INVMAT8_COUNT * V64);

        cout << "VECMatrices8 : invmat8 : orig = " << time_orig
             << ", opt = " << time_opt << endl;
        DEBUG_CHECK(MATHS_IS_NEAR(check_orig, check_opt, 0.01), "invmat8 check failed");
        cout << "VECMatrices8 : invmat8 check : " << check_orig << endl;
        cout << "------------------------------" << endl;
    }

    // *----------*
    // | invmat16 |
    // *----------*

    if (INVMAT16_COUNT > 0)
    {
        // Original.
        clean_res();
        timer->Init();
        for (int r = 0; r < repeats_count; r++)
        {
            arrays_copy(invmat16_m, invmat16_t, INVMAT16_COUNT * V256);
            run2(INVMAT16_COUNT, invmat16_orig, invmat16_t, invmat16_r, V256, V256);
        }
        for (int r = 0; r < repeats_count; r++)
        {
            arrays_copy(invmat16_m, invmat16_t, INVMAT16_COUNT * V256);
            timer->Start();
            run2(INVMAT16_COUNT, invmat16_orig, invmat16_t, invmat16_r, V256, V256);
            timer->Stop();
        }
        time_orig = timer->Time();
        check_orig = array_sum(invmat16_r, INVMAT16_COUNT * V256);

        // Optimized.
        clean_res();
        timer->Init();
        for (int r = 0; r < repeats_count; r++)
        {
            arrays_copy(invmat16_m, invmat16_t, INVMAT16_COUNT * V256);
            run2(INVMAT16_COUNT, invmat16_opt, invmat16_t, invmat16_r, V256, V256);
        }
        for (int r = 0; r < repeats_count; r++)
        {
            arrays_copy(invmat16_m, invmat16_t, INVMAT16_COUNT * V256);
            timer->Start();
            run2(INVMAT16_COUNT, invmat16_opt, invmat16_t, invmat16_r, V256, V256);
            timer->Stop();
        }
        time_opt = timer->Time();
        check_opt = array_sum(invmat16_r, INVMAT16_COUNT * V256);

        cout << "VECMatrices8 : invmat16 : orig = " << time_orig
             << ", opt = " << time_opt << endl;
        DEBUG_CHECK(MATHS_IS_NEAR(check_orig, check_opt, 0.01), "invmat16 check failed");
        cout << "VECMatrices8 : invmat16 check : " << check_orig << endl;
        cout << "------------------------------" << endl;
    }

    delete timer;

    cout << "VECMatrices8 : test end" << endl;
}