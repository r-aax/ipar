/// \file
/// \brief Matrices operations vectorization.

#include "../../Utils/IO.h"
#include "../../Utils/Randoms.h"
#include "../../Utils/Timer.h"
#include "../../Utils/Maths.h"
#include "../../Utils/Debug.h"
#include "matrices.h"
#include <stdlib.h>

/// \brief matvec8 test cases count.
#define MATVEC8_COUNT 120000

/// \brief matvec16 test cases count.
#define MATVEC16_COUNT 60000

/// \brief matmat8 test cases count.
#define MATMAT8_COUNT 15000

/// \brief matmat16 test cases cout.
#define MATMAT16_COUNT 15000

/// \brief invmat8 test cases count.
#define INVMAT8_COUNT 10000

/// \brief invmat16 test cases count.
#define INVMAT16_COUNT 2000

/// \brief Matrices for matvec8 test.
#ifdef INTEL
__declspec(align(64)) float matvec8_matr[MATVEC8_COUNT * V64];
#else
float matvec8_matr[MATVEC8_COUNT * V64];
#endif

/// \brief Vectors for matvec8 test.
#ifdef INTEL
__declspec(align(64)) float matvec8_vect[MATVEC8_COUNT * V8];
#else
float matvec8_vect[MATVEC8_COUNT * V8];
#endif

/// \brief Matrices for matvec8 results.
#ifdef INTEL
__declspec(align(64)) float matvec8_matv[MATVEC8_COUNT * V8];
#else
float matvec8_matv[MATVEC8_COUNT * V8];
#endif

/// \brief Matrices for matvec16 test.
#ifdef INTEL
__declspec(align(64)) float matvec16_matr[MATVEC16_COUNT * V256];
#else
float matvec16_matr[MATVEC16_COUNT * V256];
#endif

/// \brief Vectors for matvec16 test.
#ifdef INTEL
__declspec(align(64)) float matvec16_vect[MATVEC16_COUNT * V16];
#else
float matvec16_vect[MATVEC16_COUNT * V16];
#endif

/// \brief Matrices for matvec16 results.
#ifdef INTEL
__declspec(align(64)) float matvec16_matv[MATVEC16_COUNT * V16];
#else
float matvec16_matv[MATVEC16_COUNT * V16];
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
__declspec(align(64)) float invmat8_tm[INVMAT8_COUNT * V64];
#else
float invmat8_tm[INVMAT8_COUNT * V64];
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
__declspec(align(64)) float invmat16_tm[INVMAT16_COUNT * V256];
#else
float invmat16_tm[INVMAT16_COUNT * V256];
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

    cout << "VECMatrices : test begin" << endl;

    Timer *timer = new Timer(Timer::OMP);
    double time_orig, time_opt;
    double check_orig, check_opt;

    // Init.
    for (int i = 0; i < MATVEC8_COUNT * V64; i++)
    {
        matvec8_matr[i] = Randoms::Rand01();
    }
    for (int i = 0; i < MATVEC8_COUNT * V8; i++)
    {
        matvec8_vect[i] = Randoms::Rand01();
    }
    for (int i = 0; i < MATVEC16_COUNT * V256; i++)
    {
        matvec16_matr[i] = Randoms::Rand01();
    }
    for (int i = 0; i < MATVEC16_COUNT * V16; i++)
    {
        matvec16_vect[i] = Randoms::Rand01();
    }
    for (int i = 0; i < MATMAT8_COUNT * V64; i++)
    {
        matmat8_a[i] = Randoms::Rand01();
        matmat8_b[i] = Randoms::Rand01();
    }
    for (int i = 0; i < MATMAT16_COUNT * V256; i++)
    {
        matmat16_a[i] = Randoms::Rand01();
        matmat16_b[i] = Randoms::Rand01();
    }
    for (int i = 0; i < INVMAT8_COUNT * V64; i++)
    {
        invmat8_m[i] = Randoms::Rand01();
    }
    for (int i = 0; i < INVMAT8_COUNT; i++)
    {
        for (int j = 0; j < V8; j++)
        {
            invmat8_m[i * V64 + j * V8 + j] *= 2.0;
        }
    }
    for (int i = 0; i < INVMAT16_COUNT * V256; i++)
    {
        invmat16_m[i] = Randoms::Rand01();
    }
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

    // Original.
    timer->Init();
    run3(repeats_count, MATVEC8_COUNT, matvec8_orig, matvec8_matr, matvec8_vect, matvec8_matv, V64, V8, V8);
    timer->Start();
    run3(repeats_count, MATVEC8_COUNT, matvec8_orig, matvec8_matr, matvec8_vect, matvec8_matv, V64, V8, V8);
    timer->Stop();
    time_orig = timer->Time();
    check_orig = array_sum(matvec8_matv, MATVEC8_COUNT * V8);

    // Optimized.
    timer->Init();
    run3(repeats_count, MATVEC8_COUNT, matvec8_opt, matvec8_matr, matvec8_vect, matvec8_matv, V64, V8, V8);
    timer->Start();
    run3(repeats_count, MATVEC8_COUNT, matvec8_opt, matvec8_matr, matvec8_vect, matvec8_matv, V64, V8, V8);
    timer->Stop();
    time_opt = timer->Time();
    check_opt = array_sum(matvec8_matv, MATVEC8_COUNT * V8);

    cout << "VECMatrices : matvec8 : orig = " << time_orig
         << ", opt = " << time_opt << endl;
    cout << "VECMatrices : matvec8 check : orig = " << check_orig
         << ", opt = " << check_opt << endl;
    DEBUG_CHECK(MATHS_IS_NEAR(check_orig, check_opt, 0.01), "matvec8 check failed");

    // *----------*
    // | matvec16 |
    // *----------*

    // Original.
    timer->Init();
    run3(repeats_count, MATVEC16_COUNT, matvec16_orig, matvec16_matr, matvec16_vect, matvec16_matv, V256, V16, V16);
    timer->Start();
    run3(repeats_count, MATVEC16_COUNT, matvec16_orig, matvec16_matr, matvec16_vect, matvec16_matv, V256, V16, V16);
    timer->Stop();
    time_orig = timer->Time();
    check_orig = array_sum(matvec16_matv, MATVEC16_COUNT * V16);

    // Optimized.
    timer->Init();
    run3(repeats_count, MATVEC16_COUNT, matvec16_opt, matvec16_matr, matvec16_vect, matvec16_matv, V256, V16, V16);
    timer->Start();
    run3(repeats_count, MATVEC16_COUNT, matvec16_opt, matvec16_matr, matvec16_vect, matvec16_matv, V256, V16, V16);
    timer->Stop();
    time_opt = timer->Time();
    check_opt = array_sum(matvec16_matv, MATVEC16_COUNT * V16);

    cout << "VECMatrices : matvec16 : orig = " << time_orig
         << ", opt = " << time_opt << endl;
    cout << "VECMatrices : matvec16 check : orig = " << check_orig
         << ", opt = " << check_opt << endl;
    DEBUG_CHECK(MATHS_IS_NEAR(check_orig, check_opt, 0.01), "matvec16 check failed");

    // *---------*
    // | matmat8 |
    // *---------*

    // Original.
    timer->Init();
    run3(repeats_count, MATMAT8_COUNT, matmat8_orig, matmat8_a, matmat8_b, matmat8_r, V64, V64, V64);
    timer->Start();
    run3(repeats_count, MATMAT8_COUNT, matmat8_orig, matmat8_a, matmat8_b, matmat8_r, V64, V64, V64);
    timer->Stop();
    time_orig = timer->Time();
    check_orig = array_sum(matmat8_r, MATMAT8_COUNT * V64);

    // Optimized.
    timer->Init();
    run3(repeats_count, MATMAT8_COUNT, matmat8_opt, matmat8_a, matmat8_b, matmat8_r, V64, V64, V64);
    timer->Start();
    run3(repeats_count, MATMAT8_COUNT, matmat8_opt, matmat8_a, matmat8_b, matmat8_r, V64, V64, V64);
    timer->Stop();
    time_opt = timer->Time();
    check_opt = array_sum(matmat8_r, MATMAT8_COUNT * V64);

    cout << "VECMatrices : matmat8 : orig = " << time_orig
         << ", opt = " << time_opt << endl;
    cout << "VECMatrices : matmat8 check : orig = " << check_orig
         << ", opt = " << check_opt << endl;
    DEBUG_CHECK(MATHS_IS_NEAR(check_orig, check_opt, 0.01), "matmat8 check failed");

    // *----------*
    // | matmat16 |
    // *----------*

    // Original.
    timer->Init();
    run3(repeats_count, MATMAT16_COUNT, matmat16_orig, matmat16_a, matmat16_b, matmat16_r, V256, V256, V256);
    timer->Start();
    run3(repeats_count, MATMAT16_COUNT, matmat16_orig, matmat16_a, matmat16_b, matmat16_r, V256, V256, V256);
    timer->Stop();
    time_orig = timer->Time();
    check_orig = array_sum(matmat16_r, MATMAT16_COUNT * V256);

    // Optimized.
    timer->Init();
    run3(repeats_count, MATMAT16_COUNT, matmat16_opt, matmat16_a, matmat16_b, matmat16_r, V256, V256, V256);
    timer->Start();
    run3(repeats_count, MATMAT16_COUNT, matmat16_opt, matmat16_a, matmat16_b, matmat16_r, V256, V256, V256);
    timer->Stop();
    time_opt = timer->Time();
    check_opt = array_sum(matmat16_r, MATMAT16_COUNT * V256);

    cout << "VECMatrices : matmat16 : orig = " << time_orig
         << ", opt = " << time_opt << endl;
    cout << "VECMatrices : matmat16 check : orig = " << check_orig
         << ", opt = " << check_opt << endl;
    DEBUG_CHECK(MATHS_IS_NEAR(check_orig, check_opt, 0.01), "matmat16 check failed");

    // *---------*
    // | invmat8 |
    // *---------*

    // Original.
    timer->Init();
    for (int r = 0; r < repeats_count; r++)
    {
        arrays_copy(invmat8_m, invmat8_tm, INVMAT8_COUNT * V64);
        run2(INVMAT8_COUNT, invmat8_orig, invmat8_tm, invmat8_r, V64, V64);
    }
    for (int r = 0; r < repeats_count; r++)
    {
        arrays_copy(invmat8_m, invmat8_tm, INVMAT8_COUNT * V64);
        timer->Start();
        run2(INVMAT8_COUNT, invmat8_orig, invmat8_tm, invmat8_r, V64, V64);
        timer->Stop();
    }
    time_orig = timer->Time();
    check_orig = array_sum(invmat8_r, INVMAT8_COUNT * V64);

    // Optimized.
    timer->Init();
    for (int r = 0; r < repeats_count; r++)
    {
        arrays_copy(invmat8_m, invmat8_tm, INVMAT8_COUNT * V64);
        run2(INVMAT8_COUNT, invmat8_opt, invmat8_tm, invmat8_r, V64, V64);
    }
    for (int r = 0; r < repeats_count; r++)
    {
        arrays_copy(invmat8_m, invmat8_tm, INVMAT8_COUNT * V64);
        timer->Start();
        run2(INVMAT8_COUNT, invmat8_opt, invmat8_tm, invmat8_r, V64, V64);
        timer->Stop();
    }
    time_opt = timer->Time();
    check_opt = array_sum(invmat8_r, INVMAT8_COUNT * V64);

    cout << "VECMatrices : invmat8 : orig = " << time_orig
         << ", opt = " << time_opt << endl;
    cout << "VECMatrices : invmat8 check : orig = " << check_orig
         << ", opt = " << check_opt << endl;
    DEBUG_CHECK(MATHS_IS_NEAR(check_orig, check_opt, 0.01), "invmat8 check failed");

    // *----------*
    // | invmat16 |
    // *----------*

    // Original.
    timer->Init();
    for (int r = 0; r < repeats_count; r++)
    {
        arrays_copy(invmat16_m, invmat16_tm, INVMAT16_COUNT * V256);
        run2(INVMAT16_COUNT, invmat16_orig, invmat16_tm, invmat16_r, V256, V256);
    }
    for (int r = 0; r < repeats_count; r++)
    {
        arrays_copy(invmat16_m, invmat16_tm, INVMAT16_COUNT * V256);
        timer->Start();
        run2(INVMAT16_COUNT, invmat16_orig, invmat16_tm, invmat16_r, V256, V256);
        timer->Stop();
    }
    time_orig = timer->Time();
    check_orig = array_sum(invmat16_r, INVMAT16_COUNT * V256);

    // Optimized.
    timer->Init();
    for (int r = 0; r < repeats_count; r++)
    {
        arrays_copy(invmat16_m, invmat16_tm, INVMAT16_COUNT * V256);
        run2(INVMAT16_COUNT, invmat16_opt, invmat16_tm, invmat16_r, V256, V256);
    }
    for (int r = 0; r < repeats_count; r++)
    {
        arrays_copy(invmat16_m, invmat16_tm, INVMAT16_COUNT * V256);
        timer->Start();
        run2(INVMAT16_COUNT, invmat16_opt, invmat16_tm, invmat16_r, V256, V256);
        timer->Stop();
    }
    time_opt = timer->Time();
    check_opt = array_sum(invmat16_r, INVMAT16_COUNT * V256);

    cout << "VECMatrices : invmat16 : orig = " << time_orig
         << ", opt = " << time_opt << endl;
    cout << "VECMatrices : invmat16 check : orig = " << check_orig
         << ", opt = " << check_opt << endl;
    DEBUG_CHECK(MATHS_IS_NEAR(check_orig, check_opt, 0.01), "invmat16 check failed");

    delete timer;

    cout << "VECMatrices : test end" << endl;
}