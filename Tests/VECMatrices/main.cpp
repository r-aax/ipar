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
#define MATVEC8_COUNT 1//120000

/// \brief matvec16 test cases count.
#define MATVEC16_COUNT 0//60000

/// \brief matmat8 test cases count.
#define MATMAT8_COUNT 0//15000

/// \brief matmat16 test cases cout.
#define MATMAT16_COUNT 0//15000

/// \brief invmat8 test cases count.
#define INVMAT8_COUNT 0//10000

/// \brief invmat16 test cases count.
#define INVMAT16_COUNT 0//2000

/// \brief Matrices for matvec8 test.
__declspec(align(64)) float matvec8_matr[MATVEC8_COUNT * V64];

/// \brief Vectors for matvec8 test.
__declspec(align(64)) float matvec8_vect[MATVEC8_COUNT * V8];

/// \brief Matrices for matvec8 results.
__declspec(align(64)) float matvec8_matv[MATVEC8_COUNT * V8];

/// \brief Matrices for matvec16 test.
__declspec(align(64)) float matvec16_matr[MATVEC16_COUNT * V256];

/// \brief Vectors for matvec16 test.
__declspec(align(64)) float matvec16_vect[MATVEC16_COUNT * V16];

/// \brief Matrices for matvec16 results.
__declspec(align(64)) float matvec16_matv[MATVEC16_COUNT * V16];

/// \brief Matrices a for matmat8 test.
__declspec(align(64)) float matmat8_a[MATMAT8_COUNT * V64];

/// \brief Matrices b for matmat8 test.
__declspec(align(64)) float matmat8_b[MATMAT8_COUNT * V64];

/// \brief Matrices r for matmat8 test.
__declspec(align(64)) float matmat8_r[MATMAT8_COUNT * V64];

/// \brief Matrices a for matmat16 test.
__declspec(align(64)) float matmat16_a[MATMAT16_COUNT * V256];

/// \brief Matrices b for matmat16 test.
__declspec(align(64)) float matmat16_b[MATMAT16_COUNT * V256];

/// \brief Matrices r for matmat16 test.
__declspec(align(64)) float matmat16_r[MATMAT16_COUNT * V256];

/// \brief Matrices m for invmat8 test.
__declspec(align(64)) float invmat8_m[INVMAT8_COUNT * V64];

/// \brief Tmp matrices m for invmat8 test.
__declspec(align(64)) float invmat8_tm[INVMAT8_COUNT * V64];

/// \brief Matrices r for invmat8 test.
__declspec(align(64)) float invmat8_r[INVMAT8_COUNT * V64];

/// \brief Matrices m for invmat16 test.
__declspec(align(64)) float invmat16_m[INVMAT16_COUNT * V256];

/// \brief Tmp matrices m for invmat16 test.
__declspec(align(64)) float invmat16_tm[INVMAT16_COUNT * V256];

/// \brief Matrices r for invmat16 test.
__declspec(align(64)) float invmat16_r[INVMAT16_COUNT * V256];

using namespace Utils;

/// \brief Sum of matvec8_matv array.
///
/// \return
/// Sum of matvec8_matv array.
static double matvec8_matv_sum()
{
    double res = 0.0;

    for (int i = 0; i < MATVEC8_COUNT * V8; i++)
    {
        res += (double)matvec8_matv[i];
    }

    return res;
}

/// \brief Sum of matvec16_matv array.
///
/// \return
/// Sum of matvec16_matv array.
static double matvec16_matv_sum()
{
    double res = 0.0;

    for (int i = 0; i < MATVEC16_COUNT * V16; i++)
    {
        res += (double)matvec16_matv[i];
    }

    return res;
}

/// \brief Sum of matmat8_r array.
///
/// \return
/// Sum of matmat8_r array.
static double matmat8_r_sum()
{
    double res = 0.0;

    for (int i = 0; i < MATMAT8_COUNT * V64; i++)
    {
        res += (double)matmat8_r[i];
    }

    return res;
}

/// \brief Sum of matmat16_r array.
///
/// \return
/// Sum of matmat16_r array.
static double matmat16_r_sum()
{
    double res = 0.0;

    for (int i = 0; i < MATMAT16_COUNT * V256; i++)
    {
        res += (double)matmat16_r[i];
    }

    return res;
}

/// \brief Sum of invmat8_r array.
///
/// \return
/// Sum of invmat8_r array.
static double invmat8_r_sum()
{
    double res = 0.0;

    for (int i = 0; i < INVMAT8_COUNT * V64; i++)
    {
        res += (double)invmat8_r[i];
    }

    return res;
}

/// \brief Sum of invmat16_r array.
///
/// \return
/// Sum of invmat16_r array.
static double invmat16_r_sum()
{
    double res = 0.0;

    for (int i = 0; i < INVMAT16_COUNT * V256; i++)
    {
        res += (double)invmat16_r[i];
    }

    return res;
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
    for (int r = 0; r < repeats_count; r++)
    {
        for (int i = 0; i < MATVEC8_COUNT; i++)
        {
            matvec8_orig(&matvec8_matr[i * V64],
                         &matvec8_vect[i * V8],
                         &matvec8_matv[i * V8]);
        }
    }
    timer->Start();
    for (int r = 0; r < repeats_count; r++)
    {
        for (int i = 0; i < MATVEC8_COUNT; i++)
        {
            matvec8_orig(&matvec8_matr[i * V64],
                         &matvec8_vect[i * V8],
                         &matvec8_matv[i * V8]);
        }
    }
    timer->Stop();
    time_orig = timer->Time();
    check_orig = matvec8_matv_sum();

    // Optimized.
    timer->Init();
    for (int r = 0; r < repeats_count; r++)
    {
        for (int i = 0; i < MATVEC8_COUNT; i++)
        {
            matvec8_opt(&matvec8_matr[i * V64],
                        &matvec8_vect[i * V8],
                        &matvec8_matv[i * V8]);
        }
    }
    timer->Start();
    for (int r = 0; r < repeats_count; r++)
    {
        for (int i = 0; i < MATVEC8_COUNT; i++)
        {
            matvec8_opt(&matvec8_matr[i * V64],
                        &matvec8_vect[i * V8],
                        &matvec8_matv[i * V8]);
        }
    }
    timer->Stop();
    time_opt = timer->Time();
    check_opt = matvec8_matv_sum();

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
    for (int r = 0; r < repeats_count; r++)
    {
        for (int i = 0; i < MATVEC16_COUNT; i++)
        {
            matvec8_orig(&matvec16_matr[i * V256],
                         &matvec16_vect[i * V16],
                         &matvec16_matv[i * V16]);
        }
    }
    timer->Start();
    for (int r = 0; r < repeats_count; r++)
    {
        for (int i = 0; i < MATVEC16_COUNT; i++)
        {
            matvec8_orig(&matvec16_matr[i * V256],
                         &matvec16_vect[i * V16],
                         &matvec16_matv[i * V16]);
        }
    }
    timer->Stop();
    time_orig = timer->Time();
    check_orig = matvec16_matv_sum();

    // Optimized.
    timer->Init();
    for (int r = 0; r < repeats_count; r++)
    {
        for (int i = 0; i < MATVEC16_COUNT; i++)
        {
            matvec8_opt(&matvec16_matr[i * V256],
                        &matvec16_vect[i * V16],
                        &matvec16_matv[i * V16]);
        }
    }
    timer->Start();
    for (int r = 0; r < repeats_count; r++)
    {
        for (int i = 0; i < MATVEC16_COUNT; i++)
        {
            matvec8_opt(&matvec16_matr[i * V256],
                        &matvec16_vect[i * V16],
                        &matvec16_matv[i * V16]);
        }
    }
    timer->Stop();
    time_opt = timer->Time();
    check_opt = matvec16_matv_sum();

    cout << "VECMatrices : matvec16 : orig = " << time_orig
         << ", opt = " << time_opt << endl;
    //cout << "VECMatrices : matvec16 check : orig = " << check_orig
    //     << ", opt = " << check_opt << endl;
    DEBUG_CHECK(MATHS_IS_NEAR(check_orig, check_opt, 0.01), "matvec16 check failed");

    // *---------*
    // | matmat8 |
    // *---------*

    // Original.
    timer->Init();
    for (int r = 0; r < repeats_count; r++)
    {
        for (int i = 0; i < MATMAT8_COUNT; i++)
        {
            matmat8_orig(&matmat8_a[i * V64],
                         &matmat8_b[i * V64],
                         &matmat8_r[i * V64]);
        }
    }
    timer->Start();
    for (int r = 0; r < repeats_count; r++)
    {
        for (int i = 0; i < MATMAT8_COUNT; i++)
        {
            matmat8_orig(&matmat8_a[i * V64],
                         &matmat8_b[i * V64],
                         &matmat8_r[i * V64]);
        }
    }
    timer->Stop();
    time_orig = timer->Time();
    check_orig = matmat8_r_sum();

    // Optimized.
    timer->Init();
    for (int r = 0; r < repeats_count; r++)
    {
        for (int i = 0; i < MATMAT8_COUNT; i++)
        {
            matmat8_opt(&matmat8_a[i * V64],
                        &matmat8_b[i * V64],
                        &matmat8_r[i * V64]);
        }
    }
    timer->Start();
    for (int r = 0; r < repeats_count; r++)
    {
        for (int i = 0; i < MATMAT8_COUNT; i++)
        {
            matmat8_opt(&matmat8_a[i * V64],
                        &matmat8_b[i * V64],
                        &matmat8_r[i * V64]);
        }
    }
    timer->Stop();
    time_opt = timer->Time();
    check_opt = matmat8_r_sum();

    cout << "VECMatrices : matmat8 : orig = " << time_orig
         << ", opt = " << time_opt << endl;
    //cout << "VECMatrices : matmat8 check : orig = " << check_orig
    //     << ", opt = " << check_opt << endl;
    DEBUG_CHECK(MATHS_IS_NEAR(check_orig, check_opt, 0.01), "matmat8 check failed");

    // *----------*
    // | matmat16 |
    // *----------*

    // Original.
    timer->Init();
    for (int r = 0; r < repeats_count; r++)
    {
        for (int i = 0; i < MATMAT16_COUNT; i++)
        {
            matmat8_orig(&matmat16_a[i * V256],
                         &matmat16_b[i * V256],
                         &matmat16_r[i * V256]);
        }
    }
    timer->Start();
    for (int r = 0; r < repeats_count; r++)
    {
        for (int i = 0; i < MATMAT16_COUNT; i++)
        {
            matmat8_orig(&matmat16_a[i * V256],
                         &matmat16_b[i * V256],
                         &matmat16_r[i * V256]);
        }
    }
    timer->Stop();
    time_orig = timer->Time();
    check_orig = matmat16_r_sum();

    // Optimized.
    timer->Init();
    for (int r = 0; r < repeats_count; r++)
    {
        for (int i = 0; i < MATMAT16_COUNT; i++)
        {
            matmat8_opt(&matmat16_a[i * V256],
                        &matmat16_b[i * V256],
                        &matmat16_r[i * V256]);
        }
    }
    timer->Start();
    for (int r = 0; r < repeats_count; r++)
    {
        for (int i = 0; i < MATMAT16_COUNT; i++)
        {
            matmat8_opt(&matmat16_a[i * V256],
                        &matmat16_b[i * V256],
                        &matmat16_r[i * V256]);
        }
    }
    timer->Stop();
    time_opt = timer->Time();
    check_opt = matmat16_r_sum();

    cout << "VECMatrices : matmat16 : orig = " << time_orig
         << ", opt = " << time_opt << endl;
    //cout << "VECMatrices : matmat16 check : orig = " << check_orig
    //     << ", opt = " << check_opt << endl;
    DEBUG_CHECK(MATHS_IS_NEAR(check_orig, check_opt, 0.01), "matmat16 check failed");

    // *---------*
    // | invmat8 |
    // *---------*

    // Original.
    timer->Init();
    for (int r = 0; r < repeats_count; r++)
    {
        for (int i = 0; i < INVMAT8_COUNT * V64; i++)
        {
            invmat8_tm[i] = invmat8_m[i];
        }
        for (int i = 0; i < INVMAT8_COUNT; i++)
        {
            invmat8_orig(&invmat8_tm[i * V64],
                         &invmat8_r[i * V64]);
        }
    }
    for (int r = 0; r < repeats_count; r++)
    {
        for (int i = 0; i < INVMAT8_COUNT * V64; i++)
        {
            invmat8_tm[i] = invmat8_m[i];
        }
        timer->Start();
        for (int i = 0; i < INVMAT8_COUNT; i++)
        {
            invmat8_orig(&invmat8_tm[i * V64],
                         &invmat8_r[i * V64]);
        }
        timer->Stop();
    }
    time_orig = timer->Time();
    check_orig = invmat8_r_sum();

    // Optimized.
    timer->Init();
    for (int r = 0; r < repeats_count; r++)
    {
        for (int i = 0; i < INVMAT8_COUNT * V64; i++)
        {
            invmat8_tm[i] = invmat8_m[i];
        }
        for (int i = 0; i < INVMAT8_COUNT; i++)
        {
            invmat8_opt(&invmat8_tm[i * V64],
                        &invmat8_r[i * V64]);
        }
    }
    for (int r = 0; r < repeats_count; r++)
    {
        for (int i = 0; i < INVMAT8_COUNT * V64; i++)
        {
            invmat8_tm[i] = invmat8_m[i];
        }
        timer->Start();
        for (int i = 0; i < INVMAT8_COUNT; i++)
        {
            invmat8_opt(&invmat8_tm[i * V64],
                        &invmat8_r[i * V64]);
        }
        timer->Stop();
    }
    time_opt = timer->Time();
    check_opt = invmat8_r_sum();

    cout << "VECMatrices : invmat8 : orig = " << time_orig
         << ", opt = " << time_opt << endl;
    //cout << "VECMatrices : invmat8 check : orig = " << check_orig
    //     << ", opt = " << check_opt << endl;
    DEBUG_CHECK(MATHS_IS_NEAR(check_orig, check_opt, 0.01), "invmat8 check failed");

    // *----------*
    // | invmat16 |
    // *----------*

    // Original.
    timer->Init();
    for (int r = 0; r < repeats_count; r++)
    {
        for (int i = 0; i < INVMAT16_COUNT * V256; i++)
        {
            invmat16_tm[i] = invmat16_m[i];
        }
        for (int i = 0; i < INVMAT16_COUNT; i++)
        {
            invmat16_orig(&invmat16_tm[i * V256],
                          &invmat16_r[i * V256]);
        }
    }
    for (int r = 0; r < repeats_count; r++)
    {
        for (int i = 0; i < INVMAT16_COUNT * V256; i++)
        {
            invmat16_tm[i] = invmat16_m[i];
        }
        timer->Start();
        for (int i = 0; i < INVMAT16_COUNT; i++)
        {
            invmat16_orig(&invmat16_tm[i * V256],
                          &invmat16_r[i * V256]);
        }
        timer->Stop();
    }
    time_orig = timer->Time();
    check_orig = invmat16_r_sum();

    // Optimized.
    timer->Init();
    for (int r = 0; r < repeats_count; r++)
    {
        for (int i = 0; i < INVMAT16_COUNT * V256; i++)
        {
            invmat16_tm[i] = invmat16_m[i];
        }
        for (int i = 0; i < INVMAT16_COUNT; i++)
        {
            invmat16_opt(&invmat16_tm[i * V256],
                         &invmat16_r[i * V256]);
        }
    }
    for (int r = 0; r < repeats_count; r++)
    {
        for (int i = 0; i < INVMAT16_COUNT * V256; i++)
        {
            invmat16_tm[i] = invmat16_m[i];
        }
        timer->Start();
        for (int i = 0; i < INVMAT16_COUNT; i++)
        {
            invmat16_opt(&invmat16_tm[i * V256],
                         &invmat16_r[i * V256]);
        }
        timer->Stop();
    }
    time_opt = timer->Time();
    check_opt = invmat16_r_sum();

    cout << "VECMatrices : invmat16 : orig = " << time_orig
         << ", opt = " << time_opt << endl;
    //cout << "VECMatrices : invmat16 check : orig = " << check_orig
    //     << ", opt = " << check_opt << endl;
    DEBUG_CHECK(MATHS_IS_NEAR(check_orig, check_opt, 0.01), "invmat16 check failed");

    delete timer;

    cout << "VECMatrices : test end" << endl;
}