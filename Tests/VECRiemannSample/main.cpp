/// \file
/// \brief Matrices operations vectorization.

#include "../../Utils/IO.h"
#include "../../Utils/Randoms.h"
#include "../../Utils/Timer.h"
#include "../../Utils/Maths.h"
#include "../../Utils/Debug.h"
#include <stdlib.h>
#include <math.h>
#include "riemann.h"

// \brief Repeats count.
#define REPEATS_COUNT 50

using namespace Utils;

// In data.

// Test data for <c>dl</c>.
float dls[] =
{

#include "test_data_dl.inc"

};

// Test data for <c>ul</c>.
float uls[] =
{

#include "test_data_ul.inc"

};

// Test data for <c>pl</c>.
float pls[] =
{

#include "test_data_pl.inc"

};

// Test data for <c>cl</c>.
float cls[] =
{

#include "test_data_cl.inc"

};

// Test data for <c>dr</c>.
float drs[] =
{

#include "test_data_dr.inc"

};

// Test data for <c>ur</c>.
float urs[] =
{

#include "test_data_ur.inc"

};

// Test data for <c>pr</c>.
float prs[] =
{

#include "test_data_pr.inc"

};

// Test data for <c>cr</c>.
float crs[] =
{

#include "test_data_cr.inc"

};

// Test data for <c>pm</c>.
float pms[] =
{

#include "test_data_pm.inc"

};

// Test data for <c>um</c>.
float ums[] =
{

#include "test_data_um.inc"

};

// Out data.

// Original out data for <c>d</c>.
float ds_orig[TESTS_COUNT];

// Original out data for <c>u</c>.
float us_orig[TESTS_COUNT];

// Original out data for <c>p</c>.
float ps_orig[TESTS_COUNT];

// Optimized out data for <c>d</c>.
float ds_opt[TESTS_COUNT];

// Optimized out data for <c>u</c>.
float us_opt[TESTS_COUNT];

// Optimized out data for <c>p</c>.
float ps_opt[TESTS_COUNT];

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

// \brief Check.
static bool check()
{
    for (int i = 0; i < TESTS_COUNT; i++)
    {
        if (!MATHS_IS_EQ(ds_orig[i], ds_opt[i]))
        {
            cout << "error : ds, i = " << i << " [" << ds_orig[i] << ", " << ds_opt[i] << endl;

            return false;
        }

        if (!MATHS_IS_EQ(us_orig[i], us_opt[i]))
        {
            cout << "error : us, i = " << i << " [" << us_orig[i] << ", " << us_opt[i] << endl;
                
            return false;
        }

        if (!MATHS_IS_EQ(ps_orig[i], ps_opt[i]))
        {
            cout << "error : ps, i = " << i << " [" << ps_orig[i] << ", " << ps_opt[i] << endl;
                
            return false;
        }        
    }

    return true;
}

/// \brief Main function.
///
/// \param argc - arguments count
/// \param argv - arguments
int main(int argc, char **argv)
{
    cout << "================================================" << endl;

    int repeats_count = 1;

    // Parse repeats count if given.
    if (argc == 2)
    {
        repeats_count = atoi(argv[1]);
    }

    cout << "VECRiemannSample : test begin " << endl;

    Timer *timer = new Timer(Timer::OMP);
    double time_orig, time_opt;
    float d, u, p;
    bool is_ok;

    // Original.
    timer->Init();
    for (int r = 0; r < REPEATS_COUNT; r++)
    {
        samples_orig(dls, uls, pls, cls, drs, urs, prs, crs, pms, ums, ds_orig, us_orig, ps_orig);
    }
    timer->Start();
    for (int r = 0; r < REPEATS_COUNT; r++)
    {
        samples_orig(dls, uls, pls, cls, drs, urs, prs, crs, pms, ums, ds_orig, us_orig, ps_orig);
    }
    timer->Stop();
    time_orig = timer->Time();

    // Optimized.
    timer->Init();
    for (int r = 0; r < REPEATS_COUNT; r++)
    {
        samples_opt(dls, uls, pls, cls, drs, urs, prs, crs, pms, ums, ds_opt, us_opt, ps_opt);
    }
    timer->Start();
    for (int r = 0; r < REPEATS_COUNT; r++)
    {
        samples_opt(dls, uls, pls, cls, drs, urs, prs, crs, pms, ums, ds_opt, us_opt, ps_opt);
    }
    timer->Stop();
    time_opt = timer->Time();

    // Check.
    is_ok = check();

    cout << "VECRiemannSample : orig = " << time_orig << ", opt = " << time_opt << endl;
    cout << "VECRiemannSample : speedup = " << ((time_orig - time_opt) / time_orig) * 100.0 << "%" << endl;
    cout << "VECRiemannSample : check = " << is_ok << endl;
    cout << "------------------------------------------------" << endl;

    delete timer;
}
