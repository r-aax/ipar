/// @file
/// @brief Vectorization of triangle and box intersection.

#include "../../Utils/IO.h"
#include "../../Utils/Randoms.h"
#include "../../Utils/Timer.h"
#include "../../Utils/Maths.h"
#include "../../Utils/Debug.h"
#include <stdlib.h>
#include <math.h>
#include "tri_box_intersect.h"

/// @brief Mini flag.
#define MINI 0

using namespace Utils;

/// @brief
float Ax[]
{

#if MINI == 0
#include "ax.txt"
#else
#include "ax_mini.txt"
#endif

};

/// @brief
const int TESTS_COUNT = sizeof(Ax) / sizeof(float);

/// @brief
float Ay[TESTS_COUNT]
{

#if MINI == 0
#include "ay.txt"
#else
#include "ay_mini.txt"
#endif

};

/// @brief
float Az[TESTS_COUNT]
{

#if MINI == 0
#include "az.txt"
#else
#include "az_mini.txt"
#endif

};

/// @brief
float Bx[TESTS_COUNT]
{

#if MINI == 0
#include "bx.txt"
#else
#include "bx_mini.txt"
#endif

};

/// @brief
float By[TESTS_COUNT]
{

#if MINI == 0
#include "by.txt"
#else
#include "by_mini.txt"
#endif

};

/// @brief
float Bz[TESTS_COUNT]
{

#if MINI == 0
#include "bz.txt"
#else
#include "bz_mini.txt"
#endif

};

/// @brief
float Cx[TESTS_COUNT]
{

#if MINI == 0
#include "cx.txt"
#else
#include "cx_mini.txt"
#endif

};

/// @brief
float Cy[TESTS_COUNT]
{

#if MINI == 0
#include "cy.txt"
#else
#include "cy_mini.txt"
#endif

};

/// @brief
float Cz[TESTS_COUNT]
{

#if MINI == 0
#include "cz.txt"
#else
#include "cz_mini.txt"
#endif

};

/// @brief
float Xl[TESTS_COUNT]
{

#if MINI == 0
#include "xl.txt"
#else
#include "xl_mini.txt"
#endif

};

/// @brief
float Xh[TESTS_COUNT]
{

#if MINI == 0
#include "xh.txt"
#else
#include "xh_mini.txt"
#endif

};

/// @brief
float Yl[TESTS_COUNT]
{

#if MINI == 0
#include "yl.txt"
#else
#include "yl_mini.txt"
#endif

};

/// @brief
float Yh[TESTS_COUNT]
{

#if MINI == 0
#include "yh.txt"
#else
#include "yh_mini.txt"
#endif

};

/// @brief
float Zl[TESTS_COUNT]
{

#if MINI == 0
#include "zl.txt"
#else
#include "zl_mini.txt"
#endif

};

/// @brief
float Zh[TESTS_COUNT]
{

#if MINI == 0
#include "zh.txt"
#else
#include "zh_mini.txt"
#endif

};

/// @brief Original results in bool.
bool R_orig_bool[TESTS_COUNT]
{
#if MINI == 0
#include "r.txt"
#else
#include "r_mini.txt"
#endif
};

/// @brief Original results.
int R_orig[TESTS_COUNT];

/// @brief Optimized results.
int R_opt[TESTS_COUNT];

/// @brief Clean array.
///
/// @param a - array
/// @param c - count of elements
static void
clean_array(float *a,
            int c)
{
    for (int i = 0; i < c; i++)
    {
        a[i] = 0.0;
    }
}

/// Check function.
///
/// @return
/// Fault range.
static float
check(int tests_count)
{
    int c = 0;

    for (int i = 0; i < tests_count; i++)
    {
        if (R_orig[i] != R_opt[i])
        {
            c++;
        }
    }

    return 100.0 * ((float)c / (float)tests_count);
}

/// @brief Main function.
///
/// @param argc - arguments count
/// @param argv - arguments
int
main(int argc,
     char **argv)
{
    for (int i = 0; i < TESTS_COUNT; i++)
    {
        R_orig[i] = (R_orig_bool[i] ? 1 : 0);
    }

    cout << "================================================" << endl;

    int tests_count = TESTS_COUNT - TESTS_COUNT % VEC_WIDTH;
    int repeats_count = 1;

    // Parse repeats count if given.
    if (argc == 2)
    {
        repeats_count = atoi(argv[1]);
    }

    cout << "VECTriBoxIntersect : test begin " << endl;
    cout << "VECTriBoxIntersect : tests_count = " << tests_count << endl;
    cout << "VECTriBoxIntersect : repeats_count = " << repeats_count << endl;

    Timer *timer = new Timer(Timer::OMP);
    double time_orig, time_opt;
    float fault_range;

    // Original.
    timer->Init();
#if 1
    for (int i = 0; i < repeats_count; i++)
    {
        tri_box_intersects_orig(Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Xl, Xh, Yl, Yh, Zl, Zh, tests_count, R_orig);
    }
#endif
    timer->Start();
#if 1
    for (int i = 0; i < repeats_count; i++)
    {
        tri_box_intersects_orig(Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Xl, Xh, Yl, Yh, Zl, Zh, tests_count, R_orig);
    }
#endif
    timer->Stop();
    time_orig = timer->Time();

    // Optimized.
    timer->Init();
    for (int i = 0; i < repeats_count; i++)
    {
        tri_box_intersects_opt(Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Xl, Xh, Yl, Yh, Zl, Zh, tests_count, R_opt);
    }
    timer->Start();
    for (int i = 0; i < repeats_count; i++)
    {
        tri_box_intersects_opt(Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Xl, Xh, Yl, Yh, Zl, Zh, tests_count, R_opt);
    }
    timer->Stop();
    time_opt = timer->Time();

    // Check.
    fault_range = check(tests_count);

    cout << "VECTriBoxIntersect : orig = " << time_orig << ", opt = " << time_opt << endl;
    cout << "VECTriBoxIntersect : speedup = " << ((time_orig - time_opt) / time_orig) * 100.0 << "%" << endl;
    cout << "VECTriBoxIntersect : fault_range = " << fault_range << "%" << endl;
    cout << "------------------------------------------------" << endl;

    delete timer;
}
