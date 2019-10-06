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

using namespace Utils;

/// @brief Datas.

float Ax[]
{
#include "ax.txt"
};
const int TESTS_COUNT = sizeof(Ax) / sizeof(float);
float Ay[TESTS_COUNT]
{
#include "ay.txt"
};
float Az[TESTS_COUNT]
{
#include "az.txt"
};
float Bx[TESTS_COUNT]
{
#include "bx.txt"
};
float By[TESTS_COUNT]
{
#include "by.txt"
};
float Bz[TESTS_COUNT]
{
#include "bz.txt"
};
float Cx[TESTS_COUNT]
{
#include "cx.txt"
};
float Cy[TESTS_COUNT]
{
#include "cy.txt"
};
float Cz[TESTS_COUNT]
{
#include "cz.txt"
};
float Xl[TESTS_COUNT]
{
#include "xl.txt"
};
float Xh[TESTS_COUNT]
{
#include "xh.txt"
};
float Yl[TESTS_COUNT]
{
#include "yl.txt"
};
float Yh[TESTS_COUNT]
{
#include "yh.txt"
};
float Zl[TESTS_COUNT]
{
#include "zl.txt"
};
float Zh[TESTS_COUNT]
{
#include "zh.txt"
};

/// @brief Original results.
bool R_orig[TESTS_COUNT]
{
#include "r.txt"
};

/// @brief Optimized results.
bool R_opt[TESTS_COUNT];

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
check()
{
    int c = 0;

    for (int i = 0; i < TESTS_COUNT; i++)
    {
        if (R_orig[i] != R_opt[i])
        {
            c++;
        }
    }

    return 100.0 * ((float)c / (float)TESTS_COUNT);
}

/// @brief Main function.
///
/// @param argc - arguments count
/// @param argv - arguments
int
main(int argc,
     char **argv)
{
    cout << "================================================" << endl;

    int tests_count = TESTS_COUNT;
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
        tri_box_intersects_orig(Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Xl, Xh, Yl, Yh, Zl, Zh, TESTS_COUNT, R_orig);
    }
#endif
    timer->Start();
#if 1
    for (int i = 0; i < repeats_count; i++)
    {
        tri_box_intersects_orig(Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Xl, Xh, Yl, Yh, Zl, Zh, TESTS_COUNT, R_orig);
    }
#endif
    timer->Stop();
    time_orig = timer->Time();

    // Optimized.
    timer->Init();
    for (int i = 0; i < repeats_count; i++)
    {
        tri_box_intersects_opt(Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Xl, Xh, Yl, Yh, Zl, Zh, TESTS_COUNT, R_opt);
    }
    timer->Start();
    for (int i = 0; i < repeats_count; i++)
    {
        tri_box_intersects_opt(Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Xl, Xh, Yl, Yh, Zl, Zh, TESTS_COUNT, R_opt);
    }
    timer->Stop();
    time_opt = timer->Time();

    // Check.
    fault_range = check();

    cout << "VECTriBoxIntersect : orig = " << time_orig << ", opt = " << time_opt << endl;
    cout << "VECTriBoxIntersect : speedup = " << ((time_orig - time_opt) / time_orig) * 100.0 << "%" << endl;
    cout << "VECTriBoxIntersect : fault_range = " << fault_range << "%" << endl;
    cout << "------------------------------------------------" << endl;

    delete timer;
}
