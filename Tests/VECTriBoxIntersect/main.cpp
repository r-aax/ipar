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

/// @brief Tests count.
#define TESTS_COUNT 10

using namespace Utils;

/// @brief Datas.

float Ax[TESTS_COUNT];
float Ay[TESTS_COUNT];
float Az[TESTS_COUNT];
float Bx[TESTS_COUNT];
float By[TESTS_COUNT];
float Bz[TESTS_COUNT];
float Cx[TESTS_COUNT];
float Cy[TESTS_COUNT];
float Cz[TESTS_COUNT];
float Xl[TESTS_COUNT];
float Xh[TESTS_COUNT];
float Yl[TESTS_COUNT];
float Yh[TESTS_COUNT];
float Zl[TESTS_COUNT];
float Zh[TESTS_COUNT];

/// @brief Original results.
bool R_orig[TESTS_COUNT];

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
/// All cases are right.
static bool
check()
{
    for (int i = 0; i < TESTS_COUNT; i++)
    {
        if (R_orig[i] != R_opt[i])
        {
            return false;
        }
    }

    return true;
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
    bool is_ok;

    // Original.
    timer->Init();
    for (int i = 0; i < repeats_count; i++)
    {
        tri_box_intersects_orig(Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Xl, Xh, Yl, Yh, Zl, Zh, TESTS_COUNT, R_orig);
    }
    timer->Start();
    for (int i = 0; i < repeats_count; i++)
    {
        tri_box_intersects_orig(Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Xl, Xh, Yl, Yh, Zl, Zh, TESTS_COUNT, R_orig);
    }
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
    is_ok = check();

    cout << "VECTriBoxIntersect : orig = " << time_orig << ", opt = " << time_opt << endl;
    cout << "VECTriBoxIntersect : speedup = " << ((time_orig - time_opt) / time_orig) * 100.0 << "%" << endl;
    cout << "VECTriBoxIntersect : check = " << is_ok << endl;
    cout << "------------------------------------------------" << endl;

    delete timer;
}
