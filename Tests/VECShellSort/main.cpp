/// \file
/// \brief Matrices operations vectorization.

#include "../../Utils/IO.h"
#include "../../Utils/Randoms.h"
#include "../../Utils/Timer.h"
#include "../../Utils/Maths.h"
#include "../../Utils/Debug.h"
#include "shell_sort.h"
#include "avx512debug.h"
#include <stdlib.h>
#include <math.h>

/// \brief Min array size.
#define MIN_ARRAY_SIZE 10000

/// \brief Max array size.
#define MAX_ARRAY_SIZE 2000000

/// \brief Array size step.
#define ARRAY_SIZE_STEP 10000

/// \brief Align.
#ifdef INTEL
#define ALIGN_64 __declspec(align(64))
#else
#define ALIGN_64
#endif

/// \brief Array.
ALIGN_64 float m[MAX_ARRAY_SIZE];

/// \brief Array for orig.
ALIGN_64 float m_orig[MAX_ARRAY_SIZE];

/// \brief Array for opt.
ALIGN_64 float m_opt[MAX_ARRAY_SIZE];

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

/// \brief Sum of float array elements with factors 1.0 or -1.0.
///
/// \param a - array
/// \param c - elements count
///
/// \return
/// Sum of elements with factors 1.0 or -1.0.
static double array_pm_sum(float *a, int c)
{
    double s = 0.0;
    double m = 1.0;

    for (int i = 0; i < c; i++)
    {
        s += m * a[i];
        m = -m;
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

/// \brief Check if array sorted.
///
/// \param a - array
/// \param c - count of elements
///
/// \return
/// true - if array is sorted, false - otherwise
static bool is_array_sorted(float *a, int c)
{
    for (int i = 0; i < c - 1; i++)
    {
        if (MATHS_IS_GT(a[i], a[i + 1]))
        {
            cout << "! " << a[i] << " <= " << a[i + 1] << endl;

            return false;
        }
    }

    return true;
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

/// \brief Alias for shell sort orig.
#define SHELL_SORT_ORIG shell_sort_fib_orig

/// \brief ALias for shell sort opt.
#define SHELL_SORT_OPT shell_sort_fib_opt

/// \brief Main function.
///
/// \param argc - arguments count
/// \param argv - arguments
int main(int argc, char **argv)
{
    cout << "================================================" << endl;

    for (int AA = MIN_ARRAY_SIZE; AA <= MAX_ARRAY_SIZE; AA += ARRAY_SIZE_STEP)
    {
    int repeats_count = 1;

    // Parse repeats count if given.
    if (argc == 2)
    {
        repeats_count = atoi(argv[1]);
    }

    cout << "VECShellSort : test begin " << AA << endl;

    Timer *timer = new Timer(Timer::OMP);
    double time_orig, time_opt, sum_orig, sum_opt, pm_sum_orig, pm_sum_opt;
    bool check_orig, check_opt;

    // Init.
    random_array(m,  AA);

    // Original.
    arrays_copy(m, m_orig, AA);
    timer->Init();
    SHELL_SORT_ORIG(m_orig, AA);
    arrays_copy(m, m_orig, AA);
    timer->Start();
    SHELL_SORT_ORIG(m_orig, AA);
    timer->Stop();
    time_orig = timer->Time();
    check_orig = is_array_sorted(m_orig, AA);
    sum_orig = array_sum(m_orig, AA);
    pm_sum_orig = array_pm_sum(m_orig, AA);

    // Optimized.
    shell_sort_opt_prepare();
    arrays_copy(m, m_opt, AA);
    timer->Init();
    SHELL_SORT_OPT(m_opt, AA);
    arrays_copy(m, m_opt, AA);
    timer->Start();
    SHELL_SORT_OPT(m_opt, AA);
    timer->Stop();
    time_opt = timer->Time();
    check_opt = is_array_sorted(m_opt, AA);
    sum_opt = array_sum(m_opt, AA);
    pm_sum_opt = array_pm_sum(m_opt, AA);

    cout << "VECShellSort : orig = " << time_orig << ", opt = " << time_opt << endl;
    DEBUG_CHECK(check_orig, "orig check failed");
    DEBUG_CHECK(check_opt, "opt check failed");
    cout << "VECShellSort : sum orig = " << sum_orig << ", sum opt = " << sum_opt << endl;
    cout << "VECShellSort : pm_sum orig = " << pm_sum_orig << ", pm_sum opt = " << pm_sum_opt << endl;
    //cout << "VECShellSort : speedup = " << ((time_orig - time_opt) / time_orig) * 100.0 << "%" << endl;
    cout << "VECShellSort : speedup = " << time_orig / time_opt << " times" << endl;
    cout << "------------------------------------------------" << endl;

    delete timer;
    }
}
