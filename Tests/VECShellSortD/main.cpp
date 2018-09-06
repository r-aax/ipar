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

/// \brief Array size.
#define ARRAY_SIZE 50000

/// \brief Align.
#ifdef INTEL
#define ALIGN_64 __declspec(align(64))
#else
#define ALIGN_64
#endif

/// \brief Array.
ALIGN_64 double m[ARRAY_SIZE];

/// \brief Array for orig.
ALIGN_64 double m_orig[ARRAY_SIZE];

/// \brief Array for opt.
ALIGN_64 double m_opt[ARRAY_SIZE];


using namespace Utils;

/// \brief Sum of float array elements.
///
/// \param a - array
/// \param c - elements count
///
/// \return
/// Sum of elements.
static double array_sum(double *a, int c)
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
static double array_pm_sum(double *a, int c)
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
static void clean_array(double *a, int c)
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
static void random_array(double *a, int c)
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
static bool is_array_sorted(double *a, int c)
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
static void arrays_copy(double *from, double *to, int c)
{
    for (int i = 0; i < c; i++)
    {
        to[i] = from[i];
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

    cout << "VECShellSort : test begin" << endl;
    cout << "------------------------------" << endl;

    Timer *timer = new Timer(Timer::OMP);
    double time_orig, time_opt, sum_orig, sum_opt, pm_sum_orig, pm_sum_opt;
    bool check_orig, check_opt;

    // Init.
    random_array(m,  ARRAY_SIZE);

    // Original.
    arrays_copy(m, m_orig, ARRAY_SIZE);
    timer->Init();
    shell_sort_orig(m_orig, ARRAY_SIZE);
    arrays_copy(m, m_orig, ARRAY_SIZE);
    timer->Start();
    shell_sort_orig(m_orig, ARRAY_SIZE);
    timer->Stop();
    time_orig = timer->Time();
    check_orig = is_array_sorted(m_orig, ARRAY_SIZE);
    sum_orig = array_sum(m_orig, ARRAY_SIZE);
    pm_sum_orig = array_pm_sum(m_orig, ARRAY_SIZE);

    // Optimized.
    shell_sort_opt_prepare();
    arrays_copy(m, m_opt, ARRAY_SIZE);
    timer->Init();
    shell_sort_opt(m_opt, ARRAY_SIZE);
    arrays_copy(m, m_opt, ARRAY_SIZE);
    timer->Start();
    shell_sort_opt(m_opt, ARRAY_SIZE);
    timer->Stop();
    time_opt = timer->Time();
    check_opt = is_array_sorted(m_opt, ARRAY_SIZE);
    sum_opt = array_sum(m_opt, ARRAY_SIZE);
    pm_sum_opt = array_pm_sum(m_opt, ARRAY_SIZE);

    cout << "VECShellSort : orig = " << time_orig << ", opt = " << time_opt << endl;
    //DEBUG_CHECK(check_orig, "orig check failed");
    //DEBUG_CHECK(check_opt, "opt check failed");
    cout << "VECShellSort : sum orig = " << sum_orig << ", sum opt = " << sum_opt << endl;
    cout << "VECShellSort : pm_sum orig = " << pm_sum_orig << ", pm_sum opt = " << pm_sum_opt << endl;
}
