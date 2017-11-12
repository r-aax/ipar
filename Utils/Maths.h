/// \file
/// \brief Mathematical primitives.

#ifndef UTILS_MATHS_H
#define UTILS_MATHS_H

#include "Debug.h"

/// \brief Small value.
#define MATHS_EPS 1e-10

/// \brief Absolute value.
///
/// \param X - value
///
/// \return
/// Absolute value.
#define MATHS_ABS(X) (((X) > 0) ? (X) : (-(X)))

/// \brief Check for doubles equal.
///
/// \param X - first value
/// \param Y - second value
///
/// \return
/// true - if values are equal,
/// false - otherwise.
#define MATHS_IS_EQ(X, Y) (MATHS_ABS((X) - (Y)) < MATHS_EPS)

/// \brief Check for first value is greater than the second.
///
/// \param X - first value
/// \param Y - second value
///
/// \return
/// true - if first value is greater than the second,
/// false - otherwise.
#define MATHS_IS_GT(X, Y) ((X) > (Y) + MATHS_EPS)

/// \brief Check for first value is less than the second.
///
/// \param X - first value
/// \param Y - second value
///
/// \return
/// true - is first value is less than the second,
/// false - otherwise.
#define MATHS_IS_LT(X, Y) ((X) < (Y) - MATHS_EPS)

/// \brief Swap values.
///
/// \param T - type
/// \param X - first value
/// \param Y - second value
#define MATHS_SWAP(T, X, Y) \
{ \
    T tmp = (X); \
    (X) = (Y); \
    (Y) = tmp; \
}

/// \brief Swap values if first value is greater than the second.
///
/// \param T - type
/// \param X - first value
/// \param Y - second value
#define MATHS_SWAP_IF_GT(T, X, Y) if ((X) > (Y)) MATHS_SWAP(T, X, Y)

namespace Utils {

/// \brief Maximum value.
///
/// \param T - type
/// \param x - first value
/// \param y - second value
///
/// \return
/// Maximum value.
template<typename T>
T Max(T x, T y)
{
    return (x >= y) ? x : y;
}

/// \brief Minimum value.
///
/// \param T - type
/// \param x - first value
/// \param y - second value
///
/// \return
/// Minimum value.
template<typename T>
T Min(T x, T y)
{
    return (x <= y) ? x : y;
}

/// \brief Maximum value.
///
/// \param T - type
/// \param x - first value
/// \param y - second value
/// \param z - third value
///
/// \return
/// Maximum value.
template<typename T>
T Max(T x, T y, T z)
{
    T m = Max<T>(x, y);

    return Max<T>(m, z);
}

/// \brief Minimum value.
///
/// \param T - type
/// \param x - first
/// \param y - second
/// \param z - third
///
/// \return
/// Minimum value.
template<typename T>
T Min(T x, T y, T z)
{
    T m = Min<T>(x, y);

    return Min<T>(m, z);
}

/// \brief Mean value.
///
/// \param T - type
/// \param x - first value
/// \param y - second value
/// \param z - third value
///
/// \return
/// Mean value.
template<typename T>
T Mean(T x, T y, T z)
{
    if ((x >= y) && (x >= z))
    {
        return Max<T>(y, z);
    }
    else if ((y >= x) && (y >= z))
    {
        return Max<T>(x, z);
    }
    else
    {
        return Max<T>(x, y);
    }
}

/// \brief Absolute value.
///
/// \param T - type
/// \param x - value
///
/// \return
/// Absolute value.
template<typename T>
T Abs(T x)
{
    return (x >= 0.0) ? x : -x;
}

/// \brief Check the vaue is in bounds.
///
/// \param T - type
/// \param x - value
/// \param a - low bound
/// \param b - high bound
///
/// \return
/// true - if the value is in bounds,
/// false - otherwise.
template<typename T>
bool IsInBounds(T x, T a, T b)
{
    return (x >= a) && (x <= b);
}

/// \brief Average value of the array.
///
/// \param T - type
/// \param a - array
/// \param count - elements count
///
/// \return
/// Average value of the array.
template<typename T>
T Avg(const T *a, int count)
{
    DEBUG_IF_THROW(count == 0, "it is impossible to find average value of empty array");

    T s = a[0];
    for (int i = 1; i < count; i++)
    {
        s += a[i];
    }

    return static_cast<T>(s / count);
}


/// \brief Sort two values.
///
/// \param T - type
/// \param x - first value
/// \param y - second value
template<typename T>
void Sort(T &x, T &y)
{
    T m[] = { x, y };

    MATHS_SWAP_IF_GT(T, m[0], m[1]);

    x = m[0];
    y = m[1];
}

/// \brief Sort three values.
///
/// \param T - type
/// \param x - first value
/// \param y - second value
/// \param z - third value
template<typename T>
void Sort(T &x, T &y, T &z)
{
    T m[] = { x, y, z };

    MATHS_SWAP_IF_GT(T, m[0], m[1]);
    MATHS_SWAP_IF_GT(T, m[1], m[2]);
    MATHS_SWAP_IF_GT(T, m[0], m[1]);

    x = m[0];
    y = m[1];
    z = m[2];
}

}

#endif
