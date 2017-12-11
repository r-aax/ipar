/// \file
/// \brief Exchange buffer implementation.

#include "ExchangeBuffer.h"
#include "IO.h"
#include "Maths.h"
#include "Debug.h"

namespace Utils {

/// \brief Constructor.
///
/// \param n - words count
ExchangeBuffer::ExchangeBuffer(int n)
    : B(NULL), N(0)
{
    Allocate(n);
}

/// \brief Destructor.
ExchangeBuffer::~ExchangeBuffer()
{
    Deallocate();
}

#ifdef DEBUG

/// \brief Set value to the buffer.
///
/// \param v - value
void ExchangeBuffer::Set(double v)
{
    for (int i = 0; i < N; i++)
    {
        B[i] = v;
    }
}

/// \brief Set 0.0 to the buffer.
void ExchangeBuffer::Set0()
{
    Set(0.0);
}

/// \brief Set 1.0 to the buffer.
void ExchangeBuffer::Set1()
{
    Set(1.0);
}

/// \brief Check buffer for the value.
///
/// \param v - value
///
/// \return
/// true - if the buffer is set to the value,
/// false - otherwise.
bool ExchangeBuffer::Is(double v) const
{
    for (int i = 0; i < N; i++)
    {
        if (!MATHS_IS_EQ(B[i], v))
        {
            return false;
        }
    }

    return true;
}

/// \brief Check for buffer set to в 0.0.
///
/// \return
/// true - if the buffer is set to 0.0,
/// false - otherwise.
bool ExchangeBuffer::Is0() const
{
    return Is(0.0);
}

/// \brief Check for buffer set to 1.0.
///
/// \return
/// true - if the buffer is set to 1.0.
/// false - otherwise.
bool ExchangeBuffer::Is1() const
{
    return Is(1.0);
}

#endif

/// \brief Memory allocation.
///
/// \param n - words count
///
/// \return
/// true - if memory is allocated,
/// false - otherwise.
bool ExchangeBuffer::Allocate(int n)
{
    Deallocate();

    DEBUG_CHECK(B == NULL, "double exchange buffer memory allocation");

    N = n;
    B = new double[N];

    return B != NULL;
}

/// \brief Memory deallocation.
void ExchangeBuffer::Deallocate()
{
    if (B != NULL)
    {
        delete [] B;
        B = NULL;
    }
}

}
