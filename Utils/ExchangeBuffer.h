/// \file
/// \brief Exchange buffer.

#ifndef UTILS_EXCHANGE_BUFFER_H
#define UTILS_EXCHANGE_BUFFER_H

namespace Utils {

/// \brief Exchange buffer.
class ExchangeBuffer
{
    friend class Iface;

public:

    // Constructor/destructor.
    ExchangeBuffer(int n);
    ~ExchangeBuffer();

#ifdef DEBUG

    // Set and check values.
    void Set(double v);
    void Set0();
    void Set1();
    bool Is(double v) const;
    bool Is0() const;
    bool Is1() const;

#endif

private:

    // Memory allocation and deallocation.
    bool Allocate(int n);
    void Deallocate();

    // Data.
    double *B; ///< buffer
    int N;     ///< doubles count

public:

    /// \brief Words buffer.
    ///
    /// \return
    /// Words buffers.
    double *BufferDoubles()
    {
        return B;
    }

    /// \brief Words count in buffer.
    ///
    /// \return
    /// Words count in buffer.
    double BufferDoublesCount()
    {
        return N;
    }
};

}

#endif
