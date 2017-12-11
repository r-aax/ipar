/// \file
/// \brief Debug.

#include <stdlib.h>
#include "IO.h"

#ifndef DEBUG_H
#define DEBUG_H

#ifdef DEBUG

/// \brief Debug exception.
///
/// \param S - diagnostic
#define DEBUG_WHAT(S) \
    ( \
        string(S) \
        + string(" (") \
        + string(__FILE__) \
        + string(", ") \
        + Utils::IO::ToString<int>(__LINE__) \
        + string(")") \
    )

/// \brief Throw exception
///
/// \param S - diagnostic
#define DEBUG_THROW(S) \
    throw runtime_error(DEBUG_WHAT(S))

/// \brief Exit.
///
/// \param S - diagnostic
#define DEBUG_EXIT(S) \
    { \
        cout << "Internal Error : " << DEBUG_WHAT(S) << endl; \
        exit(1); \
    }

/// \brief Conditional throw exception.
///
/// \param C - condition
/// \param S - diagnostic
#define DEBUG_IF_THROW(C, S) \
    if (C) \
    { \
        DEBUG_THROW(S); \
    }

/// \brief Debug checker.
///
/// \param С - condition
/// \param S - diagnostic
#define DEBUG_CHECK(C, S) \
    { \
        if (!(C)) \
        { \
            DEBUG_EXIT(S); \
        } \
    }

#else

/// \brief Throw exception with diag.
///
/// \param S - diagnostic
#define DEBUG_THROW(S)

/// \brief Exit.
///
/// \param S - diagnostic
#define DEBUG_EXIT(S)

/// \brief Conditional throw exception.
///
/// \param C - condition
/// \param S - diagnostic
#define DEBUG_IF_THROW(C, S)

/// \brief Debug checker.
///
/// \param С - condition
/// \param S - diagnostic
#define DEBUG_CHECK(C, S)

#endif

#endif
