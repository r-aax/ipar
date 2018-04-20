/// \file
/// \brief Bits manipulation realization.

#ifndef UTILS_BITS_H
#define UTILS_BITS_H

/// \brief Mask construction from 16 bits values.
///
/// \param B15 - 15-th bit value
/// \param B14 - 14-th bit value
/// \param B13 - 13-th bit value
/// \param B12 - 12-th bit value
/// \param B11 - 11-th bit value
/// \param B10 - 10-th bit value
/// \param B09 - 9-th bit value
/// \param B08 - 8-th bit value
/// \param B07 - 7-th bit value
/// \param B06 - 6-th bit value
/// \param B05 - 5-th bit value
/// \param B04 - 4-th bit value
/// \param B03 - third bit value
/// \param B02 - second bit value
/// \param B01 - first bit value
/// \param B00 - zero bit value
#define BITS_MASK_16(B15, B14, B13, B12, B11, B10, B09, B08,       \
                     B07, B06, B05, B04, B03, B02, B01, B00)       \
(   ((B00) <<  0) | ((B01) <<  1) | ((B02) <<  2) | ((B03) <<  3)  \
  | ((B04) <<  4) | ((B05) <<  5) | ((B06) <<  6) | ((B07) <<  7)  \
  | ((B08) <<  8) | ((B09) <<  9) | ((B10) << 10) | ((B11) << 11)  \
  | ((B12) << 12) | ((B13) << 13) | ((B14) << 14) | ((B15) << 15)) )

#endif