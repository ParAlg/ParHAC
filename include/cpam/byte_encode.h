#pragma once

namespace cpam {

#define LAST_BIT_SET(b) (b & (0x80))
#define SIZE_PER_BYTE 7

// Reads K's (an unsigned type).
template <class K>
inline K decodeUnsigned(uint8_t*& start) {
  K value = 0;
  uint8_t* ptr = start;
  uint8_t shiftAmount = 0;
  while (1) {
    uint8_t b = *ptr++;
    value += (static_cast<K>(b & 0x7f) << shiftAmount);
    if (!LAST_BIT_SET(b)) break;
    shiftAmount += SIZE_PER_BYTE;
  }
  start = ptr;
  return value;
}

template <class K>
inline long encodeUnsigned(uint8_t* start, long curOffset, K diff) {
  uint8_t curByte = diff & 0x7f;
  int bytesUsed = 0;
  while ((curByte > 0) || (diff > 0)) {
    bytesUsed++;
    uint8_t toWrite = curByte;
    diff = diff >> 7;
    // Check to see if there's any bits left to represent
    curByte = diff & 0x7f;
    if (diff > 0) {
      toWrite |= 0x80;
    }
    start[curOffset] = toWrite;
    curOffset++;
  }
  return curOffset;
}

}  // namespace cpam
