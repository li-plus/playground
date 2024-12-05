// unpack packed int4 to packed int8

#include <cassert>
#include <chrono>
#include <climits>
#include <iostream>

inline uint64_t naive_unpack(uint32_t x) {
    uint32_t sign = x & 0x08080808;
    sign = (sign << 1) | (sign << 2);
    sign = sign | (sign << 2);
    uint32_t lo = x | sign;

    sign = x & 0x80808080;
    sign |= (sign >> 1);
    sign |= (sign >> 2);
    uint32_t hi = (x >> 4) | sign;

    return (uint64_t(hi) << 32) | lo;
}

inline uint64_t optimized_unpack(uint32_t x) {
    uint32_t sign = x & 0x08080808;
    uint32_t lo = ((sign << 5) - sign) | x;

    uint32_t hi = x >> 4;
    sign = hi & 0x08080808;
    hi = ((sign << 5) - sign) | hi;

    return (uint64_t(hi) << 32) | lo;
}

int main() {
    for (size_t x = 0; x <= UINT_MAX; x++) {
        assert(naive_unpack(x) == optimized_unpack(x));
    }
    return 0;
}