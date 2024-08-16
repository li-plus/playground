#include <cassert>
#include <iostream>

inline int float_as_int(float x) { return *(int *)&x; }

inline float int_as_float(int x) { return *(float *)&x; }

void inspect_float(float x) {
    // sign 1, exponent 8, fraction 23
    const unsigned ix = float_as_int(x);
    printf("value=%f, hex=0x%08x, sign=%d, exp=0x%02x, frac=0x%06x\n", x, ix, ix >> 31, (ix >> 23) & 0xff,
           ix & 0x7fffff);
}

// https://forums.developer.nvidia.com/t/type-conversion-throughput-latency/281663/2
float fast_int_to_float(int x) {
    // assert(-(1 << 22) <= x && x <= (1 << 22));
    const float fmagic = (1 << 23) + (1 << 22);
    const int imagic = float_as_int(fmagic);
    return int_as_float(imagic + x) - fmagic;
}

int main() {
    inspect_float(0.f);
    inspect_float(1.f);
    inspect_float(2.f);
    inspect_float(3.f);

    inspect_float(-0.f);
    inspect_float(-1.f);
    inspect_float(-2.f);
    inspect_float(-3.f);

    for (int i = -(1 << 22); i <= (1 << 22); i++) {
        assert(fast_int_to_float(i) == float(i));
    }

    return 0;
}