/*
Compile without cache:
$ time g++ ccache_test.cpp -ftemplate-depth=10000000
14.31s user 0.73s system 99% cpu 15.044 total

Compile with ccache the second time:
$ time ccache g++ ccache_test.cpp -c -o ccache_test.o -ftemplate-depth=10000000
0.01s user 0.02s system 97% cpu 0.027 total
$ time g++ ccache_test.o -o ccache_test
0.21s user 0.04s system 99% cpu 0.249 total
*/

#include <stdio.h>

template <int x>
int cumsum() {
    return cumsum<x - 1>() + x;
}

template <>
int cumsum<1>() {
    return 1;
}

int main() {
    constexpr int N = 1024 * 16;
    printf("N! = %d\n", cumsum<N>());
}
