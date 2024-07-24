#include <cstring>
#include <iostream>

struct A {
    A() = default; // will ptr be initialized to nullptr by default? NO!

    int *ptr;
};

struct DefaultInit {
    struct Object {
        Object(int val) { printf("Object init with value %d\n", val); }
    };

    DefaultInit() = default;

    DefaultInit(int val) : obj(val) {}

    DefaultInit(int val, bool) { obj = Object(val); }

    Object obj = Object(0);
};

int main() {
    char buf[128];
    memset(buf, 0xff, sizeof(buf));

    {
        A a;
        printf("a.ptr: %p\n", a.ptr);
    }
    {
        A a{};
        printf("a.ptr: %p\n", a.ptr);
    }

    // default initialization
    {
        printf("Creating a\n");
        DefaultInit a; // prints 0
        printf("Creating b\n");
        DefaultInit b(1); // prints 1
        printf("Creating c\n");
        DefaultInit c(1, true); // prints 0 & 1
    }

    return 0;
}
