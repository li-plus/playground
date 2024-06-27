#include <cstring>
#include <iostream>

class A {
  public:
    A() = default; // will ptr be initialized to nullptr by default? NO!

    int *ptr;
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

    return 0;
}
