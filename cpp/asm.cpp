// asm extension:
// https://gcc.gnu.org/onlinedocs/gcc/extensions-to-the-c-language-family/how-to-use-inline-assembly-language-in-c-code.html
// x86 instructions: https://www.felixcloutier.com/x86/

#include <iostream>
#include <x86intrin.h>

uint64_t rdtsc() { return __rdtsc(); }

// https://stackoverflow.com/questions/34810392/assembler-instruction-rdtsc
uint64_t rdtsc_asm1() {
    uint64_t msr;
    asm volatile("rdtsc\n\t"          // Returns the time in EDX:EAX.
                 "shl $32, %%rdx\n\t" // Shift the upper bits left.
                 "or %%rdx, %0"       // 'Or' in the lower bits.
                 : "=a"(msr)
                 :
                 : "rdx");
    return msr;
}

uint64_t rdtsc_asm2() {
    uint32_t eax, edx;
    asm volatile("rdtsc\n\t" : "=a"(eax), "=d"(edx));
    return (uint64_t)eax | (uint64_t)edx << 32;
}

int inc(int a) {
    asm volatile("inc %0\n\t" : "+r"(a));
    return a;
}

int add(int a, int b) {
    asm volatile("add %1,%0\n\t" : "+r"(a) : "r"(b));
    return a;
}

void swap(int *a, int *b) {
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

void swap_asm(int *a, int *b) {
    int tmp;
    asm volatile("mov %0,%2\n\t"
                 "mov %1,%0\n\t"
                 "mov %2,%1\n\t"
                 : "+rm"(*a), "+rm"(*b), "=r"(tmp)
                 :
                 : "memory");
}

int main() {
    printf("rdtsc: %zu\n", rdtsc());
    printf("rdtsc1: %zu\n", rdtsc_asm1());
    printf("rdtsc2: %zu\n", rdtsc_asm2());

    // inc
    printf("inc: 6+1=%d\n", inc(6));
    printf("add: 6+7=%d\n", add(6, 7));

    int a = 1, b = 2;
    printf("before swap: a=%d, b=%d\n", a, b);
    swap(&a, &b);
    printf("swap: a=%d, b=%d\n", a, b);
    swap_asm(&a, &b);
    printf("swap-asm: a=%d, b=%d\n", a, b);

    return 0;
}