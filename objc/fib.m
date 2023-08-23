#include <stdio.h>

static int fib(int n) {
    int prev = 1;
    int curr = 1;
    for (int i = 2; i < n; i++) {
        int next = prev + curr;
        prev = curr;
        curr = next;
    }
    return curr;
}

int main() {
    printf("fib array: ");
    for (int i = 1; i <= 10; i++) {
        printf("%d ", fib(i));
    }
    printf("\n");
    return 0;
}
