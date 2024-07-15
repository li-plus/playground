#include <iostream>
#include <vector>

struct AutoMove {
    int size;
    std::vector<int> data;

    AutoMove(int size) : size(size), data(size) {}
};

struct CustomMove {
    int size;
    std::vector<int> data;

    CustomMove(int size) : size(size), data(size) {}

    CustomMove(CustomMove &&other) { *this = std::move(other); }

    CustomMove &operator=(CustomMove &&other) {
        size = other.size;
        data = std::move(other.data);
        other.size = 0; // manually clear states of the moved object
        return *this;
    }
};

int main() {
    {
        AutoMove a(16);
        AutoMove b = std::move(a);
        // a.size will not automaticaly become zero
        std::cout << "AutoMove\n"
                  << "a.size = " << a.size << ", a.data.size = " << a.data.size() << "\n"
                  << "b.size = " << b.size << ", b.data.size = " << b.data.size() << "\n";
    }
    {
        CustomMove a(16);
        CustomMove b = std::move(a);
        // a.size will become zero since it's been moved
        std::cout << "CustomMove\n"
                  << "a.size = " << a.size << ", a.data.size = " << a.data.size() << "\n"
                  << "b.size = " << b.size << ", b.data.size = " << b.data.size() << "\n";
    }

    return 0;
}