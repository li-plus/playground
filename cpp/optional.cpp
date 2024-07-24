#include <iostream>
#include <optional>
#include <vector>

int main() {
    // optional class
    std::cout << "sizeof(std::optional<char>)   = " << sizeof(std::optional<char>) << "\n"
              << "sizeof(std::optional<int>)    = " << sizeof(std::optional<int>) << "\n"
              << "sizeof(std::optional<double>) = " << sizeof(std::optional<double>) << "\n"
              << "sizeof(vector<int>)           = " << sizeof(std::vector<int>) << "\n"
              << "sizeof(std::optional<vector<int>>)    = " << sizeof(std::optional<std::vector<int>>) << "\n";

    std::optional<int> a;
    std::cout << "a.has_value() = " << a.has_value() << "\n";
    std::optional<int> b = std::nullopt;
    std::cout << "b.has_value() = " << b.has_value() << "\n";
    std::optional<int> c = 12345;
    std::cout << "c.has_value() = " << c.has_value() << "\n";
    std::optional<int> d = c;
    std::cout << "d.has_value() = " << d.has_value() << ", c.has_value() = " << c.has_value() << "\n";
    std::optional<int> e = std::move(d);
    std::cout << "e.has_value() = " << e.has_value() << ", d.has_value() = " << d.has_value() << "\n";

    return 0;
}
