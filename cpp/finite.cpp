#include <cmath>
#include <iostream>
#include <limits>

int main() {
    std::cout << "0.f / 0.f     : " << 0.f / 0.f << "\n"
              << "1.f / 0.f     : " << 1.f / 0.f << "\n"
              << "-1.f / 0.f    : " << -1.f / 0.f << "\n"
              << "\n";

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    std::cout << "nan           : " << nan << "\n"
              << "inf           : " << inf << "\n"
              << "-inf          : " << -inf << "\n"
              << "\n";

    std::cout << "isfinite(nan) : " << std::isfinite(nan) << "\n"
              << "isfinite(inf) : " << std::isfinite(inf) << "\n"
              << "isfinite(-inf): " << std::isfinite(-inf) << "\n"
              << "\n";

    std::cout << "isnan(nan)    : " << std::isnan(nan) << "\n"
              << "isnan(inf)    : " << std::isnan(inf) << "\n"
              << "isnan(-inf)   : " << std::isnan(-inf) << "\n"
              << "\n";

    std::cout << "isinf(nan)    : " << std::isinf(nan) << "\n"
              << "isinf(inf)    : " << std::isinf(inf) << "\n"
              << "isinf(-inf)   : " << std::isinf(-inf) << "\n"
              << "\n";

    std::cout << "nan == nan    : " << (nan == nan) << "\n"
              << "nan < nan     : " << (nan < nan) << "\n"
              << "nan > nan     : " << (nan > nan) << "\n"
              << "nan == 0      : " << (nan == 0) << "\n"
              << "nan < 0       : " << (nan < 0) << "\n"
              << "nan > 0       : " << (nan > 0) << "\n"
              << "\n";

    std::cout << "max(nan, 0)   : " << std::max(nan, 0.f) << "\n"
              << "max(0, nan)   : " << std::max(0.f, nan) << "\n"
              << "max(nan, nan) : " << std::max(nan, nan) << "\n"
              << "\n";

    std::cout << "fmax(nan, 0)  : " << std::fmax(nan, 0.f) << "\n"
              << "fmax(0, nan)  : " << std::fmax(0.f, nan) << "\n"
              << "fmax(nan, nan): " << std::fmax(nan, nan) << "\n"
              << "\n";

    std::cout << "inf == inf    : " << (inf == inf) << "\n"
              << "inf > 0       : " << (inf > 0) << "\n"
              << "inf < 0       : " << (inf < 0) << "\n"
              << "\n";

    std::cout << "exp(nan)      : " << std::exp(nan) << "\n"
              << "exp(inf)      : " << std::exp(inf) << "\n"
              << "exp(-inf)     : " << std::exp(-inf) << "\n"
              << "\n";

    std::cout << "log(nan)      : " << std::log(nan) << "\n"
              << "log(-inf)     : " << std::log(-inf) << "\n"
              << "log(0)        : " << std::log(0) << "\n"
              << "log(inf)      : " << std::exp(inf) << "\n"
              << "\n";

    return 0;
}
