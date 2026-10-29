
/*
 * Research 02 — Independent C++ numerical kernels.
 *
 * Abnormal returns, cumulative abnormal returns,
 * and Holm multiple-testing adjustment.
 *
 * C++17, standard library only.
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace quassar::research02 {

using Vector = std::vector<double>;

Vector abnormal_returns(
    const Vector& observed,
    const Vector& expected
) {
    if (observed.empty() || observed.size() != expected.size()) {
        throw std::invalid_argument("Invalid input dimensions");
    }

    Vector result;
    result.reserve(observed.size());

    for (std::size_t i = 0; i < observed.size(); ++i) {
        if (!std::isfinite(observed[i]) ||
            !std::isfinite(expected[i])) {
            throw std::invalid_argument("Nonfinite input");
        }

        const double value = observed[i] - expected[i];

        if (!std::isfinite(value)) {
            throw std::overflow_error("Abnormal-return overflow");
        }

        result.push_back(value);
    }

    return result;
}

double cumulative_abnormal_return(
    const Vector& abnormal,
    std::size_t start,
    std::size_t stop
) {
    if (start >= stop || stop > abnormal.size()) {
        throw std::invalid_argument("Invalid event window");
    }

    if (!std::all_of(
            abnormal.begin(),
            abnormal.end(),
            [](double x) { return std::isfinite(x); }
        )) {
        throw std::invalid_argument("Nonfinite abnormal return");
    }

    double sum = 0.0;

    for (std::size_t i = start; i < stop; ++i) {
        sum += abnormal[i];
    }

    if (!std::isfinite(sum)) {
        throw std::overflow_error("CAR overflow");
    }

    return sum;
}

Vector holm_adjustment(const Vector& pvalues) {
    if (pvalues.empty()) {
        throw std::invalid_argument("Empty hypothesis family");
    }

    const std::size_t m = pvalues.size();

    for (double p : pvalues) {
        if (!std::isfinite(p) || p < 0.0 || p > 1.0) {
            throw std::invalid_argument("Invalid p-value");
        }
    }

    std::vector<std::size_t> order(m);
    std::iota(order.begin(), order.end(), 0);

    std::sort(
        order.begin(),
        order.end(),
        [&](std::size_t a, std::size_t b) {
            if (pvalues[a] == pvalues[b]) {
                return a < b;
            }
            return pvalues[a] < pvalues[b];
        }
    );

    Vector adjusted(m, 0.0);
    double running_max = 0.0;

    for (std::size_t rank = 0; rank < m; ++rank) {
        const std::size_t index = order[rank];

        const double multiplier =
            static_cast<double>(m - rank);

        const double candidate = std::min(
            1.0,
            pvalues[index] * multiplier
        );

        running_max = std::max(running_max, candidate);
        adjusted[index] = running_max;
    }

    return adjusted;
}

}  // namespace quassar::research02

#ifdef QUASSAR_RESEARCH02_TEST

int main() {
    using namespace quassar::research02;

    const auto abnormal = abnormal_returns(
        {0.03, -0.01, 0.04},
        {0.01, 0.00, 0.02}
    );

    assert(abnormal.size() == 3);
    assert(std::abs(abnormal[0] - 0.02) < 1e-12);
    assert(std::abs(abnormal[1] + 0.01) < 1e-12);
    assert(std::abs(abnormal[2] - 0.02) < 1e-12);

    const double car = cumulative_abnormal_return(
        {0.01, 0.02, -0.01, 0.03},
        1,
        4
    );

    assert(std::abs(car - 0.04) < 1e-12);

    const auto adjusted = holm_adjustment(
        {0.01, 0.04, 0.03, 0.20}
    );

    const Vector expected = {0.04, 0.09, 0.09, 0.20};

    for (std::size_t i = 0; i < expected.size(); ++i) {
        assert(std::abs(adjusted[i] - expected[i]) < 1e-12);
    }

    return 0;
}

#endif
