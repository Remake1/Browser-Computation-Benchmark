// Native timing driver for wasm/src/reduction.cpp. Prints the median kernel time and
// GB/s read per size (the same sweep as web/src/workloads/reduction), to compare with
// the browser's compute-phase numbers. Inputs are multiples of 1/256 in (−1, 1), like
// the browser's, so every summation order is exact and the check is equality.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

extern "C" {
int reduction_init(int threads);
void reduction_shutdown();
float reduction(const float* x, int n);
}

int main() {
  const int threads = reduction_init(std::min(8, static_cast<int>(std::thread::hardware_concurrency())));
  std::printf("threads=%d\n%10s %12s %10s %6s\n", threads, "n", "median_ms", "GB/s", "exact");
  std::mt19937 gen(1);
  std::uniform_int_distribution<int> dist(-255, 255);
  for (int n = 1 << 10; n <= 1 << 26; n = n % 3 == 0 ? n / 3 * 4 : n / 2 * 3) {
    std::vector<float> x(n);
    double ref = 0;
    for (auto& v : x) {
      v = dist(gen) / 256.0f;
      ref += v;
    }
    // Odd lengths exercise the tails of the vector loop and the thread split.
    bool exact = reduction(x.data(), n) == ref && reduction(x.data(), n - 7) == ref - (x[n - 7] + x[n - 6] + x[n - 5] + x[n - 4] + x[n - 3] + x[n - 2] + x[n - 1]);

    // Repeat until ≥ 0.3 s and ≥ 5 runs (capped at 1000); batch small sizes like the browser runner.
    const int batch = std::max(1, (1 << 20) / n);
    std::vector<double> ms;
    double spent = 0;
    volatile float sink = 0;
    while ((spent < 300 || ms.size() < 5) && ms.size() < 1000) {
      const auto t0 = std::chrono::steady_clock::now();
      for (int r = 0; r < batch; ++r) sink = reduction(x.data(), n);
      const double dt = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      ms.push_back(dt / batch);
      spent += dt;
    }
    std::nth_element(ms.begin(), ms.begin() + ms.size() / 2, ms.end());
    const double med = ms[ms.size() / 2];
    std::printf("%10d %12.4f %10.2f %6s\n", n, med, 4.0 * n / (med * 1e6), exact ? "yes" : "NO");
  }
  reduction_shutdown();
}
