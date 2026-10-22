// Native timing driver for wasm/src/matmul.cpp. Prints the median kernel time and
// GFLOP/s per size (the same sweep as web/src/workloads/matmul), to compare with
// the browser's compute-phase numbers.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

extern "C" {
int matmul_init(int threads);
void matmul_shutdown();
void matmul(const float* a, const float* b, float* c, int n);
}

int main() {
  const int threads = matmul_init(std::min(8, static_cast<int>(std::thread::hardware_concurrency())));
  std::printf("threads=%d\n%6s %12s %10s %10s\n", threads, "n", "median_ms", "GFLOP/s", "max_err");
  std::mt19937 gen(1);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (int n : {16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048}) {
    const std::size_t nn = static_cast<std::size_t>(n) * n;
    std::vector<float> a(nn), b(nn), c(nn);
    for (auto& x : a) x = dist(gen);
    for (auto& x : b) x = dist(gen);

    // Spot-check row 0 against f64, relative to (|A|·|B|) like the browser verifier.
    matmul(a.data(), b.data(), c.data(), n);
    double max_err = 0;
    for (int j = 0; j < n; ++j) {
      double ref = 0, bound = 0;
      for (int k = 0; k < n; ++k) {
        ref += double(a[k]) * b[std::size_t(k) * n + j];
        bound += std::fabs(double(a[k]) * b[std::size_t(k) * n + j]);
      }
      max_err = std::max(max_err, std::fabs(c[j] - ref) / bound);
    }

    // Repeat until ≥ 0.3 s and ≥ 5 runs (capped at 1000); batch tiny sizes like the browser runner.
    const int batch = n <= 64 ? 100 : 1;
    std::vector<double> ms;
    double spent = 0;
    while ((spent < 300 || ms.size() < 5) && ms.size() < 1000) {
      const auto t0 = std::chrono::steady_clock::now();
      for (int r = 0; r < batch; ++r) matmul(a.data(), b.data(), c.data(), n);
      const double dt = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      ms.push_back(dt / batch);
      spent += dt;
    }
    std::nth_element(ms.begin(), ms.begin() + ms.size() / 2, ms.end());
    const double med = ms[ms.size() / 2];
    std::printf("%6d %12.4f %10.2f %10.1e\n", n, med, 2.0 * n * n * n / (med * 1e6), max_err);
  }
  matmul_shutdown();
}
