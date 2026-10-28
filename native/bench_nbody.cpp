// Native timing driver for wasm/src/nbody.cpp. Prints the median kernel time and
// GFLOP/s (20 flops per interaction) per size (the same sweep as
// web/src/workloads/nbody), to compare with the browser's compute-phase numbers.
// Checks every body against f64 with the browser's componentwise bound
// |a − R| ≤ 1e-3 · Σ_j |term_j|.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

extern "C" {
int nbody_init(int threads);
void nbody_shutdown();
void nbody(const float* b, float* acc, int n);
}

namespace {

std::mt19937 gen(1);

/** Positions uniform in [−1, 1)³, masses in [0.5, 1). */
std::vector<float> bodies(int n) {
  std::uniform_real_distribution<float> pos(-1.0f, 1.0f), mass(0.5f, 1.0f);
  std::vector<float> b(4 * static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) b[4 * i] = pos(gen), b[4 * i + 1] = pos(gen), b[4 * i + 2] = pos(gen), b[4 * i + 3] = mass(gen);
  return b;
}

/** Largest err / bound over every component of every body (≤ 1e-3 passes). */
double max_rel_err(const std::vector<float>& b, int n) {
  std::vector<float> acc(b.size());
  nbody(b.data(), acc.data(), n);
  double worst = 0;
  for (int i = 0; i < n; ++i) {
    double ref[3] = {0, 0, 0}, bound[3] = {0, 0, 0};
    for (int j = 0; j < n; ++j) {
      double d[3];
      for (int c = 0; c < 3; ++c) d[c] = double(b[4 * j + c]) - b[4 * i + c];
      const double d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + double(0.01f);
      const double s = b[4 * j + 3] / (d2 * std::sqrt(d2));
      for (int c = 0; c < 3; ++c) ref[c] += d[c] * s, bound[c] += std::fabs(d[c] * s);
    }
    for (int c = 0; c < 3; ++c) {
      const double err = std::fabs(acc[4 * i + c] - ref[c]);
      if (err > 0) worst = std::max(worst, bound[c] > 0 ? err / bound[c] : INFINITY);
    }
    if (acc[4 * i + 3] != 0) worst = INFINITY;
  }
  return worst;
}

}  // namespace

int main() {
  const int threads = nbody_init(std::min(8, static_cast<int>(std::thread::hardware_concurrency())));
  // Sizes below one SIMD block, odd tails and uneven thread splits.
  double odd = 0;
  for (int n : {1, 2, 3, 5, 7, 33, 101, 1001}) odd = std::max(odd, max_rel_err(bodies(n), n));
  std::printf("threads=%d odd sizes max rel err: %.2e\n%6s %12s %10s %10s\n", threads, odd, "n", "median_ms", "GFLOP/s", "rel_err");
  for (int n = 64; n <= 65536; n = n % 3 == 0 ? n / 3 * 4 : n / 2 * 3) {
    const std::vector<float> b = bodies(n);
    std::vector<float> acc(b.size());
    const double err = n <= 16384 ? max_rel_err(b, n) : NAN;  // f64 reference is O(n²): skip the largest

    // Repeat until ≥ 0.3 s and ≥ 5 runs (capped at 1000); batch small sizes like the browser runner.
    const int batch = std::max(1, (1 << 20) / (n * n));
    std::vector<double> ms;
    double spent = 0;
    while ((spent < 300 || ms.size() < 5) && ms.size() < 1000) {
      const auto t0 = std::chrono::steady_clock::now();
      for (int r = 0; r < batch; ++r) nbody(b.data(), acc.data(), n);
      const double dt = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      ms.push_back(dt / batch);
      spent += dt;
    }
    std::nth_element(ms.begin(), ms.begin() + ms.size() / 2, ms.end());
    const double med = ms[ms.size() / 2];
    std::printf("%6d %12.4f %10.2f %10.2e\n", n, med, 20.0 * n * n / (med * 1e6), err);
  }
  nbody_shutdown();
}
