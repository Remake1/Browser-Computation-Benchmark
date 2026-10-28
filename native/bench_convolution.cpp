// Native timing driver for wasm/src/convolution.cpp. Prints the median kernel time,
// GFLOP/s and GB/s per size (the same sweep as web/src/workloads/convolution), to
// compare with the browser's compute-phase numbers. Pixels are multiples of 1/256
// in [0, 1), like the browser's, so every tap order is exact and the check is equality
// against a per-tap clamped reference.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

extern "C" {
int convolution_init(int threads);
void convolution_shutdown();
void convolution(const float* src, float* dst, int w);
}

namespace {

std::mt19937 gen(1);

std::vector<float> image(int w) {
  std::uniform_int_distribution<int> dist(0, 255);
  std::vector<float> x(static_cast<std::size_t>(w) * w);
  for (auto& v : x) v = dist(gen) / 256.0f;
  return x;
}

/** Runs convolution() once and compares every pixel with an independent f64 reference. */
bool exact(const std::vector<float>& src, int w) {
  static const double b[5] = {1, 4, 6, 4, 1};
  std::vector<float> dst(src.size());
  convolution(src.data(), dst.data(), w);
  for (int y = 0; y < w; ++y)
    for (int x = 0; x < w; ++x) {
      double s = 0;
      for (int ky = 0; ky < 5; ++ky)
        for (int kx = 0; kx < 5; ++kx) {
          const int sy = std::clamp(y + ky - 2, 0, w - 1), sx = std::clamp(x + kx - 2, 0, w - 1);
          s += b[ky] * b[kx] / 256 * src[static_cast<std::size_t>(sy) * w + sx];
        }
      if (dst[static_cast<std::size_t>(y) * w + x] != s) return false;
    }
  return true;
}

}  // namespace

int main() {
  const int threads = convolution_init(std::min(8, static_cast<int>(std::thread::hardware_concurrency())));
  // Sizes smaller than the kernel, odd sizes and ones that split unevenly across threads.
  bool odd_ok = true;
  for (int w : {1, 2, 3, 4, 5, 7, 33, 101, 257}) odd_ok = odd_ok && exact(image(w), w);
  std::printf("threads=%d odd sizes exact: %s\n%6s %12s %10s %10s %6s\n", threads, odd_ok ? "yes" : "NO", "w", "median_ms", "GFLOP/s",
              "GB/s", "exact");
  for (int w = 64; w <= 4096; w = w % 3 == 0 ? w / 3 * 4 : w / 2 * 3) {
    const std::vector<float> src = image(w);
    std::vector<float> dst(src.size());
    const bool ok = exact(src, w);

    // Repeat until ≥ 0.3 s and ≥ 5 runs (capped at 1000); batch small sizes like the browser runner.
    const int batch = std::max(1, (1 << 18) / (w * w));
    std::vector<double> ms;
    double spent = 0;
    while ((spent < 300 || ms.size() < 5) && ms.size() < 1000) {
      const auto t0 = std::chrono::steady_clock::now();
      for (int r = 0; r < batch; ++r) convolution(src.data(), dst.data(), w);
      const double dt = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      ms.push_back(dt / batch);
      spent += dt;
    }
    std::nth_element(ms.begin(), ms.begin() + ms.size() / 2, ms.end());
    const double med = ms[ms.size() / 2];
    const double px = static_cast<double>(w) * w;
    std::printf("%6d %12.4f %10.2f %10.2f %6s\n", w, med, 50 * px / (med * 1e6), 8 * px / (med * 1e6), ok ? "yes" : "NO");
  }
  convolution_shutdown();
}
