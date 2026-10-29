// Native timing driver for wasm/src/editdistance.cpp. Prints the median time and
// Mcells/s of ed_rows and ed_tiled per size (the same sweep as
// web/src/workloads/editdistance), to compare with the browser's compute-phase
// numbers. Strings are uniform over {A, C, G, T} (codes 0–3), like the browser's.
// Both functions' last rows must equal a full-matrix DP's exactly.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

extern "C" {
int ed_init(int threads);
void ed_shutdown();
int ed_scratch_elements(int la, int lb);
void ed_rows(const std::uint8_t* a, int la, const std::uint8_t* b, int lb, std::uint16_t* scratch, std::uint16_t* out);
void ed_tiled(const std::uint8_t* a, int la, const std::uint8_t* b, int lb, std::uint16_t* scratch, std::uint16_t* out);
}

namespace {

using Fn = void (*)(const std::uint8_t*, int, const std::uint8_t*, int, std::uint16_t*, std::uint16_t*);

std::mt19937 gen(1);

std::vector<std::uint8_t> dna(int n) {
  std::vector<std::uint8_t> s(n);
  for (auto& c : s) c = gen() & 3;
  return s;
}

/** Last row of the full DP matrix, written independently of editdistance.cpp. */
std::vector<std::uint16_t> reference(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b) {
  const std::size_t la = a.size(), lb = b.size();
  std::vector<int> d((la + 1) * (lb + 1));
  auto at = [&](std::size_t i, std::size_t j) -> int& { return d[i * (lb + 1) + j]; };
  for (std::size_t i = 0; i <= la; ++i) at(i, 0) = static_cast<int>(i);
  for (std::size_t j = 0; j <= lb; ++j) at(0, j) = static_cast<int>(j);
  for (std::size_t i = 1; i <= la; ++i)
    for (std::size_t j = 1; j <= lb; ++j) at(i, j) = std::min({at(i - 1, j - 1) + (a[i - 1] != b[j - 1]), at(i - 1, j) + 1, at(i, j - 1) + 1});
  return {d.begin() + la * (lb + 1), d.end()};
}

std::vector<std::uint16_t> last_row(Fn f, const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b) {
  const int la = static_cast<int>(a.size()), lb = static_cast<int>(b.size());
  std::vector<std::uint16_t> scratch(ed_scratch_elements(la, lb)), out(lb + 1);
  f(a.data(), la, b.data(), lb, scratch.data(), out.data());
  return out;
}

bool matches(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b) {
  const auto ref = reference(a, b);
  return last_row(ed_rows, a, b) == ref && last_row(ed_tiled, a, b) == ref;
}

/** Median ms of one call. */
double median_ms(Fn f, const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b) {
  const int la = static_cast<int>(a.size()), lb = static_cast<int>(b.size());
  std::vector<std::uint16_t> scratch(ed_scratch_elements(la, lb)), out(lb + 1);
  std::vector<double> ms;
  double spent = 0;
  while ((spent < 300 || ms.size() < 5) && ms.size() < 1000) {
    const auto t0 = std::chrono::steady_clock::now();
    f(a.data(), la, b.data(), lb, scratch.data(), out.data());
    const double dt = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    ms.push_back(dt);
    spent += dt;
  }
  std::nth_element(ms.begin(), ms.begin() + ms.size() / 2, ms.end());
  return ms[ms.size() / 2];
}

}  // namespace

int main() {
  const int threads = ed_init(std::min(8, static_cast<int>(std::thread::hardware_concurrency())));
  // Odd and unequal lengths: partial tiles, tile-edge sizes, a single row or column.
  bool odd = true;
  const int lengths[] = {1, 2, 3, 7, 17, 255, 256, 257, 300, 511, 513, 1000};
  for (int la : lengths)
    for (int lb : lengths) odd = odd && matches(dna(la), dna(lb));
  // Similar strings (small distances) and identical ones.
  std::vector<std::uint8_t> a = dna(2000), b = a;
  for (int k = 0; k < 100; ++k) b[gen() % b.size()] = gen() & 3;
  odd = odd && matches(a, b) && matches(a, a);
  std::printf("threads=%d odd sizes exact: %s\n%7s %12s %10s %12s %10s %6s\n", threads, odd ? "yes" : "NO", "L", "rows_ms", "Mcells/s", "tiled_ms",
              "Mcells/s", "exact");
  for (int n = 64; n <= 32768; n = n % 3 == 0 ? n / 3 * 4 : n / 2 * 3) {
    const auto x = dna(n), y = dna(n);
    // The full-matrix reference needs 4n² bytes; above 8192 compare the two functions.
    const bool exact = n <= 8192 ? matches(x, y) : last_row(ed_rows, x, y) == last_row(ed_tiled, x, y);
    const double rows = median_ms(ed_rows, x, y);
    const double tiled = median_ms(ed_tiled, x, y);
    const double cells = static_cast<double>(n) * n;
    std::printf("%7d %12.4f %10.1f %12.4f %10.1f %6s\n", n, rows, cells / (rows * 1e3), tiled, cells / (tiled * 1e3), exact ? "yes" : "NO");
  }
  ed_shutdown();
}
