// Native timing driver for wasm/src/sort.cpp. Prints the median time and Mkeys/s of
// sort_radix and sort_std per size (the same sweep as web/src/workloads/sort), to
// compare with the browser's compute-phase numbers. Keys are uniform u32, like the
// browser's; every radix result must equal the std::sort result exactly.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

extern "C" {
int sort_init(int threads);
void sort_shutdown();
void sort_radix(std::uint32_t* a, std::uint32_t* tmp, int n);
void sort_std(std::uint32_t* a, int n);
}

namespace {

std::mt19937 gen(1);

std::vector<std::uint32_t> keys(int n) {
  std::vector<std::uint32_t> a(n);
  for (auto& x : a) x = gen();
  return a;
}

bool radix_matches(std::vector<std::uint32_t> a) {
  std::vector<std::uint32_t> ref = a, tmp(a.size());
  std::sort(ref.begin(), ref.end());
  sort_radix(a.data(), tmp.data(), static_cast<int>(a.size()));
  return a == ref;
}

/** Median ms of one sort of a copy of `in`; the copy is untimed, like the browser's upload phase. */
template <class F>
double median_ms(const std::vector<std::uint32_t>& in, F sort) {
  std::vector<std::uint32_t> a(in.size());
  std::vector<double> ms;
  double spent = 0;
  while ((spent < 300 || ms.size() < 5) && ms.size() < 1000) {
    std::copy(in.begin(), in.end(), a.begin());
    const auto t0 = std::chrono::steady_clock::now();
    sort(a.data());
    const double dt = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    ms.push_back(dt);
    spent += dt;
  }
  std::nth_element(ms.begin(), ms.begin() + ms.size() / 2, ms.end());
  return ms[ms.size() / 2];
}

}  // namespace

int main() {
  const int threads = sort_init(std::min(8, static_cast<int>(std::thread::hardware_concurrency())));
  // Tiny and odd lengths: uneven thread splits, chunks with no elements.
  bool odd = true;
  for (int n : {0, 1, 2, 3, 7, 17, 255, 257, 1001, 65537}) odd = odd && radix_matches(keys(n));
  // Few distinct keys: long runs in one bucket.
  std::vector<std::uint32_t> dup = keys(100000);
  for (auto& x : dup) x &= 0x0300000f;
  odd = odd && radix_matches(dup);
  std::printf("threads=%d odd sizes exact: %s\n%9s %12s %10s %12s %10s %6s\n", threads, odd ? "yes" : "NO", "n", "radix_ms", "Mkeys/s",
              "std_ms", "Mkeys/s", "exact");
  for (int n = 1 << 10; n <= 1 << 24; n = n % 3 == 0 ? n / 3 * 4 : n / 2 * 3) {
    const std::vector<std::uint32_t> in = keys(n);
    std::vector<std::uint32_t> tmp(n);
    const bool exact = radix_matches(in);
    const double radix = median_ms(in, [&](std::uint32_t* a) { sort_radix(a, tmp.data(), n); });
    const double stdsort = median_ms(in, [&](std::uint32_t* a) { sort_std(a, n); });
    std::printf("%9d %12.4f %10.1f %12.4f %10.1f %6s\n", n, radix, n / (radix * 1e3), stdsort, n / (stdsort * 1e3), exact ? "yes" : "NO");
  }
  sort_shutdown();
}
