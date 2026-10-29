// Sort: u32[n] ascending, in place (PLAN.md §7).
//
// sort_radix  LSD radix sort, 8-bit digits, 4 passes ping-ponging between the array
//             and a caller-provided scratch buffer of n, so the result ends up back
//             in the array. The same algorithm and loop order as the js-radix
//             baseline: one counting pass builds all four digit histograms, then
//             each pass takes an exclusive prefix sum and scatters stably.
//   scalar / simd  that loop. Radix sort has no vector form in simd128 (no gather or
//             scatter, and the histogram increments conflict), so the simd build is
//             the same code compiled with -msimd128: the single-thread baseline of
//             the threads build, which uses the same flags.
//   threads   parallel LSD over contiguous chunks of the persistent pool: per pass,
//             every thread counts its chunk, the caller turns the per-thread counts
//             into offsets in (digit, thread) order, and every thread scatters its
//             chunk. Still stable, so the output is identical. The chunk histograms
//             change after every pass, so each pass counts again instead of one
//             counting pass for all four digits.
// sort_std    std::sort (libc++ introsort), the C++ library sort of PLAN.md §7.
//
// Also compiled natively by native/Makefile.
#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>

#if defined(__EMSCRIPTEN_PTHREADS__) || defined(NATIVE_THREADS)
#define SORT_THREADS 1
#include <memory>
#include <vector>

#include "threadpool.hpp"
#endif

namespace {

constexpr int RADIX = 256;
constexpr int PASSES = 4;

#ifdef SORT_THREADS

/** One thread's counts, then offsets, for the current pass; a whole number of cache lines. */
struct alignas(64) Histogram {
  std::uint32_t count[RADIX];
};

std::unique_ptr<ThreadPool> pool;
std::vector<Histogram> hist;

void radix_sort(std::uint32_t* a, std::uint32_t* tmp, std::size_t n) {
  const std::size_t threads = pool->size();
  Histogram* h = hist.data();
  std::uint32_t* src = a;
  std::uint32_t* dst = tmp;
  for (int pass = 0; pass < PASSES; ++pass) {
    const int shift = 8 * pass;
    auto count = [=](int t) {
      const std::size_t i0 = t * n / threads, i1 = (t + 1) * n / threads;
      std::uint32_t* c = h[t].count;
      std::fill(c, c + RADIX, 0u);
      for (std::size_t i = i0; i < i1; ++i) ++c[(src[i] >> shift) & 0xff];
    };
    pool->run(count);
    // Digit-major, thread-minor: thread t's elements of digit d follow those of threads < t.
    std::uint32_t sum = 0;
    for (int d = 0; d < RADIX; ++d) {
      for (std::size_t t = 0; t < threads; ++t) {
        const std::uint32_t c = h[t].count[d];
        h[t].count[d] = sum;
        sum += c;
      }
    }
    auto scatter = [=](int t) {
      const std::size_t i0 = t * n / threads, i1 = (t + 1) * n / threads;
      std::uint32_t* offset = h[t].count;
      for (std::size_t i = i0; i < i1; ++i) {
        const std::uint32_t x = src[i];
        dst[offset[(x >> shift) & 0xff]++] = x;
      }
    };
    pool->run(scatter);
    std::swap(src, dst);
  }
}

#else

void radix_sort(std::uint32_t* a, std::uint32_t* tmp, std::size_t n) {
  std::uint32_t count[PASSES][RADIX] = {};
  for (std::size_t i = 0; i < n; ++i) {
    const std::uint32_t x = a[i];
    ++count[0][x & 0xff];
    ++count[1][(x >> 8) & 0xff];
    ++count[2][(x >> 16) & 0xff];
    ++count[3][x >> 24];
  }
  std::uint32_t* src = a;
  std::uint32_t* dst = tmp;
  for (int pass = 0; pass < PASSES; ++pass) {
    const int shift = 8 * pass;
    std::uint32_t* offset = count[pass];
    std::uint32_t sum = 0;
    for (int d = 0; d < RADIX; ++d) {
      const std::uint32_t c = offset[d];
      offset[d] = sum;
      sum += c;
    }
    for (std::size_t i = 0; i < n; ++i) {
      const std::uint32_t x = src[i];
      dst[offset[(x >> shift) & 0xff]++] = x;
    }
    std::swap(src, dst);
  }
}

#endif

}  // namespace

extern "C" {

/**
 * Creates the thread pool (threads build); returns the number of threads sort_radix()
 * will use. Call once before sorting; calling again replaces the pool.
 */
EMSCRIPTEN_KEEPALIVE int sort_init(int threads) {
#ifdef SORT_THREADS
  pool.reset();
  pool = std::make_unique<ThreadPool>(threads < 1 ? 1 : threads);
  hist.assign(pool->size(), Histogram{});
  return pool->size();
#else
  (void)threads;
  return 1;
#endif
}

EMSCRIPTEN_KEEPALIVE void sort_shutdown() {
#ifdef SORT_THREADS
  pool.reset();
#endif
}

/** Sorts a[0, n) in place; tmp must hold n elements and is overwritten. */
EMSCRIPTEN_KEEPALIVE void sort_radix(std::uint32_t* a, std::uint32_t* tmp, int n) {
  radix_sort(a, tmp, n < 0 ? 0 : static_cast<std::size_t>(n));
}

/** Sorts a[0, n) in place with std::sort (single-threaded in every build). */
EMSCRIPTEN_KEEPALIVE void sort_std(std::uint32_t* a, int n) {
  std::sort(a, a + (n < 0 ? 0 : n));
}

}  // extern "C"
