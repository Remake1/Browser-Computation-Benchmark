// Matrix multiply C = A·B, square N×N, row-major f32 (PLAN.md §7).
//
// i-k-j loop order, the same as the JS implementation: the inner loop streams a
// row of B into a row of C, which is cache-friendly and vectorizes without
// -ffast-math (each C element is still summed over k in order, so no float
// reassociation is needed). The simd build gets f32x4 from the auto-vectorizer;
// the threads build splits rows of C across a persistent pool.
//
// Also compiled natively by native/Makefile as a sanity ceiling for the WASM numbers.
#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

#include <cstddef>

#if defined(__EMSCRIPTEN_PTHREADS__) || defined(NATIVE_THREADS)
#define MATMUL_THREADS 1
#include <memory>

#include "threadpool.hpp"
#endif

namespace {

/** Rows [i0, i1) of C. */
void matmul_rows(const float* __restrict a, const float* __restrict b, float* __restrict c, int n, int i0, int i1) {
  for (int i = i0; i < i1; ++i) {
    float* __restrict ci = c + static_cast<std::size_t>(i) * n;
    const float* __restrict ai = a + static_cast<std::size_t>(i) * n;
    for (int j = 0; j < n; ++j) ci[j] = 0.0f;
    for (int k = 0; k < n; ++k) {
      const float aik = ai[k];
      const float* __restrict bk = b + static_cast<std::size_t>(k) * n;
      for (int j = 0; j < n; ++j) ci[j] += aik * bk[j];
    }
  }
}

#ifdef MATMUL_THREADS
std::unique_ptr<ThreadPool> pool;
#endif

}  // namespace

extern "C" {

/**
 * Creates the thread pool (threads build); returns the number of threads matmul()
 * will use. Call once before matmul(); calling again replaces the pool.
 */
EMSCRIPTEN_KEEPALIVE int matmul_init(int threads) {
#ifdef MATMUL_THREADS
  pool.reset();
  pool = std::make_unique<ThreadPool>(threads < 1 ? 1 : threads);
  return pool->size();
#else
  (void)threads;
  return 1;
#endif
}

EMSCRIPTEN_KEEPALIVE void matmul_shutdown() {
#ifdef MATMUL_THREADS
  pool.reset();
#endif
}

EMSCRIPTEN_KEEPALIVE void matmul(const float* a, const float* b, float* c, int n) {
#ifdef MATMUL_THREADS
  // Contiguous blocks of rows; thread t gets rows [t·n/T, (t+1)·n/T).
  const int threads = pool->size();
  auto task = [=](int t) { matmul_rows(a, b, c, n, t * n / threads, (t + 1) * n / threads); };
  pool->run(task);
#else
  matmul_rows(a, b, c, n, 0, n);
#endif
}

}  // extern "C"
