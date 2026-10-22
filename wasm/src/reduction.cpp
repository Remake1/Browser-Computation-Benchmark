// Reduction: sum of f32[n] (PLAN.md §7).
//
// scalar   the simple loop: one f32 accumulator, in order, like the JS version.
//          Without -ffast-math the compiler may not reassociate, so it neither
//          vectorizes nor splits the chain: one add latency per element.
// simd     hand-written f32x4 with four vector accumulators (16 lanes), so the
//          adds of consecutive vectors don't wait on each other. Float sums
//          never auto-vectorize without fast-math, hence the intrinsics.
// threads  the simd kernel on contiguous chunks across a persistent pool; the
//          caller adds the per-thread partial sums in thread order.
//
// Also compiled natively by native/Makefile, where NATIVE_SIMD selects the same
// kernel written with clang vector extensions (NEON / SSE instead of simd128).
#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

#include <cstddef>

#if defined(__wasm_simd128__)
#include <wasm_simd128.h>
#define REDUCTION_SIMD 1
#elif defined(NATIVE_SIMD)
#define REDUCTION_SIMD 1
#endif

#if defined(__EMSCRIPTEN_PTHREADS__) || defined(NATIVE_THREADS)
#define REDUCTION_THREADS 1
#include <memory>
#include <vector>

#include "threadpool.hpp"
#endif

namespace {

#ifdef REDUCTION_SIMD

#ifdef __wasm_simd128__
using f32x4 = v128_t;
inline f32x4 zero() { return wasm_f32x4_splat(0.0f); }
inline f32x4 load(const float* p) { return wasm_v128_load(p); }
inline f32x4 add(f32x4 a, f32x4 b) { return wasm_f32x4_add(a, b); }
inline float hsum(f32x4 v) {
  return (wasm_f32x4_extract_lane(v, 0) + wasm_f32x4_extract_lane(v, 1)) +
         (wasm_f32x4_extract_lane(v, 2) + wasm_f32x4_extract_lane(v, 3));
}
#else
typedef float f32x4 __attribute__((vector_size(16)));
inline f32x4 zero() { return f32x4{0.0f, 0.0f, 0.0f, 0.0f}; }
inline f32x4 load(const float* p) {
  f32x4 v;
  __builtin_memcpy(&v, p, sizeof v);  // unaligned load
  return v;
}
inline f32x4 add(f32x4 a, f32x4 b) { return a + b; }
inline float hsum(f32x4 v) { return (v[0] + v[1]) + (v[2] + v[3]); }
#endif

/** Sum of x[0, n): 4 × f32x4 accumulators, then a fixed-order horizontal sum and the scalar tail. */
float sum_range(const float* x, std::size_t n) {
  f32x4 a0 = zero(), a1 = zero(), a2 = zero(), a3 = zero();
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    a0 = add(a0, load(x + i));
    a1 = add(a1, load(x + i + 4));
    a2 = add(a2, load(x + i + 8));
    a3 = add(a3, load(x + i + 12));
  }
  for (; i + 4 <= n; i += 4) a0 = add(a0, load(x + i));
  float s = hsum(add(add(a0, a1), add(a2, a3)));
  for (; i < n; ++i) s += x[i];
  return s;
}

#else

float sum_range(const float* x, std::size_t n) {
  float s = 0.0f;
  for (std::size_t i = 0; i < n; ++i) s += x[i];
  return s;
}

#endif

#ifdef REDUCTION_THREADS
std::unique_ptr<ThreadPool> pool;
std::vector<float> partial;
#endif

}  // namespace

extern "C" {

/**
 * Creates the thread pool (threads build); returns the number of threads reduction()
 * will use. Call once before reduction(); calling again replaces the pool.
 */
EMSCRIPTEN_KEEPALIVE int reduction_init(int threads) {
#ifdef REDUCTION_THREADS
  pool.reset();
  pool = std::make_unique<ThreadPool>(threads < 1 ? 1 : threads);
  partial.assign(pool->size(), 0.0f);
  return pool->size();
#else
  (void)threads;
  return 1;
#endif
}

EMSCRIPTEN_KEEPALIVE void reduction_shutdown() {
#ifdef REDUCTION_THREADS
  pool.reset();
#endif
}

EMSCRIPTEN_KEEPALIVE float reduction(const float* x, int n) {
  const std::size_t len = n < 0 ? 0 : static_cast<std::size_t>(n);
#ifdef REDUCTION_THREADS
  // Contiguous chunks; thread t gets [t·n/T, (t+1)·n/T).
  const std::size_t threads = pool->size();
  float* out = partial.data();
  auto task = [=](int t) {
    const std::size_t i0 = t * len / threads, i1 = (t + 1) * len / threads;
    out[t] = sum_range(x + i0, i1 - i0);
  };
  pool->run(task);
  float s = 0.0f;
  for (std::size_t t = 0; t < threads; ++t) s += out[t];
  return s;
#else
  return sum_range(x, len);
#endif
}

}  // extern "C"
