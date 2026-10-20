// Milestone 0 smoke-test module. Built three times (scalar / simd / threads) from
// this one source; each export reports what the build actually got, so the env
// report proves the flags took effect rather than trusting the file name.
#include <emscripten/emscripten.h>

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#endif

#ifndef WASM_VARIANT
#error "WASM_VARIANT must be defined by the build (scalar|simd|threads)"
#endif
#define STR_(x) #x
#define STR(x) STR_(x)

extern "C" {

EMSCRIPTEN_KEEPALIVE const char* hello_variant() { return STR(WASM_VARIANT); }

EMSCRIPTEN_KEEPALIVE int hello_simd_enabled() {
#ifdef __wasm_simd128__
  return 1;
#else
  return 0;
#endif
}

EMSCRIPTEN_KEEPALIVE int hello_threads_enabled() {
#ifdef __EMSCRIPTEN_PTHREADS__
  return 1;
#else
  return 0;
#endif
}

// Float sums don't auto-vectorize without -ffast-math, so the SIMD path is
// hand-written. Four lanes = four accumulators; the scalar path mirrors that
// so both builds produce bit-identical results.
EMSCRIPTEN_KEEPALIVE float hello_sum(const float* x, int n) {
  int i = 0;
#ifdef __wasm_simd128__
  v128_t acc = wasm_f32x4_splat(0.0f);
  for (; i + 4 <= n; i += 4) acc = wasm_f32x4_add(acc, wasm_v128_load(x + i));
  float a0 = wasm_f32x4_extract_lane(acc, 0), a1 = wasm_f32x4_extract_lane(acc, 1);
  float a2 = wasm_f32x4_extract_lane(acc, 2), a3 = wasm_f32x4_extract_lane(acc, 3);
#else
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
  for (; i + 4 <= n; i += 4) {
    a0 += x[i];
    a1 += x[i + 1];
    a2 += x[i + 2];
    a3 += x[i + 3];
  }
#endif
  float s = (a0 + a1) + (a2 + a3);
  for (; i < n; ++i) s += x[i];
  return s;
}

// Spawns `n` threads that each bump a shared counter; returns the final count.
// Without pthreads this runs inline and returns 1. `n` must not exceed
// PTHREAD_POOL_SIZE: joining blocks the calling worker, so a thread that needs a
// freshly spawned Web Worker would never start.
EMSCRIPTEN_KEEPALIVE int hello_parallel(int n) {
  std::atomic<int> counter{0};
#ifdef __EMSCRIPTEN_PTHREADS__
  std::vector<std::thread> threads;
  threads.reserve(n);
  for (int t = 0; t < n; ++t) threads.emplace_back([&counter] { counter.fetch_add(1); });
  for (auto& th : threads) th.join();
#else
  (void)n;
  counter.fetch_add(1);
#endif
  return counter.load();
}

}  // extern "C"
