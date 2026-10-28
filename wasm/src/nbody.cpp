// N-body, one step: the O(n²) softened gravitational acceleration of every body
// (PLAN.md §7). Bodies are vec4 (x, y, z, mass); the output is vec4 (ax, ay, az, 0).
//
//   a_i = Σ_j m_j (p_j − p_i) / (|p_j − p_i|² + ε²)^{3/2},   G = 1
//
// j runs over all bodies including i itself, whose term is exactly 0 thanks to the
// softening, so the inner loop has no branch.
//
// scalar   the simple double loop, the same as the JS version.
// simd     hand-written f32x4 over four bodies i at a time, broadcasting body j.
//          The inner loop sums over j, which doesn't vectorize without
//          -ffast-math (reassociation); vectorizing over i keeps each body's sum in
//          j order, so every lane computes exactly what the scalar loop computes.
// threads  the simd kernel on contiguous blocks of bodies across a persistent pool.
//
// Also compiled natively by native/Makefile, where NATIVE_SIMD selects the same
// kernel written with clang vector extensions (NEON / SSE instead of simd128).
#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

#include <cmath>

#if defined(__wasm_simd128__)
#include <wasm_simd128.h>
#define NBODY_SIMD 1
#elif defined(NATIVE_SIMD)
#define NBODY_SIMD 1
#endif

#if defined(__EMSCRIPTEN_PTHREADS__) || defined(NATIVE_THREADS)
#define NBODY_THREADS 1
#include <memory>

#include "threadpool.hpp"
#endif

namespace {

/** Softening ε², so close pairs (and the self-term) stay finite. */
constexpr float EPS2 = 0.01f;

/** Acceleration of body i, summed over j in order. */
inline void body(const float* b, float* acc, int n, int i) {
  const float xi = b[4 * i], yi = b[4 * i + 1], zi = b[4 * i + 2];
  float ax = 0.0f, ay = 0.0f, az = 0.0f;
  for (int j = 0; j < n; ++j) {
    const float dx = b[4 * j] - xi, dy = b[4 * j + 1] - yi, dz = b[4 * j + 2] - zi;
    const float d2 = dx * dx + dy * dy + dz * dz + EPS2;
    const float inv = 1.0f / std::sqrt(d2);
    const float s = b[4 * j + 3] * inv * inv * inv;
    ax += dx * s;
    ay += dy * s;
    az += dz * s;
  }
  acc[4 * i] = ax;
  acc[4 * i + 1] = ay;
  acc[4 * i + 2] = az;
  acc[4 * i + 3] = 0.0f;
}

#ifdef NBODY_SIMD

#ifdef __wasm_simd128__
using f32x4 = v128_t;
inline f32x4 splat(float v) { return wasm_f32x4_splat(v); }
inline f32x4 splat_load(const float* p) { return wasm_v128_load32_splat(p); }
inline f32x4 make(float a, float b, float c, float d) { return wasm_f32x4_make(a, b, c, d); }
inline f32x4 add(f32x4 a, f32x4 b) { return wasm_f32x4_add(a, b); }
inline f32x4 sub(f32x4 a, f32x4 b) { return wasm_f32x4_sub(a, b); }
inline f32x4 mul(f32x4 a, f32x4 b) { return wasm_f32x4_mul(a, b); }
inline f32x4 div(f32x4 a, f32x4 b) { return wasm_f32x4_div(a, b); }
inline f32x4 sqrt(f32x4 a) { return wasm_f32x4_sqrt(a); }
template <int L>
inline float lane(f32x4 v) { return wasm_f32x4_extract_lane(v, L); }
#else
typedef float f32x4 __attribute__((vector_size(16)));
inline f32x4 splat(float v) { return f32x4{v, v, v, v}; }
inline f32x4 splat_load(const float* p) { return splat(*p); }
inline f32x4 make(float a, float b, float c, float d) { return f32x4{a, b, c, d}; }
inline f32x4 add(f32x4 a, f32x4 b) { return a + b; }
inline f32x4 sub(f32x4 a, f32x4 b) { return a - b; }
inline f32x4 mul(f32x4 a, f32x4 b) { return a * b; }
inline f32x4 div(f32x4 a, f32x4 b) { return a / b; }
inline f32x4 sqrt(f32x4 a) { return __builtin_elementwise_sqrt(a); }
template <int L>
inline float lane(f32x4 v) { return v[L]; }
#endif

template <int L>
inline void store_lane(float* acc, int i, f32x4 ax, f32x4 ay, f32x4 az) {
  acc[4 * (i + L)] = lane<L>(ax);
  acc[4 * (i + L) + 1] = lane<L>(ay);
  acc[4 * (i + L) + 2] = lane<L>(az);
  acc[4 * (i + L) + 3] = 0.0f;
}

/** Accelerations of bodies [i0, i1): four at a time (one per lane), then the scalar tail. */
void bodies(const float* b, float* acc, int n, int i0, int i1) {
  const f32x4 eps2 = splat(EPS2), one = splat(1.0f);
  int i = i0;
  for (; i + 4 <= i1; i += 4) {
    const f32x4 xi = make(b[4 * i], b[4 * i + 4], b[4 * i + 8], b[4 * i + 12]);
    const f32x4 yi = make(b[4 * i + 1], b[4 * i + 5], b[4 * i + 9], b[4 * i + 13]);
    const f32x4 zi = make(b[4 * i + 2], b[4 * i + 6], b[4 * i + 10], b[4 * i + 14]);
    f32x4 ax = splat(0.0f), ay = splat(0.0f), az = splat(0.0f);
    for (int j = 0; j < n; ++j) {
      const float* bj = b + 4 * j;
      const f32x4 dx = sub(splat_load(bj), xi), dy = sub(splat_load(bj + 1), yi), dz = sub(splat_load(bj + 2), zi);
      const f32x4 d2 = add(add(add(mul(dx, dx), mul(dy, dy)), mul(dz, dz)), eps2);
      const f32x4 inv = div(one, sqrt(d2));
      const f32x4 s = mul(mul(mul(splat_load(bj + 3), inv), inv), inv);
      ax = add(ax, mul(dx, s));
      ay = add(ay, mul(dy, s));
      az = add(az, mul(dz, s));
    }
    store_lane<0>(acc, i, ax, ay, az);
    store_lane<1>(acc, i, ax, ay, az);
    store_lane<2>(acc, i, ax, ay, az);
    store_lane<3>(acc, i, ax, ay, az);
  }
  for (; i < i1; ++i) body(b, acc, n, i);
}

#else

void bodies(const float* b, float* acc, int n, int i0, int i1) {
  for (int i = i0; i < i1; ++i) body(b, acc, n, i);
}

#endif

#ifdef NBODY_THREADS
std::unique_ptr<ThreadPool> pool;
#endif

}  // namespace

extern "C" {

/**
 * Creates the thread pool (threads build); returns the number of threads nbody()
 * will use. Call once before nbody(); calling again replaces the pool.
 */
EMSCRIPTEN_KEEPALIVE int nbody_init(int threads) {
#ifdef NBODY_THREADS
  pool.reset();
  pool = std::make_unique<ThreadPool>(threads < 1 ? 1 : threads);
  return pool->size();
#else
  (void)threads;
  return 1;
#endif
}

EMSCRIPTEN_KEEPALIVE void nbody_shutdown() {
#ifdef NBODY_THREADS
  pool.reset();
#endif
}

/** acc[i] = acceleration of body i, for n bodies of 4 floats (x, y, z, mass) each. */
EMSCRIPTEN_KEEPALIVE void nbody(const float* b, float* acc, int n) {
#ifdef NBODY_THREADS
  // Contiguous blocks of bodies; thread t gets [t·n/T, (t+1)·n/T).
  const int threads = pool->size();
  auto task = [=](int t) { bodies(b, acc, n, t * n / threads, (t + 1) * n / threads); };
  pool->run(task);
#else
  bodies(b, acc, n, 0, n);
#endif
}

}  // extern "C"
