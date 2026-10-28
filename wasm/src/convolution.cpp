// Convolution: W×W f32 image with a fixed 5×5 Gaussian, clamped edges (PLAN.md §7).
//
// Direct 2D loops, the same as the JS implementation: per output row, the five
// source rows are clamped once; the interior columns then sum the 25 taps with no
// clamping, and only the two border columns on each side clamp per tap. Each output
// is summed in the same tap order (row-major over the kernel) everywhere.
//
// scalar   one output at a time.
// simd     the same loops; with the 25 taps fully unrolled, the interior loop over x
//          auto-vectorizes to f32x4 (four outputs per vector, each still summed in
//          tap order, so no float reassociation is needed).
// threads  the simd loops on contiguous blocks of output rows across a persistent pool.
//
// Also compiled natively by native/Makefile as a sanity ceiling for the WASM numbers.
#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

#include <cstddef>

#if defined(__EMSCRIPTEN_PTHREADS__) || defined(NATIVE_THREADS)
#define CONVOLUTION_THREADS 1
#include <memory>

#include "threadpool.hpp"
#endif

namespace {

constexpr int R = 2;          // kernel radius
constexpr int K = 2 * R + 1;  // kernel width

/** Binomial approximation of a Gaussian: [1 4 6 4 1]ᵀ[1 4 6 4 1] / 256, exact in f32. */
constexpr float KERNEL[K * K] = {
    1 / 256.0f, 4 / 256.0f,  6 / 256.0f,  4 / 256.0f,  1 / 256.0f,  //
    4 / 256.0f, 16 / 256.0f, 24 / 256.0f, 16 / 256.0f, 4 / 256.0f,  //
    6 / 256.0f, 24 / 256.0f, 36 / 256.0f, 24 / 256.0f, 6 / 256.0f,  //
    4 / 256.0f, 16 / 256.0f, 24 / 256.0f, 16 / 256.0f, 4 / 256.0f,  //
    1 / 256.0f, 4 / 256.0f,  6 / 256.0f,  4 / 256.0f,  1 / 256.0f,
};

inline int clamp(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; }

/** Output columns [x0, x1), all taps in range (x0 ≥ R, x1 ≤ w − R). */
void interior(const float* __restrict r0, const float* __restrict r1, const float* __restrict r2, const float* __restrict r3,
              const float* __restrict r4, float* __restrict out, int x0, int x1) {
  const float* rows[K] = {r0, r1, r2, r3, r4};
  for (int x = x0; x < x1; ++x) {
    float s = 0.0f;
    for (int ky = 0; ky < K; ++ky)
      for (int kx = 0; kx < K; ++kx) s += KERNEL[ky * K + kx] * rows[ky][x + kx - R];
    out[x] = s;
  }
}

/** Output columns [x0, x1) with each tap's column clamped to [0, w). */
void border(const float* const* rows, float* out, int w, int x0, int x1) {
  for (int x = x0; x < x1; ++x) {
    float s = 0.0f;
    for (int ky = 0; ky < K; ++ky)
      for (int kx = 0; kx < K; ++kx) s += KERNEL[ky * K + kx] * rows[ky][clamp(x + kx - R, 0, w - 1)];
    out[x] = s;
  }
}

/** Output rows [y0, y1). */
void convolve_rows(const float* src, float* dst, int w, int y0, int y1) {
  // [0, x0) and [x1, w) are border columns; both are empty-safe for w < 2R.
  const int x0 = w < R ? w : R;
  const int x1 = w - R > x0 ? w - R : x0;
  for (int y = y0; y < y1; ++y) {
    const float* rows[K];
    for (int ky = 0; ky < K; ++ky) rows[ky] = src + static_cast<std::size_t>(clamp(y + ky - R, 0, w - 1)) * w;
    float* out = dst + static_cast<std::size_t>(y) * w;
    border(rows, out, w, 0, x0);
    interior(rows[0], rows[1], rows[2], rows[3], rows[4], out, x0, x1);
    border(rows, out, w, x1, w);
  }
}

#ifdef CONVOLUTION_THREADS
std::unique_ptr<ThreadPool> pool;
#endif

}  // namespace

extern "C" {

/**
 * Creates the thread pool (threads build); returns the number of threads convolution()
 * will use. Call once before convolution(); calling again replaces the pool.
 */
EMSCRIPTEN_KEEPALIVE int convolution_init(int threads) {
#ifdef CONVOLUTION_THREADS
  pool.reset();
  pool = std::make_unique<ThreadPool>(threads < 1 ? 1 : threads);
  return pool->size();
#else
  (void)threads;
  return 1;
#endif
}

EMSCRIPTEN_KEEPALIVE void convolution_shutdown() {
#ifdef CONVOLUTION_THREADS
  pool.reset();
#endif
}

/** dst = src ∗ KERNEL for a w×w row-major image; src and dst must not overlap. */
EMSCRIPTEN_KEEPALIVE void convolution(const float* src, float* dst, int w) {
#ifdef CONVOLUTION_THREADS
  // Contiguous blocks of output rows; thread t gets rows [t·w/T, (t+1)·w/T).
  const int threads = pool->size();
  auto task = [=](int t) { convolve_rows(src, dst, w, t * w / threads, (t + 1) * w / threads); };
  pool->run(task);
#else
  convolve_rows(src, dst, w, 0, w);
#endif
}

}  // extern "C"
