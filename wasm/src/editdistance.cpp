// Edit distance (Levenshtein, unit costs) between a[0, la) and b[0, lb) (PLAN.md §7).
// Every function writes the last row of the DP matrix,
//
//   D[i][j] = min(D[i−1][j−1] + (a[i−1] ≠ b[j−1]), D[i−1][j] + 1, D[i][j−1] + 1),
//   D[i][0] = i,  D[0][j] = j,
//
// out[j] = D[la][j] for j in [0, lb]: the distance from a to every prefix of b, so
// out[lb] is the edit distance. Cells are u16, so la, lb ≤ 65535.
//
// ed_rows   the two-row DP, row by row, the same loop as the JS baseline. Each cell
//           depends on its left neighbour, so the row loop has no vector form.
// ed_tiled  blocked anti-diagonal wavefront. The matrix is cut into TILE × TILE tiles;
//           tiles on one anti-diagonal of tiles are independent. Inside a tile the
//           cells of one anti-diagonal are independent too, and that loop
//           auto-vectorizes (simd build: i16x8, 8 cells per instruction).
//   scalar / simd  tiles one after another, in wavefront order.
//   threads   the tiles of each tile anti-diagonal split across the persistent pool;
//           pool->run() returning is the barrier between tile anti-diagonals.
//
// Tiles pass their boundaries through `scratch` (ed_scratch_elements()): rows[r] holds
// D[min(r·TILE, la)][0..lb] and cols[c] holds D[0..la][min(c·TILE, lb)]. Tile (r, c)
// reads rows[r][j0..j1] and cols[c][i0+1..i1] and writes rows[r+1][j0+1..j1] and
// cols[c+1][i0+1..i1], so tiles on the same anti-diagonal never touch the same entry.
// rows[nr] is the last row.
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
#include <vector>

#if defined(__EMSCRIPTEN_PTHREADS__) || defined(NATIVE_THREADS)
#define ED_THREADS 1
#include <memory>

#include "threadpool.hpp"
#endif

namespace {

using u8 = std::uint8_t;
using u16 = std::uint16_t;

constexpr int TILE = 512;

/** One thread's tile workspace: three anti-diagonals and the tile's slice of b, reversed. */
struct Workspace {
  u16 diag[3][TILE + 1];
  u8 brev[TILE];
};

std::vector<Workspace> workspaces(1);

#ifdef ED_THREADS
std::unique_ptr<ThreadPool> pool;
#endif

int tiles(int n) { return (n + TILE - 1) / TILE; }

/** Interior cells i in [lo, hi] of anti-diagonal d (j = d − i), from the two before it. */
void diagonal(const u16* __restrict pp, const u16* __restrict p, u16* __restrict cur, const u8* __restrict a, const u8* __restrict brev, int lo,
              int hi, int off) {
  for (int i = lo; i <= hi; ++i) {
    // b[j − 1] = brev[w − j] = brev[off + i], so both strings are read ascending.
    const unsigned sub = pp[i - 1] + (a[i - 1] != brev[off + i]);
    const unsigned gap = std::min(p[i - 1], p[i]) + 1u;
    cur[i] = static_cast<u16>(std::min(sub, gap));
  }
}

/** Tile (r, c) of the wavefront: its cells anti-diagonal by anti-diagonal, then its boundaries. */
void tile(const u8* a, int la, const u8* b, int lb, u16* rows, u16* cols, int r, int c, Workspace& ws) {
  const int i0 = r * TILE, j0 = c * TILE;
  const int h = std::min(TILE, la - i0), w = std::min(TILE, lb - j0);
  const u16* top = rows + static_cast<std::size_t>(r) * (lb + 1) + j0;  // D[i0][j0 + j]
  const u16* left = cols + static_cast<std::size_t>(c) * (la + 1) + i0;  // D[i0 + i][j0]
  u16* bottom = rows + static_cast<std::size_t>(r + 1) * (lb + 1) + j0;
  u16* right = cols + static_cast<std::size_t>(c + 1) * (la + 1) + i0;
  for (int k = 0; k < w; ++k) ws.brev[k] = b[j0 + w - 1 - k];
  u16* pp = ws.diag[0];
  u16* p = ws.diag[1];
  u16* cur = ws.diag[2];
  for (int d = 0; d <= h + w; ++d) {
    diagonal(pp, p, cur, a + i0, ws.brev, std::max(1, d - w), std::min(h, d - 1), w - d);
    if (d <= w) cur[0] = top[d];
    if (d <= h) cur[d] = left[d];
    if (d > h) bottom[d - h] = cur[h];
    if (d > w) right[d - w] = cur[d - w];
    u16* t = pp;
    pp = p;
    p = cur;
    cur = t;
  }
}

}  // namespace

extern "C" {

/**
 * Creates the thread pool (threads build); returns the number of threads ed_tiled()
 * will use. Call once before ed_tiled(); calling again replaces the pool.
 */
EMSCRIPTEN_KEEPALIVE int ed_init(int threads) {
#ifdef ED_THREADS
  pool.reset();
  pool = std::make_unique<ThreadPool>(threads < 1 ? 1 : threads);
  workspaces.resize(pool->size());
  return pool->size();
#else
  (void)threads;
  return 1;
#endif
}

EMSCRIPTEN_KEEPALIVE void ed_shutdown() {
#ifdef ED_THREADS
  pool.reset();
#endif
}

/** u16 elements of scratch that ed_rows() and ed_tiled() need for these lengths. */
EMSCRIPTEN_KEEPALIVE int ed_scratch_elements(int la, int lb) {
  const int tiled = (tiles(la) + 1) * (lb + 1) + (tiles(lb) + 1) * (la + 1);
  return std::max(lb + 1, tiled);
}

/** Two-row DP; out and scratch hold lb + 1 elements each. */
EMSCRIPTEN_KEEPALIVE void ed_rows(const u8* a, int la, const u8* b, int lb, u16* scratch, u16* out) {
  // Rows alternate between the buffers; start so that row la lands in out.
  u16* prev = la % 2 ? scratch : out;
  u16* cur = la % 2 ? out : scratch;
  for (int j = 0; j <= lb; ++j) prev[j] = static_cast<u16>(j);
  for (int i = 1; i <= la; ++i) {
    const u8 ai = a[i - 1];
    unsigned diag = prev[0];
    unsigned left = static_cast<unsigned>(i);
    cur[0] = static_cast<u16>(i);
    for (int j = 1; j <= lb; ++j) {
      const unsigned up = prev[j];
      left = std::min(std::min(diag + (ai != b[j - 1]), up + 1), left + 1);
      cur[j] = static_cast<u16>(left);
      diag = up;
    }
    std::swap(prev, cur);
  }
}

/** Blocked anti-diagonal wavefront; scratch holds ed_scratch_elements(la, lb), out lb + 1 elements. */
EMSCRIPTEN_KEEPALIVE void ed_tiled(const u8* a, int la, const u8* b, int lb, u16* scratch, u16* out) {
  const int nr = tiles(la), nc = tiles(lb);
  u16* rows = scratch;
  u16* cols = scratch + static_cast<std::size_t>(nr + 1) * (lb + 1);
  // The matrix edges: D[0][j] = j, D[i][0] = i.
  for (int j = 0; j <= lb; ++j) rows[j] = static_cast<u16>(j);
  for (int r = 1; r <= nr; ++r) rows[static_cast<std::size_t>(r) * (lb + 1)] = static_cast<u16>(std::min(r * TILE, la));
  for (int i = 0; i <= la; ++i) cols[i] = static_cast<u16>(i);
  for (int c = 1; c <= nc; ++c) cols[static_cast<std::size_t>(c) * (la + 1)] = static_cast<u16>(std::min(c * TILE, lb));

  for (int k = 0; k < nr + nc - 1; ++k) {
    // Tiles (r, k − r) for r in [r0, r1].
    const int r0 = std::max(0, k - nc + 1), r1 = std::min(k, nr - 1);
#ifdef ED_THREADS
    const int threads = pool->size();
    auto task = [=](int t) {
      for (int r = r0 + t; r <= r1; r += threads) tile(a, la, b, lb, rows, cols, r, k - r, workspaces[t]);
    };
    pool->run(task);
#else
    for (int r = r0; r <= r1; ++r) tile(a, la, b, lb, rows, cols, r, k - r, workspaces[0]);
#endif
  }
  std::copy(rows + static_cast<std::size_t>(nr) * (lb + 1), rows + static_cast<std::size_t>(nr + 1) * (lb + 1), out);
}

}  // extern "C"
