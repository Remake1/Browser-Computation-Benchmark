// Blocked wavefront of the edit-distance DP: one workgroup per T × T tile, one
// dispatch per anti-diagonal of tiles (k = r + c), la/T + lb/T − 1 dispatches in one
// compute pass (255 at la = lb = 32768) instead of one per anti-diagonal of cells.
//
// Tiles pass their boundaries like ed_tiled in wasm/src/editdistance.cpp: rows[r]
// holds D[min(r·T, la)][0..lb] and cols[c] holds D[0..la][min(c·T, lb)]. Tile (r, c)
// reads rows[r][j0..j0+w] and cols[c][i0+1..i0+h] and writes rows[r+1][j0+1..j0+w]
// and cols[c+1][i0+1..i0+h], so the tiles of one dispatch never touch the same entry.
// The edges (rows[0], cols[0], index 0 of every row and column) are written by the
// host once, and rows[nr] is the last row.
//
// Inside a tile, invocation t owns tile row i = t + 1 and computes its cells left to
// right, cell (i, j) at step s = t + j − 1: a wavefront with one workgroupBarrier per
// step (h + w − 1 steps). It keeps D[i][j−1] (left) and D[i−1][j−1] (diag) in
// registers and reads D[i−1][j] (up) from invocation t − 1's result of the step
// before, in a double-buffered workgroup array; invocation 0 reads up from rows[r].

const T: u32 = 256u;

struct Params {
  la: u32,
  lb: u32,
  /** Anti-diagonal of tiles. */
  k: u32,
  /** First tile row on it; workgroup x handles tile (r0 + x, k − r0 − x). */
  r0: u32,
}

/** Characters 0–3, packed four per u32 (little-endian bytes). */
@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read_write> rows: array<u32>;
@group(0) @binding(3) var<storage, read_write> cols: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;

/** The tile's slice of b. */
var<workgroup> tb: array<u32, T>;
/** Each invocation's latest cell, by step parity. */
var<workgroup> last: array<array<u32, T>, 2>;

fn char_a(k: u32) -> u32 {
  return (a[k >> 2u] >> ((k & 3u) * 8u)) & 0xffu;
}

fn char_b(k: u32) -> u32 {
  return (b[k >> 2u] >> ((k & 3u) * 8u)) & 0xffu;
}

@compute @workgroup_size(T)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let la = params.la;
  let lb = params.lb;
  let r = params.r0 + wg.x;
  let c = params.k - r;
  let i0 = r * T;
  let j0 = c * T;
  let h = min(T, la - i0);
  let w = min(T, lb - j0);
  let top = r * (lb + 1u) + j0; // rows[r][j0]
  let bottom = (r + 1u) * (lb + 1u) + j0;
  let left = c * (la + 1u) + i0; // cols[c][i0]
  let right = (c + 1u) * (la + 1u) + i0;

  if (t < w) {
    tb[t] = char_b(j0 + t);
  }
  let owns_row = t < h;
  var ai = 0u;
  var cell = 0u; // D[i][j − 1]
  var diag = 0u; // D[i − 1][j − 1]
  if (owns_row) {
    ai = char_a(i0 + t);
    cell = cols[left + t + 1u];
    diag = select(cols[left + t], rows[top], t == 0u);
  }
  workgroupBarrier();

  for (var s = 0u; s < h + w - 1u; s++) {
    if (owns_row && s >= t && s - t < w) {
      let j = s - t + 1u;
      var up: u32;
      if (t == 0u) {
        up = rows[top + j];
      } else {
        up = last[(s + 1u) & 1u][t - 1u];
      }
      cell = min(min(diag + select(1u, 0u, ai == tb[j - 1u]), up + 1u), cell + 1u);
      diag = up;
      last[s & 1u][t] = cell;
      if (t == h - 1u) {
        rows[bottom + j] = cell;
      }
      if (j == w) {
        cols[right + t + 1u] = cell;
      }
    }
    workgroupBarrier();
  }
}
