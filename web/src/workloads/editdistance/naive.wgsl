// One anti-diagonal d of the edit-distance DP (cells i + j = d), one invocation per
// cell. The host dispatches d = 0, 1, …, la + lb in one compute pass (65,537
// dispatches at la = lb = 32768), each depending on the two before it:
//
//   D[i][j] = min(D[i−1][j−1] + (a[i−1] ≠ b[j−1]), D[i−1][j] + 1, D[i][j−1] + 1)
//
// `diags` holds three anti-diagonals of la + 1 cells, indexed by i, rotating d mod 3.
// Invocation g handles i = lo + g; the edges D[0][d] and D[d][0] are d. Cells of the
// last row (i = la) also go to `out[j]`.

struct Params {
  d: u32,
  /** max(0, d − lb): the first i on this anti-diagonal. */
  lo: u32,
  la: u32,
  lb: u32,
}

/** Characters 0–3, packed four per u32 (little-endian bytes). */
@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read_write> diags: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;

fn char_a(k: u32) -> u32 {
  return (a[k >> 2u] >> ((k & 3u) * 8u)) & 0xffu;
}

fn char_b(k: u32) -> u32 {
  return (b[k >> 2u] >> ((k & 3u) * 8u)) & 0xffu;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let d = params.d;
  let la = params.la;
  let i = params.lo + gid.x;
  if (i > min(la, d)) {
    return;
  }
  let j = d - i;
  let n = la + 1u;
  let cur = (d % 3u) * n;
  var v = d;
  if (i > 0u && j > 0u) {
    let prev = ((d + 2u) % 3u) * n; // d − 1
    let pprev = ((d + 1u) % 3u) * n; // d − 2
    let sub = diags[pprev + i - 1u] + select(1u, 0u, char_a(i - 1u) == char_b(j - 1u));
    v = min(min(sub, diags[prev + i - 1u] + 1u), diags[prev + i] + 1u);
  }
  diags[cur + i] = v;
  if (i == la) {
    out[j] = v;
  }
}
