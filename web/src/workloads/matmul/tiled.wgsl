// Tiled matmul: each 16×16 workgroup computes a 16×16 block of C. Per step it
// stages a 16×16 tile of A and of B in workgroup memory (one element per
// invocation), so each global element is loaded once per workgroup instead of
// 16 times. Out-of-range elements load as 0, which handles N not divisible by 16.

struct Params { n: u32 }

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const TILE: u32 = 16u;
var<workgroup> tileA: array<array<f32, TILE>, TILE>;
var<workgroup> tileB: array<array<f32, TILE>, TILE>;

@compute @workgroup_size(TILE, TILE)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let n = params.n;
  let row = gid.y;
  let col = gid.x;
  var sum = 0.0;
  // No early return: every invocation must reach the barriers.
  let tiles = (n + TILE - 1u) / TILE;
  for (var t = 0u; t < tiles; t++) {
    let ak = t * TILE + lid.x;
    let bk = t * TILE + lid.y;
    var av = 0.0;
    var bv = 0.0;
    if (row < n && ak < n) {
      av = a[row * n + ak];
    }
    if (bk < n && col < n) {
      bv = b[bk * n + col];
    }
    tileA[lid.y][lid.x] = av;
    tileB[lid.y][lid.x] = bv;
    workgroupBarrier();
    for (var k = 0u; k < TILE; k++) {
      sum += tileA[lid.y][k] * tileB[k][lid.x];
    }
    workgroupBarrier();
  }
  if (row < n && col < n) {
    c[row * n + col] = sum;
  }
}
