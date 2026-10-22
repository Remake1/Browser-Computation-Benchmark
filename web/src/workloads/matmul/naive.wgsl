// Naive matmul: one invocation per element of C, reading A and B straight from
// storage. x is the column, so neighbouring invocations read neighbouring
// elements of B's row k.

struct Params { n: u32 }

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let n = params.n;
  let row = gid.y;
  let col = gid.x;
  if (row >= n || col >= n) {
    return;
  }
  var sum = 0.0;
  for (var k = 0u; k < n; k++) {
    sum += a[row * n + k] * b[k * n + col];
  }
  c[row * n + col] = sum;
}
