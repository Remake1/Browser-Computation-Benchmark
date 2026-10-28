// Tiled N-body: one invocation per body i, as in naive.wgsl, but the workgroup
// stages the bodies j in tiles of WG in workgroup memory (one load per invocation)
// and every invocation then reads the whole tile from there. Each body is read from
// storage once per workgroup instead of once per invocation. The last tile is
// padded with zero-mass bodies, whose terms are exactly 0, so the inner loop has no
// bounds check.

struct Params { n: u32 }

@group(0) @binding(0) var<storage, read> bodies: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> acc: array<vec4f>;
@group(0) @binding(2) var<uniform> params: Params;

const EPS2 = 0.01;
const WG = 128u;
var<workgroup> tile: array<vec4f, WG>;

@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let n = params.n;
  let i = gid.x;
  // No early return: every invocation must reach the barriers. Invocations past n
  // help load tiles and compute a result that is never written.
  let pi = bodies[min(i, n - 1u)].xyz;
  var a = vec3f(0.0);
  for (var base = 0u; base < n; base += WG) {
    let j = base + lid.x;
    var bj = vec4f(0.0);
    if (j < n) {
      bj = bodies[j];
    }
    tile[lid.x] = bj;
    workgroupBarrier();
    for (var k = 0u; k < WG; k++) {
      let b = tile[k];
      let d = b.xyz - pi;
      let d2 = d.x * d.x + d.y * d.y + d.z * d.z + EPS2;
      let inv = 1.0 / sqrt(d2);
      a += d * (b.w * inv * inv * inv);
    }
    workgroupBarrier();
  }
  if (i < n) {
    acc[i] = vec4f(a, 0.0);
  }
}
