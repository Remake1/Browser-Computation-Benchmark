// Naive N-body: one invocation per body i, looping over every body j (including i,
// whose term is 0) straight from storage. All invocations read the same j at the
// same time, so the loads are broadcasts.

struct Params { n: u32 }

@group(0) @binding(0) var<storage, read> bodies: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> acc: array<vec4f>;
@group(0) @binding(2) var<uniform> params: Params;

const EPS2 = 0.01;
const WG = 128u;

@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let n = params.n;
  let i = gid.x;
  if (i >= n) {
    return;
  }
  let pi = bodies[i].xyz;
  var a = vec3f(0.0);
  for (var j = 0u; j < n; j++) {
    let bj = bodies[j];
    let d = bj.xyz - pi;
    let d2 = d.x * d.x + d.y * d.y + d.z * d.z + EPS2;
    let inv = 1.0 / sqrt(d2);
    a += d * (bj.w * inv * inv * inv);
  }
  acc[i] = vec4f(a, 0.0);
}
