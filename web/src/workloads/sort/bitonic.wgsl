// One compare-exchange step (k, j) of a global-memory bitonic sort. The host pads the
// keys to a power of two N with 0xFFFFFFFF, which sorts to the end, and dispatches
// log₂N·(log₂N + 1)/2 steps in one compute pass (300 at N = 2²⁴), each one through
// the whole array: for k = 2, 4, …, N and j = k/2, …, 1.
//
// Invocation t handles the pair (i, i + j), where i is t with a 0 bit inserted at j.
// The pair is sorted ascending if bit k of i is 0, descending otherwise. Above the
// per-dimension dispatch limit the host dispatches a 2D grid.

struct Params {
  k: u32,
  j: u32,
}

@group(0) @binding(0) var<storage, read_write> keys: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;

const WG: u32 = 256u;

@compute @workgroup_size(WG)
fn main(
  @builtin(global_invocation_id) gid: vec3u,
  @builtin(num_workgroups) nwg: vec3u,
) {
  let t = gid.y * nwg.x * WG + gid.x;
  if (t >= arrayLength(&keys) / 2u) {
    return;
  }
  let j = params.j;
  let i = ((t & ~(j - 1u)) << 1u) | (t & (j - 1u));
  let a = keys[i];
  let b = keys[i + j];
  if ((a > b) == ((i & params.k) == 0u)) {
    keys[i] = b;
    keys[i + j] = a;
  }
}
