// One pass of the naive multi-pass reduction: each 256-invocation workgroup loads
// one element per invocation into workgroup memory, tree-reduces it (sequential
// addressing), and writes one partial sum. The host repeats passes, 256 → 1 each
// time, until one value is left (4 passes at n = 2²⁶).
//
// dst's binding size is the number of workgroups. Above the per-dimension dispatch
// limit the host dispatches a 2D grid, which may hold a few spare workgroups.

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;

const WG: u32 = 256u;
var<workgroup> partial: array<f32, WG>;

@compute @workgroup_size(WG)
fn main(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(num_workgroups) nwg: vec3u,
  @builtin(local_invocation_index) lid: u32,
) {
  let group = wid.y * nwg.x + wid.x;
  let i = group * WG + lid;
  var v = 0.0;
  if (i < arrayLength(&src)) {
    v = src[i];
  }
  partial[lid] = v;
  workgroupBarrier();
  for (var stride = WG / 2u; stride > 0u; stride >>= 1u) {
    if (lid < stride) {
      partial[lid] += partial[lid + stride];
    }
    workgroupBarrier();
  }
  // Spare workgroups of a 2D grid must not write: an out-of-range write may be clamped onto a real element.
  if (lid == 0u && group < arrayLength(&dst)) {
    dst[group] = partial[0];
  }
}
