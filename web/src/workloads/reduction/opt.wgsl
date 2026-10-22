// One pass of the optimized reduction. A fixed grid of at most 1024 workgroups
// strides over the input in vec4 loads, so each invocation sums many elements in
// registers before the single workgroup-level step. The host runs two passes:
// n → (one partial per workgroup) → 1 with a single workgroup, or one pass when
// a single workgroup covers the whole input.
//
// src is bound as vec4s; the host zero-pads it to a multiple of 4 floats.
// workgroup_sum() is appended by the host: opt-subgroups.wgsl when the device has
// the 'subgroups' feature, otherwise opt-tree.wgsl.

@group(0) @binding(0) var<storage, read> src: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;

const WG: u32 = 256u;

@compute @workgroup_size(WG)
fn main(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(num_workgroups) nwg: vec3u,
  @builtin(local_invocation_index) lid: u32,
) {
  let n = arrayLength(&src);
  let stride = nwg.x * WG;
  var acc = vec4f(0.0);
  for (var i = wid.x * WG + lid; i < n; i += stride) {
    acc += src[i];
  }
  let total = workgroup_sum((acc.x + acc.y) + (acc.z + acc.w), lid);
  if (lid == 0u) {
    dst[wid.x] = total;
  }
}
